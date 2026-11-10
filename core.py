#!/usr/bin/env python3
"""Artefact: a 1D radiative transfer model for optically thin dust discs.

Calculates the thermal emission and scattered light from a dust disc around a
star in the optically thin limit, for several radial surface density
distributions and a power-law grain size distribution.
"""

import json
import warnings
from dataclasses import dataclass

import astropy.constants as const
import miepython as mpy
import numpy as np
from numba import njit
from scipy.optimize import brentq, curve_fit

from .fitting import Photometer
from .image import (
    ASPECT_RATIO,
    HG_G1,
    HG_G2,
    HG_WEIGHT,
    MIN_ASPECT,
    MIN_INC,
    ModelImage,
    convolve_beam,
    disc_surface_brightness,
    two_component_hg,
)
from .phoenix import FEH_PRIOR, LOGG_PRIOR, fit_star, load_grid


# Physical constants (SI units)
H_PLANCK = const.h.value  # 6.626e-34 J s
C_LIGHT = const.c.value  # 299792458.0 m / s
K_B = const.k_B.value  # 1.38e-23 J / K
SIGMA_SB = const.sigma_sb.value  # 5.67e-8 W / K^4 / m^2
AU = const.au.value  # 1.495978707e11 m
PC = const.pc.value  # 3.0857e16 m
L_SUN = const.L_sun.value  # 3.828e26 W
R_SUN = const.R_sun.value  # 6.96342e8 m
M_EARTH = const.M_earth.value  # 5.97237e24 kg
UM = 1e-6  # metres per micron, for wavelengths in microns

# Unit conversions and model constants
MJY_PER_SI = 1e3 * 1e26  # 1 W/m2/Hz expressed in mJy
SI_PER_MJY = 1e-26 * 1e-3  # 1 mJy expressed in W/m2/Hz
CGS_FLAM_TO_SI = 1e-7 * 1e4 * 1e10  # erg/s/cm2/A -> W/m2/m
G_CM3_TO_KG_M3 = 1e3  # grain density: g/cm3 -> kg/m3
FWHM_PER_SIGMA = np.sqrt(8.0 * np.log(2.0))
FWHM_TO_SIGMA  = 1.0 / np.sqrt(8.0 * np.log(2.0))
T_CMB = 2.73  # K, lower bound of the dust temperature search
T_BB_1AU = 278.0  # K, blackbody temperature at 1 au from a 1 L_sun star
SCALE_WAVE_MIN = 0.44  # um, lower limit of photometry used by scale_star
SCALE_WAVE_MAX = 5.0  # um, upper limit of photometry used by scale_star

# Three-slope size distribution of Krivov & Wyatt (2021, MNRAS 500, 718), sizes in um
S_MM = 1e3  # 1 mm, upper size of the "dust" that the sub-mm flux measures
S_KM = 1e9  # 1 km, transition from the collisional cascade to primordial slope
S_MAX = 2e11  # 200 km, largest planetesimal
Q_MED = 3.7  # slope of the collisional cascade, s_mm < s < s_km
Q_BIG = 2.8  # slope of the primordial planetesimals, s_km < s < s_max

# Radiation pressure: grains whose radiation-to-gravity force ratio beta exceeds 0.5
# are unbound when released from a parent body on a circular orbit.
BETA_BLOWOUT = 0.5
DEFAULT_NRING = 30  # annuli per belt in a multi-belt disc, if a belt gives none
BELT_TYPES = ("gauss", "onepl", "twopl", "arbit")  # shapes of a single belt

@njit(cache=True)
def planck_lam(wav, T):
    """
    Calculate Planck function in wavelength units.

    Parameters
    ----------
    wav : float or array
        Wavelengths in metres.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    intensity : float or array
        B(wav,T) W/m^2/m/sr
    """
    a = 2.0 * H_PLANCK * C_LIGHT**2
    b = H_PLANCK * C_LIGHT / (wav * K_B * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity


@njit(cache=True)
def planck_nu(freq, T):
    """
    Calculate Planck function in frequency units

    Parameters
    ----------
    freq : float or array
        Frequencies in Hertz.
    T : float
        Temperature in Kelvin.

    Returns
    -------
    intensity : float or array
        B(nu,T) W/m^2/Hz/sr
    """
    a = 2.0 * H_PLANCK * freq**3
    b = H_PLANCK * freq / (K_B * T)
    intensity = a / ((C_LIGHT**2) * (np.exp(b) - 1.0))
    return intensity


@njit(cache=True)
def modified_blackbody(wav, T, lam0, beta):
    """
    Calculate modified blackbody defined as 
    I = B(lam,T) 
    before the break wavelength lam0 and 
    I = B(lam,T) * (lam0/lam)**beta 
    beyond the break wavelength lam0.

    Parameters
    ----------
    wav : float or array
        Wavelengths in microns.
    T : float
        Temperature in Kelvin.
    lam0 : float
        Break wavelength in microns.
    beta : float
        Exponent of modified emission beyond lam0.
    
    Returns
    -------
    intensity : float or array
        I(wav,T) W/m^2/m/sr
    """

    intensity = planck_lam(wav*UM, T)

    intensity[np.where(wav > lam0)] *= (lam0/wav)**beta

    return intensity


def calculate_tbb(rdust,lstar):
    """
    Calculate blackbody temperature for dust (in Kelvin) at a distance Rbb 
    (in au) from a star of luminosity Lstar (in Lsol).
    
    Parameters
    ----------
    rdust : float or array
        Radius of dust in au.
    lstar : float
        Luminosity of star in Lsol.
    
    Returns
    -------
    tbb : float
        Tbb K

    """

    tbb = T_BB_1AU * (lstar**0.25) * (rdust ** (-0.5))

    return tbb


def calculate_rbb(tdust,lstar):
    """
    Calculate blackbody radius for dust (in au) at a blackbody temperature 
    Tbb (in Kelvin) around a star of luminosity Lstar (in Lsol).
    
    Parameters
    ----------
    tdust : float or array
        Temperature of dust in Kelvin.
    lstar : float
        Luminosity of star in Lsol.
    
    Returns
    -------
    rbb : float
        Rbb au
    """

    rbb = (lstar**0.5) * ((tdust / T_BB_1AU) ** (-2.0))

    return rbb


def calculate_rre(lstar):
    """
    Calculate estimated radius for dust (in au) around a star of luminosity 
    Lstar (in Lsol) using the Rd-Ls relation inferred from REASONS (Matra+2025).
    
    Parameters
    ----------
    lstar : float
        Luminosity of star in Lsol.
    
    Returns
    -------
    rre : float
        Rre au
    """

    R0 = 92.0 # +/- 6 au
    gamma = 0.14 # +/- 0.03

    rre = R0 * lstar**gamma

    return rre


@njit
def blowout_size(lstar, mstar, density, qpr=1.0, beta=BETA_BLOWOUT):
    """
    Radiation pressure blow-out size of a spherical grain.

    The ratio of the radiation force to gravity is independent of distance,

        beta = 3 L Q_pr / (16 pi G M c rho s)

    (Burns, Lamy & Soter 1979), so grains smaller than the size at which
    beta = ``beta`` (0.5 for grains released on circular orbits, 1 for grains at
    rest) are expelled on hyperbolic orbits. Solving for s gives

        s_blow = 3 L Q_pr / (16 pi G M c rho beta).

    Parameters
    ----------
    lstar : float
        Stellar luminosity in solar units.
    mstar : float
        Stellar mass in solar units.
    density : float
        Grain bulk density in g/cm3.
    qpr : float
        Radiation pressure efficiency, Q_abs + Q_sca (1 - <cos theta>). It is 1
        for large grains; use a value below 1 for grains close to the blow-out
        size, whose Q_pr is smaller than the geometric limit.
    beta : float
        Critical force ratio (0.5 by default).

    Returns
    -------
    float
        Blow-out size in um.

    """
    for name, value in (("lstar", lstar), ("mstar", mstar), ("density", density),
                        ("qpr", qpr), ("beta", beta)):  # fmt: skip
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}.")

    size = (
        3.0
        * lstar
        * L_SUN
        * qpr
        / (16.0 * np.pi * G_GRAV * mstar * M_SUN * C_LIGHT * density * G_CM3_TO_KG_M3 * beta)
    )  # m
    return size / UM


def get_belt_profile(params):
    """Read and validate a tabulated radial profile: radii in au, relative density."""
    if "profile_file" in params:
        radii, density = np.loadtxt(
            params["profile_file"], comments="#", usecols=(0, 1), unpack=True, ndmin=2
        )
    elif "profile_r" in params and "profile_sigma" in params:
        radii = np.atleast_1d(np.asarray(params["profile_r"], dtype=float))
        density = np.atleast_1d(np.asarray(params["profile_sigma"], dtype=float))
    else:
        raise ValueError(
            "An 'arbit' profile needs 'profile_file', or both 'profile_r' and "
            "'profile_sigma'."
        )

    if radii.ndim != 1 or radii.shape != density.shape or radii.size < 2:
        raise ValueError(
            "The profile radii and densities must be 1D arrays of equal length "
            "with at least two points."
        )
    if not (np.all(np.isfinite(radii)) and np.all(np.isfinite(density))):
        raise ValueError("The profile contains non-finite values.")
    if radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("The profile radii must be positive and strictly increasing.")
    if np.any(density < 0.0) or not np.any(density > 0.0):
        raise ValueError("The profile densities must be non-negative and not all zero.")
    return radii, density


def make_belt_profile(params):
    """
    Extent and shape of one belt from a dictionary of parameters.

    Parameters
    ----------
    spec : dict
        ``dtype`` and its parameters (the same names as the single-belt model):

        - "gauss": ``rpeak``, ``rfwhm`` (spans rpeak +/- 3 sigma);
        - "onepl": ``rin``, ``rout``, ``alpha_out``;
        - "twopl": ``rin``, ``rout``, ``rpeak``, ``alpha_in``, ``alpha_out``;
        - "arbit": a table of radii (au) and relative surface number density,
          given as ``profile_file`` (two columns) or ``profile_r`` with
          ``profile_sigma``. The belt spans the table, or ``rin`` to ``rout``
          within it, and the density is interpolated linearly.

        ``nring`` is the number of annuli.

    Returns
    -------
    rmin, rmax : float
        Inner and outer edge in au.
    nring : int
        Number of annuli.
    density : callable
        Relative surface number density N(r) at radii in au. Its normalisation
        is arbitrary.

    """
    dtype = params["dtype"]
    nring = int(params["nring"])
    if nring < 1:
        raise ValueError(f"'nring' must be at least 1, got {nring!r}.")

    match dtype:
        case "gauss":
            rpeak = params["rpeak"]
            sigma = params["rfwhm"] / FWHM_PER_SIGMA
            rmin, rmax = rpeak - 3.0 * sigma, rpeak + 3.0 * sigma
            if rmin < 0.0:
                rmin = 1.0

            def density(radii):
                return np.exp(-0.5 * ((radii - rpeak) / sigma) ** 2)

        case "onepl":
            rmin, rmax = spec["rin"], spec["rout"]
            alpha = spec["alpha_out"]

            def density(radii):
                return (radii / rmin) ** alpha

        case "twopl":
            rmin, rmax = spec["rin"], spec["rout"]
            rpeak, alpha, gamma = spec["rpeak"], spec["alpha_in"], spec["alpha_out"]

            def density(radii):
                return np.where(
                    radii > rpeak, (radii / rpeak) ** gamma, (radii / rpeak) ** alpha
                )

        case "arbit":
            table_r, table_density = get_belt_profile(params)
            rmin = params.get("rin", table_r[0])
            rmax = params.get("rout", table_r[-1])
            if rmin < table_r[0] or rmax > table_r[-1]:
                raise ValueError(
                    f"rin and rout ({rmin!r}, {rmax!r} au) must lie within the "
                    f"profile ({table_r[0]!r} to {table_r[-1]!r} au)."
                )

            def density(radii):
                return np.interp(radii, table_r, table_density)

        case _:
            raise ValueError(
                f"'dtype' must be one of {BELT_TYPES}, got {dtype!r}."
            )

    if not 0.0 < rmin < rmax:
        raise ValueError(f"A belt needs 0 < rin < rout, got {rmin!r} and {rmax!r} au.")
    return rmin, rmax, nring, density


def mie_efficiencies(m, x):
    """
    Mie efficiencies (qext, qsca, qback, g) of spheres with refractive index m
    and size parameters x.

    miepython 3 renamed ``mie`` to ``efficiencies_mx`` (same arguments and
    return values); this uses whichever the installed version has.
    """
    if hasattr(mpy, "efficiencies_mx"):
        return mpy.efficiencies_mx(m, x)
    return mpy.mie(m, x)


def extrapolate_wavelengths(lambdas, wavelengths, photospheres, nwav=100):
    """
    Extrapolate stellar photosphere model using Rayleigh-Jeans approximation
    if longest wavelength of stellar model is smaller than the longest
    wavelength of observations.

    Parameters
    ----------
    lambdas : ndarray
        Wavelengths of supplied model grid.
    wavelengths : ndarray
        Wavelengths of observations.
    photospheres : ndarray
        Stellar photosphere model.

    Returns
    -------
    lambda_ext : ndarray
        Extrapolated wavelength array for photosphere model.
    photosphere_ext : ndarray
        Extrapolated photosphere array for photosphere model.

    """

    interp_lam_arr = np.geomspace(lambdas[-1], 1.1 * wavelengths[-1], num=nwav)

    interp_pht_arr = photospheres[-1] * (lambdas[-1] / interp_lam_arr) ** 4  # FLAM

    lambda_ext = np.append(lambdas, interp_lam_arr)
    photosphere_ext = np.append(photospheres, interp_pht_arr)

    return lambda_ext, photosphere_ext


def _size_integral(s0, s1, exponent):
    """Integral of s**(exponent - 1) ds from s0 to s1, i.e. (s1^e - s0^e) / e."""
    if abs(exponent) < 1e-12:
        return np.log(s1 / s0)
    return (s1**exponent - s0**exponent) / exponent


def disc_mass_three_slope(
    mdust,
    q,
    amin,
    amax,
    smm=S_MM,
    skm=S_KM,
    smax=S_MAX,
    q_med=Q_MED,
    q_big=Q_BIG,
    rho_ratio=1.0,
    exact=False,
):
    """
    Total disc mass from the dust mass, for the three-slope size distribution of
    Krivov & Wyatt (2021, MNRAS 500, 718; arXiv:2008.07406, Sect. 2.4).

    The size distribution n(s) ~ s**-q has slope q for dust (s < smm), q_med for
    the collisional cascade (smm < s < skm) and q_big for primordial planetesimals
    (skm < s < smax), and is continuous in mass at each break. The dust mass M_d
    of the paper is the mass in grains up to smm, so the model dust mass, which
    covers amin to amax, is first rescaled to that range using the dust slope.

    Parameters
    ----------
    mdust : float
        Dust mass of the model (any mass unit; the result has the same unit).
    q : float
        Dust size-distribution exponent as used by RTModel, dN/da ~ a**q
        (negative, e.g. -3.5). The paper's slope is -q.
    amin, amax : float
        Smallest and largest grain size of the model in um.
    smm, skm, smax : float
        Break sizes and largest body in um (defaults: 1 mm, 1 km, 200 km).
    q_med, q_big : float
        Slopes (positive) of the cascade and the primordial segments.
    rho_ratio : float
        rho_big / rho_dust, applied if the number rather than the mass
        distribution is continuous at s_km (paper, below Eq. 9). Default 1.
    exact : bool
        False (default) uses the paper's Eq. 9, which neglects the mass in the
        dust and cascade segments (a few per cent). True integrates all three
        segments.

    Returns
    -------
    mdisc : float
        Total mass of bodies up to smax, in the unit of ``mdust``.

    """
    if q >= 0:
        raise ValueError(f"q must be negative (dN/da ~ a**q), got {q!r}.")
    qd = -q
    if not amin < smm < skm < smax:
        raise ValueError(
            "Sizes must satisfy amin < smm < skm < smax, got "
            f"{amin!r}, {smm!r}, {skm!r}, {smax!r}."
        )
    if max(qd, q_med, q_big) >= 4.0:
        raise ValueError("All slopes must be below 4 for the mass to be finite.")
    if amax <= amin:
        raise ValueError(f"amax must exceed amin, got {amax!r} and {amin!r}.")

    # M_d: mass of grains with s <= smm (mass per ds ~ s**(3 - q)), scaled from the
    # model range amin..amax. Extrapolates if amax < smm and clips if amax > smm.
    e_dust = 4.0 - qd
    mass_d = (
        mdust * _size_integral(amin, smm, e_dust) / _size_integral(amin, amax, e_dust)
    )

    if not exact:  # Eq. 9
        factor = (
            (4.0 - qd)
            / (4.0 - q_big)
            * (skm / smm) ** (4.0 - q_med)
            * (smax / skm) ** (4.0 - q_big)
        )
        return mass_d * factor * rho_ratio

    # All three segments, with n(s) continuous at smm and skm
    i_dust = _size_integral(amin, smm, e_dust)
    i_med = smm ** (q_med - qd) * _size_integral(smm, skm, 4.0 - q_med)
    i_big = (
        smm ** (q_med - qd)
        * skm ** (q_big - q_med)
        * _size_integral(skm, smax, 4.0 - q_big)
    )
    return mass_d * (i_dust + i_med + rho_ratio * i_big) / i_dust


def ring_grid(rmin, rmax, nring):
    """
    Divide [rmin, rmax] into nring equal-width annuli.

    Parameters
    ----------
    rmin, rmax : float
        Inner and outer edge of the disc in au.
    nring : int
        Number of annuli.

    Returns
    -------
    edges : float array
        The nring + 1 annulus boundaries in au.
    radii : float array
        The nring annulus mid-points in au, where the surface density is
        evaluated and the dust temperature is calculated.

    """
    edges = np.linspace(rmin, rmax, num=nring + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:])


@njit
def calculate_luminosity(rstar, tstar):
    """
    Calculate the luminosity of the source.

    Parameters
    ----------
    rstar : float
        Radius of the star in R_sol.
    tstar : float
        Temperature of the star in Kelvin.

    Returns
    -------
    lphot : float
        L_star in L_solar
    """
    lphot = (4.0 * np.pi * SIGMA_SB * (rstar * R_SUN) ** 2 * tstar**4) / L_SUN
    return lphot

@jit
def calculate_mass(lstar):
    """Mass (solar) of a main-sequence star from its luminosity (inverts Salaris & Cassisi 2005).

    L = 0.23 M^2.3 (M < 0.43), M^4 (0.43 <= M < 2), 1.4 M^3.5 (2 <= M < 55), 32000 M (M >= 55).
    """
    lum = np.atleast_1d(np.asarray(lstar, dtype=float))
    mass = np.empty_like(lum)
    low, mid_high = 0.23 * 0.43**2.3, 2.0**4  # luminosity at the segment boundaries (0.43 and 2 Msun)
    top = 1.4 * 55.0**3.5
    mass[lum < low] = (lum[lum < low] / 0.23) ** (1 / 2.3)
    sel = (lum >= low) & (lum < mid_high)
    mass[sel] = lum[sel] ** 0.25
    sel = (lum >= mid_high) & (lum < top)
    mass[sel] = (lum[sel] / 1.4) ** (1 / 3.5)
    mass[lum >= top] = lum[lum >= top] / 32000.0
    return mass


class StarModel:
    """
    Stellar photosphere model as its own class to avoid repetition in modified
    blackbody and radiative transfer model classes.

    Model parameters are read from a plain text file by `get_parameters` and
    stored in the `parameters` dictionary.

    Attributes
    ----------
    parameters : dict
        Model parameters, keyed by name.
    sed_wave : ndarray
        Model wavelengths in microns.
    sed_star : ndarray
        Stellar photosphere on the `sed_wave` grid.
    """

    def __init__(self):
        self.parameters = {}

        # Spectra
        self.wavelengths = None
        self.photosphere = None

        # Observations
        self.obs_flux = None    # flux densities in mJy
        self.obs_uncs = None    # uncertainties in mJy
        self.obs_wave = None    # wavelengths in um
        self.obs_band = None    # optional band names (data/filters.yaml), for filter-integrated fits
        self.stellarfit = None  # the PHOENIX fit (with its uncertainties), when stype is "phoenix"

        # Scaling
        self._in_fnu = False

    def parse_star(self,parameters):
        """
        Create a stellar photosphere model in FLAM units.
        Sets self.wavelengths and self.photosphere to be returned to either a
        ModBBModel or RTModel.

        """

        self._in_fnu = False #reset _in_fnu flag

        # rstar and tstar are outputs of the "phoenix" fit, so they need not be given for it
        rstar = parameters.get("rstar")
        tstar = parameters.get("tstar")
        dstar = parameters["dstar"]

        lmin = parameters["lmin"]
        lmax = parameters["lmax"]
        nwav = int(parameters["nwav"])

        wavelengths = np.geomspace(lmin, lmax, num=nwav)  # um

        stype = parameters["stype"]
        valid_types = ["blackbody", "spectrum", "json", "phoenix"]

        match stype:
            case "blackbody":
                photosphere = planck_lam(wavelengths * UM, tstar)  # W/m2/sr/m
                photosphere = (
                    photosphere
                    * np.pi
                    * MJY_PER_SI
                    * ((rstar * R_SUN) / (dstar * PC)) ** 2
                )  # W/m2/m

            case "json":
                lambdas, photosphere = self.read_json()

                if np.max(wavelengths) > np.max(lambdas):
                    lambdas, photosphere = extrapolate_wavelengths(
                        lambdas, wavelengths, photosphere, nwav=100
                    )

                photosphere = np.interp(wavelengths, lambdas, photosphere)
                photosphere *= MJY_PER_SI

            case "spectrum":
                lambdas, photosphere = self.read_star()

                if np.max(wavelengths) > np.max(lambdas):
                    lambdas, photosphere = extrapolate_wavelengths(
                        lambdas, wavelengths, photosphere, nwav=100
                    )

                photosphere = np.interp(wavelengths, lambdas, photosphere)
                photosphere = (
                    photosphere * MJY_PER_SI * ((rstar * R_SUN) / (dstar * PC)) ** 2
                )

            case "phoenix":
                _, photosphere = self.fit_phoenix(wavelengths)
                photosphere *= MJY_PER_SI

            case _:
                raise ValueError(
                    f"'stype' must be one of {valid_types}, got {stype!r}."
                )

        self.wavelengths = wavelengths
        self.photosphere = photosphere

    def read_json(self):
        """
        Read in a stellar photosphere model from a JSON file.

        Stellar photosphere model is assumed to have wavelengths in microns,
        and flux density in Jy.

        Returns
        -------
        model_waves : float array
            Wavelengths in microns in ascending order.
        model_spect : float array
            Photospheric flux density in W/m2/m in ascending order.

        """
        spectrum_file = self.parameters["model"]

        with open(spectrum_file) as spec_file:
            data = json.load(spec_file)

        model_waves = np.asarray(data["star_spec"]["wavelength"], dtype=float)  # um
        model_spect = np.asarray(data["star_spec"]["fnujy"], dtype=float)  # Jy

        # Jy -> W/m2/m
        model_spect *= 1e-26 * C_LIGHT / (model_waves * UM) ** 2  # W/m2/m

        return model_waves, model_spect

    def read_star(self):
        """
        Read in a stellar photosphere model from the SVO database.

        Stellar photosphere model is assumed to have wavelengths in Angstroms,
        and flux density in erg/s/cm^2/A.

        Returns
        -------
        model_waves : float array
            Wavelengths in microns in ascending order.
        model_spect : float array
            Photospheric flux density in W/m2/m in ascending order.

        """
        path = self.parameters["model"]

        model_waves, model_spect = np.loadtxt(
            path, comments="#", usecols=(0, 1), unpack=True
        )

        model_waves *= 1e-4  # Angstrom -> um
        model_spect *= CGS_FLAM_TO_SI  # W/m2/m

        return model_waves, model_spect


    def fit_phoenix(self,model_waves=None):
        """
        Fit a stellar photosphere model from a pre-compiled PHOENIX grid file.

        PHOENIX models have wavelengths in Angstroms and fluxes in ergs/cm-2/s-1/A-1/.

        Returns
        -------
        model_waves : float array
            Wavelengths in microns in ascending order.
        model_spect : float array
            Photospheric flux density in W/m2/m in ascending order.

        """

        if model_waves is None:
            model_waves = np.geomspace(self.parameters["lmin"], self.parameters["lmax"],
                                       num=int(self.parameters["nwav"]))
        model_waves = np.asarray(model_waves)

        if self.obs_wave is None or self.obs_flux is None or self.obs_uncs is None:
            raise ValueError("fit_phoenix needs obs_wave, obs_flux and obs_uncs (mJy).")

        # asarray first, so lists of observations work too
        wave_um = np.asarray(self.obs_wave, dtype=float)
        flux_jy = np.asarray(self.obs_flux, dtype=float) * 1e-3  # mJy -> Jy
        err_jy = np.asarray(self.obs_uncs, dtype=float) * 1e-3

        if self.obs_band is not None:
            # bands with a filter in data/filters.yaml are integrated over it
            phoenix_photometry = Photometer.from_bands(self.obs_band, wave_um, flux_jy, err_jy)
        else:
            # no band names: the model is compared at each point's wavelength
            phoenix_photometry = Photometer(wave_um, flux_jy, err_jy)

        logg_prior = (self.parameters.get("logg_prior",LOGG_PRIOR[0]),
                      self.parameters.get("logg_prior_sigma",LOGG_PRIOR[1]))

        feh_prior = (self.parameters.get("feh_prior",FEH_PRIOR[0]),
                     self.parameters.get("feh_prior_sigma",FEH_PRIOR[1]))

        stellarfit = fit_star(phoenix_photometry,
                                self.parameters["dstar"],
                                load_grid(),
                                distance_err_pc=self.parameters.get("dstar_err"),
                                logg=logg_prior[0],
                                mh=feh_prior[0],
                                logg_prior=logg_prior,
                                mh_prior=feh_prior,
                                fit_surface = True,
                              )

        model_spect = stellarfit.flux_at(model_waves)

        #write parameters into model object.
        self.parameters["lstar"] = stellarfit.luminosity_lsun.value
        self.parameters["tstar"] = stellarfit.teff.value
        self.parameters["rstar"] = stellarfit.radius_rsun.value
        self.parameters["logg"]  = stellarfit.logg.value
        self.parameters["feh"]   = stellarfit.feh.value
        self.stellarfit          = stellarfit #for uncertainties

        model_spect *= 1e-26 * C_LIGHT / (model_waves * UM) ** 2  # Jy -> W/m2/m

        return model_waves, model_spect

    def scale_star(self, lmin=SCALE_WAVE_MIN, lmax=SCALE_WAVE_MAX):
        """
        Scale the photosphere model to the observed photometry within a specified
        wavelength range.

        Parameters
        ----------
        lmin,lmax : Float
             Lower and upper bounds on wavelength range of photometry to scale
             photosphere model.

        Raises
        ------
        ValueError
            If the star has already been converted to F_nu by flam_to_fnu
            (scaling compares the F_lam model with the photometry).

        """

        if self._in_fnu:
            raise ValueError(
                "Star is already in Fnu units, scaling requires Flam units: "
                "call scale_star before flam_to_fnu."
            )

        # Observations and stellar photosphere model
        fobs = np.asarray(self.obs_flux)
        lobs = np.asarray(self.obs_wave)
        uobs = np.asarray(self.obs_uncs)

        mask = (lobs >= lmin) & (lobs < lmax)

        if not mask.any():
            raise ValueError(f"No photometry between {lmin} and {lmax} um.")
        if lobs[mask].min() < self.sed_wave[0] or lobs[mask].max() > self.sed_wave[-1]:
            raise ValueError("Photometry to be fitted lies outside the model grid.")

        model = (np.interp(lobs[mask], self.sed_wave, self.sed_star)
                * (lobs[mask] * UM) ** 2 / C_LIGHT)

        popt, _ = curve_fit(lambda x, a: a * x, model, fobs[mask], sigma=uobs[mask])

        self.sed_star *= popt[0]


class ModBBModel:
    """Optically thin radiative transfer model of a dusty circumstellar disc.

    Model parameters are read from a plain text file by `get_parameters` and
    stored in the `parameters` dictionary. The methods depend on each other's
    results, so they are intended to be called in this order:

        get_parameters -> make_star -> (scale_star) -> make_dust
        -> read_optical_constants -> make_disc -> calculate_surface_density
        -> calculate_qabs -> calculate_dust_scatter -> calculate_dust_emission
        -> flam_to_fnu

    Attributes
    ----------
    parameters : dict
        Model parameters, keyed by name.
    sed_wave : ndarray
        Model wavelengths in microns.
    sed_star, sed_modbb : ndarray
        Stellar photosphere and modified blackbody component spectra on the 
        `sed_wave` grid. Total disc emission only set by flam_to_fnu.
    obs_wave, obs_flux, obs_uncs : array-like or None
        Observed wavelengths (um), fluxes and uncertainties.
    tdust, adust, rbb, rre, lam0, beta : ndarray or float
        Dust temperature, scaling factor, blackbody radius, inferred radius, 
        break wavelength, beta.
    _in_fnu : bool
        Flag if the conversion from FLAM to FNU has been carried out yet.
    """

    # Retained so that RTModel.planck_lam(...) etc. keep working.
    planck_lam = staticmethod(planck_lam)
    planck_nu = staticmethod(planck_nu)
    calculate_luminosity = staticmethod(calculate_luminosity)

    def __init__(self):
        self.parameters = {}

        # Spectra
        self.sed_wave = None
        self.sed_star = None
        self.sed_modb = None # no scattered light -> sed_modb == sed_disc by default

        # Observations
        self.obs_flux = None
        self.obs_uncs = None
        self.obs_wave = None
        self.obs_band = None  # optional band names (data/filters.yaml), for filter-integrated fits
        self.stellarfit = None  # the PHOENIX fit (with its uncertainties), when stype is "phoenix"

        # Disc emission
        self.tdust = None # dust temperature in Kelvin
        self.rbb   = None # dust radius assuming blackbody emission
        self.rre   = None # dust radius assuming R-L relationship from Matra+2025
        self.adust = None # scaling factor for emission in au**2
        self.lam0  = None # break wavelength for modified blackbody emission
        self.beta  = None # modified blacbody emission exponent beyond lam0

        # Scaling
        self._in_fnu = False

    def get_parameters(self, filename):
        """
        Parameters
        ----------
        filename : str
            Filename containing plain text name/value pairs for model values

        Returns
        -------
        parameters : Dictionary
            Dictionary of parameters for the model.

        """
        with open(filename, encoding="utf-8") as f:
            for raw_line in f:
                text = raw_line.split("#", 1)[0].strip()
                if not text:
                    continue
                key, _, value = text.partition(":")
                key = key.strip()
                value = value.strip()
                try:
                    self.parameters[key] = float(value)
                except ValueError:
                    self.parameters[key] = value.strip("'\"")

        return self.parameters

    def make_star(self):

        self._in_fnu = False

        star = StarModel()
        star = star.parse_star(self.parameters)
        star = star.scale_star(lmin=SCALE_WAVE_MIN, lmax=SCALE_WAVE_MAX)

        if stype == "phoenix":
            self.parameters = self.parameters | star.parameters

        self.sed_wave = wavelengths
        self.sed_star = photosphere


    def calculate_modified_blackbody(self):
        """
        Calculate scaled modified blackbody emission profile for dust emission.
        """

        self.sed_modbb = (self.adust*AU**2)*modified_blackbody(self.sed_wave, self.tbb, self.lam0, self.beta)

    def calculate_radii(self):
        """
        Calculate blackbody radius and R-L relationship radius.
        """

        self.rbb = calculate_rbb(self.tdust,self.lstar)
        self.rre = calculate_rre(self.lstar)

    def flam_to_fnu(self):
        """
        Convert calculated stellar and disc emission (in F_lam units) to F_nu (in mJy).
        Sets self.sed_disc .

        """

        if self._in_fnu :
            print("Conversion to Fnu already called for this model.")
            return
        
        convert_factor = (self.sed_wave * UM) ** 2 / C_LIGHT

        # Star
        self.sed_star *= convert_factor
        
        # Disc Emission
        if self.sed_emit is not None:
            self.sed_emit *= scalefactor

        self._in_fnu = True

class RTModel:
    """Optically thin radiative transfer model of a dusty circumstellar disc.

    Model parameters are read from a plain text file by `get_parameters` and
    stored in the `parameters` dictionary. The methods depend on each other's
    results, so they are intended to be called in this order:

        get_parameters -> make_star -> (scale_star) -> make_dust
        -> read_optical_constants -> make_disc -> calculate_surface_density
        -> calculate_qabs -> calculate_dust_scatter -> calculate_dust_emission
        -> flam_to_fnu

    Attributes
    ----------
    parameters : dict
        Model parameters, keyed by name.
    sed_wave : ndarray
        Model wavelengths in microns.
    sed_star, sed_emit, sed_scat, sed_disc : ndarray
        Stellar photosphere, thermal emission, scattered light and total disc
        spectra on the `sed_wave` grid. Total disc emission only set by flam_to_fnu.
    sed_rings, sed_ringe : ndarray
        Per-ring, per grain scattered light and per-ring, per-grain thermal
        emission.
    obs_wave, obs_flux, obs_uncs : array-like or None
        Observed wavelengths (um), fluxes and uncertainties.
    ag, ng, mg : ndarray
        Grain sizes (um), numbers and masses (kg).
    oc_nk : ndarray
        Complex refractive index of the dust on the `sed_wave` grid.
    qext, qsca, qabs : ndarray
        Grain extinction, scattering and absorption efficiencies.
    scale, radii, dr, areas, sigma, tdust : ndarray or float
        Radial disc structure, surface number density and dust temperatures.
    _in_fnu : bool
        Flag if the conversion from FLAM to FNU has been carried out yet.
    """

    # Retained so that RTModel.planck_lam(...) etc. keep working.
    planck_lam = staticmethod(planck_lam)
    planck_nu = staticmethod(planck_nu)
    calculate_luminosity = staticmethod(calculate_luminosity)

    def __init__(self):
        self.parameters = {}

        # Spectra
        self.sed_wave = None
        self.sed_star = None
        self.sed_emit = None
        self.sed_scat = None
        self.sed_disc = None
        self.sed_rings = None
        self.sed_ringe = None

        # Observations
        self.obs_flux = None
        self.obs_uncs = None
        self.obs_wave = None
        self.obs_band = None  # optional band names (data/filters.yaml), for filter-integrated fits
        self.stellarfit = None  # the PHOENIX fit (with its uncertainties), when stype is "phoenix"

        # Dust grains
        self.ag = None
        self.ng = None
        self.mg = None
        self.oc_nk = None
        self.qext = None
        self.qsca = None
        self.qabs = None

        # Disc structure
        self.scale = None
        self.radii = None
        self.edges = None
        self.dr = None
        self.areas = None
        self.sigma = None
        self.tdust = None
        self.mdisc = None

        # Scaling
        self._in_fnu = False

    def get_parameters(self, filename):
        """
        Parameters
        ----------
        filename : str
            Filename containing plain text name/value pairs for model values

        Returns
        -------
        parameters : Dictionary
            Dictionary of parameters for the model.

        """
        with open(filename, encoding="utf-8") as f:
            for raw_line in f:
                text = raw_line.split("#", 1)[0].strip()
                if not text:
                    continue
                key, _, value = text.partition(":")
                key = key.strip()
                value = value.strip()
                try:
                    self.parameters[key] = float(value)
                except ValueError:
                    self.parameters[key] = value.strip("'\"")

        return self.parameters

    def make_star(self):
        """
        Create a stellar photosphere model in FLAM units.
        Sets self.sed_wave and self.sed_star .

        """
        self._in_fnu = False

        star = StarModel()
        star = star.parse_star(self.parameters)
        star = star.scale_star(lmin=SCALE_WAVE_MIN, lmax=SCALE_WAVE_MAX)

        if stype == "phoenix":
            self.parameters = self.parameters | star.parameters
            self.stellarfit = star.stellarfit

        self.sed_wave = wavelengths
        self.sed_star = photosphere


    def calculate_blowout_size(self):
        """
        Radiation pressure blow-out size in um; see ``blowout_size``.

        Uses ``lstar`` and ``mstar`` (solar units), ``density`` (g/cm3), and the
        optional ``qpr`` (default 1) and ``beta_blowout`` (default 0.5). Sets and
        returns self.ablow.

        """
        if "mstar" not in self.parameters and "lstar" not in self.parameters:
            raise ValueError("Calculation of blowout size requires Lstar, Mstar.")
            self.parameters["mstar"] = calculate_mass(self.parameters["lstar"])

        elif "mstar" not in self.parameters and "lstar" in self.parameters:
            warnings.warn("Warning: Mstar not provided, calculating from Lstar.")
            lstar

        self.ablow = blowout_size(
            self.parameters["lstar"],
            self.parameters["mstar"],
            self.parameters["density"],
            qpr=self.parameters.get("qpr", 1.0),
            beta=self.parameters.get("beta_blowout", BETA_BLOWOUT),
        )
        return self.ablow

    def _amin(self):
        """Minimum grain size in um: ``parameters["amin"]``, or the blow-out size."""
        amin = self.parameters["amin"]
        if isinstance(amin, str):
            if amin.strip().lower() != "blowout":
                raise ValueError(f"'amin' must be a number or 'blowout', got {amin!r}.")
            return self.calculate_blowout_size()

        amin = float(amin)
        if "mstar" in self.parameters:
            ablow = self.calculate_blowout_size()
            if amin < ablow:
                warnings.warn(
                    f"amin = {amin:g} um is below the blow-out size of {ablow:.3g} um; "
                    "grains this small would be unbound. Use amin = 'blowout' to start "
                    "the size distribution there.",
                    stacklevel=3,
                )
        return amin


    def make_dust(self):
        """
        Calculate dust grain sizes, numbers, and masses, assuming a power law
        size distribution.

        The parameter ``q`` is the exponent of the differential size
        distribution, dN/da ~ a**q (Mueller 2007, Eq. 2.9, where q = -eta;
        a collisional cascade has q = -3.5). Sets self.ag (um), self.ng (number
        of grains of each size in the whole disc, normalised so that their total
        mass is ``mdust`` Earth masses) and self.mg (kg per grain).

        """
        amin = self.parameters["amin"]
        amax = self.parameters["amax"]
        rho = self.parameters["density"]
        ngrain = int(self.parameters["ngrain"])
        mdust = self.parameters["mdust"]
        q = self.parameters["q"]

        if amax <= amin:
            raise ValueError(
                f"amax ({amax!r} um) must exceed the minimum grain size ({amin!r} um)."
            )

        if q > 0:
            warnings.warn("q must be negative, got {q!r}")
            q *= -1

        grain_sizes = np.geomspace(amin, amax, num=ngrain)

        # The number of grains in a bin is (dN/da) * da. On a logarithmic grid
        # da is proportional to a, so the count scales as a**(q + 1). The
        # constant d(ln a) cancels in the mass normalisation below.
        grain_numbers = grain_sizes ** (q + 1.0)

        grain_masses = (
            rho * G_CM3_TO_KG_M3 * (4.0 / 3.0) * np.pi * ((UM * grain_sizes) ** 3)
        )  # kg
        disc_masses = grain_masses * grain_numbers
        disc_mass_scale = mdust * M_EARTH / np.sum(disc_masses)
        grain_numbers = grain_numbers * disc_mass_scale

        self.ag = grain_sizes
        self.ng = grain_numbers
        self.mg = grain_masses

    def read_optical_constants(self):
        """
        Read in (n,k) optical constants from a text file into self.oc_nk.

        """
        path = self.parameters["composition"] + ".lnk"

        dl, dn, dk = np.loadtxt(path, comments="#", usecols=(0, 1, 2), unpack=True)

        dust_n = np.interp(self.sed_wave, dl, dn)
        dust_k = np.interp(self.sed_wave, dl, dk)
        dust_nk = dust_n - 1j * np.abs(dust_k)

        self.oc_nk = dust_nk

    def make_disc(self):
        """
        Calculate the radial dust density distribution.

        The disc is split into ``nring`` equal-width annuli that tile the
        range between the inner and outer edge exactly. The radial profile is
        the surface number density N(r) evaluated at the annulus mid-points, and 
        the number of grains in an annulus is N(r) times its area.

        Sets self.edges and self.radii (au), self.dr (au), self.areas (m^2) and
        self.scale (the fraction of all grains in each annulus; sums to one).


        ``dtype`` is one of "gauss", "onepl", "twopl", an arbitrary tabulated
        profile "arbit", or "multi" for several belts. See ``make_belt_profile`` for
        the parameters of one belt and ``make_multi_belt_disc`` for several.
        """
        dtype = self.parameters["dtype"]

        if dtype not in BELT_TYPES:
            raise ValueError(
                f"Belt type needs to be one of {BELT_TYPES!r} ,got {dtype!r}."
            )

        if dtype == "multi":
            self._make_multi_belt_profile()
            return

        rmin, rmax, nring, density = make_belt_profile(self.parameters)
        edges, radii = ring_grid(rmin, rmax, nring)
        rings = density(radii)

        self.edges = edges
        self.radii = radii
        self.dr = edges[1] - edges[0]
        self.areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * AU**2  # m^2

        grains_per_ring = rings * self.areas  # N(r) * area, Eq. 2.17
        self.scale = grains_per_ring / np.sum(grains_per_ring)
        self.belt_scale = self.scale[None, :]

        self.edges = edges
        self.radii = radii
        self.dr = edges[1] - edges[0]
        self.areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * AU**2  # m^2

        grains_per_ring = rings * self.areas  # N(r) * area, Eq. 2.17
        self.scale = grains_per_ring / np.sum(grains_per_ring)


    def _read_belts(self):
        """The list of belt dictionaries from ``belts`` or the JSON ``belts_file``."""
        if "belts" in self.parameters:
            belts = self.parameters["belts"]
        elif "belts_file" in self.parameters:
            with open(self.parameters["belts_file"], encoding="utf-8") as f:
                belts = json.load(f)
        else:
            raise ValueError(
                "dtype 'multi' needs the belts: set 'belts' to a list of "
                "dictionaries, or 'belts_file' to a JSON file holding that list."
            )
        if not isinstance(belts, (list, tuple)) or len(belts) == 0:
            raise ValueError("'belts' must be a non-empty list of dictionaries.")
        return list(belts)


    def _make_multi_belt_profle(self):
        """
        Radial structure of a disc made of several belts.

        Each belt is a dictionary with its own ``dtype`` ("gauss", "onepl",
        "twopl" or "arbit") and parameters, see ``belt_profile``, plus an optional
        ``mass`` giving its share of the dust (relative weights, default 1) and
        ``nring`` (default ``DEFAULT_NRING``). All belts share the grain size
        distribution, so a belt's share of the dust mass is also its share of the
        grains.

        The annuli come from the union of every belt's ring edges, so belts may
        overlap or be separated by gaps; a gap is a single annulus that holds no
        dust. Sets self.edges, self.radii, self.areas, self.scale (as for one belt;
        self.dr is an array, as the annuli are not equally wide) and
        self.belt_scale, the (nbelt, nring) fraction of all grains that each belt
        puts in each annulus.

        """
        belts = self.read_belts()
        profiles, weights, boundaries = [], [], []
        for index, belt in enumerate(belts):
            spec = dict(belt)
            if spec.get("dtype") == "multi":
                raise ValueError("A belt cannot itself be of dtype 'multi'.")
            spec.setdefault("nring", DEFAULT_NRING)
            try:
                rmin, rmax, nring, density = belt_profile(spec)
            except (KeyError, ValueError) as error:
                raise ValueError(f"Belt {index}: {error}") from error

            weight = float(spec.get("mass", 1.0))
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError(f"Belt {index}: 'mass' must be non-negative.")
            profiles.append((rmin, rmax, density))
            weights.append(weight)
            boundaries.append(np.linspace(rmin, rmax, nring + 1))

        weights = np.asarray(weights)
        if weights.sum() <= 0.0:
            raise ValueError("At least one belt needs a positive 'mass'.")
        weights = weights / weights.sum()

        edges = np.unique(np.concatenate(boundaries))
        keep = np.concatenate(([True], np.diff(edges) > 1e-9 * edges[-1]))
        edges = edges[keep]
        radii = 0.5 * (edges[:-1] + edges[1:])
        areas_au2 = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

        belt_scale = np.zeros((len(profiles), len(radii)))
        for index, (rmin, rmax, density) in enumerate(profiles):
            inside = (radii > rmin) & (radii < rmax)
            grains = np.where(inside, density(radii), 0.0) * areas_au2
            if not grains.sum() > 0.0:
                raise ValueError(f"Belt {index} contains no dust.")
            belt_scale[index] = weights[index] * grains / grains.sum()

        self.edges = edges
        self.radii = radii
        self.dr = np.diff(edges)
        self.areas = areas_au2 * AU**2  # m^2
        self.belt_scale = belt_scale
        self.scale = belt_scale.sum(axis=0)


    def calculate_surface_density(self):
        """
        Calculate the number of grains in each annulus and size bin, self.sigma.

        Despite its name, self.sigma is a dimensionless grain count with shape
        (nring, ngrain), not a surface density: it is the fraction of the disc in
        the annulus (self.scale) times the number of grains of that size in the
        whole disc (self.ng). Its sum over the rings is self.ng.

        """

        self.sigma = np.outer(self.scale, self.ng)

    def calculate_dust_temperature(self, radius, qabs, mode="full", tolerance=1e-6):
        """
        Calculate the temperature of a dust grain at a given distance from the star.

        Parameters
        ----------
        radius : Float
            Element of self.radii to fit temperature.
        qabs : float array
            Absorption coefficients for grain size ag across all wavelengths.
        mode : str
            Keyword for implementing iterative dust temperature calculation
            either "bb" (blackbody approximation) or "full".
        tolerance :
            Relative tolerance of brentq determination of temperature.

        Returns
        -------
        td : float
            Dust temperature in Kelvin.

        """
        lstar = self.parameters["lstar"]
        rstar = self.parameters["rstar"]
        tstar = self.parameters["tstar"]
        dstar = self.parameters["dstar"]

        tbb = T_BB_1AU * (lstar**0.25) * (radius ** (-0.5))

        match mode:
            case "bb":
                return tbb

            case "full":
                wav_m = self.sed_wave * UM

                factor = 0.5 * ((rstar * R_SUN) / AU)

                stype = self.parameters["stype"]
                valid_types = ["blackbody", "json", "spectrum"]

                match stype:
                    case "blackbody":
                        dust_absr = np.trapezoid(qabs * planck_lam(wav_m, tstar), wav_m)

                    case "spectrum" | "json":
                        # Scaling the observed flux by (d / R)^2 gives the surface
                        # flux, pi * B_lambda; dividing by pi gives the intensity
                        # B_lambda used in the other branch.
                        dust_absr = np.trapezoid(
                            qabs
                            * self.sed_star
                            * SI_PER_MJY
                            * ((dstar * PC) / (rstar * R_SUN)) ** 2
                            / np.pi,
                            wav_m,
                        )

                    case _:
                        raise ValueError(
                            f"Spectrum must be one of {valid_types}, got {stype!r}"
                        )

                def residual(td):
                    dust_emit = np.trapezoid(qabs * planck_lam(wav_m, td), wav_m)

                    return factor * (dust_absr / dust_emit) ** 0.5 - radius

                res_lo = residual(T_CMB)
                res_hi = residual(tstar)

                if res_lo * res_hi >= 0:
                    raise ValueError(
                        f"Temperature determination at {radius} au"
                        " requires limits f(Tcmb), f(Tstar) have different sign."
                    )

                return brentq(residual, T_CMB, tstar, rtol=tolerance)

            case _:
                raise ValueError(f"'mode' must be one of 'bb' or 'full', got {mode!r}.")

    def calculate_qabs(self):
        """
        Calculate the qabs, qsca values for dust grains in the model using Mie theory.
        Sets self.qabs, self.qsca, and self.qext .

        """
        self.qext = np.zeros(
            (int(self.parameters["ngrain"]), int(self.parameters["nwav"]))
        )
        self.qsca = np.zeros(
            (int(self.parameters["ngrain"]), int(self.parameters["nwav"]))
        )

        for ii in range(int(self.parameters["ngrain"])):
            x = 2.0 * np.pi * self.ag[ii] / self.sed_wave
            qext, qsca, *_ = mie_efficiencies(self.oc_nk, x)

            self.qext[ii, :] = qext
            self.qsca[ii, :] = qsca

        self.qabs = self.qext - self.qsca

    def calculate_dust_scatter(self):
        """
        Calculate the scattered light contribution to the total emission from the disc.

        Starlight is scattered isotropically, and the disc is optically thin. A grain
        at distance r intercepts pi a^2 Q_sca L / (4 pi r^2) and re-radiates it over
        4 pi sr, so the flux at the observer is the stellar flux times
        N pi a^2 Q_sca / (4 pi r^2) = N a^2 Q_sca / (4 r^2) for N grains. Q_sca
        already contains the albedo (Q_sca = albedo * Q_ext). Sets self.sed_scat and
        self.sed_rings.

        """
        self.sed_scat = np.zeros(int(self.parameters["nwav"]))
        self.sed_rings = np.zeros(
            (
                int(self.parameters["nring"]),
                int(self.parameters["ngrain"]),
                int(self.parameters["nwav"]),
            )
        )

        for ii in range(int(self.parameters["nring"])):
            for ij in range(int(self.parameters["ngrain"])):
                scalefactor = (
                    self.qsca[ij, :]
                    * self.sigma[ii, ij]
                    * (self.ag[ij] * UM) ** 2
                    / (4.0 * (self.radii[ii] * AU) ** 2)
                )

                self.sed_rings[ii, ij, :] += scalefactor * self.sed_star
                self.sed_scat += self.sed_rings[ii, ij, :]

    def calculate_dust_emission(self, mode="full", tolerance=1e-6):
        """
        Calculate the continuum emission contribution to the total emission from the
        disc.

        Parameters
        ----------
        mode : 'bb', 'full'
            Keyword for implementing iterative dust temperature calculation.
        tolerance :
            Relative tolerance of brentq determination of temperature.

        """

        self.tdust = np.zeros(
            (int(self.parameters["nring"]), int(self.parameters["ngrain"]))
        )

        self.sed_emit = np.zeros(int(self.parameters["nwav"]))
        self.sed_ringe = np.zeros(
            (
                int(self.parameters["nring"]),
                int(self.parameters["ngrain"]),
                int(self.parameters["nwav"]),
            )
        )

        dstar_m = self.parameters["dstar"] * PC

        # Calculate emission
        for ii in range(int(self.parameters["nring"])):
            for ij in range(int(self.parameters["ngrain"])):
                # Each grain of radius a emits 4 pi a^2 Q_abs pi B_lambda, so it
                # contributes pi a^2 Q_abs B_lambda / d^2 to the flux at the
                # observer.
                scalefactor = (
                    MJY_PER_SI
                    * np.pi
                    * (self.ag[ij] * UM) ** 2
                    * self.sigma[ii, ij]
                    / dstar_m**2
                )

                self.tdust[ii, ij] = self.calculate_dust_temperature(
                    self.radii[ii], self.qabs[ij, :], mode=mode, tolerance=tolerance
                )

                self.sed_ringe[ii, ij, :] = (
                    scalefactor
                    * self.qabs[ij, :]
                    * planck_lam(self.sed_wave * UM, self.tdust[ii, ij])
                )

                self.sed_emit += self.sed_ringe[ii, ij, :]

    def flam_to_fnu(self):
        """
        Convert calculated stellar and disc emission (in F_lam units) to F_nu (in mJy).
        Sets self.sed_disc .

        """

        if self._in_fnu :
            print("Conversion to Fnu already called for this model.")
            return
        
        convert_factor = (self.sed_wave * UM) ** 2 / C_LIGHT

        # Star
        self.sed_star *= convert_factor
        
        # Disc Emission
        if self.sed_emit is not None:
            self.sed_emit *= convert_factor
            self.sed_ringe *= convert_factor
        
        # Disc Scattering
        if self.sed_scat is not None:
            self.sed_scat *= convert_factor
            self.sed_rings *= convert_factor
        
        self._in_fnu = True

        # Total disc model
        parts = [s for s in (self.sed_emit, self.sed_scat) if s is not None]
        self.sed_disc = sum(parts) if parts else None


    def calculate_disc_mass(
        self,
        smm=S_MM,
        skm=S_KM,
        smax=S_MAX,
        q_med=Q_MED,
        q_big=Q_BIG,
        rho_big=None,
        exact=False,
        mdust=None,
    ):
        """
        Estimate the total disc mass from the fitted dust mass.

        Run this after the SED fit has set ``parameters["mdust"]``. It extends the
        dust size distribution (slope ``-q``, grains ``amin`` to ``amax``) to
        planetesimals with the three-slope approximation of Krivov & Wyatt (2021,
        Eq. 9); see ``disc_mass_three_slope``. Sets and returns self.mdisc.

        Parameters
        ----------
        smm, skm, smax : float
            Break sizes and largest body in um (1 mm, 1 km, 200 km by default).
        q_med, q_big : float
            Slopes of the cascade and primordial segments (3.7 and 2.8).
        rho_big : float, optional
            Density of the large bodies in g/cm3, if the number rather than the
            mass distribution is continuous at ``skm``.
        exact : bool
            Integrate all segments instead of using the paper's Eq. 9.
        mdust : float, optional
            Dust mass in Earth masses; defaults to ``parameters["mdust"]``.

        Returns
        -------
        mdisc : float
            Total disc mass in Earth masses.

        """
        if mdust is None:
            mdust = self.parameters["mdust"]
        rho_ratio = 1.0 if rho_big is None else rho_big / self.parameters["density"]

        self.mdisc = disc_mass_three_slope(
            mdust,
            self.parameters["q"],
            self.parameters["amin"],
            self.parameters["amax"],
            smm=smm,
            skm=skm,
            smax=smax,
            q_med=q_med,
            q_big=q_big,
            rho_ratio=rho_ratio,
            exact=exact,
        )
        return self.mdisc


    def _ring_fluxes(self, wavelength, component):
        """
        Thermal and scattered flux of each annulus at one wavelength, in mJy,
        summed over grain sizes.

        Works whether or not ``flam_to_fnu`` has been called; it never changes the
        stored spectra.

        Returns
        -------
        emit, scat : float arrays
            Flux of each annulus; zeros for a part that was not asked for.

        """
        valid = ("emit", "scat", "total")
        if component not in valid:
            raise ValueError(f"'component' must be one of {valid}, got {component!r}.")
        if self.sed_wave is None:
            raise ValueError("Run make_star first.")
        if not self.sed_wave[0] <= wavelength <= self.sed_wave[-1]:
            raise ValueError(
                f"Wavelength {wavelength} um is outside the model grid "
                f"({self.sed_wave[0]:.3g}-{self.sed_wave[-1]:.3g} um)."
            )

        nring = len(self.radii)
        parts = {"emit": np.zeros(nring), "scat": np.zeros(nring)}
        sources = {"emit": self.sed_ringe, "scat": self.sed_rings}
        wanted = ("emit", "scat") if component == "total" else (component,)
        found = False
        for name in wanted:
            if sources[name] is None:
                continue
            found = True
            spectra = sources[name].sum(axis=1)  # (nring, nwav)
            if not self._in_fnu:
                spectra = spectra * ((self.sed_wave * UM) ** 2 / C_LIGHT)
            parts[name] = np.array(
                [np.interp(wavelength, self.sed_wave, row) for row in spectra]
            )
        if not found:
            raise ValueError(
                f"No '{component}' disc light to image: run calculate_dust_emission "
                "and/or calculate_dust_scatter first."
            )
        return parts["emit"], parts["scat"]

    def make_image(
        self,
        wavelength,
        component="total",
        inc=0.0,
        pa=0.0,
        npix=128,
        pixscale=None,
        beam=None,
        unit="mJy/pixel",
        oversample=3,
        include_star=False,
        aspect=None,
        g1=None,
        g2=None,
        hg_weight=None,
        nlos=48,
    ):
        """
        Make a resolved image of the disc at one wavelength.

        Each annulus of the radial model emits (or scatters) a flux F_i, spread over
        its area A_i, so its face-on flux per unit area is F_i / A_i. The dust has a
        Gaussian vertical profile with standard deviation H = aspect * r, and the
        disc is optically thin, so the image is the integral of the emissivity along
        each line of sight (see ``disc_surface_brightness``). This conserves the
        total flux. A thin disc seen at inclination ``inc`` is stretched by cos(inc)
        across the major axis and brightened by 1 / cos(inc); a thick disc is also
        blurred across the minor axis by H sin(inc).

        Thermal emission is isotropic. Scattered light is weighted by a two-component
        Henyey-Greenstein phase function, 4 pi p(theta) relative to the isotropic
        ``calculate_dust_scatter`` flux, so the flux of the scattered image changes
        with the viewing geometry (forward-scattering grains brighten the near side).
        The weighting is applied to the scattered light only: to ``component="scat"``
        and to the scattered part of ``component="total"``. The side of the disc at
        position angle ``pa + 90`` degrees is the near side. Run
        ``calculate_dust_emission`` and/or ``calculate_dust_scatter`` first.

        Parameters
        ----------
        wavelength : float
            Wavelength in um, within the model grid.
        component : {"total", "emit", "scat"}
            Which disc light to image; "total" sums those that have been computed.
        inc : float
            Inclination in degrees (0 is face-on), from 0 up to but excluding 90.
        pa : float
            Position angle of the major axis in degrees, east of north.
        npix : int
            Number of pixels along each side of the (square) image.
        pixscale : float, optional
            Pixel size in arcsec. Default: a field of view of 2.4 outer radii.
        beam : float, 2D array or kernel object, optional
            Circular Gaussian beam FWHM in arcsec, or a PSF array or astropy-style
            kernel object (with a 2D ``array`` attribute) sampled on the image grid.
        unit : {"mJy/pixel", "mJy/arcsec2", "mJy/beam"}
            Unit of the returned image; "mJy/beam" needs ``beam``.
        oversample : int
            Sub-pixels per pixel along each axis, to resolve sharp ring edges.
        include_star : bool
            Add the stellar photosphere as a point source at the centre.
        aspect : float, optional
            Scale height over radius, H / r, of the Gaussian vertical profile
            exp(-z^2 / 2 H^2). Default: ``parameters["aspect"]`` if set, else 0.03.
            Zero (or very small) gives a razor-thin disc.
        g1, g2, hg_weight : float, optional
            Asymmetry parameters of the forward and backward Henyey-Greenstein
            components and the weight of the first. Defaults: the model parameters
            ``hg_g1``, ``hg_g2`` and ``hg_weight`` if set, else 0.7, -0.2 and 0.7;
            these are placeholders to be set from observations. Setting
            ``g1 = g2 = 0`` gives isotropic scattering.
        nlos : int
            Line-of-sight samples per sub-pixel.

        Returns
        -------
        image : ModelImage

        """
        valid_units = ("mJy/pixel", "mJy/arcsec2", "mJy/beam")
        if unit not in valid_units:
            raise ValueError(f"'unit' must be one of {valid_units}, got {unit!r}.")
        if unit == "mJy/beam" and beam is None:
            raise ValueError("unit 'mJy/beam' needs a beam.")
        if not 0.0 <= inc < 90.0:
            raise ValueError(f"'inc' must be in [0, 90) degrees, got {inc!r}.")
        if int(npix) < 2 or int(oversample) < 1 or int(nlos) < 2:
            raise ValueError(
                "'npix' must be at least 2, 'oversample' at least 1 and 'nlos' "
                "at least 2."
            )
        if self.edges is None:
            raise ValueError("Run make_disc first.")
        npix, oversample, nlos = int(npix), int(oversample), int(nlos)

        if aspect is None:
            aspect = self.parameters.get("aspect", ASPECT_RATIO)
        if aspect < 0.0:
            raise ValueError(f"'aspect' must not be negative, got {aspect!r}.")
        if g1 is None:
            g1 = self.parameters.get("hg_g1", HG_G1)
        if g2 is None:
            g2 = self.parameters.get("hg_g2", HG_G2)
        if hg_weight is None:
            hg_weight = self.parameters.get("hg_weight", HG_WEIGHT)
        two_component_hg(0.0, g1, g2, hg_weight)  # validate the phase function

        flux_emit, flux_scat = self._ring_fluxes(wavelength, component)  # mJy
        scattered = component in ("scat", "total") and np.any(flux_scat)
        phase = None
        if scattered:

            def phase(cos_theta):
                return two_component_hg(cos_theta, g1, g2, hg_weight)

        dstar = self.parameters["dstar"]  # pc, so 1 arcsec is 1 au per pc
        if pixscale is None:
            pixscale = 2.4 * self.edges[-1] / dstar / npix
        pix_au = pixscale * dstar

        # Face-on flux per unit disc area (mJy/au^2) in each annulus
        edges = self.edges
        annulus_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)

        # Sky offsets of the sub-pixel centres (au), north up and east left
        ngrid = npix * oversample
        offset = ((np.arange(ngrid) + 0.5) / oversample - npix / 2.0) * pix_au
        north, east = offset[:, None], -offset[None, :]
        pa_rad = np.radians(pa)
        along = east * np.sin(pa_rad) + north * np.cos(pa_rad)  # major axis
        across = -east * np.cos(pa_rad) + north * np.sin(pa_rad)  # minor axis

        # A face-on or razor-thin disc is the limit of a very small tilt or height
        brightness = disc_surface_brightness(
            np.broadcast_to(along, (ngrid, ngrid)).ravel(),
            np.broadcast_to(across, (ngrid, ngrid)).ravel(),
            edges,
            flux_emit / annulus_area,
            flux_scat / annulus_area,
            max(np.radians(inc), MIN_INC),
            max(aspect, MIN_ASPECT),
            phase=phase,
            nlos=nlos,
        ).reshape(ngrid, ngrid)
        data = (brightness * (pix_au / oversample) ** 2).reshape(
            npix, oversample, npix, oversample
        )
        data = data.sum(axis=(1, 3))  # mJy per pixel

        if include_star:
            if self.sed_star is None:
                raise ValueError("Run make_star first.")
            spectrum = self.sed_star
            if not self._in_fnu:  # convert on the model grid, then interpolate
                spectrum = spectrum * (self.sed_wave * UM) ** 2 / C_LIGHT
            star = np.interp(wavelength, self.sed_wave, spectrum)
            centre = [npix // 2 - 1, npix // 2] if npix % 2 == 0 else [npix // 2]
            data[np.ix_(centre, centre)] += star / len(centre) ** 2

        beam_area = None  # in pixels
        if beam is not None:
            if hasattr(beam, "array"):
                psf = beam.array
                data = convolve_beam(data, psf)
                beam_area = np.sum(psf) / np.max(psf)  # integral over peak
            elif np.ndim(beam) == 0:
                data = convolve_beam(data, beam / pixscale)
                beam_area = 2.0 * np.pi * (beam / pixscale * FWHM_TO_SIGMA) ** 2
            else:
                data = convolve_beam(data, beam)
                beam_area = np.sum(beam) / np.max(beam)  # integral over peak

        if unit == "mJy/arcsec2":
            data = data / pixscale**2
        elif unit == "mJy/beam":
            data = data * beam_area

        return ModelImage(
            data,
            pixscale,
            wavelength,
            unit,
            component,
            inc,
            pa,
            beam,
            aspect,
            (g1, g2, hg_weight) if scattered else None,
        )
