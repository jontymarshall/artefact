#!/usr/bin/env python3
"""Artefact: a 1D radiative transfer model for optically thin dust discs.

Calculates the thermal emission and scattered light from a dust disc around a
star in the optically thin limit, for several radial surface density
distributions and a power-law grain size distribution.
"""

__version__ = "0.0.6"
__author__ = "Jonathan P. Marshall"
__email__ = ""
__copyright__ = "2023-26, Jonathan P. Marshall"
__license__ = "GNU GPL V3"
__url__ = "https://github.com/jontymarshall/artefact"
__citation__ = r"""@ARTICLE{2023MNRAS.521.5940M,
    author = {{Marshall}, Jonathan P. and {Milli}, J. and {Choquet}, E. and {del Burgo}, C. and {Kennedy}, G.~M. and {Kemper}, F. and {Wyatt}, M.~C. and {Kral}, Q. and {Soummer}, R.},
    title = '{Stirred but not shaken: a multiwavelength view of HD 16743's debris disc}',
    journal = {\mnras},
    keywords = {planet-disc interactions, circumstellar matter, stars: individual: HD 16743, radio continuum: planetary systems, Astrophysics - Earth and Planetary Astrophysics},
    year = 2023,
    month = jun,
    volume = {521},
    number = {4},
    pages = {5940-5951},
    doi = {10.1093/mnras/stad913},
    archivePrefix = {arXiv},
    eprint = {2303.17128},
    primaryClass = {astro-ph.EP},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.5940M},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}"""

import json

import astropy.constants as const
import miepython as mpy
import numpy as np
from numba import njit
from scipy.optimize import brentq, curve_fit


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
FWHM_PER_SIGMA = 2.355  # approximately 2 * sqrt(2 * ln(2))
T_CMB = 2.73  # K, lower bound of the dust temperature search
T_BB_1AU = 278.0  # K, blackbody temperature at 1 au from a 1 L_sun star
SCALE_WAVE_MIN = 0.44  # um, lower limit of photometry used by scale_star
SCALE_WAVE_MAX = 5.0  # um, upper limit of photometry used by scale_star

# Deprecated lowercase aliases, retained for backwards compatibility only.
h, c, k, sb = H_PLANCK, C_LIGHT, K_B, SIGMA_SB
au, pc, lsol, rsol = AU, PC, L_SUN, R_SUN
MEarth, um = M_EARTH, UM


@njit(cache=True)
def planck_lam(wav, T):
    """
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


def extrapolate_wavelengths(lambdas, wavelengths, photospheres, nwav=100):
    """
    Extrapolate stellar photosphere model using Rayleigh-Jeans approximation
    if longest wavelength of stellar model is smaller than the longest
    wavelength of requested model grid.

    Parameters
    ----------
    lambdas : ndarray
        Wavelengths of supplied stellar model.
    wavelengths : ndarray
        Wavelengths of model grid.
    photospheres : ndarray
        Flux of supplied stellar photosphere model.

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


@njit
def calc_luminosity(rstar, tstar):
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


class RTModel:
    """Optically thin radiative transfer model of a dusty circumstellar disc.

    Model parameters are read from a plain text file by `get_parameters` and
    stored in the `parameters` dictionary. The methods depend on each other's
    results, so they are intended to be called in this order::

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
    """

    # Retained so that RTModel.planck_lam(...) etc. keep working.
    planck_lam = staticmethod(planck_lam)
    planck_nu = staticmethod(planck_nu)
    calc_luminosity = staticmethod(calc_luminosity)

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
        self.dr = None
        self.areas = None
        self.sigma = None
        self.tdust = None

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
        Create a stellar photosphere model Flam in W/m2/m x 1e29 (mJy compatible).
        Sets self.sed_wave and self.sed_star .

        """

        rstar = self.parameters["rstar"]
        tstar = self.parameters["tstar"]
        dstar = self.parameters["dstar"]

        lmin = self.parameters["lmin"]
        lmax = self.parameters["lmax"]
        nwav = int(self.parameters["nwav"])

        wavelengths = np.geomspace(lmin, lmax, num=nwav)  # um

        stype = self.parameters["stype"]
        valid_types = ["blackbody", "spectrum", "json"]

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
                        lambdas, wavelengths, photosphere, nwav=nwav
                    )

                photosphere = np.interp(wavelengths, lambdas, photosphere)
                photosphere *= MJY_PER_SI

            case "spectrum":
                lambdas, photosphere = self.read_star()

                if np.max(wavelengths) > np.max(lambdas):
                    lambdas, photosphere = extrapolate_wavelengths(
                        lambdas, wavelengths, photosphere, nwav=nwav
                    )

                photosphere = np.interp(wavelengths, lambdas, photosphere)
                photosphere = (
                    photosphere * MJY_PER_SI * ((rstar * R_SUN) / (dstar * PC)) ** 2
                )

            case "starfish":
                raise NotImplementedError("Not yet implemented.")

            case _:
                raise ValueError(
                    f"'stype' must be one of {valid_types}, got {stype!r}."
                )

        self.sed_wave = wavelengths
        self.sed_star = photosphere
        self._in_fnu = False

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

    def scale_star(self, lmin=SCALE_WAVE_MIN, lmax=SCALE_WAVE_MAX):
        """
        Scale the photosphere model to the observed photometry within a specified
        wavelength range. Must be run before flux conversion using flam_to_fnu.

        Parameters
        ----------
        lmin,lmax : Float
             Lower and upper bounds on wavelength range of photometry to scale
             photosphere model.

        """

        if self._in_fnu: 
            raise RuntimeError("Conversion to Fnu already called for this model."
                               " Function scale_star expects self.sed_star in Flam.")

        # Observations and stellar photosphere model
        fobs = np.asarray(self.obs_flux)
        lobs = np.asarray(self.obs_wave)
        uobs = np.asarray(self.obs_uncs)

        mask = (lobs >= lmin) & (lobs < lmax)

        if not mask.any():
            raise ValueError(f"No photometry between {lmin} and {lmax} um.")
        if lobs[mask].min() < self.sed_wave[0] or lobs[mask].max() > self.sed_wave[-1]:
            raise ValueError("Photometry to be fitted lies outside the model grid.")

        model = (
                np.interp(lobs[mask], self.sed_wave, self.sed_star) 
                * (lobs[mask] * UM) ** 2 / C_LIGHT
                )

        popt, _ = curve_fit(lambda x, a: a * x, model, fobs[mask], sigma=uobs[mask])

        self.sed_star *= popt[0]

    def make_dust(self):
        """
        Calculate dust grain sizes, numbers, and masses, assuming a power law
        size distribution.

        """
        amin = self.parameters["amin"]
        amax = self.parameters["amax"]
        rho = self.parameters["density"]
        ngrain = int(self.parameters["ngrain"])
        mdust = self.parameters["mdust"]
        q = self.parameters["q"]

        grain_sizes = np.geomspace(amin, amax, num=ngrain)

        grain_numbers = grain_sizes**q

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
        Sets self.scale, self.dr, self.radii, and self.areas .

        """
        valid_types = ["gauss", "onepl", "twopl", "arbit"]

        dtype = self.parameters["dtype"]

        match dtype:
            case "gauss":
                # Gaussian ring surface density model
                rpeak = self.parameters["rpeak"]
                rfwhm = self.parameters["rfwhm"]
                nring = int(self.parameters["nring"])

                lower = rpeak - 3.0 * (rfwhm / FWHM_PER_SIGMA)
                upper = rpeak + 3.0 * (rfwhm / FWHM_PER_SIGMA)
                if lower < 0.0:
                    lower = 1.0
                radii = np.linspace(lower, upper, num=nring, endpoint=True)
                rings = np.exp(-0.5 * ((radii - rpeak) / (rfwhm / FWHM_PER_SIGMA)) ** 2)
                scale = rings / np.sum(rings)

            case "onepl":
                # Single power law surface density model
                rin = self.parameters["rin"]
                rout = self.parameters["rout"]
                alpha = self.parameters["alpha_out"]
                nring = int(self.parameters["nring"])

                radii = np.linspace(rin, rout, num=nring, endpoint=True)
                rings = (radii / rin) ** alpha
                scale = rings / np.sum(rings)

            case "twopl":
                # Two power law surface density model
                rin = self.parameters["rin"]
                rout = self.parameters["rout"]
                rpeak = self.parameters["rpeak"]
                alpha = self.parameters["alpha_in"]
                gamma = self.parameters["alpha_out"]
                nring = int(self.parameters["nring"])

                radii = np.linspace(rin, rout, num=nring, endpoint=True)
                rings = (radii / rpeak) ** alpha
                outer = np.where(radii > rpeak)
                rings[outer] = (radii[outer] / rpeak) ** gamma
                scale = rings / np.sum(rings)

            case "arbit":
                raise NotImplementedError(
                    "Arbitrary density distribution has not yet been implemented."
                )

            case _:
                raise ValueError(
                    f"'dtype' must be one of {valid_types}, got {dtype!r}."
                )

        self.scale = scale
        self.radii = radii

        self.dr = (radii[-1] - radii[0]) / nring

        self.areas = (
            (self.radii + 0.5 * self.dr) ** 2 - (self.radii - 0.5 * self.dr) ** 2
        ) * AU**2

        self.scale = self.scale * (self.areas / self.areas[0])
        self.scale = self.scale / np.sum(self.scale)

    def calculate_surface_density(self):
        """
        Calculate radial surface number density of the disc, self.sigma.

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

        if self._in_fnu: 
            raise RuntimeError("Conversion to Fnu already called for this model."
                               " Function calculate_dust_temperature expects" 
                               " self.sed_star in Flam.")

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
                        dust_absr = np.trapezoid(
                            qabs
                            * self.sed_star
                            * SI_PER_MJY
                            * ((dstar * PC) / (rstar * R_SUN)) ** 2 / np.pi,
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
            qext, qsca, *_ = mpy.efficiencies_mx(self.oc_nk, x)

            self.qext[ii, :] = qext
            self.qsca[ii, :] = qsca

        self.qabs = self.qext - self.qsca

    def calculate_dust_scatter(self):
        """
        Calculate the scattered light contribution to the total emission from the disc.

        """

        if self._in_fnu: 
            raise RuntimeError("Conversion to Fnu already called for this model."
                               " Function calculate_dust_scatter expects" 
                               " self.sed_star in Flam.")

        self.sed_scat = np.zeros(int(self.parameters["nwav"]))
        self.sed_rings = np.zeros(
            (
                int(self.parameters["nring"]),
                int(self.parameters["ngrain"]),
                int(self.parameters["nwav"]),
            )
        )

        alb = self.qsca / self.qext

        for ii in range(int(self.parameters["nring"])):
            for ij in range(int(self.parameters["ngrain"])):
                scalefactor = (
                    self.qsca[ij, :]
                    * alb[ij, :]
                    * self.sigma[ii, ij]
                    * np.pi
                    * ((self.ag[ij] * UM) ** 2)
                    / (2 * self.radii[ii] * AU) ** 2
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

        # Calculate emission
        for ii in range(int(self.parameters["nring"])):
            for ij in range(int(self.parameters["ngrain"])):
                scalefactor = (
                    MJY_PER_SI
                    * (2.0 * np.pi**2 / ((self.parameters["dstar"] * PC) ** 2))
                    * self.sigma[ii, ij]
                    * self.radii[ii] ** 2
                    * (self.ag[ij] * UM) ** 3
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

        # Unit state
        if self._in_fnu:
            raise RuntimeError("Conversion to Fnu already called for this model.")

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
