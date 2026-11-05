"""
Stellar SED fit: effective temperature, radius and luminosity from photometry, plus mass relations.

The photosphere is a PHOENIX model scaled by (R / d)^2 and reddened by a Fitzpatrick (1999) curve. Stars
within 100 pc get no extinction; beyond that the pysed rule applies (Av = distance in kpc, Rv = 3.1).
Fitting uses lmfit; asymmetric errors come from sampling the fit covariance (plus the distance error) 
and taking the 16th and 84th percentiles of each fitted or derived quantity.
"""

import hashlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lmfit import Parameters, minimize

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHOENIX_DIR = . / "data" / "stellar_models" / "phoenix"
CACHE_PATH = . / "data" / "stellar_models" / "phoenix_grid.npz"
SHIPPED_PATH = . / "data" / "stellar_models" / "phoenix_grid.npz"  # the committed copy of the cache
SHIPPED_CHECKSUM_PATH = . / "data" / "stellar_models" / "phoenix_grid.sha256"
WAVE_FILE = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
METALLICITY_FOLDERS = {-0.5: "Z-0.5", 0.0: "Z-0.0", 0.5: "Z+0.5"}
BIN_EDGES_UM = np.logspace(np.log10(0.05), np.log10(5.4), 1201)  # 1200 bins
LIGHT_SPEED_CM_S = 2.99792458e10
_FILE_NAME = re.compile(r"lte(\d+)-(\d\.\d\d)[+-]\d\.\d")


SIGMA_SB = 5.670374419e-8  # W / m^2 / K^4
L_SUN_W = 3.828e26
R_SUN_M = 6.957e8
PARSEC_M = 3.0856775814913673e16
NO_EXTINCTION_WITHIN_PC = 100.0
RV = 3.1

STELLAR_MIN_UM = 0.3  # shortward of this, chromospheric / UV excess
STELLAR_MAX_UM = 4.0  # longward of this, disks start to contribute
MIN_STELLAR_POINTS = 4
MISSING_ERROR_FRACTION = 0.20  # assumed uncertainty of a flux whose catalog gives no error (e.g. IRAS)
ERROR_FLOOR = 0.03  # photometry errors are never smaller than 3% of the flux (calibration, saturation)
SAMPLES = 2000
# Priors on log g and [Fe/H] for the fit (mean, sigma). Photometry alone barely constrains them and, left
# free, they inflate the errors of Teff and the radius tenfold; these keep the fit physical for a nearby star.
LOGG_PRIOR_SIGMA_GAIA = 0.3  # around the Gaia GSP-Phot log g, when there is one
LOGG_PRIOR = (4.4, 0.5)  # dwarfs and subgiants, when there is no Gaia value
FEH_PRIOR = (0.0, 0.3)  # solar neighbourhood

@dataclass
class PhoenixGrid:
    """Binned model spectra: `log_flux_nu[mh, logg, teff, wave]` is log10 of the surface F_nu in Jy.

    A surface flux times (R / d)^2 is the flux a distant observer sees.
    """

    teff: np.ndarray  # K, ascending
    logg: np.ndarray
    mh: np.ndarray
    wave_um: np.ndarray  # bin centres, ascending
    log_flux_nu: np.ndarray

    @staticmethod
    def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        """The two neighbouring nodes of `axis` around `value` (clamped to its range) and the weight of the
        upper one; a one-node axis gives (0, 0, 0)."""
        if len(axis) == 1:
            return 0, 0, 0.0
        value = float(np.clip(value, axis[0], axis[-1]))
        upper = int(np.clip(np.searchsorted(axis, value), 1, len(axis) - 1))
        return upper - 1, upper, (value - axis[upper - 1]) / (axis[upper] - axis[upper - 1])

    def describe(self, logg: float, mh: float) -> str:
        return f"PHOENIX-ACES logg={logg:.2f} [M/H]={mh:+.2f} (interpolated between grid nodes)"

    def surface_flux_jy(
        self, wave_um: np.ndarray, teff: float, logg: float = 4.5, mh: float = 0.0
    ) -> np.ndarray:
        """Surface F_nu (Jy) at the given wavelengths for a temperature, log g and [M/H] between grid nodes.

        The log flux is interpolated linearly in Teff, log g and [M/H]; values outside the grid take its
        edge. Wavelengths past the model's end follow a Rayleigh-Jeans tail from its last bin.
        """
        t_low, t_high, t_weight = self._bracket(self.teff, teff)
        g_low, g_high, g_weight = self._bracket(self.logg, logg)
        m_low, m_high, m_weight = self._bracket(self.mh, mh)
        log_flux = 0.0
        for i_mh, w_mh in ((m_low, 1 - m_weight), (m_high, m_weight)):
            for i_logg, w_logg in ((g_low, 1 - g_weight), (g_high, g_weight)):
                spectra = self.log_flux_nu[i_mh, i_logg]
                at_teff = (1 - t_weight) * spectra[t_low] + t_weight * spectra[t_high]
                log_flux = log_flux + w_mh * w_logg * at_teff

        wave = np.asarray(wave_um, dtype=float)
        inside = np.interp(np.log10(wave), np.log10(self.wave_um), log_flux)
        tail = log_flux[-1] - 2.0 * np.log10(wave / self.wave_um[-1])
        return 10 ** np.where(wave > self.wave_um[-1], tail, inside)


def load_grid(
    directory: Path = PHOENIX_DIR,
    cache: Path = CACHE_PATH,
    shipped: Path = SHIPPED_PATH,
    checksum_path: Path = SHIPPED_CHECKSUM_PATH,
) -> PhoenixGrid:
    """The binned grid: the cache if there is one, else the committed copy (checked against its SHA-256),
    else built from the model files (minutes) and cached. Raises FileNotFoundError if there is none."""
    if cache.exists():
        return _read_cache(cache)
    if shipped.exists():
        verify_shipped_grid(shipped, checksum_path)
        return _read_cache(shipped)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no PHOENIX grid: none of {cache}, {shipped} or the model files in {directory} exist"
        )
    logger.warning("building the PHOENIX cache from %s: this takes a few minutes", directory)
    grid = build_grid(directory)
    save_grid(grid, cache)
    return grid

def _read_wavelength_um(directory: Path) -> np.ndarray:
    from astropy.io import fits

    return fits.getdata(directory / WAVE_FILE).astype(float) * 1e-4  # Angstrom -> micron


def _read_cache(path: Path) -> PhoenixGrid:
    data = np.load(path)
    return PhoenixGrid(data["teff"], data["logg"], data["mh"], data["wave_um"], data["log_flux_nu"])


def file_checksum(path: Path) -> str:
    """SHA-256 of a file, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_shipped_grid(path: Path = SHIPPED_PATH, checksum_path: Path = SHIPPED_CHECKSUM_PATH) -> None:
    """Raise ValueError unless the committed grid matches its recorded SHA-256 (a changed or damaged file)."""
    expected = checksum_path.read_text().split()[0]
    if file_checksum(path) != expected:
        raise ValueError(f"{path} does not match {checksum_path.name}: the grid was changed or damaged")


def build_grid(directory: Path = PHOENIX_DIR) -> PhoenixGrid:
    """Read every model file under `directory` and bin it. Slow (minutes): see `load_grid`."""
    from astropy.io import fits

    wave_um = _read_wavelength_um(directory)
    edges = BIN_EDGES_UM
    centres = np.sqrt(edges[:-1] * edges[1:])
    counts, _ = np.histogram(wave_um, bins=edges)
    mh_values = sorted(METALLICITY_FOLDERS)
    found: dict[tuple[int, float, float], Path] = {}
    for mh, folder in METALLICITY_FOLDERS.items():
        for path in (directory / folder).glob("lte*.fits"):
            match = _FILE_NAME.match(path.name)
            if match:
                found[(int(match[1]), float(match[2]), mh)] = path
    if not found:
        raise FileNotFoundError(f"no PHOENIX model files under {directory}")

    teffs = sorted({key[0] for key in found})
    loggs = sorted({key[1] for key in found})
    grid = np.full((len(mh_values), len(loggs), len(teffs), len(centres)), np.nan, dtype=np.float32)
    for (teff, logg, mh), path in found.items():
        flux_lambda = fits.getdata(path).astype(float)  # erg/s/cm^2/cm
        binned = np.histogram(wave_um, bins=edges, weights=flux_lambda)[0] / np.maximum(counts, 1)
        flux_nu_jy = binned * (centres * 1e-4) ** 2 / LIGHT_SPEED_CM_S * 1e23  # erg/s/cm^2/Hz -> Jy
        grid[mh_values.index(mh), loggs.index(logg), teffs.index(teff)] = np.log10(
            np.maximum(flux_nu_jy, 1e-30)
        )
        logger.info("binned %s", path.name)
    return PhoenixGrid(np.array(teffs, float), np.array(loggs), np.array(mh_values), centres, grid)


def save_grid(grid: PhoenixGrid, path: Path = CACHE_PATH) -> None:
    """Save photosphere model grid to .npz file."""
    np.savez_compressed(
        path, teff=grid.teff, logg=grid.logg, mh=grid.mh, wave_um=grid.wave_um, log_flux_nu=grid.log_flux_nu
    )


@dataclass
class Quantity:
    """A value with positive lower and upper 1-sigma errors (None when not determined)."""

    value: float
    err_lo: float | None = None
    err_hi: float | None = None


@dataclass
class StellarFit:
    teff: Quantity
    radius_rsun: Quantity
    luminosity_lsun: Quantity
    chi2: float
    dof: int
    n_points: int
    av: float
    model_grid: str
    logg: Quantity = field(default_factory=lambda: Quantity(4.5))  # log10 g in cgs, from the model grid
    feh: Quantity = field(default_factory=lambda: Quantity(0.0))  # [Fe/H] = the grid's [M/H]
    luminosity_samples: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    flux_at: "callable" = field(repr=False, default=None)  # wavelength (micron) -> best-fit stellar flux (Jy)


def extinction_av(distance_pc: float) -> float:
    """Visual extinction assumed for a star: none within 100 pc, else 1 mag per kpc (the pysed rule)."""
    return 0.0 if distance_pc <= NO_EXTINCTION_WITHIN_PC else distance_pc / 1000.0


def reddening_factor(wave_um: np.ndarray, av: float) -> np.ndarray:
    """Multiplicative flux factor 10^(-0.4 A_lambda) from a Fitzpatrick (1999) curve.

    The curve is defined for 0.1-3.3 micron; longward of that the extinction is taken as zero (it is below 4%
    of Av there) and shortward of it the 0.1 micron value is used.
    """
    wave = np.asarray(wave_um, dtype=float)
    if av <= 0:
        return np.ones_like(wave)
    from dust_extinction.parameter_averages import F99

    model = F99(Rv=RV)
    x = 1.0 / np.clip(wave, 0.1, None)  # inverse micron
    inside = (x >= 0.3) & (x <= 10.0)
    a_lambda = np.zeros_like(wave)
    a_lambda[inside] = av * model.evaluate(x[inside], RV)
    return 10 ** (-0.4 * a_lambda)


def effective_errors(flux: np.ndarray, err: np.ndarray, floor: float = ERROR_FLOOR) -> np.ndarray:
    """Errors with the calibration floor applied; a flux without an error counts as 20% uncertain."""
    err = np.where(np.isfinite(err) & (err > 0), err, MISSING_ERROR_FRACTION * flux)
    return np.maximum(err, floor * flux)


def sample_covariance(
    mean: list[float], covar: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray | None:
    """Draws from a normal distribution with the fit's covariance, or None when there is no usable covariance.

    A covariance that is not quite positive semi-definite (common when a parameter sits on a bound) is
    repaired by clipping its negative eigenvalues to zero.
    """
    if covar is None or not np.all(np.isfinite(covar)):
        return None
    symmetric = (covar + covar.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    repaired = (vectors * np.clip(values, 0.0, None)) @ vectors.T
    return rng.multivariate_normal(mean, repaired, SAMPLES, check_valid="ignore")


def percentile_errors(value: float, samples: np.ndarray) -> Quantity:
    """`value` with lower/upper errors from the 16th and 84th percentiles of `samples` (never negative)."""
    samples = samples[np.isfinite(samples)]
    if len(samples) < 20:
        return Quantity(value)
    low, high = np.percentile(samples, [16, 84])
    return Quantity(value, max(value - low, 0.0), max(high - value, 0.0))


def stellar_luminosity_lsun(radius_rsun: np.ndarray, teff: np.ndarray) -> np.ndarray:
    """L = 4 pi R^2 sigma Teff^4 in solar luminosities."""
    return 4 * np.pi * (radius_rsun * R_SUN_M) ** 2 * SIGMA_SB * np.asarray(teff, dtype=float) ** 4 / L_SUN_W


def _model_flux(
    grid: PhoenixGrid, wave_um: np.ndarray, teff: float, radius_rsun: float, distance_pc: float,
    logg: float, mh: float, av: float,
) -> np.ndarray:  # fmt: skip
    solid = (radius_rsun * R_SUN_M / (distance_pc * PARSEC_M)) ** 2
    return grid.surface_flux_jy(wave_um, teff, logg, mh) * solid * reddening_factor(wave_um, av)


def fit_star(
    phot: Photometer,
    distance_pc: float,
    grid: PhoenixGrid,
    *,
    distance_err_pc: float | None = None,
    teff_start: float | None = None,
    logg: float = 4.5,
    mh: float = 0.0,
    fit_surface: bool = True,
    logg_prior: tuple[float, float] | None = None,
    mh_prior: tuple[float, float] | None = None,
) -> StellarFit:
    """Fit Teff, radius, log g and [Fe/H] to the photometry between 0.3 and 4 micron (limits excluded).

    The start temperature is the best node of the grid (each with its best scale) unless `teff_start` is
    given; `logg` and `mh` are the start values of log g and [Fe/H] (the Gaia log g is a good start). Both are
    fitted within the grid's range, between its nodes, unless `fit_surface` is False or the grid has a single
    node on that axis. Photometry constrains them weakly, so their errors are wide and a fit that reaches the
    edge of the grid has no error on that side. `logg_prior` and `mh_prior` are (mean, sigma) Gaussian priors
    that pull the fitted value towards the mean; with a prior the errors are those of the combined fit. Raises
    ValueError when fewer than four usable points are available.
    """
    keep = (
        (phot.wave_um >= STELLAR_MIN_UM)
        & (phot.wave_um <= STELLAR_MAX_UM)
        & ~phot.is_limit
        & (phot.flux_jy > 0)
    )
    data = phot.select(keep)
    if len(data.wave_um) < MIN_STELLAR_POINTS:
        raise ValueError(
            f"only {len(data.wave_um)} photometry points between 0.3 and 4 micron; need {MIN_STELLAR_POINTS}"
        )
    sigma = effective_errors(data.flux_jy, data.err_jy)
    av = extinction_av(distance_pc)
    logg = float(np.clip(logg, grid.logg[0], grid.logg[-1]))
    mh = float(np.clip(mh, grid.mh[0], grid.mh[-1]))

    def chi2_for_node(teff: float) -> tuple[float, float]:
        shape = data.evaluate(  # per solar radius
            lambda wave: _model_flux(grid, wave, teff, 1.0, distance_pc, logg, mh, av)
        )
        scale = np.sum(data.flux_jy * shape / sigma**2) / np.sum(shape**2 / sigma**2)  # best (R/Rsun)^2
        return float(np.sum(((data.flux_jy - scale * shape) / sigma) ** 2)), float(np.sqrt(max(scale, 1e-12)))

    if teff_start is None:
        scores = [(chi2_for_node(t)[0], t) for t in grid.teff]
        teff_start = min(scores)[1]
    radius_start = chi2_for_node(teff_start)[1]

    params = Parameters()
    params.add("teff", value=teff_start, min=float(grid.teff[0]), max=float(grid.teff[-1]))
    params.add("radius", value=radius_start, min=0.02, max=1000.0)
    for name, value, axis in (("logg", logg, grid.logg), ("mh", mh, grid.mh)):
        if fit_surface and len(axis) > 1:
            params.add(name, value=value, min=float(axis[0]), max=float(axis[-1]))
        else:  # a one-node axis (or a fixed surface): the start value stays, with no bounds to violate
            params.add(name, value=value, vary=False)

    def model_for(p: Parameters, wave_um: np.ndarray) -> np.ndarray:
        return _model_flux(
            grid, wave_um, p["teff"].value, p["radius"].value, distance_pc, p["logg"].value, p["mh"].value, av
        )

    def data_residual(p: Parameters) -> np.ndarray:
        return (data.flux_jy - data.evaluate(lambda wave: model_for(p, wave))) / sigma

    def residual(p: Parameters) -> np.ndarray:
        extra = [
            (p[name].value - prior[0]) / prior[1]
            for name, prior in (("logg", logg_prior), ("mh", mh_prior))
            if prior is not None and p[name].vary
        ]  # a Gaussian prior is one more residual
        return np.append(data_residual(p), extra)

    result = minimize(residual, params)
    fitted = result.params
    teff, radius = fitted["teff"].value, fitted["radius"].value
    best_logg, best_mh = fitted["logg"].value, fitted["mh"].value
    luminosity = float(stellar_luminosity_lsun(radius, teff))

    # draws from the covariance of the parameters that were fitted (in lmfit's order)
    rng = np.random.default_rng(0)
    names = list(result.var_names)
    sampled = sample_covariance([fitted[n].value for n in names], result.covar, rng)
    draws = {n: np.full(SAMPLES, fitted[n].value) for n in ("teff", "radius", "logg", "mh")}
    if sampled is not None:
        for column, name in enumerate(names):
            draws[name] = sampled[:, column]
    draws["teff"] = np.clip(draws["teff"], grid.teff[0], grid.teff[-1])
    draws["radius"] = np.abs(draws["radius"])
    draws["logg"] = np.clip(draws["logg"], grid.logg[0], grid.logg[-1])
    draws["mh"] = np.clip(draws["mh"], grid.mh[0], grid.mh[-1])
    # R follows from the angular size and the distance, so the distance error widens R and L
    distance_draws = np.full(SAMPLES, distance_pc)
    if distance_err_pc:
        distance_draws = np.abs(rng.normal(distance_pc, distance_err_pc, SAMPLES))
    radii = draws["radius"] * distance_draws / distance_pc
    luminosities = stellar_luminosity_lsun(radii, draws["teff"])

    def surface_quantity(name: str, value: float) -> Quantity:
        if name not in names or sampled is None:  # not fitted, or no usable covariance: no error to give
            return Quantity(value)
        return percentile_errors(value, draws[name])

    def flux_at(wave_um: np.ndarray) -> np.ndarray:
        return model_for(fitted, np.asarray(wave_um, dtype=float))

    return StellarFit(
        teff=percentile_errors(teff, draws["teff"]),
        radius_rsun=percentile_errors(radius, radii),
        luminosity_lsun=percentile_errors(luminosity, luminosities),
        chi2=float(np.sum(data_residual(fitted) ** 2)),  # the data only, not the priors
        dof=max(len(data.wave_um) - len(names), 1),
        n_points=len(data.wave_um),
        av=av,
        model_grid=grid.describe(best_logg, best_mh),
        logg=surface_quantity("logg", best_logg),
        feh=surface_quantity("mh", best_mh),
        luminosity_samples=luminosities,
        flux_at=flux_at,
    )


# --- mass ---
def main_sequence_mass(luminosity_lsun: np.ndarray) -> np.ndarray:
    """Mass (solar) of a main-sequence star from its luminosity (inverts Salaris & Cassisi 2005).

    L = 0.23 M^2.3 (M < 0.43), M^4 (0.43 <= M < 2), 1.4 M^3.5 (2 <= M < 55), 32000 M (M >= 55).
    """
    lum = np.atleast_1d(np.asarray(luminosity_lsun, dtype=float))
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