"""
Artefact: a 1D radiative transfer model for optically thin dust discs.

The model lives in :mod:`artefact.core`; this module exposes the public API so
that users can write::

    import artefact

    model = artefact.RTModel()
    model.get_parameters("parameters.txt")
    model.make_star()
    ...
    image = model.make_image(70.0, inc=60.0, pa=30.0)

Importing the package pulls in numba, astropy and miepython, so the first
import takes a few seconds while numba loads.
"""

# metadata
__author__ = "Jonathan P. Marshall"
__version__ = "0.3.0"
__copyright__ = "2023-26, Jonathan P. Marshall"
__email__ = ""
__license__ = "GNU GPL V3"
__url__ = "https://github.com/jontymarshall/artefact"

# package contents
# Stellar-fit settings (priors, sample counts, ...) are not re-exported here;
# import them from :mod:`artefact.phoenix`.
from ._data import data_path, list_data
from .core import (
    AU,
    C_LIGHT,
    CGS_FLAM_TO_SI,
    FWHM_PER_SIGMA,
    FWHM_TO_SIGMA,
    G_CM3_TO_KG_M3,
    H_PLANCK,
    K_B,
    L_SUN,
    M_EARTH,
    MJY_PER_SI,
    PC,
    Q_BIG,
    Q_MED,
    R_SUN,
    S_KM,
    S_MAX,
    S_MM,
    SCALE_WAVE_MAX,
    SCALE_WAVE_MIN,
    SI_PER_MJY,
    SIGMA_SB,
    T_BB_1AU,
    T_CMB,
    UM,
    ModBBModel,
    RTModel,
    StarModel,
    calculate_luminosity,
    calculate_rbb,
    calculate_rre,
    calculate_tbb,
    disc_mass_three_slope,
    extrapolate_wavelengths,
    modified_blackbody,
    planck_lam,
    planck_nu,
    ring_grid,
)
from .fitting import BandResponse, FilterCurve, Photometer, band_weights, read_filter
from .image import (
    ARCSEC_PER_DEG,
    ASPECT_RATIO,
    HG_G1,
    HG_G2,
    HG_WEIGHT,
    MIN_ASPECT,
    MIN_INC,
    ModelImage,
    convolve_beam,
    disc_surface_brightness,
    henyey_greenstein,
    two_component_hg,
)
from .phoenix import (
    PhoenixGrid,
    Quantity,
    StellarFit,
    build_grid,
    effective_errors,
    extinction_av,
    fit_star,
    load_grid,
    main_sequence_mass,
    percentile_errors,
    sample_covariance,
    save_grid,
    stellar_luminosity_lsun,
    verify_shipped_grid,
)

# public API
__all__ = [
    "__version__",
    # physical constants (SI) and unit conversions
    "AU",
    "C_LIGHT",
    "CGS_FLAM_TO_SI",
    "FWHM_PER_SIGMA",
    "FWHM_TO_SIGMA",
    "G_CM3_TO_KG_M3",
    "H_PLANCK",
    "K_B",
    "L_SUN",
    "M_EARTH",
    "MJY_PER_SI",
    "PC",
    "R_SUN",
    "SI_PER_MJY",
    "SIGMA_SB",
    "T_CMB",
    "UM",
    # model defaults
    "Q_BIG",
    "Q_MED",
    "S_KM",
    "S_MAX",
    "S_MM",
    "SCALE_WAVE_MAX",
    "SCALE_WAVE_MIN",
    "T_BB_1AU",
    "ARCSEC_PER_DEG",
    "ASPECT_RATIO",
    "HG_G1",
    "HG_G2",
    "HG_WEIGHT",
    "MIN_ASPECT",
    "MIN_INC",
    # models and images
    "RTModel",
    "StarModel",
    "ModBBModel",
    "ModelImage",
    # bundled data
    "data_path",
    "list_data",
    # physics helpers
    "calculate_luminosity",
    "calculate_rbb",
    "calculate_rre",
    "calculate_tbb",
    "disc_mass_three_slope",
    "extrapolate_wavelengths",
    "modified_blackbody",
    "planck_lam",
    "planck_nu",
    "ring_grid",
    # imaging
    "convolve_beam",
    "disc_surface_brightness",
    "henyey_greenstein",
    "two_component_hg",
    # synthetic photometry
    "BandResponse",
    "FilterCurve",
    "Photometer",
    "band_weights",
    "read_filter",
    # stellar fitting (PHOENIX grid)
    "PhoenixGrid",
    "Quantity",
    "StellarFit",
    "build_grid",
    "effective_errors",
    "extinction_av",
    "fit_star",
    "load_grid",
    "main_sequence_mass",
    "percentile_errors",
    "sample_covariance",
    "save_grid",
    "stellar_luminosity_lsun",
    "verify_shipped_grid",
]
