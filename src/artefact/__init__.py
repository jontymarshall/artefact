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
from .core import (
    # model and image classes
    StarModel,
    ModBBModel,
    RTModel,
    # stellar and dust helpers
    calculate_luminosity,
    calculate_tbb,
    calculate_rbb,
    calculate_rre,
    extrapolate_wavelengths,
    planck_lam,
    planck_nu,
    modified_blackbody,
    ring_grid,
    # disc mass
    disc_mass_three_slope,
    # physical constants (SI) and unit conversions
    AU,
    C_LIGHT,
    CGS_FLAM_TO_SI,
    G_CM3_TO_KG_M3,
    FWHM_PER_SIGMA,
    FWHM_TO_SIGMA,
    H_PLANCK,
    K_B,
    L_SUN,
    M_EARTH,
    MJY_PER_SI,
    PC,
    R_SUN,
    SI_PER_MJY,
    SIGMA_SB,
    UM,
    #model defaults
    Q_BIG,
    Q_MED,
    S_KM,
    S_MAX,
    S_MM,
    SCALE_WAVE_MAX,
    SCALE_WAVE_MIN,
    T_BB_1AU,
    T_CMB,
)

from .image import (
    ModelImage,
    # imaging
    convolve_beam,
    disc_surface_brightness,
    henyey_greenstein,
    two_component_hg,
    # model defaults
    ARCSEC_PER_DEG,
    ASPECT_RATIO,
    HG_G1,
    HG_G2,
    HG_WEIGHT,
    MIN_ASPECT,
    MIN_INC,
)


from .fitting import (
    Photometer,
    BandResponse,
    FilterCurve,
    band_weights,
    read_filter,
)


from .phoenix import (
    PhoenixGrid,
    Quantity,
    StellarFit,
    # grid functions
    load_grid,
    save_grid,
    verify_shipped_grid,
    build_grid,
    # model constants
    L_SUN_W,
    R_SUN_M,
    PARSEC_M,
    # model defaults
    NO_EXTINCTION_WITHIN_PC,
    RV,
    STELLAR_MIN_UM,
    STELLAR_MAX_UM,
    MIN_STELLAR_POINTS,
    MISSING_ERROR_FRACTION,
    ERROR_FLOOR,
    SAMPLES,
    LOGG_PRIOR,
    FEH_PRIOR,
    # star functions
    extinction_av,
    sample_covariance,
    effective_errors,
    percentile_errors,
    stellar_luminosity_lsun,
    fit_star,
    main_sequence_mass,
)


from ._data import (
    data_path,
    list_data,
)


# public API
__all__ = [
    "ModelImage",
    "StarModel",
    "ModBBModel",
    "RTModel",
    "calculate_luminosity",
    "calculate_tbb",
    "calculate_rbb",
    "calculate_rre",
    "convolve_beam",
    "disc_mass_three_slope",
    "disc_surface_brightness",
    "extrapolate_wavelengths",
    "planck_lam",
    "planck_nu",
    "ring_grid",
    "henyey_greenstein",
    "two_component_hg",
    "__version__",
]
