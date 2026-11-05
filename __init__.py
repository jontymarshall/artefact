"""
Artefact: a 1D radiative transfer model for optically thin dust discs.

The model lives in :mod:`artefact.core`; this module exposes the public API so
that users can write::

    import artefact

    model = artefact.RTModel()
    model.read_json("parameters.json")
    model.make_star()
    ...
    image = model.make_image(70.0, inc=60.0, pa=30.0)

Importing the package pulls in numba, astropy and miepython, so the first
import takes a few seconds while numba loads.
"""
# metadata
__author__ = "Jonathan P. Marshall"
__version__ = "0.0.8"
__copyright__ = "2023-26, Jonathan P. Marshall"
__email__ = ""
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

# package contents
from .core import (
    # model and image classes
    RTModel,
    # stellar and dust helpers
    calc_luminosity,
    extrapolate_wavelengths,
    planck_lam,
    planck_nu,
    ring_grid,
    # disc mass
    disc_mass_three_slope,
    # physical constants (SI) and unit conversions
    AU,
    ARCSEC_PER_DEG,
    C_LIGHT,
    CGS_FLAM_TO_SI,
    G_CM3_TO_KG_M3,
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
    ASPECT_RATIO,
    FWHM_PER_SIGMA,
    FWHM_TO_SIGMA,
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
    SIGMA_SB,
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


# public API
__all__ = [
    "ModelImage",
    "RTModel",
    "calc_luminosity",
    "convolve_beam",
    "disc_mass_three_slope",
    "disc_surface_brightness",
    "extrapolate_wavelengths",
    "henyey_greenstein",
    "planck_lam",
    "planck_nu",
    "ring_grid",
    "two_component_hg",
    "__version__",
]
