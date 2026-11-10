"""Shared helpers for the artefact example scripts and notebooks.

The helpers only wrap the public ``RTModel`` workflow so the examples stay short:

    run_model()        build and run a complete SED model from a parameter dict
    plot_sed()         plot star, dust emission, scattered light and photometry
    belt_sed()         split a multi-belt model into the SED of each belt
    model_at()         interpolate a model SED onto observed wavelengths

Paths are resolved relative to the repository, so the examples work from any
working directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
try:
    from artefact import RTModel
except ImportError:  # not pip-installed: the repository directory *is* the package
    sys.path.insert(0, str(REPO.parent))
    from artefact import RTModel

DATA = REPO / "data"
EXAMPLE = REPO / "example"
OUTPUT = REPO / "example" / "output"

# RTModel appends ".lnk" to ``composition`` itself, so no extension here.
ASTROSIL = str(DATA / "optical_constants" / "astrosil")
G2V_SPECTRUM = str(DATA / "stellar_models" / "G2V_spectrum.txt")
A0V_SPECTRUM = str(DATA / "stellar_models" / "A0V_spectrum.txt")


def base_parameters(**overrides):
    """A fast, sensible starting model: a solar twin at 10 pc with a single belt.

    The grids (``nwav``, ``ngrain``, ``nring``) are deliberately coarse so each model
    takes seconds; increase them for publication-quality results.
    """
    parameters = dict(
        # star
        stype="spectrum",  # "blackbody", "spectrum" (SVO-style), "json" (SDB) or "phoenix"
        model=G2V_SPECTRUM,
        lstar=1.0,  # L_sun
        rstar=1.0,  # R_sun
        tstar=5770.0,  # K
        dstar=10.0,  # pc
        # dust
        composition=ASTROSIL,
        density=3.3,  # g cm^-3
        amin=1.0,  # um
        amax=1000.0,  # um
        ngrain=41,
        q=-3.5,  # dN/da ~ a^q; -3.5 is a collisional cascade
        mdust=1e-3,  # Earth masses of dust between amin and amax
        # wavelength grid
        lmin=0.3,  # um
        lmax=3000.0,  # um
        nwav=150,
        # belt
        dtype="onepl",
        nring=31,
        rin=40.0,  # au
        rout=60.0,  # au
        alpha_out=-1.0,
    )
    parameters.update(overrides)
    return parameters


def two_belt_parameters(
    log_mdust=-1.3,
    q=-3.5,
    warm_fraction=0.02,
    rpeak_warm=5.0,
    rpeak_cold=80.0,
    fwhm_warm=2.0,
    fwhm_cold=30.0,
):
    """A F-type star at 30 pc with a warm inner belt and a cold outer belt.

    The belts share one grain size distribution; ``warm_fraction`` is the warm belt's
    share of the total dust mass ``10**log_mdust`` (Earth masses). Each belt is a
    Gaussian ring given by its peak radius and FWHM in au.
    """
    return base_parameters(
        stype="blackbody",
        tstar=6700.0,
        rstar=1.6,
        lstar=4.0,
        dstar=30.0,
        amin=2.0,
        ngrain=31,
        nwav=120,
        q=q,
        mdust=10**log_mdust,
        dtype="multi",
        belts=[
            dict(dtype="gauss", rpeak=rpeak_warm, rfwhm=fwhm_warm, nring=10,
                 mass=warm_fraction),
            dict(dtype="gauss", rpeak=rpeak_cold, rfwhm=fwhm_cold, nring=20,
                 mass=1.0 - warm_fraction),
        ],
    )


def run_model(
    parameters,
    photometry=None,
    scale_wave=None,
    emission_mode="full",
    tolerance=1e-3,
):
    """Run the full RTModel workflow and return the finished model.

    Parameters
    ----------
    parameters : dict
        Model parameters, see ``base_parameters``.
    photometry : tuple of arrays, optional
        ``(wavelength_um, flux_mJy, uncertainty_mJy)``, attached to the model.
    scale_wave : tuple of float, optional
        ``(lmin, lmax)`` in um. If given, the photosphere is scaled to the photometry
        in this range (needs ``photometry``).
    emission_mode : {"full", "bb"}
        "full" solves the grain temperature with the real absorption efficiencies;
        "bb" uses the blackbody approximation, which is much quicker, so it suits
        fitting loops.
    tolerance : float
        Relative tolerance of the temperature solver.

    Returns
    -------
    RTModel
        With ``sed_wave``, ``sed_star``, ``sed_emit``, ``sed_scat`` and ``sed_disc``
        all in mJy.
    """
    model = RTModel()
    model.parameters.update(parameters)

    if photometry is not None:
        model.obs_wave, model.obs_flux, model.obs_uncs = (
            np.asarray(column, dtype=float) for column in photometry
        )

    model.make_star()
    if scale_wave is not None:
        model.scale_star(lmin=scale_wave[0], lmax=scale_wave[1])

    model.make_dust()
    model.read_optical_constants()
    model.make_disc()
    # A multi-belt disc has its own number of annuli (the union of each belt's edges),
    # so tell the emission and scattering loops how many there really are.
    model.parameters["nring"] = len(model.radii)
    model.calculate_surface_density()
    model.calculate_qabs()
    model.calculate_dust_scatter()
    model.calculate_dust_emission(mode=emission_mode, tolerance=tolerance)
    model.flam_to_fnu()
    return model


def total_sed(model):
    """Star + thermal emission + scattered light in mJy on ``model.sed_wave``."""
    return model.sed_star + model.sed_emit + model.sed_scat


def model_at(model, wavelengths):
    """Total model flux (mJy) at the given wavelengths, interpolated in log-log."""
    return np.exp(
        np.interp(np.log(wavelengths), np.log(model.sed_wave), np.log(total_sed(model)))
    )


def belt_sed(model):
    """Thermal emission (mJy) of each belt of a multi-belt model.

    ``model.belt_scale[b, i]`` is the fraction of all grains that belt ``b`` puts in
    annulus ``i``, and ``model.scale[i]`` is the total fraction in that annulus, so
    belt ``b`` owns ``belt_scale / scale`` of the light of each annulus. Annuli with
    no dust (the gap between belts) are skipped.

    Returns
    -------
    ndarray, shape (nbelt, nwav)
    """
    per_ring = model.sed_ringe.sum(axis=1)  # (nring, nwav), summed over grain size
    share = np.divide(
        model.belt_scale,
        model.scale[None, :],
        out=np.zeros_like(model.belt_scale),
        where=model.scale[None, :] > 0,
    )
    return share @ per_ring


def plot_sed(model, ax=None, belts=False, title=None):
    """Log-log SED with the stellar, thermal and scattered components and the photometry."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    wave = model.sed_wave
    ax.loglog(wave, model.sed_star, "k-.", lw=1, label="Photosphere")
    ax.loglog(wave, model.sed_emit, "r:", label="Thermal emission")
    ax.loglog(wave, model.sed_scat, "b:", label="Scattered light")
    ax.loglog(wave, total_sed(model), "k-", label="Total")

    if belts and getattr(model, "belt_scale", None) is not None:
        for index, emission in enumerate(belt_sed(model)):
            ax.loglog(wave, emission, "--", lw=1.2, label=f"Belt {index + 1} emission")

    if model.obs_wave is not None:
        ax.errorbar(
            model.obs_wave,
            model.obs_flux,
            yerr=model.obs_uncs,
            fmt="o",
            mfc="white",
            mec="k",
            ecolor="k",
            label="Photometry",
        )

    disc = np.nanmax(model.sed_emit + model.sed_scat)
    star = np.nanmax(model.sed_star)
    ax.set_xlim(wave[0], wave[-1])
    ax.set_ylim(min(disc, star) * 1e-4, max(disc, star) * 10)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Flux density (mJy)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def show_image(image, ax=None, log=False, cmap="inferno", title=None):
    """Display a ``ModelImage`` with north up and east left, axes in arcsec."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    data = image.data
    half = 0.5 * data.shape[0] * image.pixscale
    if log:
        positive = data[data > 0]
        vmin = positive.max() * 1e-3 if positive.size else None
        data = np.where(data > vmin, data, vmin) if vmin else data
        norm = plt.matplotlib.colors.LogNorm(vmin=vmin)
    else:
        norm = None

    shown = ax.imshow(
        data,
        origin="lower",
        extent=[half, -half, -half, half],  # east is to the left
        cmap=cmap,
        norm=norm,
    )
    ax.set_xlabel(r"$\Delta$RA (arcsec)")
    ax.set_ylabel(r"$\Delta$Dec (arcsec)")
    ax.set_title(title or f"{image.wavelength:g} $\\mu$m ({image.component})")
    plt.colorbar(shown, ax=ax, label=image.unit, fraction=0.046, pad=0.04)
    return ax


def save_figure(fig, name, outdir=OUTPUT):
    """Save ``fig`` as ``outdir/name`` (creating the directory) and return the path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
