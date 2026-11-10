"""Example 4: a two-belt system, from SED to fit to images.

1. Model a star with a warm inner belt and a cold outer belt (``dtype="multi"``).
2. Generate synthetic photometry from that model, then fit it back with lmfit to
   recover the dust mass, size slope, warm-belt mass fraction and belt radii.
3. Image the best-fit system at wavelengths that pick out each belt.

Run from anywhere:  python example/04_two_belt_system.py [--no-show]
"""

import argparse

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from example_utils import (
    model_at,
    plot_sed,
    run_model,
    save_figure,
    show_image,
    two_belt_parameters,
)

# Photometric bands (um) of a typical dataset: optical, near-IR, WISE, MIPS, PACS, ALMA
WAVELENGTHS = np.array(
    [0.44, 0.55, 0.79, 1.25, 1.65, 2.2, 3.4, 4.6, 12.0, 22.0, 24.0, 70.0, 100.0, 160.0,
     450.0, 870.0, 1300.0]
)
TRUTH = dict(log_mdust=-1.3, q=-3.5, warm_fraction=0.02, rpeak_warm=5.0, rpeak_cold=80.0)


def synthetic_photometry(seed=42):
    """Photometry (um, mJy, mJy) sampled from the true model with 5% noise."""
    truth = run_model(two_belt_parameters(**TRUTH), emission_mode="bb")
    rng = np.random.default_rng(seed)
    flux = model_at(truth, WAVELENGTHS)
    unc = 0.05 * flux
    return WAVELENGTHS, flux + rng.normal(0.0, unc), unc


def main(show=True):
    # --- 1. the model --------------------------------------------------------------
    model = run_model(two_belt_parameters(**TRUTH))
    print(f"{len(model.radii)} annuli, {model.belt_scale.shape[0]} belts")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_sed(model, ax=ax, belts=True, title="Two-belt system (truth)")
    save_figure(fig, "04_two_belt_truth.png")

    # --- 2. fit the synthetic photometry --------------------------------------------
    photometry = synthetic_photometry()
    wave, flux, unc = photometry
    fit_mask = wave > 8.0  # the star is known; fit the infrared excess

    params = lmfit.Parameters()
    params.add("log_mdust", value=-1.0, min=-4.0, max=1.0)
    params.add("q", value=-3.6, min=-4.5, max=-3.0)
    params.add("warm_fraction", value=0.05, min=1e-4, max=0.5)
    params.add("rpeak_warm", value=4.0, min=2.0, max=15.0)  # keeps rpeak - 3 sigma > 0
    params.add("rpeak_cold", value=70.0, min=40.0, max=150.0)

    def residual(theta):
        model = run_model(
            two_belt_parameters(**theta.valuesdict()), emission_mode="bb"
        )
        return (model_at(model, wave[fit_mask]) - flux[fit_mask]) / unc[fit_mask]

    result = lmfit.minimize(residual, params, method="least_squares")
    print(lmfit.fit_report(result))
    for name, true_value in TRUTH.items():
        print(f"{name:>14}: fit {result.params[name].value:8.3g}   true {true_value:8.3g}")

    best = run_model(two_belt_parameters(**result.params.valuesdict()))
    best.obs_wave, best.obs_flux, best.obs_uncs = photometry

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_sed(best, ax=ax, belts=True, title="Two-belt system (fit)")
    save_figure(fig, "04_two_belt_fit.png")

    # --- 3. images of the best fit ---------------------------------------------------
    # The warm belt is ~0.3 arcsec across at 30 pc, the cold belt several arcsec, so
    # zoom in on the warm belt with a fine pixel scale (a 1.3 arcsec field of view).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    show_image(
        best.make_image(
            24.0, component="emit", inc=50.0, pa=20.0, npix=128, pixscale=0.01,
            beam=0.04,
        ),
        ax=axes[0],
        title="24 um, zoomed: warm belt",
    )
    show_image(
        best.make_image(70.0, component="emit", inc=50.0, pa=20.0, beam=1.0),
        ax=axes[1],
        title="70 um: both belts",
    )
    show_image(
        best.make_image(870.0, component="emit", inc=50.0, pa=20.0, beam=0.5),
        ax=axes[2],
        title="870 um: cold belt",
    )
    save_figure(fig, "04_two_belt_images.png")

    if show:
        plt.show()
    return best, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures only")
    args = parser.parse_args()
    main(show=not args.no_show)
