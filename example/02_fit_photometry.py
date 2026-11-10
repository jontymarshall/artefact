"""Example 2: fit a model to photometry (HD 104860).

Scales the photosphere to the optical/near-IR photometry, then fits the dust mass,
grain size slope and belt radius to the infrared excess with lmfit. The thermal
emission uses the quick blackbody-temperature mode during the fit and the full
solution for the final model.

Run from anywhere:  python example/02_fit_photometry.py [--no-show]
"""

import argparse

import lmfit
import matplotlib.pyplot as plt
import numpy as np

from example_utils import (
    EXAMPLE,
    base_parameters,
    model_at,
    plot_sed,
    run_model,
    save_figure,
)

TARGET = "HD104860"
SCALE_RANGE = (0.45, 5.0)  # um, photometry used to scale the photosphere
EXCESS_MIN = 10.0  # um, photometry fitted for the disc


def load_photometry(target=TARGET):
    """Wavelength (um), flux (mJy) and uncertainty (mJy) from the example file."""
    return np.loadtxt(
        EXAMPLE / f"{target}_photometry.txt",
        delimiter=",",
        comments="#",
        usecols=(0, 1, 2),
        unpack=True,
    )


def parameters_for(theta):
    """HD 104860 model parameters, with the fitted values in ``theta`` applied."""
    return base_parameters(
        # F8V star at 48 pc; the G2V template is rescaled to the photometry
        lstar=1.318,
        rstar=0.94,
        tstar=5950.0,
        dstar=48.0,
        ngrain=31,
        nwav=120,
        amin=4.3,
        dtype="gauss",
        nring=21,
        rfwhm=60.0,
        mdust=10 ** theta["log_mdust"],
        q=theta["q"],
        rpeak=theta["rpeak"],
    )


def main(show=True):
    wave, flux, unc = load_photometry()
    photometry = (wave, flux, unc)
    excess = wave > EXCESS_MIN

    params = lmfit.Parameters()
    params.add("log_mdust", value=-2.0, min=-5.0, max=0.0)
    params.add("q", value=-3.5, min=-4.5, max=-3.0)
    params.add("rpeak", value=110.0, min=20.0, max=250.0)

    def residual(theta):
        model = run_model(
            parameters_for(theta),
            photometry=photometry,
            scale_wave=SCALE_RANGE,
            emission_mode="bb",
        )
        return (model_at(model, wave[excess]) - flux[excess]) / unc[excess]

    result = lmfit.minimize(residual, params, method="least_squares")
    print(lmfit.fit_report(result))

    # Re-run the best fit with the full temperature solution
    best = run_model(
        parameters_for(result.params.valuesdict()),
        photometry=photometry,
        scale_wave=SCALE_RANGE,
        emission_mode="full",
    )
    print(f"Reduced chi^2 (excess points): {result.redchi:.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_sed(best, ax=ax, title=f"{TARGET} best fit")
    save_figure(fig, "02_fit_photometry.png")

    # Optional: sample the posterior with emcee (pip install emcee corner)
    #
    #   fit = lmfit.minimize(residual, result.params, method="emcee",
    #                        nwalkers=32, steps=1000, burn=300, thin=5)
    #   import corner
    #   corner.corner(fit.flatchain, labels=list(fit.var_names))

    if show:
        plt.show()
    return result, best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures only")
    args = parser.parse_args()
    main(show=not args.no_show)
