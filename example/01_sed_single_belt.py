"""Example 1: the SED of a star with a single debris belt.

Builds a solar twin at 10 pc with a 40-60 au belt, runs the radiative transfer and
plots the stellar, thermal and scattered-light components, then shows how the
dust temperature varies with radius and grain size.

Run from anywhere:  python example/01_sed_single_belt.py [--no-show]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from example_utils import base_parameters, plot_sed, run_model, save_figure


def main(show=True):
    parameters = base_parameters(
        mdust=1e-3,  # Earth masses of dust
        dtype="onepl",  # single power-law belt
        rin=40.0,
        rout=60.0,
        alpha_out=-1.0,
    )
    model = run_model(parameters)

    # Dust mass is for grains up to amax; extrapolate to the whole planetesimal belt.
    mdisc = model.calculate_disc_mass()
    print(f"Dust mass (<{parameters['amax']:g} um): {parameters['mdust']:g} M_earth")
    print(f"Total belt mass (Krivov & Wyatt 2021): {mdisc:.3g} M_earth")

    fig, (ax_sed, ax_temp) = plt.subplots(1, 2, figsize=(12, 5))
    plot_sed(model, ax=ax_sed, title="Single belt, 40-60 au")

    # tdust has shape (nring, ngrain): temperature of each grain size at each radius
    for index in (0, len(model.ag) // 2, len(model.ag) - 1):
        ax_temp.semilogx(
            model.radii, model.tdust[:, index], label=f"a = {model.ag[index]:.3g} um"
        )
    ax_temp.set_xlabel("Radius (au)")
    ax_temp.set_ylabel("Dust temperature (K)")
    ax_temp.legend()
    ax_temp.set_title("Grain temperatures")

    path = save_figure(fig, "01_sed_single_belt.png")
    print(f"Saved {path}")
    if show:
        plt.show()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures only")
    args = parser.parse_args()
    main(show=not args.no_show)
