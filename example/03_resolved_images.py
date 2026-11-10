"""Example 3: resolved model images of a debris belt.

Uses the HD 104860-like belt (60-170 au at 48 pc) to make thermal images at 70 um and
1.3 mm, a scattered-light image at 1.6 um, the effect of inclination and beam size,
and writes a FITS file.

Run from anywhere:  python example/03_resolved_images.py [--no-show]
"""

import argparse

import matplotlib.pyplot as plt

from example_utils import OUTPUT, base_parameters, run_model, save_figure, show_image


def main(show=True):
    parameters = base_parameters(
        lstar=1.318,
        rstar=0.94,
        tstar=5950.0,
        dstar=48.0,
        amin=4.3,
        mdust=7.2e-3,
        dtype="onepl",
        rin=60.0,
        rout=170.0,
        alpha_out=0.0,
        nring=41,
    )
    model = run_model(parameters)

    # 1. Thermal emission and scattered light at different wavelengths
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    show_image(
        model.make_image(70.0, component="emit", inc=60.0, pa=30.0, beam=5.0),
        ax=axes[0],
        title="70 um thermal, 5\" beam",
    )
    show_image(
        model.make_image(1300.0, component="emit", inc=60.0, pa=30.0, beam=1.0),
        ax=axes[1],
        title="1.3 mm thermal, 1\" beam",
    )
    show_image(
        model.make_image(1.6, component="scat", inc=60.0, pa=30.0, g1=0.7, g2=-0.2),
        ax=axes[2],
        log=True,
        title="1.6 um scattered light",
    )
    save_figure(fig, "03_images_wavelengths.png")

    # 2. Inclination and position angle (thin disc, no beam)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, inc in zip(axes, (0.0, 45.0, 80.0)):
        show_image(
            model.make_image(70.0, component="emit", inc=inc, pa=30.0),
            ax=ax,
            title=f"inc = {inc:g} deg",
        )
    save_figure(fig, "03_images_inclination.png")

    # 3. Write a FITS file with a sky WCS in the header
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = model.make_image(850.0, inc=60.0, pa=30.0, beam=2.0, unit="mJy/beam")
    image.writeto(OUTPUT / "03_hd104860_850um.fits")
    print(f"Wrote {OUTPUT / '03_hd104860_850um.fits'}")

    if show:
        plt.show()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-show", action="store_true", help="save figures only")
    args = parser.parse_args()
    main(show=not args.no_show)
