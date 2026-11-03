#!/usr/bin/env python3
"""Generate a stellar photosphere for artefact with Starfish (astrostarfish).

Workflow
--------
1. ``build``  - download a PHOENIX subgrid, convert it to a low-resolution HDF5 grid
   that keeps *absolute* fluxes, train a Starfish spectral emulator and save it.
   Slow (minutes to hours); run once per region of parameter space.
2. ``export`` - reconstruct the spectrum of one star from the emulator and write it as
   a plain text file that artefact's existing ``stype: spectrum`` reads. Needs only
   Starfish, so it works even when Starfish and artefact cannot share a Python
   environment (see the note below).
3. ``run``    - drive artefact with ``stype: starfish`` (requires the changes to
   ``make_star`` described in the notes, and an environment in which both packages
   import) and compare with a blackbody star of the same Teff.

Environment note
----------------
astrostarfish 0.4.2 declares ``Python >=3.7,<3.10`` on PyPI, whereas artefact needs
Python >=3.10 (``match`` statements) and NumPy >=2.0. As published they cannot be
installed in one environment. Use ``build`` and ``export`` in a Python 3.9 environment
with astrostarfish, and run artefact (``stype: spectrum``) in your main environment.

Examples
--------
    python starfish_example.py build  --teff-range 5000 6500 --logg-range 4.0 5.0 \\
                                      --feh-range -0.5 0.5 --emulator sun_like_emu.hdf5
    python starfish_example.py export --emulator sun_like_emu.hdf5 \\
                                      --teff 5772 --logg 4.44 --feh 0.0 --out sun.txt
    python starfish_example.py run    --emulator sun_like_emu.hdf5 \\
                                      --teff 5772 --logg 4.44 --feh 0.0

Nothing here has been run against the real astrostarfish package; the calls follow the
Starfish 0.4.x documentation and source (grid_tools, emulator).
"""

import argparse

import numpy as np

SPEED_OF_LIGHT_KMS = 299_792.458
PER_CM_TO_PER_ANGSTROM = 1e-8  # erg/s/cm2/cm -> erg/s/cm2/A


# ---------------------------------------------------------------------------
# Step 1: build the emulator
# ---------------------------------------------------------------------------


def build_emulator(
    grid_dir,
    hdf5_name,
    emulator_name,
    ranges,
    resolving_power=100.0,
    wl_range=(3000.0, 54000.0),
    download=True,
    max_training_attempts=3,
):
    """Download, pre-process and emulate a PHOENIX subgrid for SED work.

    Parameters
    ----------
    ranges : list of (min, max)
        Limits of (Teff [K], logg [cgs dex], [Fe/H] [dex]). The emulator can only be
        queried inside this box, and the box must contain the star.
    resolving_power : float
        Resolving power R of the pre-processed grid. A photosphere for an SED needs
        no more than R ~ 100-500; the native PHOENIX R ~ 500,000 would give ~10^6
        pixels per spectrum and an impractically large PCA.
    wl_range : (float, float)
        Wavelength range in Angstrom. Widen the short end for hot stars, whose UV
        heats the dust.
    """
    from Starfish.emulator import Emulator
    from Starfish.grid_tools import (
        HDF5Creator,
        Instrument,
        PHOENIXGridInterfaceNoAlpha,
        download_PHOENIX_models,
    )

    class AbsoluteFluxPHOENIX(PHOENIXGridInterfaceNoAlpha):
        """PHOENIX grid that keeps absolute surface fluxes (erg/s/cm2/cm).

        HDF5Creator calls ``load_flux(params, header=True)``, whose default
        ``norm=True`` rescales every spectrum to the solar bolometric flux. That would
        discard the Teff^4 scaling that artefact needs, so it is switched off here.
        """

        def load_flux(self, parameters, header=False, norm=False):
            return super().load_flux(parameters, header=header, norm=False)

    if download:
        # Skips files already on disk. Full libraries exceed 10 GB; keep `ranges` small.
        download_PHOENIX_models(path=grid_dir, ranges=ranges)

    # air=False keeps vacuum wavelengths, which is what an SED model wants.
    grid = AbsoluteFluxPHOENIX(path=grid_dir, air=False, wl_range=wl_range)

    # Gaussian smoothing to the chosen resolving power; FWHM in km/s.
    sed_resolution = Instrument(
        name="SED", FWHM=SPEED_OF_LIGHT_KMS / resolving_power, wl_range=wl_range
    )
    HDF5Creator(
        grid, hdf5_name, instrument=sed_resolution, wl_range=wl_range, ranges=ranges
    ).process_grid()

    emulator = Emulator.from_grid(hdf5_name)
    for _ in range(max_training_attempts):
        emulator.train()
        if "Trained: True" in repr(emulator):  # the docs advise re-running if it fails
            break
    else:
        print("WARNING: the emulator did not report 'Trained: True'; inspect it.")
    print(emulator)
    emulator.save(emulator_name)
    return emulator


# ---------------------------------------------------------------------------
# Step 2: reconstruct a spectrum
# ---------------------------------------------------------------------------


def mean_spectrum(emulator, params):
    """Deterministic spectrum at ``params`` = (Teff, logg, [Fe/H]).

    ``Emulator.load_flux`` returns a *random draw* from the emulator's Gaussian
    process, so two calls differ. A model grid should be reproducible, so this uses
    the mean of the weight distribution instead, following the reconstruction recipe in
    the Starfish documentation.

    Returns
    -------
    wavelength : ndarray
        Angstrom.
    flux : ndarray
        Surface flux density F_lambda in erg/s/cm2/cm (PHOENIX units).
    """
    params = np.asarray(params, dtype=float)
    low, high = emulator.min_params, emulator.max_params
    if np.any(params < low) or np.any(params > high):
        raise ValueError(
            f"{tuple(params)} is outside the emulator grid {list(zip(low, high))}"
        )
    mean_weights, _ = emulator(params, full_cov=False)
    weights = np.reshape(mean_weights, (-1, emulator.ncomps))
    flux = weights @ (emulator.eigenspectra * emulator.flux_std) + emulator.flux_mean
    flux = np.squeeze(flux * emulator.norm_factor(params)[:, np.newaxis])
    return np.asarray(emulator.wl), flux


def check_bolometric_flux(wavelength_angstrom, flux_per_cm, teff):
    """Ratio of the emulated bolometric flux to sigma*Teff^4 (should be ~0.9-1.0)."""
    sigma_sb_cgs = 5.670374419e-5  # erg/s/cm2/K4
    flux_per_angstrom = flux_per_cm * PER_CM_TO_PER_ANGSTROM
    bolometric = np.trapezoid(flux_per_angstrom, wavelength_angstrom)
    return bolometric / (sigma_sb_cgs * teff**4)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_build(args):
    ranges = [args.teff_range, args.logg_range, args.feh_range]
    build_emulator(
        args.grid_dir,
        args.hdf5,
        args.emulator,
        ranges,
        resolving_power=args.resolving_power,
        wl_range=tuple(args.wl_range),
        download=not args.no_download,
    )


def cmd_export(args):
    from Starfish.emulator import Emulator

    emulator = Emulator.load(args.emulator)
    params = [args.teff, args.logg, args.feh]
    wavelength, flux = mean_spectrum(emulator, params)

    ratio = check_bolometric_flux(wavelength, flux, args.teff)
    print(f"bolometric flux / (sigma Teff^4) = {ratio:.3f}")
    if not 0.9 < ratio < 1.1:
        print("WARNING: far from 1 - check the grid normalisation and wavelength range.")

    header = (
        f"Starfish emulator spectrum, Teff={args.teff} logg={args.logg} [Fe/H]={args.feh}\n"
        "wavelength[Angstrom]  surface_flux[erg/s/cm2/A]"
    )
    np.savetxt(
        args.out, np.c_[wavelength, flux * PER_CM_TO_PER_ANGSTROM], header=header
    )
    print(f"wrote {args.out}; use 'stype : spectrum' and 'model : {args.out}' in artefact")


def cmd_run(args):
    import artefact

    common = dict(
        tstar=args.teff, rstar=args.rstar, lstar=args.lstar, dstar=args.dstar,
        lmin=0.5, lmax=2000.0, nwav=300,
    )  # fmt: skip
    starfish = artefact.RTModel()
    starfish.parameters.update(
        common,
        stype="starfish", model=args.emulator, logg=args.logg, feh=args.feh,
    )  # fmt: skip
    blackbody = artefact.RTModel()
    blackbody.parameters.update(common, stype="blackbody")

    for model in (starfish, blackbody):
        model.make_star()
        model.flam_to_fnu()  # F_lambda -> F_nu in mJy

    print(f"{'lambda [um]':>12} {'Starfish [mJy]':>16} {'blackbody [mJy]':>16} {'ratio':>7}")
    for wav in (0.55, 1.25, 2.2, 24.0, 70.0, 160.0):
        sf = np.interp(wav, starfish.sed_wave, starfish.sed_star)
        bb = np.interp(wav, blackbody.sed_wave, blackbody.sed_star)
        print(f"{wav:12.2f} {sf:16.4g} {bb:16.4g} {sf / bb:7.3f}")
    print(
        "\nAt wavelengths beyond the emulator's coverage the Planck tail makes the two "
        "agree closely; in the optical the ratio departs from 1 because of the real "
        "photospheric lines and opacity that a blackbody lacks."
    )


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_star_arguments(p):
        p.add_argument("--emulator", default="emulator.hdf5")
        p.add_argument("--teff", type=float, default=5772.0, help="K")
        p.add_argument("--logg", type=float, default=4.44, help="cgs dex")
        p.add_argument("--feh", type=float, default=0.0, help="[Fe/H], dex")

    build = sub.add_parser("build", help="download, pre-process and train")
    build.add_argument("--emulator", default="emulator.hdf5", help="output file")
    build.add_argument("--grid-dir", default="PHOENIX")
    build.add_argument("--hdf5", default="sed_grid.hdf5")
    build.add_argument("--teff-range", type=float, nargs=2, default=[5000.0, 6500.0])
    build.add_argument("--logg-range", type=float, nargs=2, default=[4.0, 5.0])
    build.add_argument("--feh-range", type=float, nargs=2, default=[-0.5, 0.5])
    build.add_argument("--resolving-power", type=float, default=100.0)
    build.add_argument("--wl-range", type=float, nargs=2, default=[3000.0, 54000.0],
                       help="Angstrom")  # fmt: skip
    build.add_argument("--no-download", action="store_true")
    build.set_defaults(func=cmd_build)

    export = sub.add_parser("export", help="write a text spectrum for stype: spectrum")
    add_star_arguments(export)
    export.add_argument("--out", default="star.txt")
    export.set_defaults(func=cmd_export)

    run = sub.add_parser("run", help="drive artefact with stype: starfish")
    add_star_arguments(run)
    run.add_argument("--rstar", type=float, default=1.0, help="R_sun")
    run.add_argument("--lstar", type=float, default=1.0, help="L_sun")
    run.add_argument("--dstar", type=float, default=10.0, help="pc")
    run.set_defaults(func=cmd_run)
    return parser


if __name__ == "__main__":
    arguments = make_parser().parse_args()
    arguments.func(arguments)
