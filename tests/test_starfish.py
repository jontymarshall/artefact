"""Tests for the ``stype="starfish"`` branch of ``RTModel.make_star``.

These tests do NOT need astrostarfish. They inject a small stand-in for
``Starfish.emulator.Emulator`` (implementing only the documented attributes the
branch uses) into ``sys.modules``, so they check artefact's own logic: parameter
handling, the reconstruction of the mean spectrum, unit conversion, the Planck tail,
and the guard rails. They do not test Starfish itself.

They assume the changes described in the accompanying notes have been applied to
``artefact.py`` (``read_starfish``, the ``"spectrum" | "starfish"`` branch,
``extrapolate_wavelengths(..., tstar=...)``) and, for the dust-temperature test, the
``/ np.pi`` fix for spectrum-type stars.

Note: two assertions in ``test_artefact.py`` describe the *old* behaviour and must be
updated once starfish is implemented:

* ``test_make_star_rejects_unknown_and_unimplemented_types``: drop the
  ``NotImplementedError`` check for ``stype="starfish"``.
* ``test_dust_temperature_error_handling``: use another invalid ``stype`` (for example
  ``"nonsense"``) instead of ``"starfish"``.
"""

import sys
import types

import numpy as np
import pytest

import artefact as art
from artefact import RTModel

UM = art.UM


def rel(expected, tol=1e-6):
    return pytest.approx(expected, rel=tol, abs=0)


BASE = dict(
    lstar=1.0, rstar=1.0, tstar=5772.0, dstar=10.0,
    lmin=0.5, lmax=2000.0, nwav=300,
    logg=4.44, feh=0.0, model="emulator.hdf5",
)  # fmt: skip


def make_stand_in(teff=5772.0, scale=1.0, limits=((5000.0, 4.0, -0.5), (6500.0, 5.0, 0.5))):
    """A stand-in 'trained emulator' that reproduces a pi*B_lambda(Teff) surface flux.

    It mimics ``Emulator.from_grid``: spectra are stored divided by their mean (the
    ``factors``), and ``load_flux``-style reconstruction multiplies them back. Fluxes
    are in PHOENIX units, erg/s/cm2/cm, on a wavelength grid in Angstrom
    (0.3-54 um, the Starfish PHOENIX default).
    """

    class StandInEmulator:
        def __init__(self):
            self.wl = np.geomspace(3000.0, 54000.0, 4000)
            lam = self.wl * 1e-10
            planck = (
                2 * art.H_PLANCK * art.C_LIGHT**2
                / lam**5
                / np.expm1(art.H_PLANCK * art.C_LIGHT / (lam * art.K_B * teff))
            )  # W/m2/m/sr
            spectrum = scale * np.pi * planck * 10.0  # 1 W/m2/m = 10 erg/s/cm2/cm
            self._factor = spectrum.mean()
            self.flux_mean = spectrum / self._factor
            self.eigenspectra = np.zeros((1, self.wl.size))
            self.flux_std = np.ones(self.wl.size)
            self.ncomps = 1
            self.min_params = np.array(limits[0])
            self.max_params = np.array(limits[1])

        @classmethod
        def load(cls, filename):
            return cls()

        def __call__(self, params, full_cov=True, reinterpret_batch=False):
            return np.zeros(self.ncomps), np.zeros(self.ncomps)

        def norm_factor(self, params):
            return np.array([self._factor])

    return StandInEmulator


@pytest.fixture
def install_emulator(monkeypatch):
    """Return a function that installs a stand-in ``Starfish.emulator`` module."""

    def _install(**kwargs):
        package = types.ModuleType("Starfish")
        module = types.ModuleType("Starfish.emulator")
        module.Emulator = make_stand_in(**kwargs)
        monkeypatch.setitem(sys.modules, "Starfish", package)
        monkeypatch.setitem(sys.modules, "Starfish.emulator", module)

    return _install


def build(stype, **overrides):
    model = RTModel()
    model.parameters.update({**BASE, "stype": stype, **overrides})
    return model


def test_planck_tail_replaces_lambda_minus_four_when_a_temperature_is_given():
    lam = np.array([3.0, 4.0, 5.4])
    flux = np.array([3.0, 2.0, 1.0])
    target = np.array([1.0, 1000.0])
    rj_lam, rj_flux = art.extrapolate_wavelengths(lam, target, flux, nwav=50)
    pl_lam, pl_flux = art.extrapolate_wavelengths(lam, target, flux, nwav=50, tstar=5772.0)

    assert pl_lam == rel(rj_lam, 1e-12)
    expected = (
        art.planck_lam(pl_lam[3:] * UM, 5772.0) / art.planck_lam(5.4 * UM, 5772.0)
    )
    assert pl_flux[3:] == rel(expected * flux[-1], 1e-12)
    # Anchored on the last model point, but ~20% above the lambda^-4 tail at long wavelength
    assert pl_flux[3] == rel(flux[-1], 1e-12)
    assert pl_flux[-1] / rj_flux[-1] == pytest.approx(1.27, rel=0.05)


def test_starfish_star_reproduces_the_blackbody_it_was_built_from(install_emulator):
    install_emulator()
    starfish, blackbody = build("starfish"), build("blackbody")
    starfish.make_star()
    blackbody.make_star()

    assert starfish.sed_wave == rel(blackbody.sed_wave, 1e-12)
    inside = starfish.sed_wave <= 5.4  # the emulator's own coverage
    assert starfish.sed_star[inside] == rel(blackbody.sed_star[inside], 1e-4)
    # Beyond it the Planck tail should stay within a couple of percent
    # (the old lambda^-4 tail would be ~20% low at 100 um).
    assert starfish.sed_star[~inside] == rel(blackbody.sed_star[~inside], 2e-2)


def test_starfish_star_gives_the_same_dust_temperature_as_a_blackbody(install_emulator):
    install_emulator()
    starfish = build("starfish", lmin=0.5, lmax=1e4, nwav=3000)
    blackbody = build("blackbody", lmin=0.5, lmax=1e4, nwav=3000)
    starfish.make_star()
    blackbody.make_star()
    ones = np.ones(3000)
    for radius in (1.0, 50.0):
        reference = blackbody.calculate_dust_temperature(radius, ones)
        assert starfish.calculate_dust_temperature(radius, ones) == rel(reference, 1e-2)


def test_starfish_rejects_parameters_outside_the_emulator_grid(install_emulator):
    install_emulator()
    for override in (dict(tstar=7000.0), dict(logg=3.0), dict(feh=1.0)):
        with pytest.raises(ValueError, match="outside the emulator grid"):
            build("starfish", **override).make_star()


def test_starfish_rejects_lmin_below_the_emulator_coverage(install_emulator):
    install_emulator()
    with pytest.raises(ValueError, match="shorter than the emulator"):
        build("starfish", lmin=0.1).make_star()


def test_starfish_warns_if_the_grid_was_built_with_normalised_fluxes(install_emulator):
    # A grid normalised to solar bolometric flux would be too faint for a hotter star;
    # scale=0.62 mimics that.
    install_emulator(scale=0.62)
    with pytest.warns(UserWarning, match="bolometric flux"):
        build("starfish").make_star()


def test_starfish_gives_a_clear_error_when_astrostarfish_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "Starfish.emulator", None)  # makes the import fail
    with pytest.raises(ImportError, match="astrostarfish"):
        build("starfish").make_star()


def test_starfish_is_a_valid_stype_and_is_listed_in_the_error_message():
    with pytest.raises(ValueError, match="starfish"):
        build("nonsense").make_star()
