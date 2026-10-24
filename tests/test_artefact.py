"""Tests for artefact.py.

Run from the repository root with::

    pip install -e ".[test]"
    pytest

The tests are grouped by the part of the model they exercise:

1. Physics helpers and unit constants, checked against analytic results
   (Wien's law, Stefan-Boltzmann, Rayleigh-Jeans, ...).
2. Parameter and file input.
3. Stellar photosphere (blackbody, SVO-style spectrum, JSON) and ``scale_star``.
4. Dust grains, optical constants, and the radial disc structure.
5. Mie efficiencies, scattered light, dust temperature and thermal emission.
6. Unit conversion and an end-to-end run.

Where possible the expected values come from an independent calculation, not
from re-implementing the code under test. ``miepython`` is replaced by a
deterministic stand-in except in ``test_real_mie_limits``, so that everything
else tests artefact's own plumbing and maths.
"""

import json

import numpy as np
import pytest

import artefact as art
from artefact import RTModel

C = art.C_LIGHT
UM = art.UM


def rel(expected, tol=1e-6):
    """Relative-only comparison.

    ``pytest.approx`` has a default absolute floor of 1e-12, which would make
    comparisons of small SI-unit values (1e-8 W/m2/m and below) vacuous.
    """
    return pytest.approx(expected, rel=tol, abs=0)


# ---------------------------------------------------------------------------
# Shared parameters, fixtures and helpers
# ---------------------------------------------------------------------------

BASE_PARAMS = dict(
    stype="blackbody",
    lstar=1.0,  # L_sun
    rstar=1.0,  # R_sun
    tstar=5772.0,  # K
    dstar=10.0,  # pc
    lmin=0.5,  # um
    lmax=2000.0,  # um
    nwav=200,
    amin=1.0,  # um
    amax=100.0,  # um
    density=2.5,  # g/cm3
    ngrain=4,
    mdust=0.01,  # Earth masses
    q=-3.5,
    dtype="onepl",
    rin=40.0,  # au
    rout=80.0,  # au
    rpeak=60.0,  # au
    rfwhm=20.0,  # au
    alpha_in=2.0,
    alpha_out=-1.0,
    nring=5,
)


def fake_mie(m, x):
    """Deterministic stand-in for ``miepython.mie``: (qext, qsca, qback, g)."""
    x = np.asarray(x, dtype=float)
    qext = 2.0 * (1.0 - np.exp(-x))
    return qext, 0.4 * qext, 0.0 * qext, 0.0 * qext


@pytest.fixture
def model_factory():
    """Return a function building an RTModel with ``BASE_PARAMS`` + overrides."""

    def _make(**overrides):
        model = RTModel()
        model.parameters.update({**BASE_PARAMS, **overrides})
        return model

    return _make


@pytest.fixture
def disc_model(model_factory, monkeypatch):
    """A model run up to (and including) ``calculate_qabs`` with mocked Mie."""
    monkeypatch.setattr(art.mpy, "mie", fake_mie)
    model = model_factory()
    model.make_star()
    model.make_dust()
    model.oc_nk = np.full(model.sed_wave.shape, 1.5 - 0.1j)
    model.make_disc()
    model.calculate_surface_density()
    model.calculate_qabs()
    return model


@pytest.fixture
def wide_blackbody(model_factory):
    """Blackbody star on a grid wide enough to hold cold-dust emission."""
    model = model_factory(lmin=0.1, lmax=1e4, nwav=3000)
    model.make_star()
    return model


def write_surface_flux_file(path, tstar, lam_min=0.01, lam_max=1e5, n=8000):
    """SVO-style file: wavelength [A], surface flux [erg/s/cm2/A] = pi*B_lambda."""
    lam = np.geomspace(lam_min, lam_max, n)  # um
    surface_si = np.pi * art.planck_lam(lam * UM, tstar)  # W/m2/m
    np.savetxt(
        path,
        np.c_[lam * 1e4, surface_si / art.CGS_FLAM_TO_SI],
        header="wavelength[A] flux[erg/s/cm2/A]",
    )


def write_observed_flux_json(path, tstar, rstar, dstar, lam_min=0.01, lam_max=1e5, n=4000):
    """JSON file holding the *observed* flux density (Jy) of a blackbody star."""
    lam = np.geomspace(lam_min, lam_max, n)  # um
    fnu_si = np.pi * art.planck_nu(C / (lam * UM), tstar)
    fnu_si = fnu_si * ((rstar * art.R_SUN) / (dstar * art.PC)) ** 2  # W/m2/Hz
    data = {"star_spec": {"wavelength": lam.tolist(), "fnujy": (fnu_si / 1e-26).tolist()}}
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# 1. Physics helpers and constants
# ---------------------------------------------------------------------------


def test_planck_lam_peaks_at_wien_wavelength():
    temperature = 5772.0
    wav = np.geomspace(1e-7, 1e-5, 20001)
    peak = wav[np.argmax(art.planck_lam(wav, temperature))]
    assert peak == rel(2.897771955e-3 / temperature, 1e-3)


def test_planck_lam_integrates_to_stefan_boltzmann():
    temperature = 5772.0
    wav = np.geomspace(1e-8, 1e-2, 200001)
    flux = np.pi * np.trapezoid(art.planck_lam(wav, temperature), wav)
    assert flux == rel(art.SIGMA_SB * temperature**4, 1e-4)


def test_planck_nu_is_planck_lam_in_frequency_units():
    wav = np.geomspace(1e-6, 1e-3, 50)
    lhs = art.planck_lam(wav, 300.0)
    rhs = art.planck_nu(C / wav, 300.0) * C / wav**2
    assert lhs == rel(rhs, 1e-10)


def test_planck_lam_rayleigh_jeans_limit():
    wav, temperature = 1.0, 300.0
    rayleigh_jeans = 2.0 * C * art.K_B * temperature / wav**4
    assert art.planck_lam(wav, temperature) == rel(rayleigh_jeans, 1e-3)


def test_calc_luminosity_solar_values_and_scaling():
    assert art.calc_luminosity(1.0, 5772.0) == rel(1.0, 2e-3)
    assert art.calc_luminosity(2.0, 5772.0) == rel(4.0 * art.calc_luminosity(1.0, 5772.0))
    assert art.calc_luminosity(1.0, 2 * 5772.0) == rel(16.0 * art.calc_luminosity(1.0, 5772.0))


def test_unit_conversion_constants():
    assert art.MJY_PER_SI * art.SI_PER_MJY == rel(1.0, 1e-12)
    assert art.CGS_FLAM_TO_SI == rel(1e7, 1e-12)  # erg/s/cm2/A -> W/m2/m
    assert art.FWHM_PER_SIGMA == rel(2 * np.sqrt(2 * np.log(2)), 1e-3)
    assert art.UM == 1e-6


def test_deprecated_lowercase_aliases_match_new_names():
    pairs = [("h", "H_PLANCK"), ("c", "C_LIGHT"), ("k", "K_B"), ("sb", "SIGMA_SB"),
             ("au", "AU"), ("pc", "PC"), ("lsol", "L_SUN"), ("rsol", "R_SUN"),
             ("MEarth", "M_EARTH"), ("um", "UM")]  # fmt: skip
    for old, new in pairs:
        assert getattr(art, old) == getattr(art, new)


def test_physics_helpers_are_reachable_from_the_class():
    expected = art.planck_lam(1e-6, 300.0)
    assert RTModel.planck_lam(1e-6, 300.0) == expected
    assert RTModel().planck_lam(1e-6, 300.0) == expected  # works on an instance too
    assert RTModel.calc_luminosity(1.0, 5772.0) == art.calc_luminosity(1.0, 5772.0)


def test_extrapolate_wavelengths_continues_as_rayleigh_jeans():
    lam = np.array([1.0, 2.0, 5.0, 10.0])
    flux = np.array([8.0, 6.0, 4.0, 2.0])
    target_grid = np.array([1.0, 50.0, 100.0])
    lam_ext, flux_ext = art.extrapolate_wavelengths(lam, target_grid, flux, nwav=7)

    assert len(lam_ext) == len(lam) + 7
    assert lam_ext[:4] == rel(lam)
    assert flux_ext[:4] == rel(flux)
    assert lam_ext[4] == rel(10.0)  # extension starts at the last model point
    assert lam_ext[-1] == rel(110.0)  # ... and ends 10% beyond the target grid
    assert flux_ext[4:] == rel(2.0 * (10.0 / lam_ext[4:]) ** 4)  # F_lam ~ lambda^-4


def test_new_model_starts_empty():
    model = RTModel()
    assert model.parameters == {}
    for name in ("sed_wave", "sed_star", "sed_emit", "sed_scat", "sed_disc",
                 "sed_rings", "sed_ringe", "ag", "ng", "mg", "oc_nk", "qext",
                 "qsca", "qabs", "scale", "radii", "dr", "areas", "sigma", "tdust"):  # fmt: skip
        assert getattr(model, name) is None, name
    assert model._in_fnu is False


# ---------------------------------------------------------------------------
# 2. Parameter input
# ---------------------------------------------------------------------------


def test_get_parameters_handles_comments_blank_lines_and_types(tmp_path):
    path = tmp_path / "params.txt"
    path.write_text(
        "# a full-line comment\n"
        "\n"
        "rstar : 1.5   # trailing comment\n"
        "stype : blackbody\n"
        "model : 'dir:file.txt'\n"
        "   nwav:  100\n"
        "\n"
    )
    model = RTModel()
    out = model.get_parameters(str(path))

    assert out is model.parameters
    assert out["rstar"] == 1.5 and isinstance(out["rstar"], float)
    assert out["nwav"] == 100.0
    assert out["stype"] == "blackbody"
    assert out["model"] == "dir:file.txt"  # quotes stripped, colon kept
    assert set(out) == {"rstar", "stype", "model", "nwav"}


# ---------------------------------------------------------------------------
# 3. Stellar photosphere
# ---------------------------------------------------------------------------


def test_make_star_blackbody_builds_log_grid(model_factory):
    model = model_factory(lmin=0.5, lmax=100.0, nwav=50)
    model.make_star()
    assert model.sed_wave.shape == model.sed_star.shape == (50,)
    assert model.sed_wave[0] == rel(0.5) and model.sed_wave[-1] == rel(100.0)
    assert np.diff(np.log(model.sed_wave)) == rel(np.full(49, np.log(200.0) / 49), 1e-9)
    assert np.all(np.isfinite(model.sed_star)) and np.all(model.sed_star > 0)


def test_make_star_blackbody_obeys_inverse_square_and_radius_scaling(model_factory):
    base, far, big = model_factory(), model_factory(dstar=20.0), model_factory(rstar=2.0)
    for model in (base, far, big):
        model.make_star()
    assert far.sed_star == rel(base.sed_star / 4, 1e-12)
    assert big.sed_star == rel(base.sed_star * 4, 1e-12)


def test_make_star_blackbody_matches_independent_flux_density_in_mjy(model_factory):
    model = model_factory()
    model.make_star()
    model.flam_to_fnu()
    nu = C / (model.sed_wave * UM)
    geometry = ((BASE_PARAMS["rstar"] * art.R_SUN) / (BASE_PARAMS["dstar"] * art.PC)) ** 2
    expected = np.pi * art.planck_nu(nu, BASE_PARAMS["tstar"]) * geometry * 1e3 * 1e26
    assert model.sed_star == rel(expected, 1e-9)


def test_make_star_spectrum_reproduces_blackbody(tmp_path, model_factory):
    write_surface_flux_file(tmp_path / "star.txt", BASE_PARAMS["tstar"])
    spectrum = model_factory(stype="spectrum", model=str(tmp_path / "star.txt"))
    blackbody = model_factory()
    spectrum.make_star()
    blackbody.make_star()
    assert spectrum.sed_wave == rel(blackbody.sed_wave, 1e-12)
    assert spectrum.sed_star == rel(blackbody.sed_star, 1e-3)


def test_make_star_spectrum_extrapolates_beyond_the_model(tmp_path, model_factory):
    write_surface_flux_file(tmp_path / "star.txt", 5772.0, lam_max=1000.0)
    model = model_factory(stype="spectrum", model=str(tmp_path / "star.txt"), lmax=4000.0)
    model.make_star()

    beyond = np.flatnonzero(model.sed_wave > 1200.0)
    first, last = beyond[0], beyond[-1]
    expected_ratio = (model.sed_wave[first] / model.sed_wave[last]) ** 4
    assert model.sed_star[last] / model.sed_star[first] == rel(expected_ratio, 2e-2)


def test_make_star_json_reproduces_blackbody(tmp_path, model_factory):
    write_observed_flux_json(tmp_path / "star.json", 5772.0, 1.0, 10.0)
    from_json = model_factory(stype="json", model=str(tmp_path / "star.json"))
    blackbody = model_factory()
    from_json.make_star()
    blackbody.make_star()
    assert from_json.sed_star == rel(blackbody.sed_star, 2e-3)


def test_read_json_converts_jy_to_si_f_lambda(tmp_path, model_factory):
    # Regression: a factor of 1e12 error crept in during refactoring.
    path = tmp_path / "star.json"
    path.write_text(json.dumps({"star_spec": {"wavelength": [10.0, 20.0], "fnujy": [1.0, 4.0]}}))
    waves, spect = model_factory(model=str(path)).read_json()
    assert waves == rel([10.0, 20.0])
    assert spect == rel([1e-26 * C / (10e-6) ** 2, 4e-26 * C / (20e-6) ** 2])


def test_read_json_accepts_integer_valued_files(tmp_path, model_factory):
    path = tmp_path / "star.json"
    path.write_text(json.dumps({"star_spec": {"wavelength": [10, 20], "fnujy": [1, 2]}}))
    _, spect = model_factory(model=str(path)).read_json()
    assert np.all(np.isfinite(spect)) and spect.dtype == float


def test_read_star_converts_angstrom_and_cgs_to_um_and_si(tmp_path, model_factory):
    path = tmp_path / "star.txt"
    path.write_text("# header\n10000.0 2.0\n20000.0 3.0\n")
    waves, spect = model_factory(model=str(path)).read_star()
    assert waves == rel([1.0, 2.0])  # Angstrom -> um
    assert spect == rel([2.0e7, 3.0e7])  # erg/s/cm2/A -> W/m2/m


def test_make_star_rejects_unknown_and_unimplemented_types(model_factory):
    with pytest.raises(ValueError, match="stype"):
        model_factory(stype="nonsense").make_star()
    with pytest.raises(NotImplementedError):
        model_factory(stype="starfish").make_star()


# ---------------------------------------------------------------------------
# 3b. scale_star
# ---------------------------------------------------------------------------


def photometry_from_model(model, factor, wavelengths):
    """Photometry (mJy) equal to ``factor`` x the model star at ``wavelengths``."""
    f_lam = np.interp(wavelengths, model.sed_wave, model.sed_star)
    return factor * f_lam * (wavelengths * UM) ** 2 / C


def test_scale_star_recovers_known_factor_and_ignores_points_outside_window(model_factory):
    model = model_factory()
    model.make_star()
    before = model.sed_star.copy()

    lobs = np.array([0.6, 1.0, 1.6, 2.2, 3.5, 4.5, 100.0])
    fobs = photometry_from_model(model, 0.8, lobs)
    fobs[-1] = 1e12  # far-IR point lies outside the default fit window
    model.obs_wave, model.obs_flux, model.obs_uncs = lobs, fobs, 0.05 * fobs

    model.scale_star()
    assert model.sed_star == rel(0.8 * before, 1e-6)


def test_scale_star_honours_custom_window(model_factory):
    model = model_factory()
    model.make_star()
    before = model.sed_star.copy()
    lobs = np.array([8.0, 10.0, 12.0, 15.0])
    fobs = photometry_from_model(model, 1.25, lobs)
    model.obs_wave, model.obs_flux, model.obs_uncs = lobs, fobs, 0.1 * fobs

    model.scale_star(lmin=5.0, lmax=20.0)
    assert model.sed_star == rel(1.25 * before, 1e-6)


def test_scale_star_requires_photometry_in_window(model_factory):
    model = model_factory()
    model.make_star()
    model.obs_wave = np.array([10.0, 20.0])
    model.obs_flux = model.obs_uncs = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="No photometry"):
        model.scale_star()


def test_scale_star_requires_model_to_cover_photometry(model_factory):
    model = model_factory(lmin=1.0)  # model grid starts at 1 um
    model.make_star()
    model.obs_wave = np.array([0.6, 1.5])
    model.obs_flux = model.obs_uncs = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="outside the model grid"):
        model.scale_star()


@pytest.mark.xfail(reason="scale_star does not yet reject a star already converted to F_nu")
def test_scale_star_after_fnu_conversion_is_rejected(model_factory):
    model = model_factory()
    model.make_star()
    model.flam_to_fnu()
    lobs = np.array([1.0, 2.0])
    model.obs_wave, model.obs_flux, model.obs_uncs = lobs, np.ones(2), np.ones(2)
    with pytest.raises((RuntimeError, ValueError)):
        model.scale_star()


# ---------------------------------------------------------------------------
# 4. Dust grains, optical constants and disc structure
# ---------------------------------------------------------------------------


def test_make_dust_size_distribution_and_mass_normalisation(model_factory):
    model = model_factory(amin=0.5, amax=500.0, ngrain=7, q=-3.5, density=3.0, mdust=0.02)
    model.make_dust()

    assert model.ag[0] == rel(0.5) and model.ag[-1] == rel(500.0)
    size_ratio = model.ag[1:] / model.ag[:-1]
    assert size_ratio == rel(np.full(6, size_ratio[0]), 1e-9)  # logarithmic spacing
    assert model.ng[1:] / model.ng[:-1] == rel(size_ratio**-3.5, 1e-9)  # power law
    assert model.mg == rel(4 / 3 * np.pi * (model.ag * UM) ** 3 * 3.0e3, 1e-12)
    assert np.sum(model.ng * model.mg) == rel(0.02 * art.M_EARTH, 1e-12)


def test_read_optical_constants_interpolates_and_forces_absorbing_sign(tmp_path, model_factory):
    (tmp_path / "dust.lnk").write_text(
        "# lambda  n  k\n0.1 1.5 0.01\n10.0 2.0 0.5\n3000.0 3.0 -2.0\n"
    )
    model = model_factory(composition=str(tmp_path / "dust"))
    model.sed_wave = np.array([0.1, 1.0, 10.0, 3000.0])
    model.read_optical_constants()

    frac = (1.0 - 0.1) / (10.0 - 0.1)
    expected = [
        1.5 - 0.01j,
        (1.5 + 0.5 * frac) - (0.01 + 0.49 * frac) * 1j,
        2.0 - 0.5j,
        3.0 - 2.0j,  # file has k = -2.0; code uses |k|
    ]
    assert model.oc_nk == rel(expected, 1e-12)


DISC_CASES = [
    ("gauss", dict(rpeak=60.0, rfwhm=20.0)),
    ("onepl", dict(rin=40.0, rout=80.0, alpha_out=-1.0)),
    ("twopl", dict(rin=40.0, rout=80.0, rpeak=60.0, alpha_in=3.0, alpha_out=-3.0)),
]


@pytest.mark.parametrize("dtype,extra", DISC_CASES)
def test_make_disc_basic_invariants(model_factory, dtype, extra):
    model = model_factory(dtype=dtype, nring=25, **extra)
    model.make_disc()
    assert model.radii.shape == model.scale.shape == model.areas.shape == (25,)
    assert np.all(np.diff(model.radii) > 0)
    assert np.all(model.scale >= 0) and np.sum(model.scale) == rel(1.0, 1e-12)
    assert model.dr == rel((model.radii[-1] - model.radii[0]) / 25)
    assert np.all(model.areas > 0)


def test_make_disc_single_power_law_weights_follow_ring_area(model_factory):
    # Linear radial spacing means ring area is proportional to radius, so the
    # ring weights must scale as r**(alpha + 1).
    model = model_factory(dtype="onepl", rin=40.0, rout=80.0, alpha_out=-1.5, nring=11)
    model.make_disc()
    assert model.radii[0] == rel(40.0) and model.radii[-1] == rel(80.0)
    expected = model.radii ** (-1.5 + 1.0)
    assert model.scale == rel(expected / expected.sum(), 1e-10)


def test_make_disc_two_power_law_has_continuous_break(model_factory):
    model = model_factory(
        dtype="twopl", rin=40.0, rout=80.0, rpeak=60.0, alpha_in=3.0, alpha_out=-3.0, nring=21
    )
    model.make_disc()
    r = model.radii
    surface = np.where(r <= 60.0, (r / 60.0) ** 3.0, (r / 60.0) ** -3.0)
    expected = surface * r
    assert model.scale == rel(expected / expected.sum(), 1e-10)


def test_make_disc_gaussian_clips_inner_edge_to_one_au(model_factory):
    model = model_factory(dtype="gauss", rpeak=5.0, rfwhm=20.0, nring=9)
    model.make_disc()
    assert model.radii[0] == rel(1.0)
    assert model.radii[-1] == rel(5.0 + 3.0 * 20.0 / art.FWHM_PER_SIGMA)


def test_make_disc_rejects_unknown_and_unimplemented_types(model_factory):
    with pytest.raises(NotImplementedError):
        model_factory(dtype="arbit").make_disc()
    with pytest.raises(ValueError, match="dtype"):
        model_factory(dtype="nonsense").make_disc()


def test_surface_density_distributes_each_grain_size_over_the_rings(model_factory):
    model = model_factory(nring=7, ngrain=5)
    model.make_dust()
    model.make_disc()
    model.calculate_surface_density()
    assert model.sigma.shape == (7, 5)
    assert model.sigma == rel(np.outer(model.scale, model.ng), 1e-12)
    assert model.sigma.sum(axis=0) == rel(model.ng, 1e-12)


# ---------------------------------------------------------------------------
# 5. Optics, scattering, temperature and emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_returned", [3, 4])
def test_calculate_qabs_accepts_three_or_four_mie_return_values(
    model_factory, monkeypatch, n_returned
):
    calls = []

    def mie(m, x):
        calls.append((m, np.array(x)))
        ones = np.ones_like(x)
        return (2.0 * ones, 0.5 * ones, 0.1 * ones, 0.0 * ones)[:n_returned]

    monkeypatch.setattr(art.mpy, "mie", mie)
    model = model_factory(ngrain=3, nwav=20)
    model.make_star()
    model.make_dust()
    model.oc_nk = np.full(20, 1.5 - 0.1j)
    model.calculate_qabs()

    assert model.qext.shape == model.qsca.shape == model.qabs.shape == (3, 20)
    assert model.qext == rel(np.full((3, 20), 2.0)) and model.qsca == rel(np.full((3, 20), 0.5))
    assert model.qabs == rel(np.full((3, 20), 1.5))
    assert len(calls) == 3
    for grain_size, (refractive_index, x) in zip(model.ag, calls):
        assert refractive_index is model.oc_nk
        assert x == rel(2 * np.pi * grain_size / model.sed_wave, 1e-12)  # size parameter


def test_real_mie_limits():
    """Uses the real miepython: Rayleigh scattering and the extinction paradox."""
    qext, qsca, *_ = art.mpy.mie(1.5 - 0.1j, np.array([0.01, 200.0]))
    qabs = qext - qsca
    assert qsca[0] < 1e-3 * qabs[0]  # tiny absorbing grains barely scatter
    assert qext[1] == pytest.approx(2.0, rel=0.1)  # large grains: Q_ext -> 2


def test_calculate_dust_scatter_structure_and_scaling(disc_model):
    model = disc_model
    model.calculate_dust_scatter()
    nring, ngrain, nwav = len(model.radii), len(model.ag), len(model.sed_wave)

    assert model.sed_rings.shape == (nring, ngrain, nwav)
    assert np.all(model.sed_rings > 0)  # every ring/grain contributes
    assert model.sed_rings.sum(axis=(0, 1)) == rel(model.sed_scat, 1e-12)

    # For a fixed grain size the contribution is proportional to the number of
    # grains in the ring and to 1/r^2.
    for j in range(ngrain):
        per_ring = model.sed_rings[:, j, :] * (model.radii**2 / model.sigma[:, j])[:, None]
        assert per_ring == rel(np.broadcast_to(per_ring[0], per_ring.shape), 1e-10)


def test_calculate_dust_scatter_is_linear_in_the_stellar_spectrum(disc_model):
    model = disc_model
    model.calculate_dust_scatter()
    first = model.sed_scat.copy()
    model.sed_star = model.sed_star * 3.0
    model.calculate_dust_scatter()
    assert model.sed_scat == rel(3.0 * first, 1e-12)


def test_dust_temperature_bb_mode_follows_the_278k_law(wide_blackbody, model_factory):
    ones = np.ones(wide_blackbody.sed_wave.size)
    assert wide_blackbody.calculate_dust_temperature(1.0, ones, mode="bb") == rel(278.0)
    assert wide_blackbody.calculate_dust_temperature(4.0, ones, mode="bb") == rel(139.0)
    luminous = model_factory(lstar=16.0)
    assert luminous.calculate_dust_temperature(1.0, ones, mode="bb") == rel(556.0)


@pytest.mark.parametrize("radius", [1.0, 4.0, 50.0])
def test_dust_temperature_of_blackbody_grains_matches_analytic_value(wide_blackbody, radius):
    ones = np.ones(wide_blackbody.sed_wave.size)
    expected = 5772.0 * np.sqrt(0.5 * art.R_SUN / (radius * art.AU))
    assert wide_blackbody.calculate_dust_temperature(radius, ones) == rel(expected, 2e-3)


def test_dust_temperature_physical_trends(wide_blackbody):
    model = wide_blackbody
    ones = np.ones(model.sed_wave.size)
    temps = [model.calculate_dust_temperature(r, ones) for r in (1.0, 2.0, 4.0, 8.0, 16.0)]
    assert np.all(np.diff(temps) < 0)  # colder further out
    assert temps[0] / temps[2] == rel(2.0, 2e-3)  # T ~ r^-1/2 for blackbody grains

    # A grey grain (constant efficiency) has the same temperature as a black one.
    assert model.calculate_dust_temperature(4.0, 0.3 * ones) == rel(temps[2], 1e-5)

    # Small grains (efficient absorbers, poor emitters) run hotter than blackbodies.
    small_grain = np.minimum(1.0, 2.0 / model.sed_wave)
    assert model.calculate_dust_temperature(4.0, small_grain) > 1.1 * temps[2]


@pytest.mark.parametrize("stype", ["spectrum", "json"])
def test_dust_temperature_does_not_depend_on_how_the_star_is_specified(
    tmp_path, model_factory, wide_blackbody, stype
):
    """The same star must heat dust equally whether given as a blackbody, an
    SVO-style spectrum (surface flux, pi*B), or an observed JSON spectrum."""
    if stype == "spectrum":
        write_surface_flux_file(tmp_path / "star.txt", 5772.0)
        extra = dict(model=str(tmp_path / "star.txt"))
    else:
        write_observed_flux_json(tmp_path / "star.json", 5772.0, 1.0, 10.0)
        extra = dict(model=str(tmp_path / "star.json"))
    other = model_factory(stype=stype, lmin=0.1, lmax=1e4, nwav=3000, **extra)
    other.make_star()

    ones = np.ones(3000)
    for radius in (1.0, 50.0):
        reference = wide_blackbody.calculate_dust_temperature(radius, ones)
        assert other.calculate_dust_temperature(radius, ones) == rel(reference, 1e-2)


def test_dust_temperature_error_handling(wide_blackbody, model_factory):
    ones = np.ones(wide_blackbody.sed_wave.size)
    with pytest.raises(ValueError, match="mode"):
        wide_blackbody.calculate_dust_temperature(1.0, ones, mode="nonsense")
    # Inside the star's own radius no temperature can balance heating and cooling.
    with pytest.raises(ValueError, match="different sign"):
        wide_blackbody.calculate_dust_temperature(1e-3, ones)

    bad = model_factory(stype="starfish")
    bad.sed_wave, bad.sed_star = np.geomspace(0.5, 100.0, 10), np.ones(10)
    with pytest.raises(ValueError, match="one of"):
        bad.calculate_dust_temperature(1.0, np.ones(10))


def test_calculate_dust_emission_structure_and_temperatures(disc_model):
    model = disc_model
    model.calculate_dust_emission()
    nring, ngrain, nwav = len(model.radii), len(model.ag), len(model.sed_wave)

    assert model.tdust.shape == (nring, ngrain)
    assert np.all(model.tdust > art.T_CMB) and np.all(model.tdust < BASE_PARAMS["tstar"])
    assert np.all(np.diff(model.tdust, axis=0) < 0)  # outer rings are colder
    assert model.sed_ringe.shape == (nring, ngrain, nwav)
    assert np.all(model.sed_ringe >= 0) and np.all(np.isfinite(model.sed_ringe))
    assert model.sed_emit == rel(model.sed_ringe.sum(axis=(0, 1)), 1e-12)


def test_calculate_dust_emission_bb_mode_and_tolerance(disc_model):
    model = disc_model
    model.calculate_dust_emission(mode="bb")
    expected = art.T_BB_1AU * BASE_PARAMS["lstar"] ** 0.25 * model.radii**-0.5
    assert model.tdust == rel(np.tile(expected[:, None], (1, len(model.ag))))

    model.calculate_dust_emission(mode="full", tolerance=1e-8)
    tight = model.tdust.copy()
    model.calculate_dust_emission(mode="full", tolerance=1e-3)
    assert model.tdust == rel(tight, 2e-3)


# ---------------------------------------------------------------------------
# 6. Unit conversion and end-to-end behaviour
# ---------------------------------------------------------------------------


def test_flam_to_fnu_converts_each_component_exactly_once(disc_model):
    model = disc_model
    model.calculate_dust_scatter()
    model.calculate_dust_emission(mode="bb")
    star, emit, scat = model.sed_star.copy(), model.sed_emit.copy(), model.sed_scat.copy()
    factor = (model.sed_wave * UM) ** 2 / C

    model.flam_to_fnu()

    assert model.sed_star == rel(star * factor, 1e-12)
    assert model.sed_emit == rel(emit * factor, 1e-12)
    assert model.sed_scat == rel(scat * factor, 1e-12)
    # Regression: the total used to be multiplied by the conversion factor twice.
    assert model.sed_disc == rel((emit + scat) * factor, 1e-12)
    assert model.sed_rings.sum(axis=(0, 1)) == rel(model.sed_scat, 1e-10)
    assert model.sed_ringe.sum(axis=(0, 1)) == rel(model.sed_emit, 1e-10)


def test_flam_to_fnu_second_call_leaves_everything_unchanged(disc_model):
    model = disc_model
    model.calculate_dust_scatter()
    model.calculate_dust_emission(mode="bb")
    model.flam_to_fnu()
    snapshot = {n: getattr(model, n).copy() for n in ("sed_star", "sed_emit", "sed_scat", "sed_disc")}
    model.flam_to_fnu()
    for name, values in snapshot.items():
        assert getattr(model, name) == rel(values, 1e-15), name


def test_flam_to_fnu_total_with_only_one_disc_component(disc_model, model_factory):
    scatter_only = disc_model
    scatter_only.calculate_dust_scatter()
    scatter_only.flam_to_fnu()
    assert scatter_only.sed_emit is None
    assert scatter_only.sed_disc == rel(scatter_only.sed_scat, 1e-15)

    star_only = model_factory()
    star_only.make_star()
    star_only.flam_to_fnu()
    assert star_only.sed_disc is None


@pytest.mark.xfail(reason="_in_fnu is not reset when make_star creates a fresh F_lam star")
def test_make_star_resets_the_unit_state(model_factory):
    model = model_factory()
    model.make_star()
    model.flam_to_fnu()
    model.make_star()
    fresh_f_lam = model.sed_star.copy()
    model.flam_to_fnu()
    assert model.sed_star == rel(fresh_f_lam * (model.sed_wave * UM) ** 2 / C, 1e-12)


def test_end_to_end_disc_is_finite_and_negligible_in_the_optical(disc_model):
    model = disc_model
    model.calculate_dust_scatter()
    model.calculate_dust_emission()
    model.flam_to_fnu()
    for name in ("sed_star", "sed_emit", "sed_scat", "sed_disc"):
        values = getattr(model, name)
        assert values.shape == model.sed_wave.shape, name
        assert np.all(np.isfinite(values)) and np.all(values >= 0), name
    assert model.sed_disc[0] < 0.1 * model.sed_star[0]  # star dominates at 0.5 um


def test_full_pipeline_from_parameter_and_optical_constant_files(tmp_path, monkeypatch):
    monkeypatch.setattr(art.mpy, "mie", fake_mie)
    (tmp_path / "dust.lnk").write_text("# lambda n k\n0.01 1.5 0.1\n10000.0 1.5 0.1\n")
    params = {**BASE_PARAMS, "composition": str(tmp_path / "dust"), "nwav": 60, "nring": 4}
    (tmp_path / "model.txt").write_text(
        "# artefact test model\n" + "\n".join(f"{k} : {v}  # comment" for k, v in params.items())
    )

    model = RTModel()
    model.get_parameters(str(tmp_path / "model.txt"))
    model.make_star()
    model.make_dust()
    model.read_optical_constants()
    model.make_disc()
    model.calculate_surface_density()
    model.calculate_qabs()
    model.calculate_dust_scatter()
    model.calculate_dust_emission(mode="bb")
    model.flam_to_fnu()

    assert model.sed_wave.shape == model.sed_disc.shape == (60,)
    assert model.sigma.shape == (4, 4)
    assert np.all(np.isfinite(model.sed_disc)) and np.all(model.sed_disc > 0)
    assert model.sed_disc == rel(model.sed_emit + model.sed_scat, 1e-12)
