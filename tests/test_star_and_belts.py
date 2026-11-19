"""Tests for the stellar model split, multi-belt discs and the blow-out size.

Run from the repository root with ``pytest``. As in ``test_artefact.py``, Mie theory is
replaced by a deterministic stand-in so that these tests exercise artefact's own
plumbing (array shapes, belt bookkeeping, parameter handling), not miepython.

1. ``StarModel`` and ``RTModel.make_star`` / ``scale_star``.
2. Multi-belt discs (``dtype="multi"``): structure, equivalence with a single belt,
   input validation, and a full run whose ring count differs from ``nring``.
3. The minimum grain size ``amin="blowout"``.
"""

import json

import numpy as np
import pytest

import artefact as art
from artefact import RTModel

C = art.C_LIGHT
UM = art.UM

BASE = dict(
    stype="blackbody",
    lstar=4.0,  # L_sun
    rstar=1.6,  # R_sun
    tstar=6700.0,  # K
    dstar=30.0,  # pc
    lmin=0.5,  # um
    lmax=2000.0,  # um
    nwav=60,
    amin=2.0,  # um
    amax=100.0,  # um
    density=2.5,  # g/cm3
    ngrain=4,
    mdust=0.01,  # Earth masses
    q=-3.5,
)
WARM = dict(dtype="gauss", rpeak=5.0, rfwhm=2.0, nring=4, mass=1.0)  # 2.45-7.55 au
COLD = dict(dtype="gauss", rpeak=80.0, rfwhm=30.0, nring=6, mass=3.0)  # ~41.8-118 au


def fake_mie(m, x):
    """Deterministic stand-in for ``artefact.core.mie_efficiencies``: (qext, qsca, qback, g)."""
    x = np.asarray(x, dtype=float)
    qext = 2.0 * (1.0 - np.exp(-x))
    return qext, 0.4 * qext, 0.0 * qext, 0.0 * qext


def make_model(**overrides):
    model = RTModel()
    model.parameters.update({**BASE, **overrides})
    return model


def disc_model(monkeypatch, **overrides):
    """A model run up to and including ``calculate_qabs``, with mocked Mie theory."""
    monkeypatch.setattr(art.core, "mie_efficiencies", fake_mie)
    model = make_model(**overrides)
    model.make_star()
    model.make_dust()
    model.oc_nk = np.full(model.sed_wave.shape, 1.5 - 0.1j)
    model.make_disc()
    model.calculate_surface_density()
    model.calculate_qabs()
    return model


# ---------------------------------------------------------------------------
# 1. Stellar model
# ---------------------------------------------------------------------------


def photometry_from_photosphere(wavelengths, photosphere, factor, lobs):
    """Fnu (mJy) of ``factor`` times the F_lambda photosphere, at wavelengths ``lobs``."""
    return factor * np.interp(lobs, wavelengths, photosphere) * (lobs * UM) ** 2 / C


def test_starmodel_scales_its_own_photosphere():
    parameters = {**BASE}
    star = art.StarModel()
    star.parameters = parameters
    star.parse_star(parameters)
    before = star.photosphere.copy()

    lobs = np.array([0.6, 1.0, 1.6, 2.2, 3.5, 4.5])
    fobs = photometry_from_photosphere(star.wavelengths, before, 0.9, lobs)
    star.obs_wave, star.obs_flux, star.obs_uncs = lobs, fobs, 0.05 * fobs

    star.scale_star()
    assert star.photosphere == pytest.approx(0.9 * before, rel=1e-6)


def test_make_star_does_not_scale_to_photometry():
    """Scaling is a separate, explicit step: make_star only builds the photosphere."""
    reference = make_model()
    reference.make_star()

    model = make_model()
    lobs = np.array([0.6, 1.0, 1.6, 2.2])
    fobs = 3.0 * photometry_from_photosphere(
        reference.sed_wave, reference.sed_star, 1.0, lobs
    )
    model.obs_wave, model.obs_flux, model.obs_uncs = lobs, fobs, 0.05 * fobs
    model.make_star()

    assert model.sed_star == pytest.approx(reference.sed_star, rel=1e-12)


def test_make_star_without_photometry_works():
    model = make_model()
    model.make_star()
    assert model.sed_wave.shape == model.sed_star.shape == (BASE["nwav"],)
    assert np.all(model.sed_star > 0)


def test_scale_star_does_not_change_the_wavelength_grid():
    model = make_model()
    model.make_star()
    grid = model.sed_wave.copy()
    lobs = np.array([0.6, 1.0, 1.6, 2.2])
    fobs = photometry_from_photosphere(model.sed_wave, model.sed_star, 1.1, lobs)
    model.obs_wave, model.obs_flux, model.obs_uncs = lobs, fobs, 0.05 * fobs

    model.scale_star()
    assert np.array_equal(model.sed_wave, grid)


# ---------------------------------------------------------------------------
# 2. Multi-belt discs
# ---------------------------------------------------------------------------


def test_multi_belt_structure(monkeypatch):
    model = disc_model(monkeypatch, dtype="multi", belts=[WARM, COLD])

    nring = len(model.radii)
    assert nring == 4 + 6 + 1  # two belts and the empty gap annulus between them
    assert model.belt_scale.shape == (2, nring)
    assert model.scale.shape == (nring,)
    assert np.all(np.diff(model.edges) > 0)
    assert model.dr.shape == (nring,)

    assert model.scale.sum() == pytest.approx(1.0, rel=1e-12)
    assert model.belt_scale.sum(axis=0) == pytest.approx(model.scale, rel=1e-12)
    # the belts' ``mass`` entries (1 and 3) are relative shares of the dust
    assert model.belt_scale.sum(axis=1) == pytest.approx([0.25, 0.75], rel=1e-12)

    # one annulus (the gap) holds no dust, and the sigma table puts none there either
    empty = np.flatnonzero(model.scale == 0.0)
    assert len(empty) == 1
    assert model.sigma.shape == (nring, BASE["ngrain"])
    assert model.sigma[empty[0]] == pytest.approx(0.0, abs=0)
    assert model.sigma.sum(axis=0) == pytest.approx(model.ng, rel=1e-12)


def test_multi_belt_of_one_matches_single_belt(monkeypatch):
    belt = dict(dtype="onepl", rin=40.0, rout=80.0, alpha_out=-1.0, nring=5)
    single = disc_model(monkeypatch, **belt)
    multi = disc_model(monkeypatch, dtype="multi", belts=[belt])

    assert multi.edges == pytest.approx(single.edges, rel=1e-12)
    assert multi.radii == pytest.approx(single.radii, rel=1e-12)
    assert multi.scale == pytest.approx(single.scale, rel=1e-12)
    assert multi.areas == pytest.approx(single.areas, rel=1e-12)


def test_overlapping_belts_have_no_gap(monkeypatch):
    first = dict(dtype="onepl", rin=40.0, rout=80.0, alpha_out=0.0, nring=4)
    second = dict(dtype="onepl", rin=60.0, rout=100.0, alpha_out=0.0, nring=4)
    model = disc_model(monkeypatch, dtype="multi", belts=[first, second])

    assert np.all(model.scale > 0.0)
    assert model.edges[0] == pytest.approx(40.0)
    assert model.edges[-1] == pytest.approx(100.0)
    assert model.belt_scale.sum() == pytest.approx(1.0, rel=1e-12)


def test_belts_can_be_read_from_a_json_file(monkeypatch, tmp_path):
    path = tmp_path / "belts.json"
    path.write_text(json.dumps([WARM, COLD]))

    from_file = disc_model(monkeypatch, dtype="multi", belts_file=str(path))
    inline = disc_model(monkeypatch, dtype="multi", belts=[WARM, COLD])

    assert from_file.edges == pytest.approx(inline.edges, rel=1e-12)
    assert from_file.belt_scale == pytest.approx(inline.belt_scale, rel=1e-12)


def test_multi_belt_run_uses_the_real_number_of_annuli(monkeypatch):
    """The emission and scattering arrays follow the annuli, not parameters['nring']."""
    model = disc_model(monkeypatch, dtype="multi", belts=[WARM, COLD], nring=99)
    nring = len(model.radii)

    model.calculate_dust_scatter()
    model.calculate_dust_emission(mode="bb")
    model.flam_to_fnu()

    shape = (nring, BASE["ngrain"], BASE["nwav"])
    assert model.sed_rings.shape == shape
    assert model.sed_ringe.shape == shape
    assert model.tdust.shape == (nring, BASE["ngrain"])

    gap = np.flatnonzero(model.scale == 0.0)[0]
    assert np.all(model.sed_ringe[gap] == 0.0)
    assert np.all(model.sed_rings[gap] == 0.0)

    assert np.all(np.isfinite(model.sed_emit)) and np.all(model.sed_emit >= 0.0)
    assert model.sed_emit.sum() > 0.0
    assert model.sed_emit == pytest.approx(model.sed_ringe.sum(axis=(0, 1)), rel=1e-10)
    assert model.sed_scat == pytest.approx(model.sed_rings.sum(axis=(0, 1)), rel=1e-10)
    # the warm belt is hotter than the cold belt
    assert model.tdust[0, 0] > model.tdust[-1, 0]


def test_multi_belt_emission_is_the_sum_of_the_belts(monkeypatch):
    """Dust emission is linear in grain number, so each belt can be run on its own."""
    # the warm belt holds 10% of the dust, the cold belt 90%
    belts = [{**WARM, "mass": 1.0}, {**COLD, "mass": 9.0}]
    both = disc_model(monkeypatch, dtype="multi", belts=belts)
    both.calculate_dust_emission(mode="bb")

    share = both.belt_scale / np.where(both.scale > 0, both.scale, 1.0)[None, :]
    per_ring = both.sed_ringe.sum(axis=1)  # (nring, nwav)
    by_belt = share @ per_ring
    assert by_belt.sum(axis=0) == pytest.approx(both.sed_emit, rel=1e-10)

    # the warm belt dominates at short wavelengths, the cold belt (more dust) at long ones
    short = np.argmin(np.abs(both.sed_wave - 20.0))
    assert by_belt[0, short] > by_belt[1, short]
    long = np.argmin(np.abs(both.sed_wave - 1000.0))
    assert by_belt[1, long] > by_belt[0, long]


@pytest.mark.parametrize(
    "parameters, message",
    [
        (dict(dtype="multi"), "belts"),
        (dict(dtype="multi", belts=[]), "non-empty"),
        (dict(dtype="multi", belts=[dict(dtype="multi", nring=3)]), "multi"),
        (
            dict(dtype="multi", belts=[{**WARM, "mass": 0.0}, {**COLD, "mass": 0.0}]),
            "positive",
        ),
        (dict(dtype="multi", belts=[{**WARM, "mass": -1.0}, COLD]), "non-negative"),
        (dict(dtype="multi", belts=[WARM, dict(dtype="gauss", rpeak=80.0)]), "Belt 1"),
        (dict(dtype="spiral"), "Belt type"),
    ],
)
def test_multi_belt_rejects_bad_input(monkeypatch, parameters, message):
    monkeypatch.setattr(art.core, "mie_efficiencies", fake_mie)
    model = make_model(**parameters)
    model.make_star()
    model.make_dust()
    with pytest.raises(ValueError, match=message):
        model.make_disc()


# ---------------------------------------------------------------------------
# 3. Minimum grain size
# ---------------------------------------------------------------------------


def test_amin_blowout_starts_the_size_distribution_at_the_blowout_size():
    model = make_model(amin="blowout", mstar=1.5)
    model.make_dust()

    ablow = model.calculate_blowout_size()
    assert ablow > 0.0
    assert model.ag[0] == pytest.approx(ablow, rel=1e-12)
    assert model.ag[-1] == pytest.approx(BASE["amax"], rel=1e-12)
    assert np.all(np.diff(model.ag) > 0)


def test_amin_blowout_estimates_mstar_from_lstar_with_a_warning():
    model = make_model(amin="blowout")
    with pytest.warns(UserWarning, match="Mstar not provided"):
        model.make_dust()

    mstar = model.parameters["mstar"]
    assert isinstance(mstar, float)
    assert mstar == pytest.approx(BASE["lstar"] ** 0.25, rel=1e-6)  # 0.43 <= M < 2 Msun


def test_blowout_size_grows_with_luminosity_and_shrinks_with_mass():
    def ablow(lstar, mstar):
        return make_model(lstar=lstar, mstar=mstar).calculate_blowout_size()

    assert ablow(8.0, 1.5) == pytest.approx(2.0 * ablow(4.0, 1.5), rel=1e-12)
    assert ablow(4.0, 3.0) == pytest.approx(0.5 * ablow(4.0, 1.5), rel=1e-12)


def test_blowout_size_needs_the_star():
    model = RTModel()
    model.parameters.update(density=2.5)
    with pytest.raises(ValueError, match="Lstar"):
        model.calculate_blowout_size()


def test_amin_must_be_below_amax():
    model = make_model(amin=200.0, amax=100.0)
    with pytest.raises(ValueError, match="amax"):
        model.make_dust()
