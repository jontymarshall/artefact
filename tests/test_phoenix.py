"""Tests for the PHOENIX photosphere fit (artefact.phoenix, artefact.fitting, RTModel stype "phoenix").

The HD 10647 values are a regression against dust_cloud (github.com/jontymarshall/dust_cloud), whose
stellar fit this code was ported from: the same six points (as dust_cloud fitted them, 2026-11-06) gave the
same Teff, radius, luminosity, log g and [Fe/H], errors included.
"""

import numpy as np
import pytest
from artefact import RTModel, fitting, phoenix

# HD 10647 (q1 Eri) between 0.3 and 4 micron: band, wavelength (um), flux (Jy), error (Jy)
HD10647_POINTS = [
    ("Johnson.B", 0.44, 16.081185272450405, 0.20735847232218002),
    ("Johnson.V", 0.55, 23.237785762091583, 0.19262511643184527),
    ("WISE.W1", 3.3526, 6.642273847768676, 2.3063956572108095),
    ("2MASS.J", 1.235, 19.323614341924667, 4.022281605899431),
    ("2MASS.H", 1.662, 17.811475157829552, 3.838764077272478),
    ("2MASS.Ks", 2.159, 12.24419941432016, 3.112541461622441),
]
HD10647_DISTANCE_PC = 17.43375155786755
HD10647_DISTANCE_ERR_PC = 0.07631654426347012
# dust_cloud's results: (value, lower error, upper error); chi2
DUST_CLOUD_WITH_FILTERS = {
    "teff": (6283.854896822706, 115.0213710428061, 107.49993653485672),
    "radius": (1.0515118513705628, 0.044879182468748224, 0.0477059160022808),
    "luminosity": (1.5531972540079129, 0.045825631304230674, 0.039851856905903205),
    "logg": (4.43240140313757, 0.35892938735656443, 0.06759859686242997),
    "feh": (0.1221365122559731, 0.21311444060585866, 0.1997040065644931),
    "chi2": 1.916332315890095,
}
DUST_CLOUD_AT_WAVELENGTHS = {
    "teff": (6281.952766818606, 95.45727992398679, 89.21524868488268),
    "radius": (1.0517007792600175, 0.03580978455206263, 0.037298748463493014),
    "luminosity": (1.5518749980882014, 0.03739229418907608, 0.03439228144899853),
    "logg": (4.4676924153498145, 0.32059192912750945, 0.03230758465018546),
    "feh": (0.062556066710175, 0.1916937798377897, 0.18942292680562361),
    "chi2": 1.6446643774053429,
}
PRIORS = {"logg_prior": phoenix.LOGG_PRIOR, "mh_prior": phoenix.FEH_PRIOR}


@pytest.fixture(scope="module")
def grid():
    return phoenix.load_grid()


def hd10647_arrays():
    bands = [point[0] for point in HD10647_POINTS]
    wave, flux, err = (
        np.array([point[i] for point in HD10647_POINTS]) for i in (1, 2, 3)
    )
    return bands, wave, flux, err


def assert_matches(fit, expected):
    quantities = {
        "teff": fit.teff,
        "radius": fit.radius_rsun,
        "luminosity": fit.luminosity_lsun,
        "logg": fit.logg,
        "feh": fit.feh,
    }
    for name, quantity in quantities.items():
        found = (quantity.value, quantity.err_lo, quantity.err_hi)
        np.testing.assert_allclose(found, expected[name], rtol=1e-5, err_msg=name)
    assert fit.chi2 == pytest.approx(expected["chi2"], rel=1e-5)


# --- the grid ---


def test_shipped_grid_matches_its_checksum():
    phoenix.verify_shipped_grid()


def test_grid_axes(grid):
    assert grid.teff[0] == 3000 and grid.teff[-1] == 12000
    assert grid.logg.tolist() == [3.5, 4.0, 4.5]
    assert grid.mh.tolist() == [-0.5, 0.0, 0.5]
    assert grid.log_flux_nu.shape == (3, 3, len(grid.teff), len(grid.wave_um))


def test_flux_beyond_the_grid_is_a_rayleigh_jeans_tail(grid):
    flux_100, flux_1000 = grid.surface_flux_jy(np.array([100.0, 1000.0]), 5800.0)
    assert flux_100 / flux_1000 == pytest.approx(100.0)  # F_nu ~ lambda^-2


def test_interpolation_between_nodes_lies_between_them(grid):
    wave = np.array([0.55])
    low, high = grid.teff[10], grid.teff[11]
    middle = grid.surface_flux_jy(wave, (low + high) / 2)
    assert grid.surface_flux_jy(wave, low) < middle < grid.surface_flux_jy(wave, high)


# --- filters ---


def test_every_mapped_filter_has_a_curve():
    mapping = fitting.load_mapping()
    specs = [*mapping.exact.values(), *(spec for _pattern, spec in mapping.patterns)]
    wanted = {spec.filter for spec in specs if spec.filter is not None}
    assert wanted <= set(fitting.load_library())


def test_a_flat_spectrum_is_unchanged_by_its_filter():
    response = fitting.response_for("2MASS.J")
    assert response.flat
    assert response.apply(np.full(len(response.wave_um), 3.0)) == pytest.approx(3.0)


def test_unknown_band_is_compared_at_its_wavelength():
    phot = fitting.Photometer.from_bands(["Nobody.X"], [1.0], [1.0], [0.1])
    assert phot.responses == [None]
    assert phot.missing_filters == ("Nobody.X",)


# --- the fit ---


def test_hd10647_with_filters_matches_dust_cloud(grid):
    bands, wave, flux, err = hd10647_arrays()
    phot = fitting.Photometer.from_bands(bands, wave, flux, err)
    fit = phoenix.fit_star(
        phot,
        HD10647_DISTANCE_PC,
        grid,
        distance_err_pc=HD10647_DISTANCE_ERR_PC,
        logg=phoenix.LOGG_PRIOR[0],
        **PRIORS,
    )
    assert_matches(fit, DUST_CLOUD_WITH_FILTERS)


def test_hd10647_at_wavelengths_matches_dust_cloud(grid):
    _bands, wave, flux, err = hd10647_arrays()
    fit = phoenix.fit_star(
        fitting.Photometer(wave, flux, err),
        HD10647_DISTANCE_PC,
        grid,
        distance_err_pc=HD10647_DISTANCE_ERR_PC,
        logg=phoenix.LOGG_PRIOR[0],
        **PRIORS,
    )
    assert_matches(fit, DUST_CLOUD_AT_WAVELENGTHS)


def test_a_star_made_from_the_grid_is_recovered(grid):
    bands = [
        "Johnson.B",
        "Johnson.V",
        "Gaia.BP",
        "Gaia.G",
        "Gaia.RP",
        "2MASS.J",
        "2MASS.H",
        "2MASS.Ks",
        "WISE.W1",
        "WISE.W2",
    ]
    wave = np.array([0.44, 0.55, 0.51, 0.62, 0.78, 1.235, 1.662, 2.159, 3.35, 4.6])

    def star(wave_um):
        return phoenix._model_flux(grid, wave_um, 5850.0, 1.1, 12.0, 4.4, 0.0, 0.0)

    flux = fitting.Photometer.from_bands(
        bands, wave, np.ones(10), np.ones(10)
    ).evaluate(star)
    fit = phoenix.fit_star(
        fitting.Photometer.from_bands(bands, wave, flux, 0.02 * flux),
        12.0,
        grid,
        **PRIORS,
    )
    assert fit.teff.value == pytest.approx(5850.0, abs=5.0)
    assert fit.radius_rsun.value == pytest.approx(1.1, abs=0.002)


def test_fewer_than_four_points_is_an_error(grid):
    phot = fitting.Photometer([0.5, 1.0, 2.0], [1.0, 1.0, 1.0], [0.1, 0.1, 0.1])
    with pytest.raises(ValueError, match="need 4"):
        phoenix.fit_star(phot, 10.0, grid)


def test_upper_limits_are_not_fitted(grid):
    _bands, wave, flux, err = hd10647_arrays()
    limits = np.zeros(len(wave), dtype=bool)
    limits[0] = True
    fit = phoenix.fit_star(
        fitting.Photometer(wave, flux, err, limits), HD10647_DISTANCE_PC, grid, **PRIORS
    )
    assert fit.n_points == len(wave) - 1


# --- RTModel ---


def phoenix_model(with_bands=True):
    bands, wave, flux, err = hd10647_arrays()
    model = RTModel()
    model.parameters.update(
        stype="phoenix", dstar=HD10647_DISTANCE_PC, dstar_err=HD10647_DISTANCE_ERR_PC,
        lmin=0.1, lmax=3000.0, nwav=300,
    )  # fmt: skip
    model.obs_wave, model.obs_flux, model.obs_uncs = wave, flux * 1e3, err * 1e3  # mJy
    model.obs_band = bands if with_bands else None
    return model


@pytest.mark.parametrize(
    "with_bands, expected",
    [(True, DUST_CLOUD_WITH_FILTERS), (False, DUST_CLOUD_AT_WAVELENGTHS)],
)
def test_make_star_fits_the_photosphere(with_bands, expected):
    model = phoenix_model(with_bands)
    model.make_star()
    assert_matches(model.stellarfit, expected)
    for name in ("tstar", "rstar", "lstar", "logg", "feh"):
        assert isinstance(model.parameters[name], float)
    assert model.parameters["tstar"] == pytest.approx(expected["teff"][0], rel=1e-5)


def test_make_star_spectrum_is_the_fitted_photosphere_in_mjy():
    model = phoenix_model()
    model.make_star()
    model.flam_to_fnu()
    expected_mjy = model.stellarfit.flux_at(model.sed_wave) * 1e3
    np.testing.assert_allclose(model.sed_star, expected_mjy, rtol=1e-10)


def test_fit_phoenix_needs_observations():
    model = phoenix_model()
    model.obs_flux = None
    with pytest.raises(ValueError, match="obs_wave"):
        model.fit_phoenix()
