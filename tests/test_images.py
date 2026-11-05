"""Tests for resolved model images (``RTModel.make_image``, ``convolve_beam``).

The expected values come from independent calculations: conservation of the total
flux, the 1 / cos(i) brightening of an inclined optically thin disc checked on
individual pixels, the orientation of the major axis checked by which pixels are
lit, and the exact area normalisation of the beam.
"""

import numpy as np
import pytest

import artefact as art
from artefact import RTModel

UM = art.UM
C = art.C_LIGHT


def rel(expected, tol=1e-6):
    return pytest.approx(expected, rel=tol, abs=0)


PARAMS = dict(
    stype="blackbody", lstar=1.0, rstar=1.0, tstar=5772.0, dstar=10.0,
    lmin=0.5, lmax=2000.0, nwav=200,
    amin=1.0, amax=100.0, density=2.5, ngrain=4, mdust=0.01, q=-3.5,
    dtype="onepl", rin=40.0, rout=80.0, alpha_out=-1.0, nring=30,
)  # fmt: skip


def fake_mie(m, x):
    x = np.asarray(x, dtype=float)
    qext = 2.0 * (1.0 - np.exp(-x))
    return qext, 0.4 * qext, 0.0 * qext, 0.0 * qext


@pytest.fixture
def disc(monkeypatch):
    """A model with thermal emission and scattered light, still in F_lambda."""
    monkeypatch.setattr(art.core, "mie_efficiencies", fake_mie)
    model = RTModel()
    model.parameters.update(PARAMS)
    model.make_star()
    model.make_dust()
    model.oc_nk = np.full(model.sed_wave.shape, 1.5 - 0.1j)
    model.make_disc()
    model.calculate_surface_density()
    model.calculate_qabs()
    model.calculate_dust_emission(mode="bb")
    model.calculate_dust_scatter()
    return model


def total_flux_mjy(model, wavelength, which=("emit", "scat")):
    """Disc flux at one wavelength in mJy, from the summed spectra."""
    spectrum = sum(getattr(model, f"sed_{name}") for name in which)
    if not model._in_fnu:
        spectrum = spectrum * (model.sed_wave * UM) ** 2 / C
    return np.interp(wavelength, model.sed_wave, spectrum)


def ring_flux_mjy(model, wavelength, component="total"):
    """Flux of each annulus in mJy for "emit", "scat" or both ("total")."""
    rings = 0.0
    if component in ("emit", "total"):
        rings = rings + model.sed_ringe.sum(axis=1)
    if component in ("scat", "total"):
        rings = rings + model.sed_rings.sum(axis=1)
    if not model._in_fnu:
        rings = rings * ((model.sed_wave * UM) ** 2 / C)
    return np.array([np.interp(wavelength, model.sed_wave, r) for r in rings])


def phase_weight(cos_theta, g1=art.HG_G1, g2=art.HG_G2, weight=art.HG_WEIGHT):
    """4 pi p(theta): the scattered flux relative to isotropic scattering."""
    return 4 * np.pi * art.two_component_hg(cos_theta, g1, g2, weight)


# ---------------------------------------------------------------------------
# Flux and brightness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inc,pa", [(0.0, 0.0), (60.0, 33.0), (80.0, 120.0)])
@pytest.mark.parametrize("aspect", [0.0, 0.03, 0.1])
def test_image_conserves_the_thermal_flux(disc, inc, pa, aspect):
    image = disc.make_image(70.0, component="emit", inc=inc, pa=pa, npix=200, aspect=aspect)
    assert image.data.sum() == rel(total_flux_mjy(disc, 70.0, ("emit",)), 1e-2)


@pytest.mark.parametrize("inc,pa", [(0.0, 0.0), (60.0, 33.0), (80.0, 120.0)])
def test_image_conserves_isotropically_scattered_flux(disc, inc, pa):
    image = disc.make_image(24.0, component="scat", inc=inc, pa=pa, npix=200, g1=0.0, g2=0.0)
    assert image.data.sum() == rel(total_flux_mjy(disc, 24.0, ("scat",)), 1e-2)


def test_near_edge_on_thick_disc_still_conserves_flux(disc):
    image = disc.make_image(70.0, component="emit", inc=86.0, pa=15.0, npix=240, aspect=0.05)
    assert image.data.sum() == rel(total_flux_mjy(disc, 70.0, ("emit",)), 3e-2)


def test_image_flux_converges_as_the_pixelation_is_refined(disc):
    expected = total_flux_mjy(disc, 70.0, ("emit",))
    kwargs = dict(component="emit", inc=50.0, pa=20.0)
    coarse = disc.make_image(70.0, npix=60, oversample=1, **kwargs)
    fine = disc.make_image(70.0, npix=240, oversample=3, **kwargs)
    assert abs(fine.data.sum() / expected - 1) < abs(coarse.data.sum() / expected - 1)
    assert fine.data.sum() == rel(expected, 3e-3)


def test_total_image_is_the_sum_of_the_emission_and_scattered_images(disc):
    total = disc.make_image(24.0, component="total", inc=40.0, pa=10.0).data
    emit = disc.make_image(24.0, component="emit", inc=40.0, pa=10.0).data
    scat = disc.make_image(24.0, component="scat", inc=40.0, pa=10.0).data
    assert total == rel(emit + scat, 1e-12)
    assert np.all(emit > 0) or np.any(emit > 0)


def test_inclined_pixel_is_brighter_by_one_over_cos_i_and_matches_the_ring_flux(disc):
    """Surface brightness of an optically thin sheet scales as 1 / cos(i)."""
    npix, c, k, inc = 201, 100, 62, 60.0
    # 1 pixel = 1 au at 10 pc; the thinnest disc allowed (aspect floor 1e-3, so the
    # tolerance covers the O(aspect^2) variation of H across the line of sight)
    kwargs = dict(npix=npix, pixscale=0.1, oversample=1, aspect=0.0, component="emit")
    face_on = disc.make_image(70.0, inc=0.0, **kwargs).data
    tilted = disc.make_image(70.0, inc=inc, pa=90.0, **kwargs).data

    edges = disc.edges
    ring = np.searchsorted(edges, k, side="right") - 1
    flux = ring_flux_mjy(disc, 70.0, "emit")[ring]
    per_area = flux / (np.pi * (edges[ring + 1] ** 2 - edges[ring] ** 2))
    expected_face_on = per_area * 1.0**2  # mJy per 1 au^2 pixel

    assert face_on[c, c - k] == rel(expected_face_on, 1e-5)
    # on the major axis (east-west for PA = 90), then on the minor axis
    assert tilted[c, c - k] == rel(expected_face_on / np.cos(np.radians(inc)), 1e-5)
    assert tilted[c + int(k * np.cos(np.radians(inc))), c] == rel(
        expected_face_on / np.cos(np.radians(inc)), 1e-5
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def lit_extent(data):
    rows, cols = np.nonzero(data)
    return cols.max() - cols.min() + 1, rows.max() - rows.min() + 1  # width, height


def test_image_is_foreshortened_along_the_minor_axis_and_follows_the_position_angle(disc):
    kwargs = dict(npix=201, pixscale=0.1, oversample=1, inc=60.0, aspect=0.0, component="emit")
    width, height = lit_extent(disc.make_image(70.0, pa=90.0, **kwargs).data)
    assert width == pytest.approx(160, abs=3) and height == pytest.approx(80, abs=3)
    width, height = lit_extent(disc.make_image(70.0, pa=0.0, **kwargs).data)
    assert width == pytest.approx(80, abs=3) and height == pytest.approx(160, abs=3)


def test_position_angle_is_measured_east_of_north_with_east_to_the_left(disc):
    image = disc.make_image(
        70.0, inc=60.0, pa=30.0, npix=201, pixscale=0.1, oversample=1, aspect=0.0,
        component="emit",
    ).data
    c = 100
    # The major axis points to (east, north) = (sin 30, cos 30) = (0.5, 0.87):
    # 35 au east and 61 au north of the star, with east to the left.
    assert image[c + 61, c - 35] > 0
    assert image[c + 61, c + 35] == 0  # the mirror image across the north axis


@pytest.mark.parametrize("kwargs", [dict(component="emit"), dict(component="scat", g1=0.0, g2=0.0)])
def test_image_is_point_symmetric_about_the_star_without_anisotropy(disc, kwargs):
    image = disc.make_image(70.0, inc=55.0, pa=25.0, npix=101, pixscale=0.2, **kwargs).data
    assert image == pytest.approx(image[::-1, ::-1], rel=1e-9, abs=1e-30)


def test_default_field_of_view_holds_the_whole_disc(disc):
    image = disc.make_image(70.0, npix=100, component="emit")
    assert image.pixscale == rel(2.4 * disc.edges[-1] / PARAMS["dstar"] / 100)
    assert np.all(image.data[0] == 0) and np.all(image.data[:, 0] == 0)


# ---------------------------------------------------------------------------
# Star, units and wavelength handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("npix", [100, 101])
def test_star_is_added_as_a_point_source_at_the_centre(disc, npix):
    without = disc.make_image(10.0, npix=npix)
    with_star = disc.make_image(10.0, npix=npix, include_star=True)
    star = np.interp(10.0, disc.sed_wave, disc.sed_star * (disc.sed_wave * UM) ** 2 / C)
    assert (with_star.data - without.data).sum() == rel(star, 1e-9)
    lit = np.argwhere(with_star.data != without.data)
    assert len(lit) == (4 if npix % 2 == 0 else 1)
    assert np.all(np.abs(lit - npix / 2 + 0.5) <= 0.5)


def test_image_is_the_same_before_and_after_flam_to_fnu(disc):
    before = disc.make_image(70.0, inc=45.0, pa=15.0, include_star=True).data
    disc.flam_to_fnu()
    after = disc.make_image(70.0, inc=45.0, pa=15.0, include_star=True).data
    assert after == rel(before, 1e-12)


def test_unit_conversions(disc):
    pix = disc.make_image(70.0, beam=0.8, npix=100, unit="mJy/pixel")
    area = disc.make_image(70.0, beam=0.8, npix=100, unit="mJy/arcsec2")
    per_beam = disc.make_image(70.0, beam=0.8, npix=100, unit="mJy/beam")
    beam_area = np.pi * 0.8**2 / (4 * np.log(2))  # arcsec^2
    assert area.data == rel(pix.data / pix.pixscale**2, 1e-12)
    assert per_beam.data == rel(pix.data * beam_area / pix.pixscale**2, 1e-12)
    assert (pix.unit, area.unit, per_beam.unit) == ("mJy/pixel", "mJy/arcsec2", "mJy/beam")


def test_beam_smooths_the_image_and_conserves_flux(disc):
    sharp = disc.make_image(70.0, npix=200, inc=30.0)
    blurred = disc.make_image(70.0, npix=200, inc=30.0, beam=1.0)
    assert blurred.data.sum() == rel(sharp.data.sum(), 1e-6)
    assert blurred.data.max() < sharp.data.max()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(component="star"), "component"),
        (dict(wavelength=1e5), "outside the model grid"),
        (dict(wavelength=0.01), "outside the model grid"),
        (dict(inc=90.0), "inc"),
        (dict(inc=-5.0), "inc"),
        (dict(npix=1), "npix"),
        (dict(oversample=0), "oversample"),
        (dict(unit="Jy"), "unit"),
        (dict(unit="mJy/beam"), "needs a beam"),
    ],
)
def test_make_image_rejects_bad_arguments(disc, kwargs, message):
    args = dict(wavelength=70.0)
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        disc.make_image(**args)


def test_make_image_asks_for_the_missing_calculations():
    model = RTModel()
    model.parameters.update(PARAMS)
    with pytest.raises(ValueError, match="make_disc"):
        model.make_image(70.0)
    model.make_star()
    model.make_dust()
    model.make_disc()
    with pytest.raises(ValueError, match="calculate_dust_emission"):
        model.make_image(70.0)


# ---------------------------------------------------------------------------
# convolve_beam and FITS output
# ---------------------------------------------------------------------------


def test_convolve_beam_gaussian_has_the_requested_fwhm_and_conserves_flux():
    image = np.zeros((201, 201))
    image[100, 100] = 5.0
    out = art.convolve_beam(image, 10.0)
    assert out.sum() == rel(5.0, 1e-9)
    profile = out[100, 100:]
    half = np.interp(0.5 * profile[0], profile[::-1], np.arange(len(profile))[::-1])
    assert 2 * half == pytest.approx(10.0, rel=0.03)


def test_convolve_beam_psf_array_matches_the_equivalent_gaussian():
    rng = np.random.default_rng(1)
    image = np.zeros((101, 101))
    image[rng.integers(30, 70, 5), rng.integers(30, 70, 5)] = 1.0
    sigma = 3.0
    y, x = np.mgrid[-20:21, -20:21]
    psf = np.exp(-0.5 * (x**2 + y**2) / sigma**2)
    from_psf = art.convolve_beam(image, 5.0 * psf)  # scale is irrelevant
    from_gauss = art.convolve_beam(image, sigma / art.FWHM_TO_SIGMA)
    assert np.max(np.abs(from_psf - from_gauss)) < 2e-3 * from_gauss.max()


@pytest.mark.parametrize("bad", [0.0, -2.0, np.zeros((5, 5)), np.ones(5), np.ones((4, 4))])
def test_convolve_beam_rejects_invalid_beams(bad):
    with pytest.raises(ValueError):
        art.convolve_beam(np.ones((8, 8)), bad)


def test_image_header_describes_a_tangent_plane_with_east_to_the_left(disc):
    image = disc.make_image(70.0, npix=64, pixscale=0.25, beam=1.5, inc=30.0, pa=40.0)
    header = image.header()
    assert header["ASPECT"] == art.ASPECT_RATIO
    assert (header["HGG1"], header["HGG2"], header["HGWEIGHT"]) == (
        art.HG_G1, art.HG_G2, art.HG_WEIGHT
    )
    assert header["CDELT1"] == rel(-0.25 / 3600) and header["CDELT2"] == rel(0.25 / 3600)
    assert header["CRPIX1"] == 32.5 and header["CRPIX2"] == 32.5
    assert header["BUNIT"] == "mJy/pixel" and header["WAVELEN"] == 70.0
    assert header["BMAJ"] == rel(1.5 / 3600) and header["BMIN"] == header["BMAJ"]
    assert (header["INC"], header["PA"]) == (30.0, 40.0)


def test_image_without_a_beam_has_no_beam_keywords(disc):
    assert "BMAJ" not in disc.make_image(70.0).header()
    # thermal emission has no phase function, so no phase keywords either
    assert "HGG1" not in disc.make_image(70.0, component="emit").header()


def test_image_round_trips_through_a_fits_file(disc, tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    image = disc.make_image(70.0, npix=32, beam=1.0)
    path = tmp_path / "disc.fits"
    image.writeto(path)
    with fits.open(path) as hdus:
        assert hdus[0].data == pytest.approx(image.data.astype(np.float32), rel=1e-6)
        assert hdus[0].header["BUNIT"] == "mJy/pixel"
        assert hdus[0].header["CDELT1"] < 0


# ---------------------------------------------------------------------------
# Scale height
# ---------------------------------------------------------------------------


def test_default_aspect_ratio_is_0_03_and_can_come_from_the_parameters(disc):
    default = disc.make_image(70.0, component="emit", inc=50.0, npix=60)
    explicit = disc.make_image(70.0, component="emit", inc=50.0, npix=60, aspect=0.03)
    assert art.ASPECT_RATIO == 0.03 and default.aspect == 0.03
    assert default.data == rel(explicit.data, 1e-12)

    disc.parameters["aspect"] = 0.1
    from_parameters = disc.make_image(70.0, component="emit", inc=50.0, npix=60)
    thick = disc.make_image(70.0, component="emit", inc=50.0, npix=60, aspect=0.1)
    assert from_parameters.aspect == 0.1
    assert from_parameters.data == rel(thick.data, 1e-12)
    assert disc.make_image(70.0, component="emit", npix=60, aspect=0.03).aspect == 0.03


def test_negative_aspect_ratio_is_rejected(disc):
    with pytest.raises(ValueError, match="aspect"):
        disc.make_image(70.0, aspect=-0.1)


def test_scale_height_does_not_change_a_face_on_thermal_image(disc):
    thin = disc.make_image(70.0, component="emit", inc=0.0, npix=100, aspect=0.0)
    thick = disc.make_image(70.0, component="emit", inc=0.0, npix=100, aspect=0.2)
    assert thick.data.sum() == rel(thin.data.sum(), 1e-3)
    # pixels straddling the disc edge may be point-sampled differently; the rest agree
    lit = thin.data > 0
    assert np.median(thick.data[lit] / thin.data[lit]) == rel(1.0, 1e-6)


@pytest.mark.parametrize("aspect", [0.0, 0.2])
def test_minor_axis_spread_follows_the_vertical_structure(disc, aspect):
    """Variance across the major axis is <y^2> cos^2(i) + <z^2> sin^2(i), with
    <y^2> = <r^2> / 2 and <z^2> = aspect^2 <r^2>; <r^2> in an annulus is (r1^2 + r2^2) / 2."""
    inc = 80.0
    image = disc.make_image(
        70.0, component="emit", inc=inc, pa=90.0, npix=160, oversample=2, aspect=aspect
    )
    n = image.data.shape[0]
    north = (np.arange(n) + 0.5 - n / 2) * image.pixscale * PARAMS["dstar"]  # au
    variance = np.sum(image.data * north[:, None] ** 2) / image.data.sum()

    edges = disc.edges
    weights = ring_flux_mjy(disc, 70.0, "emit")
    mean_r2 = np.sum(weights * 0.5 * (edges[1:] ** 2 + edges[:-1] ** 2)) / weights.sum()
    c, s = np.cos(np.radians(inc)), np.sin(np.radians(inc))
    assert variance == rel(mean_r2 * (0.5 * c**2 + aspect**2 * s**2), 3e-2)


def test_thicker_disc_is_taller_on_the_sky_but_not_wider(disc):
    kwargs = dict(component="emit", inc=75.0, pa=90.0, npix=200, oversample=1)
    thin = disc.make_image(70.0, aspect=0.0, **kwargs).data
    thick = disc.make_image(70.0, aspect=0.1, **kwargs).data
    width_thin, height_thin = lit_extent(thin > 1e-3 * thin.max())
    width_thick, height_thick = lit_extent(thick > 1e-3 * thick.max())
    assert height_thick > height_thin + 8
    assert width_thick == pytest.approx(width_thin, abs=3)
    assert thick.max() < thin.max()


# ---------------------------------------------------------------------------
# Two-component Henyey-Greenstein phase function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "g1,g2,weight", [(0.0, 0.0, 1.0), (0.7, -0.2, 0.7), (0.9, -0.5, 0.3), (-0.3, 0.4, 0.5)]
)
def test_two_component_hg_is_normalised_over_the_sphere(g1, g2, weight):
    mu = np.linspace(-1.0, 1.0, 400001)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    total = 2 * np.pi * integrate(art.two_component_hg(mu, g1, g2, weight), mu)
    assert total == rel(1.0, 1e-4)


def test_henyey_greenstein_limits_and_closed_form():
    assert art.henyey_greenstein(0.3, 0.0) == rel(1 / (4 * np.pi), 1e-12)
    g = 0.7
    assert art.henyey_greenstein(0.0, g) == rel((1 - g**2) / (4 * np.pi * (1 + g**2) ** 1.5))
    assert art.henyey_greenstein(1.0, g) > art.henyey_greenstein(-1.0, g)
    assert art.henyey_greenstein(-1.0, -g) > art.henyey_greenstein(1.0, -g)
    mixed = art.two_component_hg(0.4, 0.8, -0.3, 0.25)
    assert mixed == rel(
        0.25 * art.henyey_greenstein(0.4, 0.8) + 0.75 * art.henyey_greenstein(0.4, -0.3), 1e-12
    )
    assert art.two_component_hg(0.4, 0.8, -0.3, 1.0) == rel(art.henyey_greenstein(0.4, 0.8), 1e-12)


@pytest.mark.parametrize(
    "call",
    [
        lambda: art.henyey_greenstein(0.0, 1.0),
        lambda: art.henyey_greenstein(0.0, -1.2),
        lambda: art.two_component_hg(0.0, 0.5, -0.5, 1.5),
        lambda: art.two_component_hg(0.0, 0.5, -0.5, -0.1),
    ],
)
def test_phase_function_rejects_invalid_parameters(call):
    with pytest.raises(ValueError):
        call()


def test_face_on_thin_disc_scatters_at_ninety_degrees(disc):
    """Starlight reaches a face-on grain from the side: theta = 90 degrees."""
    g1, g2, w = 0.8, -0.3, 0.6
    image = disc.make_image(
        24.0, component="scat", inc=0.0, npix=200, aspect=0.0, g1=g1, g2=g2, hg_weight=w
    )
    expected = total_flux_mjy(disc, 24.0, ("scat",)) * phase_weight(0.0, g1, g2, w)
    assert image.data.sum() == rel(expected, 1e-2)
    assert image.phase == (g1, g2, w)


def test_near_and_far_sides_of_an_inclined_thin_ring_have_the_exact_phase_weights(disc):
    """For a razor-thin disc a grain on the minor axis has cos(theta) = +/- sin(i);
    the side at PA + 90 degrees (east for PA = 0) is nearer the observer."""
    inc, k, c = 60.0, 62, 100
    image = disc.make_image(
        24.0, component="scat", inc=inc, pa=0.0, npix=201, pixscale=0.1,
        oversample=1, aspect=0.0,
    ).data
    edges = disc.edges
    ring = np.searchsorted(edges, k, side="right") - 1
    flux = ring_flux_mjy(disc, 24.0, "scat")[ring]
    isotropic = flux / (np.pi * (edges[ring + 1] ** 2 - edges[ring] ** 2)) / np.cos(np.radians(inc))
    sin_i = np.sin(np.radians(inc))
    east, west = image[c, c - 31], image[c, c + 31]  # minor axis: 62 cos(60) = 31 au
    assert east == rel(isotropic * phase_weight(+sin_i), 1e-4)
    assert west == rel(isotropic * phase_weight(-sin_i), 1e-4)
    assert east > west


def test_forward_scattering_brightens_the_near_side_only_when_anisotropic(disc):
    def east_over_west(**kwargs):
        data = disc.make_image(24.0, component="scat", inc=60.0, pa=0.0, npix=200, **kwargs).data
        return data[:, :100].sum() / data[:, 100:].sum()  # east is the left half

    assert east_over_west(g1=0.0, g2=0.0) == rel(1.0, 1e-9)
    assert east_over_west() > 1.5


def test_phase_function_applies_to_scattered_light_only(disc):
    default = disc.make_image(70.0, component="emit", inc=60.0, npix=80)
    changed = disc.make_image(70.0, component="emit", inc=60.0, npix=80, g1=0.95, g2=-0.9)
    assert changed.data == rel(default.data, 1e-12)
    assert changed.phase is None

    kwargs = dict(inc=60.0, pa=30.0, npix=80)
    total = disc.make_image(24.0, component="total", **kwargs).data
    parts = disc.make_image(24.0, component="emit", **kwargs).data + disc.make_image(
        24.0, component="scat", **kwargs
    ).data
    assert total == rel(parts, 1e-12)


def test_average_over_viewing_directions_recovers_the_isotropic_flux(disc):
    """p integrates to 1 over the sphere, so the scattered flux averaged over
    viewing directions (uniform in cos i) is the isotropic flux."""
    nodes, weights = np.polynomial.legendre.leggauss(8)
    mu, weights = 0.5 * (nodes + 1), 0.5 * weights
    flux = [
        disc.make_image(
            24.0, component="scat", inc=np.degrees(np.arccos(m)), pa=0.0,
            npix=120, oversample=1, aspect=0.1,
        ).data.sum()
        for m in mu
    ]
    assert np.sum(weights * np.array(flux)) == rel(total_flux_mjy(disc, 24.0, ("scat",)), 3e-2)


def test_phase_function_parameters_can_come_from_the_model_parameters(disc):
    explicit = disc.make_image(24.0, component="scat", inc=50.0, npix=60, g1=0.9, g2=-0.4, hg_weight=0.2)
    disc.parameters.update(hg_g1=0.9, hg_g2=-0.4, hg_weight=0.2)
    from_parameters = disc.make_image(24.0, component="scat", inc=50.0, npix=60)
    assert from_parameters.data == rel(explicit.data, 1e-12)
    assert from_parameters.phase == (0.9, -0.4, 0.2)
    override = disc.make_image(24.0, component="scat", inc=50.0, npix=60, g1=0.0, g2=0.0)
    assert override.phase == (0.0, 0.0, 0.2)


@pytest.mark.parametrize("kwargs", [dict(g1=1.0), dict(g2=-1.5), dict(hg_weight=1.2), dict(nlos=1)])
def test_make_image_rejects_invalid_phase_and_sampling_arguments(disc, kwargs):
    with pytest.raises(ValueError):
        disc.make_image(24.0, component="scat", **kwargs)


# ---------------------------------------------------------------------------
# Kernel objects as beams
# ---------------------------------------------------------------------------


class FakeKernel:
    """Stand-in for an astropy Kernel2D, which exposes the kernel as ``.array``."""

    def __init__(self, array):
        self.array = array


def gaussian_array(size=31, sigma=3.0, sigma_y=None):
    y, x = np.mgrid[-(size // 2) : size // 2 + 1, -(size // 2) : size // 2 + 1]
    sigma_y = sigma if sigma_y is None else sigma_y
    return np.exp(-0.5 * ((x / sigma) ** 2 + (y / sigma_y) ** 2))


def test_convolve_beam_accepts_kernel_objects_through_their_array():
    image = np.zeros((81, 81))
    image[30, 40] = 2.0
    psf = gaussian_array()
    assert art.convolve_beam(image, FakeKernel(psf)) == rel(art.convolve_beam(image, psf), 1e-12)


def test_elliptical_kernel_object_gives_an_elliptical_image_and_beam_area(disc):
    psf = gaussian_array(size=41, sigma=1.5, sigma_y=4.0)
    kernel = FakeKernel(psf)
    kwargs = dict(component="emit", inc=0.0, npix=100)
    plain = disc.make_image(70.0, **kwargs)
    blurred = disc.make_image(70.0, beam=kernel, **kwargs)
    assert blurred.data.sum() == rel(plain.data.sum(), 1e-3)  # a little leaves the edge
    per_beam = disc.make_image(70.0, beam=kernel, unit="mJy/beam", **kwargs)
    assert per_beam.data == rel(blurred.data * (np.sum(psf) / np.max(psf)), 1e-9)
    assert "BMAJ" not in blurred.header()

    probe = np.zeros((101, 101))
    probe[50, 50] = 1.0
    out = art.convolve_beam(probe, kernel)
    assert out.sum() == rel(1.0, 1e-9)
    assert np.sum(out[:, 50] > 0.5 * out.max()) > np.sum(out[50, :] > 0.5 * out.max())


def test_convolve_beam_matches_astropy_for_astropy_kernels():
    conv = pytest.importorskip("astropy.convolution")
    rng = np.random.default_rng(3)
    image = np.zeros((101, 101))
    image[rng.integers(30, 70, 6), rng.integers(30, 70, 6)] = rng.uniform(1, 3, 6)
    for kernel in (
        conv.Gaussian2DKernel(x_stddev=3.0),
        conv.Gaussian2DKernel(x_stddev=4.0, y_stddev=1.5, theta=0.6),
    ):
        expected = conv.convolve_fft(image, kernel, boundary="fill", fill_value=0.0)
        assert art.convolve_beam(image, kernel) == pytest.approx(expected, abs=1e-6 * image.max())
    assert art.convolve_beam(image, 3.0 / art.FWHM_TO_SIGMA) == pytest.approx(
        art.convolve_beam(image, conv.Gaussian2DKernel(x_stddev=3.0)), abs=2e-3 * image.max()
    )
