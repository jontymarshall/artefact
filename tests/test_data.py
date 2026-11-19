"""Tests for the bundled-data resolver (``artefact._data``) and the files it finds.

These guard the failure that only shows up after ``pip install``: a data file that is
missing from the package, or a bare name such as ``composition: astrosil`` that is only
found when Python happens to run from the repository.

1. The bundled data is present (what the wheel must ship).
2. ``data_path`` lookup order, suffixes and error messages.
3. The readers use it: ``RTModel.read_optical_constants`` and ``StarModel.read_star``
   / ``read_json``, with bundled names, local files and absolute paths.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import artefact as art
from artefact._data import DATA_DIR, KINDS, data_path, list_data

# Every file the examples and the PHOENIX fit rely on. If one is dropped on purpose,
# remove it here and from the readers; if it is missing from a wheel, this fails.
EXPECTED = {
    "optical_constants": ["astrosil.lnk"],
    "stellar_models": [
        "A0V_spectrum.npz",
        "G2V_spectrum.npz",
        "phoenix_grid.npz",
        "phoenix_grid.sha256",
    ],
}


# ---------------------------------------------------------------------------
# 1. Bundled data
# ---------------------------------------------------------------------------


def test_data_directory_layout():
    assert DATA_DIR.is_dir()
    assert (DATA_DIR / "filters.yaml").is_file()
    for kind in KINDS:
        assert (DATA_DIR / kind).is_dir(), kind


@pytest.mark.parametrize(
    "kind, name", [(k, n) for k, names in EXPECTED.items() for n in names]
)
def test_expected_files_are_bundled(kind, name):
    assert name in list_data(kind)
    assert (DATA_DIR / kind / name).stat().st_size > 0


def test_filter_library_is_bundled():
    curves = list_data("filters")
    assert len(curves) > 10
    assert all(name.endswith(".csv") for name in curves)


def test_list_data_is_sorted_and_only_lists_files():
    for kind in KINDS:
        names = list_data(kind)
        assert names == sorted(names)
        assert all((DATA_DIR / kind / name).is_file() for name in names)


def test_package_paths_point_at_the_bundled_data():
    from artefact import fitting, phoenix

    assert fitting.FILTER_DIR.is_dir()
    assert fitting.DEFAULT_MAPPING_PATH.is_file()
    assert phoenix.SHIPPED_PATH.is_file()
    assert phoenix.SHIPPED_CHECKSUM_PATH.is_file()
    phoenix.verify_shipped_grid()  # the grid matches its checksum


# ---------------------------------------------------------------------------
# 2. data_path
# ---------------------------------------------------------------------------


def test_bare_name_resolves_to_the_bundled_file():
    path = data_path("optical_constants", "astrosil", suffixes=(".lnk",))
    assert path == DATA_DIR / "optical_constants" / "astrosil.lnk"
    assert path.is_file()


def test_bare_name_resolves_from_any_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = data_path("optical_constants", "astrosil", suffixes=(".lnk",))
    assert path.is_file()
    assert path.resolve() == (DATA_DIR / "optical_constants" / "astrosil.lnk").resolve()


def test_name_with_extension_needs_no_suffix():
    path = data_path("stellar_models", "phoenix_grid.npz")
    assert path == DATA_DIR / "stellar_models" / "phoenix_grid.npz"


def test_suffixes_are_tried_in_order():
    # "astrosil" itself does not exist, so the empty suffix misses and ".lnk" hits
    path = data_path("optical_constants", "astrosil", suffixes=("", ".lnk"))
    assert path.name == "astrosil.lnk"


def test_absolute_path_with_and_without_suffix(tmp_path):
    target = tmp_path / "mydust.lnk"
    target.write_text("1.0 1.5 0.1\n")

    assert data_path("optical_constants", str(target)) == target
    assert data_path("optical_constants", str(tmp_path / "mydust"), (".lnk",)) == target


def test_local_file_shadows_the_bundled_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "astrosil.lnk").write_text("1.0 1.5 0.1\n")

    path = data_path("optical_constants", "astrosil", suffixes=(".lnk",))
    assert path.resolve() == (tmp_path / "astrosil.lnk").resolve()


def test_name_with_a_folder_does_not_fall_back_to_the_bundled_data():
    with pytest.raises(FileNotFoundError):
        data_path("optical_constants", "somewhere/astrosil", suffixes=(".lnk",))


def test_missing_file_error_lists_what_was_tried_and_what_is_bundled():
    with pytest.raises(FileNotFoundError) as error:
        data_path("optical_constants", "no_such_dust", suffixes=(".lnk",))

    message = str(error.value)
    assert "no_such_dust.lnk" in message
    assert str(DATA_DIR / "optical_constants") in message
    assert "astrosil.lnk" in message  # the bundled alternatives are named


def test_unknown_kind_is_an_error():
    with pytest.raises(FileNotFoundError):
        data_path("not_a_kind", "anything")


def test_directories_are_not_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        data_path("optical_constants", str(tmp_path))


def test_public_helpers_are_exported():
    assert art.data_path is data_path
    assert art.list_data is list_data


# ---------------------------------------------------------------------------
# 3. The readers use it
# ---------------------------------------------------------------------------

STAR = dict(stype="blackbody", tstar=5772.0, rstar=1.0, dstar=10.0, lmin=0.5, lmax=2000.0, nwav=40)


def test_bundled_optical_constants_by_bare_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # nothing named astrosil here: must come from the package
    model = art.RTModel()
    model.parameters.update({**STAR, "composition": "astrosil"})
    model.make_star()
    model.read_optical_constants()

    assert model.oc_nk.shape == model.sed_wave.shape
    assert np.all(model.oc_nk.real > 0)
    assert np.all(model.oc_nk.imag <= 0)  # n - ik convention


def test_optical_constants_from_a_user_file(tmp_path):
    (tmp_path / "custom.lnk").write_text("# wavelength n k\n0.01 1.5 0.1\n1e5 1.5 0.1\n")
    model = art.RTModel()
    model.parameters.update({**STAR, "composition": str(tmp_path / "custom")})
    model.make_star()
    model.read_optical_constants()

    assert model.oc_nk == pytest.approx(1.5 - 0.1j)


def test_optical_constants_missing_file_names_the_bundled_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = art.RTModel()
    model.parameters.update({**STAR, "composition": "olivine"})
    model.make_star()
    with pytest.raises(FileNotFoundError, match="astrosil"):
        model.read_optical_constants()


def test_read_star_with_a_bundled_spectrum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    star = art.StarModel()
    star.parameters = {"model": "G2V_spectrum.npz"}
    waves, spectrum = star.read_star()

    assert waves.min() < 0.5 < 5.0 < waves.max()  # wavelengths are in um
    assert np.all(np.diff(waves) >= 0)  # non-decreasing: the SVO tables repeat some points
    assert np.all(spectrum >= 0)

    # the extension may be left off
    star.parameters = {"model": "G2V_spectrum"}
    waves_again, spectrum_again = star.read_star()
    assert np.array_equal(waves, waves_again)
    assert np.array_equal(spectrum, spectrum_again)


@pytest.mark.parametrize("name", ["G2V_spectrum", "A0V_spectrum"])
def test_bundled_spectra_are_complete_physical_models(name):
    star = art.StarModel()
    star.parameters = {"model": name}
    waves, spectrum = star.read_star()

    assert len(waves) == len(spectrum) > 100_000
    assert waves.min() < 0.01 and waves.max() > 1000.0  # um: UV to far-IR
    assert np.all(np.isfinite(waves)) and np.all(np.isfinite(spectrum))
    assert spectrum.max() > 0.0


def test_read_star_npz_matches_the_text_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wave_angstrom = np.array([1e3, 1e4, 1e5, 1e6])
    flux = np.array([3.0, 2.0, 1.0, 0.5])
    np.savetxt(tmp_path / "table.txt", np.c_[wave_angstrom, flux])
    np.savez(tmp_path / "table.npz", wavelength=wave_angstrom, flux=flux)

    from_text = art.StarModel()
    from_text.parameters = {"model": "table.txt"}
    from_npz = art.StarModel()
    from_npz.parameters = {"model": "table.npz"}

    for a, b in zip(from_text.read_star(), from_npz.read_star(), strict=True):
        assert np.array_equal(a, b)


def test_read_star_with_a_local_spectrum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wave_angstrom = np.array([1e3, 1e4, 1e5])
    np.savetxt(tmp_path / "mystar.txt", np.c_[wave_angstrom, [3.0, 2.0, 1.0]])

    star = art.StarModel()
    star.parameters = {"model": "mystar.txt"}
    waves, spectrum = star.read_star()

    assert waves == pytest.approx(wave_angstrom * 1e-4)
    assert spectrum == pytest.approx(np.array([3.0, 2.0, 1.0]) * art.CGS_FLAM_TO_SI)


@pytest.mark.parametrize("name", ["mystar.json", "mystar"])
def test_read_json_with_and_without_extension(tmp_path, monkeypatch, name):
    monkeypatch.chdir(tmp_path)
    data = {"star_spec": {"wavelength": [1.0, 10.0, 100.0], "fnujy": [4.0, 2.0, 1.0]}}
    (tmp_path / "mystar.json").write_text(json.dumps(data))

    star = art.StarModel()
    star.parameters = {"model": name}
    waves, spectrum = star.read_json()

    assert waves == pytest.approx([1.0, 10.0, 100.0])
    expected = np.array([4.0, 2.0, 1.0]) * 1e-26 * art.C_LIGHT / (waves * art.UM) ** 2
    assert spectrum == pytest.approx(expected)
