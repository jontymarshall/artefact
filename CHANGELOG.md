# Changelog

All notable changes to artefact are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

First release as an installable package.

### Added
- `dtype="multi"`: discs of several belts (Gaussian, power-law, two-power-law or tabulated) that share one grain size distribution, each with its own relative mass, from `belts` or a JSON `belts_file`.
- `dtype="arbit"`: a tabulated radial profile from `profile_file` or `profile_r` / `profile_sigma`.
- `amin="blowout"`: start the grain size distribution at the radiation-pressure blow-out size.
- `RTModel.make_image` for resolved thermal and scattered-light images, and `ModelImage.writeto` for FITS output.
- `RTModel.calculate_disc_mass`, extending the dust mass to the planetesimal belt (Krivov & Wyatt 2021).
- PHOENIX photosphere fitting (`stype="phoenix"`) and filter-integrated photometry (`Photometer`).
- Bundled data (dust optical constants, stellar spectra, PHOENIX grid, filter curves) found by name with `artefact.data_path` and `artefact.list_data`.
- Four example scripts and Jupyter notebooks: SED modelling, fitting photometry, resolved imaging, and a two-belt system.
- Test suite, CI on Python 3.10-3.13 (Linux, Windows, macOS), and a trusted-publishing release workflow.
- `CITATION.cff` and `NOTICE` (provenance of the bundled data).

### Changed
- The code is a `src/artefact` package; install it with `pip install artefact`.
- The stellar model is its own class: `StarModel` holds `read_star`, `read_json`, `fit_phoenix` and `scale_star`, and `RTModel.make_star` uses it. `RTModel.scale_star` is now a separate, explicit step; `make_star` no longer scales to the photometry.
- The bundled G2V and A0V spectra are stored as compressed `.npz` archives (about 1.4 MB each instead of 13 MB). `read_star` reads either `.npz` or a text table.
- Dust emission and scattering size their arrays from the real number of annuli, so multi-belt discs work.

### Fixed
- `blowout_size` and `calculate_mass` are no longer compiled with numba, which could not handle them.
- Files named in a parameter file (`composition`, `model`) are looked up in the bundled data as well as the working directory.

### Removed
- Scripts using the pre-0.3 API (`RTModel.get_parameters(model, ...)` workflow).
