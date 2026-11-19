# Analytical Radiative Transfer Engine Focused on the Architectures of Circumstellar Things

Radiative transfer code to calculate the continuum emission from a dust disc around a star in the optically thin limit. 

## Introduction ##

The repository contains the artefact code along with a worked example including the necessary data files. The code can take input stellar models from a variety of sources including direct user defined input, a blackbody assumption, VOSA stellar models (https://svo2.cab.inta-csic.es/theory/newov2/), and from SDB JSON files (https://sdb.drgmk.com). Disc architecture is likewise flexibly defined with 1- and 2- power law models, a Gaussian belt, or user defined distribution. Dust grain scattering and emission is calculated from optical constants using Mie theory with a power law size distribution of dust grains in conjunction with the belt architecture.

If you use artefact in your research, please cite Marshall et al. (2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.5940M). This code was written by Jonathan Marshall in collaboration with Robin Young and Shane Hengst. Artefact makes use of several Python packages for its calculations, most notably AstroPy, Numpy, SciPy, and MiePython. In the examples given here I have used a version of PySED (https://github.com/jontymarshall/pysed) to obtain stellar data and photometry for HD 104860 from Simbad and Vizier. I also used the Python packages Emcee and Corner to explore parameter space for modelling and return maximum amplitude probability results.

## Installation ##

From PyPI:

```bash
pip install artefact
pip install "artefact[examples]"   # also matplotlib, jupyter, emcee and corner, for the examples
```

From source, for development:

```bash
git clone https://github.com/jontymarshall/artefact.git
cd artefact
pip install -e ".[dev,examples]"
pytest
```

artefact needs Python 3.10 or later. The first import takes a few seconds while numba loads.

## Quickstart ##

The dust optical constants (`astrosil`), the solar-type and A-star spectra (`G2V_spectrum.npz`, `A0V_spectrum.npz`), the PHOENIX grid and the filter curves ship with the package and are found by name from any directory; `artefact.list_data("stellar_models")` lists them. Your own files work too, by path.

```python
import artefact

model = artefact.RTModel()
model.parameters.update(
    stype="spectrum",
    model="G2V_spectrum.npz",  # star
    lstar=1.0,
    rstar=1.0,
    tstar=5770.0,
    dstar=10.0,  # L_sun, R_sun, K, pc
    composition="astrosil",
    density=3.3,  # dust: g cm^-3
    amin=1.0,
    amax=1000.0,
    ngrain=41,
    q=-3.5,  # grain sizes in um, dN/da ~ a^q
    mdust=1e-3,  # Earth masses of dust
    lmin=0.3,
    lmax=3000.0,
    nwav=150,  # wavelength grid in um
    dtype="onepl",
    rin=40.0,
    rout=60.0,
    alpha_out=-1.0,
    nring=31,  # belt in au
)

model.make_star()  # photosphere
model.make_dust()  # grain sizes and numbers
model.read_optical_constants()
model.make_disc()  # annuli
model.calculate_surface_density()
model.calculate_qabs()  # Mie theory
model.calculate_dust_scatter()
model.calculate_dust_emission(mode="full", tolerance=1e-3)
model.flam_to_fnu()  # model.sed_wave (um); sed_star, sed_emit, sed_scat, sed_disc (mJy)

image = model.make_image(
    70.0, inc=60.0, pa=30.0, beam=5.0
)  # resolved image, image.data
```

Worked examples, each as a script in `examples/` and as a Jupyter notebook in `notebooks/`:

1. SED of a star with a single debris belt.
2. Fitting a model to photometry (HD 104860) with lmfit, with the minimum grain size as a free parameter.
3. Resolved images: thermal emission and scattered light, inclination, beams and FITS output.
4. A two-belt system (`dtype="multi"`): SED, fit and images.

Run a script with `python examples/01_sed_single_belt.py`.

API documentation is in the docstrings (`help(artefact.RTModel)`); to build HTML pages, run `pip install pdoc` and `pdoc artefact -o docs`.

## Repository Overview ##

The structure of the repository is laid out as follows:

```
artefact/                       <- repository root
├── pyproject.toml
├── README.md
├── LICENSE
├── NOTICE                      <- provenance and licences of bundled data
├── CITATION.cff                <- citation
├── CHANGELOG.md
├── MANIFEST.in                 <- what the sdist includes beyond the wheel
├── .gitattributes  .gitignore
├── .github/workflows/
│   ├── ci.yml                  <- ruff + pytest, Python 3.10-3.13, Linux/Windows/macOS
│   └── publish.yml             <- build + PyPI trusted publishing on a tagged release
├── src/
│   └── artefact/               <- the only thing that goes in the wheel
│       ├── __init__.py         <- public API and __version__
│       ├── core.py             <- RTModel, StarModel, ModBBModel
│       ├── image.py
│       ├── fitting.py
│       ├── phoenix.py
│       ├── _data.py            <- data_path() resolver
│       └── data/
│           ├── filters.yaml
│           ├── filters/*.csv
│           ├── optical_constants/astrosil.lnk
│           └── stellar_models/     <- G2V and A0V spectra (.npz), PHOENIX grid
├── tests/                      <- test_artefact.py, test_images.py, test_phoenix.py, test_star_and_belts.py
├── examples/                   <- example scripts, parameter files, photometry, example_utils.py
├── notebooks/                  <- the same examples as Jupyter notebooks
├── docs/                       <- documentation
└── contrib/                    <- unmaintained scripts (ryoung/) and svo_to_npz.py; not packaged
```
