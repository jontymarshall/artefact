# Analytical Radiative Transfer Engine Focused on the Architectures of Circumstellar Things

Radiative transfer code to calculate the continuum emission from a dust disc around a star in the optically thin limit. 

## Introduction ##

The repository contains the artefact code along with a worked example including the necessary data files. The code can take input stellar models from a variety of sources including direct user defined input, a blackbody assumption, VOSA stellar models (https://svo2.cab.inta-csic.es/theory/newov2/), and from SDB JSON files (https://sdb.drgmk.com). Disc architecture is likewise flexibly defined with 1- and 2- power law models, a Gaussian belt, or user defined distribution. Dust grain scattering and emission is calculated from optical constants using Mie theory with a power law size distribution of dust grains in conjunction with the belt architecture.

If you use artefact in your research, please cite Marshall et al. (2023) (https://ui.adsabs.harvard.edu/abs/2023MNRAS.521.5940M). This code was written by Jonathan Marshall in collaboration with Robin Young and Shane Hengst. Artefact makes use of several Python packages for its calculations, most notably AstroPy, Numpy, SciPy, and MiePython. In the examples given here I have used a version of PySED (https://github.com/jontymarshall/pysed) to obtain stellar data and photometry for HD 104860 from Simbad and Vizier. I also used the Python packages Emcee and Corner to explore parameter space for modelling and return maximum amplitude probability results.

## Installation ##

```bash
git clone https://github.com/jontymarshall/artefact.git
cd artefact
pip install -e .
```

## Quickstart ##

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
│       ├── _data.py            <- data_path() resolver (see below)
│       └── data/
│           ├── filters.yaml
│           ├── filters/*.csv
│           ├── optical_constants/astrosil.lnk
│           └── stellar_models/ ...
├── tests/                      <- test_artefact.py, test_images.py, test_phoenix.py, test_star_and_belts.py
├── examples/                   <- example scripts, parameter files, example_utils.py
├── notebooks/                  <- example scripts in Jupyter notebook format
├── docs/                       <- documentation
└── contrib/                    <- the old robin/ scripts, unmaintained and not packaged
```
