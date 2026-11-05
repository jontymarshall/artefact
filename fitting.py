"""
Photometry class with filters for artefact fitting stellar photosphere and dust excess models.

A catalog quotes a flux density for a reference spectrum (F_nu constant, or proportional to 1/nu, ...): the
flux density of that spectrum which gives the signal the instrument saw. For a model spectrum F_nu(lambda) the
equivalent quoted flux is

    F_quoted = sum_i F_nu(lambda_i) w_i,    w_i = T_i g_i dlambda_i / sum_j R_j T_j g_j dlambda_j

where T is the filter's transmission, g = 1/lambda for a photon-counting detector (1/lambda^2 for an energy
detector), and R the reference spectrum scaled to 1 at its reference wavelength. `response_for` gives the
wavelengths lambda_i and weights w_i of a band, so a fit evaluates its model once on those wavelengths and
takes one dot product. Band names are mapped to filters by `data/filters.yaml`; the curves are in
`data/filters/`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml

DATA_DIR = Path(__file__).resolve().parent / "data"
FILTER_DIR = DATA_DIR / "filters"
DEFAULT_MAPPING_PATH = DATA_DIR / "filters.yaml"
MAX_SAMPLES = (
    400  # a longer filter curve is thinned to this many wavelengths for evaluation
)
DETECTOR_TYPES = ("photon", "energy")
HC_OVER_K_UM_K = 14387.77  # h c / k in micron kelvin


# --- filter curves ---


@dataclass(frozen=True)
class FilterCurve:
    """One filter: transmission against wavelength (micron), as the detector responds to it."""

    name: str  # the SVO identifier, e.g. "WISE/WISE.W3", or "ALMA/ALMA.B6.REASONS"
    wave_um: np.ndarray = None  # type: ignore[assignment]
    transmission: np.ndarray = None  # type: ignore[assignment]
    detector: str = "photon"  # "photon" (counts photons) or "energy" (bolometer)
    reference_um: float | None = (
        None  # the filter's reference wavelength, when the source gives one
    )
    source: str = ""  # where the curve came from, with the download date

    def __post_init__(self) -> None:
        if self.detector not in DETECTOR_TYPES:
            raise ValueError(
                f"{self.name}: detector must be one of {DETECTOR_TYPES}, not {self.detector!r}"
            )
        if len(self.wave_um) != len(self.transmission) or len(self.wave_um) < 2:
            raise ValueError(
                f"{self.name}: needs matching wavelength and transmission columns"
            )
        if np.any(np.diff(self.wave_um) <= 0):
            raise ValueError(f"{self.name}: wavelengths must increase")

    @property
    def filename(self) -> str:
        return file_stem(self.name) + ".csv"


def file_stem(name: str) -> str:
    """The file name (without ".csv") of the filter called `name`."""
    return name.replace("/", "_")


def read_filter(path: Path) -> FilterCurve:
    """Read one filter file."""
    facts: dict[str, str] = {}
    waves: list[float] = []
    values: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            key, _, text = line[1:].partition(":")
            facts[key.strip()] = text.strip()
        elif line and not line.startswith("wavelength"):
            wave, value = line.split(",")
            waves.append(float(wave))
            values.append(float(value))
    reference = facts.get("reference_um")
    return FilterCurve(
        name=facts["name"],
        wave_um=np.array(waves),
        transmission=np.array(values),
        detector=facts.get("detector", "photon"),
        reference_um=float(reference) if reference else None,
        source=facts.get("source", ""),
    )


@cache
def load_library(directory: Path = FILTER_DIR) -> dict[str, FilterCurve]:
    """Every filter in `directory`, by name (cached)."""
    curves = (read_filter(path) for path in sorted(directory.glob("*.csv")))
    return {curve.name: curve for curve in curves}


# --- band names -> filters ---


@dataclass(frozen=True)
class BandSpec:
    """One line of data/filters.yaml."""

    filter: (
        str | None
    )  # None for a band that is a flat flux density at one wavelength (needs no filter)
    index: float = 0.0  # reference spectrum F_nu ~ nu^-index
    blackbody_k: float | None = None  # or a blackbody of this temperature
    reference_um: float | None = None


@dataclass(frozen=True)
class BandMap:
    exact: dict[str, BandSpec]
    patterns: tuple[tuple[re.Pattern[str], BandSpec], ...]

    def spec_for(self, band: str) -> BandSpec | None:
        if band in self.exact:
            return self.exact[band]
        return next(
            (spec for pattern, spec in self.patterns if pattern.fullmatch(band)), None
        )


def _spec(entry: dict) -> BandSpec:
    return BandSpec(
        filter=None if entry.get("monochromatic") else entry["filter"],
        index=float(entry.get("index", 0.0)),
        blackbody_k=entry.get("blackbody_k"),
        reference_um=entry.get("reference_um"),
    )


@cache
def load_mapping(path: Path = DEFAULT_MAPPING_PATH) -> BandMap:
    """Read data/filters.yaml (cached)."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    exact = {
        band: _spec(entry) for band, entry in (document.get("bands") or {}).items()
    }
    patterns = tuple(
        (re.compile(item["match"]), _spec(item))
        for item in (document.get("patterns") or [])
    )
    return BandMap(exact, patterns)


@dataclass(frozen=True)
class BandResponse:
    """What a band needs from a model: evaluate it at `wave_um`, then `weights @ flux` is the quoted flux."""

    band: str
    filter: str
    wave_um: np.ndarray
    weights: np.ndarray
    flat: bool  # the reference spectrum is constant in F_nu, so no colour correction ever applies

    def apply(self, flux_jy: np.ndarray) -> float:
        return float(self.weights @ flux_jy)


def _reference_spectrum(
    wave_um: np.ndarray, spec: BandSpec, reference_um: float
) -> np.ndarray:
    """The catalog's reference spectrum F_nu(lambda), equal to 1 at `reference_um`."""
    if spec.blackbody_k is not None:

        def planck(wave: np.ndarray) -> np.ndarray:
            return wave**-3 / np.expm1(HC_OVER_K_UM_K / (wave * spec.blackbody_k))

        return planck(wave_um) / planck(np.array(reference_um))
    return (wave_um / reference_um) ** spec.index


def _thinned(curve: FilterCurve) -> tuple[np.ndarray, np.ndarray]:
    if len(curve.wave_um) <= MAX_SAMPLES:
        return curve.wave_um, curve.transmission
    wave = np.geomspace(curve.wave_um[0], curve.wave_um[-1], MAX_SAMPLES)
    return wave, np.interp(wave, curve.wave_um, curve.transmission)


def band_weights(band: str, curve: FilterCurve, spec: BandSpec) -> BandResponse:
    """Wavelengths and weights of `curve` for a catalog whose reference spectrum is `spec`'s."""
    wave, transmission = _thinned(curve)
    reference_um = (
        spec.reference_um or curve.reference_um or float(np.sqrt(wave[0] * wave[-1]))
    )
    detector = 1.0 / wave if curve.detector == "photon" else 1.0 / wave**2
    edges = np.concatenate(([wave[0]], (wave[1:] + wave[:-1]) / 2, [wave[-1]]))
    width = np.diff(edges)
    signal = np.clip(transmission, 0.0, None) * detector * width
    norm = float(np.sum(_reference_spectrum(wave, spec, reference_um) * signal))
    if norm <= 0:
        raise ValueError(f"filter {curve.name} passes nothing")
    return BandResponse(
        band,
        curve.name,
        wave,
        signal / norm,
        flat=spec.blackbody_k is None and spec.index == 0,
    )


def response_for(
    band: str,
    library: dict[str, FilterCurve] | None = None,
    mapping: BandMap | None = None,
) -> BandResponse | None:
    """The band's response, or None when the band has no filter in the mapping or the library lacks it."""
    library = load_library() if library is None else library
    spec = (mapping or load_mapping()).spec_for(band)
    if spec is None or spec.filter is None or spec.filter not in library:
        return None
    return band_weights(band, library[spec.filter], spec)


def missing_bands(
    bands: Iterable[str],
    library: dict[str, FilterCurve] | None = None,
    mapping: BandMap | None = None,
) -> list[str]:
    """The band names (each once, sorted) that have no usable filter, so they are fitted at one wavelength.

    A band listed as `monochromatic` (a spectrum bin, a radio flux density) never needs one."""
    chosen = mapping or load_mapping()
    missing = set()
    for band in bands:
        spec = chosen.spec_for(band)
        if spec is not None and spec.filter is None:
            continue
        if response_for(band, library, mapping) is None:
            missing.add(band)
    return sorted(missing)


# --- photometry ---


@dataclass
class Photometer:
    """Photometry as arrays: wavelength (micron), flux and error (Jy), upper-limit flag."""

    wave_um: np.ndarray
    flux_jy: np.ndarray
    err_jy: np.ndarray
    is_limit: np.ndarray | None = (
        None  # True for an upper limit; None means all are detections
    )
    # per point: the band's filter response, or None to compare the model at the point's wavelength
    responses: list[BandResponse | None] | None = None
    missing_filters: tuple[
        str, ...
    ] = ()  # fitted bands that have no filter (set on the whole data set)

    def __post_init__(self) -> None:
        self.wave_um = np.asarray(self.wave_um, dtype=float)
        self.flux_jy = np.asarray(self.flux_jy, dtype=float)
        self.err_jy = np.asarray(self.err_jy, dtype=float)
        if self.is_limit is None:
            self.is_limit = np.zeros(len(self.wave_um), dtype=bool)
        self.is_limit = np.asarray(self.is_limit, dtype=bool)

    @classmethod
    def from_bands(
        cls,
        bands: Iterable[str | None],
        wave_um: np.ndarray,
        flux_jy: np.ndarray,
        err_jy: np.ndarray,
        is_limit: np.ndarray | None = None,
    ) -> Photometer:
        """Photometry whose points are named bands (data/filters.yaml): a band with a filter is compared with
        the model integrated over it, a band without one (or a None name) at its wavelength."""
        names = list(bands)
        responses = [None if name is None else response_for(name) for name in names]
        missing = tuple(missing_bands(name for name in names if name is not None))
        return cls(wave_um, flux_jy, err_jy, is_limit, responses, missing)

    def select(self, mask: np.ndarray) -> Photometer:
        responses = None
        if self.responses is not None:
            responses = [self.responses[i] for i in np.flatnonzero(mask)]
        return Photometer(
            self.wave_um[mask],
            self.flux_jy[mask],
            self.err_jy[mask],
            self.is_limit[mask],
            responses,
        )

    def evaluate(self, model: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """What the instrument would report at each point for `model` (wavelength in micron -> Jy).

        Points with a filter response get the model integrated over the band, as the catalog defines its
        flux; the others get the model at the point's wavelength. The model is called once.
        """
        if not self.responses or all(r is None for r in self.responses):
            return np.asarray(model(self.wave_um), dtype=float)
        grids = [r.wave_um for r in self.responses if r is not None]
        values = np.asarray(model(np.concatenate([self.wave_um, *grids])), dtype=float)
        result = values[: len(self.wave_um)].copy()
        start = len(self.wave_um)
        for i, response in enumerate(self.responses):
            if response is not None:
                stop = start + len(response.wave_um)
                result[i] = response.apply(values[start:stop])
                start = stop
        return result
