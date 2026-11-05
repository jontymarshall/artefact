"""
Photometry class with filters for artefact fitting stellar photosphere and dust excess models.
"""

import sys
import types
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml

FILTER_DIR = Path(__file__).resolve().parents[3] / "data" / "filters"
DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[3] / "data" / "filters.yaml"
MAX_SAMPLES = 400  # a longer filter curve is thinned to this many wavelengths for evaluation


HC_OVER_K_UM_K = 14387.77  # h c / k in micron kelvin

@dataclass
class Photometer:
    """Photometry as arrays: wavelength (micron), flux and error (Jy)."""

    wave_um: np.ndarray
    flux_jy: np.ndarray
    err_jy: np.ndarray
    # per point: the band's filter response, or None to compare the model at the point's wavelength
    responses: list[BandResponse | None] | None = None
    missing_filters: tuple[str, ...] = ()  # fitted bands that have no filter (set on the whole data set)

    def select(self, mask: np.ndarray) -> "Photometer":
        responses = None
        if self.responses is not None:
            responses = [self.responses[i] for i in np.flatnonzero(mask)]
        return Photometer(
            self.wave_um[mask], self.flux_jy[mask], self.err_jy[mask], self.is_limit[mask], responses
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


@dataclass(frozen=True)
class FilterCurve:
    """One filter: transmission against wavelength (micron), as the detector responds to it."""

    name: str  # the SVO identifier, e.g. "WISE/WISE.W3", or "ALMA/ALMA.B6.REASONS"
    wave_um: np.ndarray = None  # type: ignore[assignment]
    transmission: np.ndarray = None  # type: ignore[assignment]
    detector: str = "photon"  # "photon" (counts photons) or "energy" (bolometer)
    reference_um: float | None = None  # the filter's reference wavelength, when the source gives one
    source: str = ""  # where the curve came from, with the download date

    def __post_init__(self) -> None:
        if self.detector not in DETECTOR_TYPES:
            raise ValueError(f"{self.name}: detector must be one of {DETECTOR_TYPES}, not {self.detector!r}")
        if len(self.wave_um) != len(self.transmission) or len(self.wave_um) < 2:
            raise ValueError(f"{self.name}: needs matching wavelength and transmission columns")
        if np.any(np.diff(self.wave_um) <= 0):
            raise ValueError(f"{self.name}: wavelengths must increase")

    @property
    def filename(self) -> str:
        return file_stem(self.name) + ".csv"


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


def _thinned(curve: FilterCurve) -> tuple[np.ndarray, np.ndarray]:
    if len(curve.wave_um) <= MAX_SAMPLES:
        return curve.wave_um, curve.transmission
    wave = np.geomspace(curve.wave_um[0], curve.wave_um[-1], MAX_SAMPLES)
    return wave, np.interp(wave, curve.wave_um, curve.transmission)


def band_weights(band: str, curve: FilterCurve, spec: BandSpec) -> BandResponse:
    """Wavelengths and weights of `curve` for a catalog whose reference spectrum is `spec`'s."""
    wave, transmission = _thinned(curve)
    reference_um = spec.reference_um or curve.reference_um or float(np.sqrt(wave[0] * wave[-1]))
    detector = 1.0 / wave if curve.detector == "photon" else 1.0 / wave**2
    edges = np.concatenate(([wave[0]], (wave[1:] + wave[:-1]) / 2, [wave[-1]]))
    width = np.diff(edges)
    signal = np.clip(transmission, 0.0, None) * detector * width
    norm = float(np.sum(_reference_spectrum(wave, spec, reference_um) * signal))
    if norm <= 0:
        raise ValueError(f"filter {curve.name} passes nothing")
    return BandResponse(
        band, curve.name, wave, signal / norm, flat=spec.blackbody_k is None and spec.index == 0
    )