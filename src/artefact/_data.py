"""Locate the data files that ship with artefact, or that the user supplies."""

from importlib.resources import files
from pathlib import Path

# Real directory inside the installed package (or src/artefact/data in an editable install)
DATA_DIR = Path(str(files("artefact") / "data"))

KINDS = ("optical_constants", "stellar_models", "filters")


def list_data(kind):
    """Names of the bundled files of one kind, for messages and discovery."""
    return sorted(p.name for p in (DATA_DIR / kind).iterdir() if p.is_file())


def data_path(kind, name, suffixes=("",)):
    """Resolve ``name`` to an existing file.

    Order of lookup, first hit wins:
      1. ``name`` as given (absolute, or relative to the working directory),
         with each suffix tried in turn;
      2. if ``name`` has no directory part, ``DATA_DIR/kind/name`` with each suffix.
    """
    name = str(name)
    tried = []

    for suffix in suffixes:
        candidate = Path(name + suffix).expanduser()
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate

    if Path(name).name == name:  # bare name, no folders
        for suffix in suffixes:
            candidate = DATA_DIR / kind / (name + suffix)
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"No {kind} file {name!r}. Looked in: {', '.join(tried)}. "
        f"Bundled {kind}: {', '.join(list_data(kind))}."
    )
    