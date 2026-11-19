"""Convert an SVO-style stellar spectrum table to the compact ``.npz`` format.

Usage:  python contrib/svo_to_npz.py spectrum.txt [spectrum.npz]

The text table has wavelength (Angstrom) and flux density (erg/cm2/s/A) in its first
two columns, with ``#`` comments. The archive stores the same numbers, unchanged, as
arrays ``wavelength`` and ``flux``, plus the comment lines as ``header``. It is about
ten times smaller and loads much faster. ``artefact`` reads either format through the
``model`` parameter.
"""

import sys
from pathlib import Path

import numpy as np


def convert(source, target=None):
    source = Path(source)
    target = Path(target) if target else source.with_suffix(".npz")

    with open(source, encoding="utf-8") as table:
        header = "".join(line for line in table if line.startswith("#"))
    wavelength, flux = np.loadtxt(source, comments="#", usecols=(0, 1), unpack=True)

    np.savez_compressed(
        target, wavelength=wavelength, flux=flux, header=np.array(header)
    )

    with np.load(target) as archive:  # check the round trip
        assert np.array_equal(archive["wavelength"], wavelength)
        assert np.array_equal(archive["flux"], flux)
    return target


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit(__doc__)
    print("Wrote", convert(*sys.argv[1:]))
