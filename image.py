# Defaults for model images
ASPECT_RATIO = 0.03  # scale height over radius, H / r
HG_G1 = 0.7  # forward-scattering asymmetry parameter of the first component
HG_G2 = -0.2  # backward-scattering asymmetry parameter of the second component
HG_WEIGHT = 0.7  # weight of the first component (the second has 1 - weight)
# Limits that keep the line-of-sight integral well conditioned: the depth z is
# divided by sin(i), and a razor-thin disc has a vertical width of aspect * r.
MIN_INC = 1e-4  # radians; face-on images are exact to about 5e-9
MIN_ASPECT = 1e-3

FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))  # sigma = FWHM * FWHM_TO_SIGMA
ARCSEC_PER_DEG = 3600.0


def convolve_beam(image, beam):
    """
    Convolve a flux image (flux per pixel) with a beam, conserving total flux.

    Parameters
    ----------
    image : 2D float array
        Image in flux per pixel.
    beam : float, 2D float array or kernel object
        A float is the FWHM of a circular Gaussian beam in pixels. An array is a
        point-spread function sampled on the same pixels. An object with a 2D
        ``array`` attribute, such as an astropy ``Kernel2D``, is used through that
        array, so ``Gaussian2DKernel(x_stddev, y_stddev, theta)`` gives an
        elliptical beam at any position angle. A PSF must have odd dimensions so
        that its peak is on the central pixel, and is normalised to unit sum.

    Returns
    -------
    convolved : 2D float array
        Image in flux per pixel, with the same shape as ``image``.

    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve

    if hasattr(beam, "array"):
        beam = beam.array

    if np.ndim(beam) == 0:
        if beam <= 0:
            raise ValueError(f"Beam FWHM must be positive, got {beam!r}.")
        return gaussian_filter(
            image, sigma=beam * FWHM_TO_SIGMA, mode="constant", cval=0.0
        )

    psf = np.asarray(beam, dtype=float)
    if psf.ndim != 2 or np.sum(psf) <= 0:
        raise ValueError("A PSF must be a 2D array with a positive sum.")
    if psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        raise ValueError(f"A PSF must have odd dimensions, got {psf.shape}.")
    return fftconvolve(image, psf / np.sum(psf), mode="same")


def henyey_greenstein(cos_theta, g):
    """
    Henyey-Greenstein phase function, normalised to 1 over the sphere.

    Parameters
    ----------
    cos_theta : float or float array
        Cosine of the scattering angle (1 is forward scattering).
    g : float
        Asymmetry parameter, between -1 (backward) and 1 (forward).

    Returns
    -------
    phase : float or float array
        p(theta) in 1/sr, so that the integral of p over 4 pi sr is 1. An isotropic
        scatterer has 1 / (4 pi).

    """
    if not -1.0 < g < 1.0:
        raise ValueError(f"The asymmetry parameter must be in (-1, 1), got {g!r}.")
    return (1.0 - g**2) / (4.0 * np.pi * (1.0 + g**2 - 2.0 * g * cos_theta) ** 1.5)


def two_component_hg(cos_theta, g1=HG_G1, g2=HG_G2, weight=HG_WEIGHT):
    """
    Two-component Henyey-Greenstein phase function, normalised to 1 over the sphere.

    The sum of a forward-peaked component (``g1``) with weight ``weight`` and a
    backward component (``g2``) with weight ``1 - weight``; see
    ``henyey_greenstein``.

    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"The component weight must be in [0, 1], got {weight!r}.")
    return weight * henyey_greenstein(cos_theta, g1) + (1.0 - weight) * (
        henyey_greenstein(cos_theta, g2)
    )


def disc_surface_brightness(
    p, q, edges, flux_emit, flux_scat, inc, aspect, phase=None, nlos=48, chunk=40000
):
    """
    Flux per unit sky area of a vertically thick, optically thin disc.

    The dust of each annulus has a Gaussian vertical profile with standard
    deviation H = aspect * r. A point (x, y, z) of the disc (x along the major axis,
    y across it in the plane, z vertical) projects onto the sky at
    (p, q) = (x, y cos i + z sin i) and lies s = z cos i - y sin i towards the
    observer. The flux at (p, q) is the integral along the line of sight of the
    emissivity, which is the face-on flux per unit disc area of the annulus times
    the vertical profile. The integral is done over y, within the extent where the
    profile is non-negligible (6 sigma) and the disc exists, with the midpoint rule.

    Scattered light is weighted by 4 pi p(theta), where theta is the angle between
    the direction of the starlight at the grain (away from the star) and the
    direction to the observer, cos(theta) = s / sqrt(x^2 + y^2 + z^2). The side of the
    disc with q < 0 is the near side.

    Parameters
    ----------
    p, q : 1D float arrays
        Sky offsets along and across the major axis, in au.
    edges : float array
        Annulus boundaries in au.
    flux_emit, flux_scat : float arrays
        Face-on thermal and scattered flux per unit disc area of each annulus,
        in mJy / au^2.
    inc : float
        Inclination in radians, greater than zero.
    aspect : float
        Scale height over radius, greater than zero.
    phase : callable or None
        Phase function p(cos theta) in 1/sr, or None for isotropic scattering.
    nlos : int
        Number of line-of-sight samples per pixel.
    chunk : int
        Number of pixels processed at once, to limit memory.

    Returns
    -------
    brightness : 1D float array
        Flux per unit sky area in mJy / au^2, with the shape of ``p``.

    """
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    nring = len(flux_emit)
    rin, rout = edges[0], edges[-1]

    # Extent of the integral over y: the chord of the disc at each p, limited to
    # 6 sigma around the point where the line of sight crosses the mid-plane.
    chord = np.sqrt(np.clip(rout**2 - p**2, 0.0, None))
    y_mid = q / cos_i
    r_ref = np.clip(np.hypot(p, y_mid), rin, rout)
    half = 6.0 * aspect * r_ref * sin_i / cos_i
    lower = np.maximum(y_mid - half, -chord)
    upper = np.minimum(y_mid + half, chord)

    brightness = np.zeros_like(p, dtype=float)
    todo = np.flatnonzero((chord > 0.0) & (upper > lower))
    fraction = (np.arange(nlos) + 0.5) / nlos

    for start in range(0, len(todo), chunk):
        sel = todo[start : start + chunk]
        ps, qs = p[sel, None], q[sel, None]
        width = (upper[sel] - lower[sel])[:, None]
        y = lower[sel, None] + width * fraction

        r = np.hypot(ps, y)
        ring = np.searchsorted(edges, r, side="right") - 1
        inside = (ring >= 0) & (ring < nring)
        ring = np.clip(ring, 0, nring - 1)

        z = (qs - y * cos_i) / sin_i
        height = np.maximum(aspect * r, 1e-12)
        vertical = np.exp(-0.5 * (z / height) ** 2) / (np.sqrt(2.0 * np.pi) * height)

        scattered = flux_scat[ring]
        if phase is not None:
            towards_observer = (qs * cos_i - y) / sin_i
            cos_theta = towards_observer / np.maximum(np.hypot(r, z), 1e-12)
            scattered = scattered * 4.0 * np.pi * phase(cos_theta)

        line = np.where(inside, (flux_emit[ring] + scattered) * vertical, 0.0)
        brightness[sel] = line.sum(axis=1) * (width[:, 0] / nlos) / sin_i

    return brightness


@dataclass
class ModelImage:
    """
    A model image of the disc on a regular sky grid.

    ``data[row, column]`` has north increasing with the row and east decreasing with
    the column, so ``imshow(data, origin="lower")`` shows north up and east left.
    The star is at the centre of the array.

    Attributes
    ----------
    data : 2D float array
        Image in the unit given by ``unit``.
    pixscale : float
        Pixel size in arcsec.
    wavelength : float
        Wavelength in um.
    unit : str
        "mJy/pixel", "mJy/arcsec2" or "mJy/beam".
    component : str
        "emit", "scat" or "total".
    inc, pa : float
        Inclination and position angle of the disc in degrees.
    beam : float, 2D array, kernel object or None
        Beam FWHM in arcsec (or the PSF array or kernel) that the image was
        convolved with.
    aspect : float or None
        Scale height over radius that the image was made with.
    phase : tuple or None
        (g1, g2, weight) of the phase function applied to the scattered light, or
        None if the scattering was isotropic or there was none.

    """

    data: np.ndarray
    pixscale: float
    wavelength: float
    unit: str
    component: str
    inc: float
    pa: float
    beam: object = None
    aspect: float = None
    phase: tuple = None

    def header(self):
        """Return FITS header cards (a dict) with a tangent-plane offset WCS."""
        npix_y, npix_x = self.data.shape
        cards = {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CRPIX1": npix_x / 2.0 + 0.5,
            "CRPIX2": npix_y / 2.0 + 0.5,
            "CDELT1": -self.pixscale / ARCSEC_PER_DEG,
            "CDELT2": self.pixscale / ARCSEC_PER_DEG,
            "BUNIT": self.unit,
            "WAVELEN": float(self.wavelength),
            "COMMENT": f"artefact model image, component {self.component}",
            "INC": float(self.inc),
            "PA": float(self.pa),
        }
        if self.aspect is not None:
            cards["ASPECT"] = float(self.aspect)
        if self.phase is not None:
            cards["HGG1"], cards["HGG2"], cards["HGWEIGHT"] = map(float, self.phase)
        if isinstance(self.beam, (int, float, np.floating)):
            cards["BMAJ"] = cards["BMIN"] = self.beam / ARCSEC_PER_DEG
            cards["BPA"] = 0.0
        return cards

    def writeto(self, filename, overwrite=True):
        """Write the image to a FITS file (requires astropy)."""
        from astropy.io import fits

        hdu = fits.PrimaryHDU(np.asarray(self.data, dtype=np.float32))
        for key, value in self.header().items():
            hdu.header[key] = value
        hdu.writeto(filename, overwrite=overwrite)
