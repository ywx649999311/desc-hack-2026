"""Shared helpers for the target-selection and cutout notebooks.

Catalog access, image search, FITS cutouts and PSF stamps for the OpenUniverse2024
Roman and Rubin simulations hosted by IRSA (public S3 bucket, no credentials needed).
Rubin PSFs need the LSST Science Pipelines (``lsst.afw``); everything else does not.
"""
import json
import pickle
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import galsim
import galsim.roman
import h5py
import numpy as np
import s3fs
import pandas as pd
import pyarrow.fs
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp
from tqdm.auto import tqdm

# Harmless, and raised for every Roman image header.
warnings.simplefilter("ignore", FITSFixedWarning)

# Simulated-data services are separate from IRSA's observed-data ones.
Irsa.sia_url = "https://irsa.ipac.caltech.edu/simulated/SIA"
Irsa.tap_url = "https://irsa.ipac.caltech.edu/simulated/TAP"

BUCKET = "nasa-irsa-simulations"
CATALOG_PREFIX = f"{BUCKET}/openuniverse2024/roman/full/roman_rubin_cats_v1.1.2_faint"
OBSEQ_URI = f"s3://{BUCKET}/openuniverse2024/roman/full/RomanTDS/Roman_TDS_obseq_11_6_23.fits"
SIA_COLLECTIONS = {
    "roman": "simulated_roman_openuniverse2024",
    "rubin": "simulated_rubin_openuniverse2024",
}

# Outputs go in <repo>/data, whatever directory a notebook runs from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGETS_PATH = DATA_DIR / "targets.parquet"

# Arrays read from each source image: (name saved in the cutout, HDU index in the source
# image, kind). The first is the science image; "error" is a per-pixel uncertainty or
# variance; "flags" is a quality bit mask, which must never be interpolated. Roman HDU 2 is
# called ERR in the source files but is the background and noise of SCI without the sources
# (roman_imsim v2.0 writes its sky_noise image there), so it is saved as BKGNOISE.
CUTOUT_LAYERS = {
    "roman": [("SCI", 1, "image"), ("BKGNOISE", 2, "image"), ("DQ", 3, "flags")],
    "rubin": [("IMAGE", 1, "image"), ("VARIANCE", 3, "error"), ("MASK", 2, "flags")],
}
# HDU order in the cutout files, so both instruments have the image at 0, the variance at 1
# and the flags at 2. Roman VARIANCE is not read but computed (see `roman_variance`).
CUTOUT_ORDER = {
    "roman": ["SCI", "VARIANCE", "DQ", "BKGNOISE"],
    "rubin": ["IMAGE", "VARIANCE", "MASK"],
}
# roman_imsim v2.0 clips pixels at this many electrons, then adds read noise.
ROMAN_SATURATION = 1e5

# Rubin images are downloaded here while they are cut out, so the LSST stack can read
# their PSF from a local file; each one is deleted once its cutouts are written.
CACHE_DIR = DATA_DIR / "cache"

# Roman PSF settings, as in the simulation (tds.yaml: galsim.roman.getPSF with n_waves=5;
# roman_imsim draws ordinary objects with pupil_bin=4).
ROMAN_PSF_NWAVES = 5
ROMAN_PSF_PUPIL_BIN = 4
# Charge diffusion, from roman_imsim's ChargeDiff photon op: a sum of three Gaussians in
# pixel units, as (weight, sigma in pixels).
ROMAN_CHARGE_DIFFUSION = [(0.17519, 0.4522 * 0.3279), (0.53146, 0.8050 * 0.3279),
                          (0.29335, 1.4329 * 0.3279)]
# Placeholder source SED, flat in photons per nm. A redder SED broadens the PSF slightly.
ROMAN_PSF_SED = galsim.SED(galsim.LookupTable([100, 2600], [1, 1], interpolant="linear"),
                           wave_type="nm", flux_type="fphotons")
# Default PSF stamp widths in pixels (odd): Roman 51 (5.5 arcsec), Rubin 41 (8.2 arcsec, the
# size of the PSFEx model itself; a wider Rubin stamp would be zero-padded around it).
PSF_SIZE_PIX = {"roman": 51, "rubin": 41}
# GalSim's aperture cache is not guaranteed thread-safe, and cutouts run in threads.
_GALSIM_LOCK = threading.Lock()


# --- Catalogs ---------------------------------------------------------------

def read_catalog(kind, region, columns=None, filters=None):
    """Read one HEALPix region (nside=32, RING) of a parquet catalog into pandas.

    ``kind`` is ``'snana'`` (transients), ``'galaxy'`` or ``'galaxy_flux'``.
    Passing ``columns``/``filters`` reads only what is needed, which is much faster.
    """
    fs = pyarrow.fs.S3FileSystem(region="us-east-1", anonymous=True)
    return pq.read_table(f"{CATALOG_PREFIX}/{kind}_{region}.parquet", filesystem=fs,
                         columns=columns, filters=filters).to_pandas()


def tds_footprint():
    """Return the ra/dec/date ranges covered by the Roman Time Domain Survey."""
    with fits.open(OBSEQ_URI, fsspec_kwargs={"anon": True}) as hdul:
        # Through a Table: pandas >= 3 rejects the big-endian FITS buffer directly.
        obseq = Table(hdul[1].data).to_pandas()
    return {"ra": (obseq["ra"].min(), obseq["ra"].max()),
            "dec": (obseq["dec"].min(), obseq["dec"].max()),
            "mjd": (obseq["date"].min(), obseq["date"].max())}


# --- Image search -----------------------------------------------------------

def get_s3_uri(cloud_access):
    """Turn the ``cloud_access`` JSON string of an SIA result row into an S3 URI."""
    aws = json.loads(cloud_access)["aws"]
    return f"s3://{aws['bucket_name']}/{aws['key']}"


def find_images(coord, instrument, band, t_min=None, t_max=None, edge_frac=0.45):
    """Find simulated images that contain ``coord``, as an astropy Table with ``s3_uri``.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
    instrument : {'roman', 'rubin'}
    band : str
        Roman ``'J129'``, ``'H158'``, ... or Rubin ``'g'``, ``'r'``, ...
    t_min, t_max : float, optional
        Keep only images observed in this MJD window.
    edge_frac : float
        Drop images where ``coord`` lies farther than ``edge_frac * s_fov`` from the
        image centre (``s_fov`` is the image diameter), so cutouts are not clipped.
    """
    res = Irsa.query_sia(pos=(coord, (1 * u.arcsec).to(u.deg)),
                         collection=SIA_COLLECTIONS[instrument])
    band_name = np.array([str(b) for b in res["energy_bandpassname"]])
    if instrument == "roman":
        obs_id = np.array([str(o) for o in res["obs_id"]])
        keep = np.array(["TDS_simple_model" in o for o in obs_id]) & \
               np.array([band in b for b in band_name])
    else:
        # Rubin band names look like "r_57"; compare only the part before the underscore.
        keep = np.array([b.split("_")[0] == band for b in band_name])
    res = res[keep]

    if t_min is not None:
        t_obs = np.asarray(res["t_min"], dtype=float)
        res = res[(t_obs >= t_min) & (t_obs <= t_max)]
    if len(res):
        centres = SkyCoord(res["s_ra"], res["s_dec"], unit="deg")
        res = res[coord.separation(centres).deg < edge_frac * np.asarray(res["s_fov"])]
    res["s3_uri"] = np.array([get_s3_uri(c) for c in res["cloud_access"]], dtype=str)
    return res


def select_epochs(images, n):
    """Pick up to ``n`` images evenly spread in time (``t_min`` is the observation MJD)."""
    images = images.copy()
    images.sort("t_min")
    idx = np.unique(np.rint(np.linspace(0, len(images) - 1, min(n, len(images)))).astype(int))
    return images[idx]


def count_images(ra, dec, peak_mjd, instrument, band, window_days=30):
    """Number of usable images of ``band`` within ``peak_mjd +/- window_days``.

    Uses the same search as the cutout notebooks. Returns -1 if the search fails, so a
    network error is not mistaken for "no coverage".
    """
    try:
        coord = SkyCoord(ra, dec, unit="deg")
        return len(find_images(coord, instrument, band, peak_mjd - window_days, peak_mjd + window_days))
    except Exception:
        return -1


def check_coverage(targets, coverage, window_days=30, n_workers=4):
    """Add an ``n_<instrument>_images`` column per entry of ``coverage`` ({instrument: band})."""
    out = targets.copy()
    rows = list(targets.itertuples())
    for instrument, band in coverage.items():
        out[f"n_{instrument}_images"] = _map(
            lambda r: count_images(r.ra, r.dec, r.peak_mjd, instrument, band, window_days),
            rows, n_workers, f"{instrument} coverage")
    return out


# --- Cutouts ----------------------------------------------------------------

def make_cutout(data, wcs, coord, size_arcsec, north_up=False, kind="image"):
    """Cut a square, ``size_arcsec`` on a side, centred on ``coord``.

    The cutout is ``size_arcsec`` wide, rounded to an odd number of pixels so the
    target sits on the central pixel. The angular size is what is fixed: detectors
    differ slightly in pixel scale, so shapes can differ by a pixel or two. By
    default the native pixel grid is kept (no rotation) and the returned WCS is the
    image's own, shifted to the cutout. With ``north_up=True`` the cutout is resampled
    onto a north-up, east-left grid at the native pixel scale, and the returned WCS
    describes that new grid, so the file stays astrometrically correct.

    ``kind`` is ``'image'`` or ``'error'`` (float32, NaN outside the image, bilinear
    resampling) or ``'flags'`` (integer mask: 0 outside the image, nearest-neighbour
    resampling, dtype kept).

    Returns ``(array, wcs)``; raises `~astropy.nddata.utils.NoOverlapError` if
    ``coord`` is off the image.
    """
    flags = kind == "flags"
    fill = 0 if flags else np.nan
    scale = np.mean(proj_plane_pixel_scales(wcs))  # deg/pixel
    npix = int(round(size_arcsec / 3600 / scale)) | 1  # force odd
    if not north_up:
        cut = Cutout2D(data, coord, (npix, npix), wcs=wcs, mode="partial", fill_value=fill)
        return (cut.data if flags else cut.data.astype(np.float32)), cut.wcs

    # Cut oversized (corners must have data to draw from), then resample.
    big = Cutout2D(data, coord, (int(np.ceil(npix * np.sqrt(2))) + 2,) * 2, wcs=wcs,
                   mode="partial", fill_value=fill)
    out = WCS(naxis=2)
    out.wcs.ctype = wcs.wcs.ctype
    out.wcs.crval = [coord.ra.deg, coord.dec.deg]
    out.wcs.crpix = [(npix + 1) / 2, (npix + 1) / 2]
    out.wcs.cdelt = [-scale, scale]  # east to the left, north up
    out.wcs.radesys = wcs.wcs.radesys or "ICRS"
    arr, _ = reproject_interp((big.data.astype(np.float64 if flags else np.float32), big.wcs), out,
                              shape_out=(npix, npix), order="nearest-neighbor" if flags else "bilinear")
    if flags:
        return np.nan_to_num(arr).astype(data.dtype), out
    return arr.astype(np.float32), out


def roman_variance(sci, bkgnoise):
    """Per-pixel variance of a Roman cutout, in counts**2, from its ``SCI`` and ``BKGNOISE``.

    ``BKGNOISE`` is the background and noise of ``SCI`` without the sources, so
    ``SCI - BKGNOISE`` is the source counts and the mean of ``BKGNOISE`` is the expected sky
    plus dark current (flat over a cutout). The variance is their Poisson variance plus the
    read noise: ``(max(SCI - BKGNOISE, 0) + mean(BKGNOISE)) / gain + (read_noise / gain)**2``,
    with the simulation's gain and read noise (``galsim.roman``). Both parts are nearly free
    of noise, so the variance does not follow the noise in ``SCI``, which would bias fits.
    Saturated pixels (within 5 read-noise sigma of ``ROMAN_SATURATION``) are set to inf, so
    they get zero weight; ``DQ`` does not flag them in these simulations.

    Returns ``(variance, header keywords, header comment lines)``.
    """
    gain, read_noise = galsim.roman.gain, galsim.roman.read_noise
    sky = float(np.nanmean(bkgnoise))
    sources = np.clip(sci.astype(np.float64) - bkgnoise, 0, None)   # NaN off the detector stays NaN
    var = (sources + sky) / gain + (read_noise / gain) ** 2
    sat_level = (ROMAN_SATURATION - 5 * read_noise) / gain
    saturated = sci >= sat_level
    var[saturated] = np.inf
    keys = {"BUNIT": ("counts**2", "variance of SCI"),
            "GAIN": (gain, "e-/count (galsim.roman.gain)"),
            "RDNOISE": (read_noise, "read noise in e- (galsim.roman.read_noise)"),
            "SKYEXP": (sky, "expected sky + dark used, counts (mean of BKGNOISE)"),
            "SATLEVEL": (sat_level, "SCI at or above this is saturated; variance inf"),
            "NSAT": (int(saturated.sum()), "saturated pixels in this cutout")}
    # FITS comment cards hold 72 characters, so keep each line shorter.
    comments = ["VARIANCE = (max(SCI - BKGNOISE, 0) + SKYEXP) / GAIN",
                "           + (RDNOISE / GAIN)**2; saturated pixels are inf.",
                "Uses the simulation's BKGNOISE plane (real data: smooth sky model)."]
    return var.astype(np.float32), keys, comments


_LAYER_NOTES = {
    "BKGNOISE": ["The source image's ERR plane: background and noise of SCI without",
                 "the sources (roman_imsim v2.0 sky_noise). Not an uncertainty;",
                 "use VARIANCE for that."],
}


# --- PSFs -------------------------------------------------------------------

def roman_psf(header0, header1, band, x, y, size_pix=PSF_SIZE_PIX["roman"]):
    """Roman PSF stamp at pixel ``(x, y)`` (0-based, as in astropy) of one Roman image.

    Drawn as the simulation drew its objects: ``galsim.roman.getPSF`` for the image's
    detector (``SCA_NUM`` in the primary header ``header0``) and ``band``, through the
    image's own WCS (from the ``SCI`` header ``header1``) at that position, convolved
    with the simulation's charge diffusion and the pixel response, for the placeholder
    ``ROMAN_PSF_SED``. The stamp is on the image's native pixel grid, ``size_pix`` (made
    odd) on a side, centred on the central pixel, and sums to 1.

    Returns ``(array, header keywords)``.
    """
    size_pix = int(size_pix) | 1
    pos = galsim.PositionD(x + 1, y + 1)  # GalSim pixels are 1-based
    local = galsim.GSFitsWCS(header=header1).local(image_pos=pos)
    diffusion = galsim.Sum([galsim.Gaussian(sigma=s, flux=w) for w, s in ROMAN_CHARGE_DIFFUSION])
    with _GALSIM_LOCK:
        optics = galsim.roman.getPSF(int(header0["SCA_NUM"]), band, SCA_pos=pos, wcs=local,
                                     pupil_bin=ROMAN_PSF_PUPIL_BIN, n_waves=ROMAN_PSF_NWAVES)
        star = galsim.Convolve(galsim.DeltaFunction() * ROMAN_PSF_SED, optics,
                               local.profileToWorld(diffusion))  # diffusion is in pixels
        arr = star.drawImage(galsim.roman.getBandpasses()[band], nx=size_pix, ny=size_pix,
                             wcs=local).array
    keys = {"PSFMODEL": ("galsim.roman.getPSF", "PSF model"),
            "SCA": (int(header0["SCA_NUM"]), "Roman detector"),
            "NWAVES": (ROMAN_PSF_NWAVES, "wavelengths sampled over the bandpass"),
            "PUPILBIN": (ROMAN_PSF_PUPIL_BIN, "pupil-plane binning"),
            "CHGDIFF": (True, "roman_imsim charge diffusion included"),
            "PSFSED": ("flat in photons/nm", "source SED (placeholder)")}
    return (arr / arr.sum()).astype(np.float32), keys


def _read_rubin_psf(path):
    """Read the PSF model, pixel origin and flux calibration of a local Rubin calexp.

    Uses the LSST stack. The calibration is the calexp's mean ``PhotoCalib`` factor in nJy
    per count (the calexp's pixel unit); it differs from visit to visit with airmass and
    sky transparency, so fluxes from different visits are only comparable after it.
    """
    from lsst.afw.image import ExposureFitsReader
    reader = ExposureFitsReader(str(path))
    xy0 = reader.readXY0()
    calib = float(reader.readPhotoCalib().getCalibrationMean())
    return reader.readPsf(), (xy0.getX(), xy0.getY()), calib


def _rubin_calib_keys(calib):
    """Primary-header keywords for a Rubin cutout's flux calibration (nJy per count)."""
    return {"BUNIT": ("count", "IMAGE unit; VARIANCE is in count**2"),
            "CALIB": (calib, "nJy per count (calexp PhotoCalib mean)"),
            "MAGZERO": (31.4 - 2.5 * np.log10(calib), "AB mag = MAGZERO - 2.5 log10(count)")}


def rubin_psf(psf_model, x, y, xy0=(0, 0), size_pix=PSF_SIZE_PIX["rubin"]):
    """Rubin PSF stamp at pixel ``(x, y)`` (0-based, as in astropy) of one calexp.

    ``psf_model`` is the image's ``PsfexPsf`` and ``xy0`` its pixel origin (both from
    ``_read_rubin_psf``). The stamp is the PSF's kernel image: on the calexp's pixel grid,
    centred on the central pixel, normalised to sum 1. The model itself covers a fixed
    odd size (41 pixels for these images, saved as ``PSFMODSZ``); the stamp is
    zero-padded to ``size_pix`` (made odd) if that is larger, or cropped if smaller.

    Returns ``(array, header keywords)``.
    """
    import lsst.geom
    arr = psf_model.computeKernelImage(lsst.geom.Point2D(x + xy0[0], y + xy0[1])).array
    model_size = arr.shape[0]
    size_pix = int(size_pix) | 1
    pad = (size_pix - model_size) // 2          # both sizes are odd, so the centre stays the centre
    arr = np.pad(arr, pad) if pad > 0 else arr[-pad:model_size + pad, -pad:model_size + pad]
    keys = {"PSFMODEL": ("PsfexPsf.computeKernelImage", "PSF model (LSST stack)"),
            "PSFMODSZ": (model_size, "pixels covered by the model; zero outside")}
    return (arr / arr.sum()).astype(np.float32), keys


def _psf_hdu(arr, keys, src_xy, wcs):
    """The ``PSF`` extension: the stamp plus where and how it was evaluated."""
    hdr = fits.Header()
    for key, (value, comment) in keys.items():
        hdr[key] = (value, comment)
    hdr["SRCX"], hdr["SRCY"] = (src_xy[0], "PSF position on source image, x (0-based)"), \
                               (src_xy[1], "PSF position on source image, y (0-based)")
    hdr["PSFSCALE"] = (float(np.mean(proj_plane_pixel_scales(wcs))) * 3600, "arcsec per stamp pixel")
    hdr["COMMENT"] = "Native detector pixel grid, centred on the central pixel, sum 1."
    return fits.ImageHDU(arr, header=hdr, name="PSF")


# --- Writing cutouts --------------------------------------------------------

def _write_cutout(arrays, wcs, job, src_uri, src_xy, size_arcsec, north_up, center_on, names,
                  psf_hdu=None, layer_keys=None):
    """Write the cutout layers to one FITS file; returns the NaN fraction of the image.

    The science image is the primary HDU (with the metadata keywords), followed by one
    image extension per remaining layer, all on the same pixel grid and WCS, and then
    the ``PSF`` extension if one is given. ``layer_keys`` maps a layer name to extra
    ``(keywords, comments)`` for its header; for the science image (the primary HDU) they
    are added to the metadata keywords.
    """
    layer_keys = layer_keys or {}
    wcs_hdr = wcs.to_header(relax=True)  # relax keeps the SIP distortion terms
    hdr = wcs_hdr.copy()
    hdr["EXTNAME"] = names[0]
    hdr["TARGETID"] = str(job["target_id"])
    hdr["BAND"] = job["band"]
    hdr["MJD"] = float(job["mjd"])
    hdr["OBSID"] = str(job["obs_id"])
    hdr["SRCIMG"] = src_uri
    hdr["SRCX"], hdr["SRCY"] = src_xy  # cutout centre on the source image (0-based pixels)
    hdr["TRA"], hdr["TDEC"] = float(job["target_ra"]), float(job["target_dec"])  # the transient
    hdr["CENTERON"] = center_on  # what the cutout is centred on: "transient" or "host"
    hdr["SIZEARCS"] = float(size_arcsec)
    hdr["NORTHUP"] = bool(north_up)
    hdr["LAYERS"] = ",".join(names)
    hdr["NANFRAC"] = float(np.isnan(arrays[0]).mean())
    if psf_hdu is not None:
        hdr["PSFSIZE"] = psf_hdu.data.shape[0]  # PSF stamp width in pixels
    keys, comments = layer_keys.get(names[0], ({}, []))
    for key, value in keys.items():
        hdr[key] = value
    for line in comments:
        hdr["COMMENT"] = line
    hdus = [fits.PrimaryHDU(arrays[0], header=hdr)]
    for arr, name in zip(arrays[1:], names[1:]):
        ext_hdr = wcs_hdr.copy()
        keys, comments = layer_keys.get(name, ({}, []))
        for key, value in keys.items():
            ext_hdr[key] = value
        for line in comments:
            ext_hdr["COMMENT"] = line
        hdus.append(fits.ImageHDU(arr, header=ext_hdr, name=name))
    if psf_hdu is not None:
        hdus.append(psf_hdu)
    Path(job["path"]).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(job["path"], overwrite=True)
    return hdr["NANFRAC"]


def _is_current(path, size_arcsec, north_up, center_on, names, psf_size_pix=None, required=()):
    """True if an existing cutout file was made with these settings, layers and PSF size,
    and its primary header has every keyword in ``required``."""
    h = fits.getheader(path)
    return (h.get("SIZEARCS") == size_arcsec and h.get("NORTHUP") == north_up
            and h.get("CENTERON") == center_on and "SRCX" in h
            and h.get("LAYERS") == ",".join(names)
            and ("PSF" not in names or h.get("PSFSIZE") == psf_size_pix)
            and all(key in h for key in required))


def _download(src_uri):
    """Copy an S3 file into ``CACHE_DIR``; returns the local path."""
    path = CACHE_DIR / Path(src_uri).name
    path.parent.mkdir(parents=True, exist_ok=True)
    s3fs.S3FileSystem(anon=True).get(src_uri, str(path))
    return path


def cutouts_from_image(src_uri, jobs, instrument, size_arcsec=10, north_up=False, center_on="transient",
                       psf=False, psf_size_pix=None):
    """Read one image once and write a FITS cutout for each job that lies on it.

    Each cutout holds the layers in ``CUTOUT_ORDER[instrument]``: the science image, its
    variance and quality flags (and for Roman the background-and-noise plane BKGNOISE, from
    which the variance is computed, see `roman_variance`), and with ``psf=True`` a
    ``PSF`` extension evaluated at the cutout centre (see `roman_psf` and `rubin_psf`;
    ``psf_size_pix`` sets the stamp width, by default ``PSF_SIZE_PIX[instrument]``).
    Rubin images are then downloaded to
    ``CACHE_DIR`` first, because the LSST stack reads only local files, and deleted after;
    their flux calibration is saved too (``CALIB``, ``MAGZERO``, see `_rubin_calib_keys`).

    ``jobs`` are dicts with ``target_id, ra, dec, target_ra, target_dec, band, obs_id, mjd,
    path``, where ``ra, dec`` is the cutout centre. Existing files made with the same
    settings are skipped, so interrupted runs can simply be repeated; files made with
    different settings are rewritten.
    Returns one manifest record (dict) per job; errors are recorded, not raised.
    """
    layers = CUTOUT_LAYERS[instrument]
    names = CUTOUT_ORDER[instrument] + (["PSF"] if psf else [])
    psf_size_pix = int(psf_size_pix or PSF_SIZE_PIX[instrument]) | 1
    required = ("CALIB",) if psf and instrument == "rubin" else ()
    records, todo = [], []
    for job in jobs:
        rec = {"target_id": str(job["target_id"]), "band": job["band"], "obs_id": str(job["obs_id"]),
               "mjd": float(job["mjd"]), "path": str(job["path"]), "src_image": src_uri}
        if Path(job["path"]).exists() and _is_current(job["path"], size_arcsec, north_up, center_on, names,
                                                      psf_size_pix, required):
            h = fits.getheader(job["path"])
            rec.update(status="exists", nan_frac=h.get("NANFRAC", np.nan), src_x=h["SRCX"], src_y=h["SRCY"])
            records.append(rec)
        else:
            todo.append((job, rec))
    if not todo:
        return records

    local = None
    try:
        if psf and instrument == "rubin":
            local = _download(src_uri)
        with fits.open(local, memmap=False) if local else \
                fits.open(src_uri, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
            wcs = WCS(hdul[layers[0][1]].header)
            data = [hdul[index].data for _, index, _ in layers]
            if local:
                psf_model, xy0, calib = _read_rubin_psf(local)
            for job, rec in todo:
                try:
                    coord = SkyCoord(job["ra"], job["dec"], unit="deg")
                    src_xy = tuple(float(v) for v in wcs.world_to_pixel(coord))
                    rec.update(src_x=src_xy[0], src_y=src_xy[1])
                    cuts = {name: make_cutout(d, wcs, coord, size_arcsec, north_up, kind)
                            for d, (name, _, kind) in zip(data, layers)}
                    layer_keys = {name: ({}, lines) for name, lines in _LAYER_NOTES.items()}
                    if instrument == "roman":
                        var, keys, comments = roman_variance(cuts["SCI"][0], cuts["BKGNOISE"][0])
                        cuts["VARIANCE"] = (var, cuts["SCI"][1])
                        layer_keys["VARIANCE"] = (keys, comments)
                    if local:
                        layer_keys[CUTOUT_ORDER[instrument][0]] = (_rubin_calib_keys(calib), [])
                    psf_hdu = None
                    if psf and instrument == "roman":
                        psf_hdu = _psf_hdu(*roman_psf(hdul[0].header, hdul[layers[0][1]].header,
                                                      job["band"], *src_xy, psf_size_pix), src_xy, wcs)
                    elif psf:
                        psf_hdu = _psf_hdu(*rubin_psf(psf_model, *src_xy, xy0, psf_size_pix), src_xy, wcs)
                    order = CUTOUT_ORDER[instrument]
                    rec.update(status="ok", nan_frac=_write_cutout(
                        [cuts[name][0] for name in order], cuts[order[0]][1], job, src_uri, src_xy,
                        size_arcsec, north_up, center_on, names, psf_hdu, layer_keys))
                except Exception as err:  # e.g. NoOverlapError: record it and carry on
                    rec.update(status=f"error: {type(err).__name__}", nan_frac=np.nan)
                records.append(rec)
    except Exception as err:  # the image itself could not be read
        for job, rec in todo:
            rec.update(status=f"error: {type(err).__name__}", nan_frac=np.nan)
            records.append(rec)
    finally:
        if local:
            local.unlink(missing_ok=True)
    return records


def _map(fn, items, n_workers, desc):
    """Apply ``fn`` to ``items`` with a thread pool (the work is network-bound)."""
    with ThreadPoolExecutor(n_workers) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))


def run_cutouts(targets, instrument, bands, window_days=30, max_epochs=10, size_arcsec=10,
                north_up=False, center_on="transient", psf=True, psf_size_pix=None, n_workers=4):
    """Find images for each target, then cut every image once for all targets on it.

    For each target and band, images observed within ``peak_mjd +/- window_days`` are
    searched, thinned to ``max_epochs`` evenly spaced epochs, and the cutouts (centred
    on the transient, or on its host galaxy if ``center_on="host"``) are
    written under ``DATA_DIR/cutouts/<instrument>/<band>/<target_id>/``. With ``psf=True``
    each cutout also gets a ``PSF`` extension, evaluated at the cutout centre on the
    native pixel grid, so it needs ``north_up=False``; Rubin PSFs need the LSST stack.

    Returns this run's manifest (one row per cutout, plus one row for each target and
    band that had nothing to cut out). The same rows are merged into the cumulative
    ``DATA_DIR/cutouts/<instrument>/manifest.parquet``.
    """
    if psf and north_up:
        raise ValueError("PSF stamps are on the native pixel grid, so they do not match north-up "
                         "cutouts; use north_up=False or psf=False")
    if psf and instrument == "rubin":
        try:
            import lsst.afw.image  # noqa: F401
        except ImportError as err:
            raise ImportError("Rubin PSFs need the LSST Science Pipelines (lsst.afw); see "
                              "setup_on_mac.md, or use psf=False") from err
    out_root = DATA_DIR / "cutouts" / instrument

    def search(args):
        row, band = args
        ra, dec = (row.host_ra, row.host_dec) if center_on == "host" else (row.ra, row.dec)
        coord = SkyCoord(ra, dec, unit="deg")
        try:
            imgs = find_images(coord, instrument, band, row.peak_mjd - window_days,
                               row.peak_mjd + window_days)
        except Exception as err:
            return [], f"error: {type(err).__name__}"
        if not len(imgs):
            return [], "no images in window"
        imgs = select_epochs(imgs, max_epochs)
        return [{"target_id": row.id, "ra": ra, "dec": dec, "target_ra": row.ra, "target_dec": row.dec,
                 "band": band,
                 "obs_id": r["obs_id"], "mjd": r["t_min"], "src": r["s3_uri"],
                 "path": out_root / band / str(row.id) / f"{r['t_min']:.4f}_{str(r['obs_id']).replace('/', '_')}.fits"}
                for r in imgs], None

    pairs = [(row, band) for row in targets.itertuples() for band in bands]
    results = _map(search, pairs, n_workers, "image search")
    jobs = [j for found, _ in results for j in found]
    # Targets with nothing to cut out are listed in the manifest rather than dropped silently.
    missing = [{"target_id": str(row.id), "band": band, "status": note, "path": ""}
               for (row, band), (_, note) in zip(pairs, results) if note]

    # Invert target -> images into image -> targets, so each large image is read once.
    by_image = {}
    for j in jobs:
        by_image.setdefault(j["src"], []).append(j)
    print(f"{len(jobs)} cutouts requested from {len(by_image)} images")

    records = [r for recs in _map(lambda kv: cutouts_from_image(kv[0], kv[1], instrument, size_arcsec, north_up,
                                                                center_on, psf, psf_size_pix),
                                  list(by_image.items()), n_workers, "cutouts")
               for r in recs]
    this_run = pd.DataFrame(records + missing)
    this_run.insert(1, "instrument", instrument)
    manifest = this_run

    manifest_path = out_root / "manifest.parquet"
    if manifest_path.exists():
        old = pd.read_parquet(manifest_path)
        done = set(zip(*manifest.loc[manifest["path"] != "", ["target_id", "band"]].T.values))
        stale = (old["path"] == "") & np.array([(t, b) in done for t, b in zip(old["target_id"], old["band"])])
        manifest = pd.concat([old[~stale], manifest]).drop_duplicates(["target_id", "band", "path"], keep="last")
    out_root.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)
    return this_run


def show_cutouts(paths, ncols=4):
    """Plot saved FITS cutouts exactly as stored (north is up only if saved north-up).

    A cyan ring marks the transient's position, which is off-centre for host-centred cutouts.
    """
    paths = list(paths)
    nrows = -(-len(paths) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, path in zip(axes.ravel(), paths):
        data, hdr = fits.getdata(path, header=True)
        lo, hi = np.nanpercentile(data, [5, 99.5])
        ax.imshow(data, origin="lower", cmap="gray", vmin=lo, vmax=hi)
        x, y = WCS(hdr).world_to_pixel(SkyCoord(hdr["TRA"], hdr["TDEC"], unit="deg"))
        ax.add_patch(plt.Circle((x, y), 0.06 * data.shape[0], fill=False, ec="#00bcd4", lw=1))
        ax.set_title(f"{hdr['TARGETID']} {hdr['BAND']}\nMJD {hdr['MJD']:.1f}", fontsize=8)
    fig.tight_layout()
    return fig


def show_psfs(paths, ncols=4):
    """Plot the ``PSF`` extension of saved cutouts on a log scale, with its FWHM.

    The FWHM is estimated from the area above half the peak, so it is coarse for
    PSFs only a few pixels wide.
    """
    paths = list(paths)
    nrows = -(-len(paths) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, path in zip(axes.ravel(), paths):
        with fits.open(path) as hdul:
            psf, hdr, band = hdul["PSF"].data, hdul["PSF"].header, hdul[0].header["BAND"]
        fwhm = 2 * np.sqrt((psf > psf.max() / 2).sum() / np.pi) * hdr["PSFSCALE"]
        ax.imshow(np.log10(np.clip(psf, psf.max() * 1e-4, None)), origin="lower", cmap="magma")
        ax.set_title(f"{band} PSF {psf.shape[0]}x{psf.shape[1]}\nFWHM ~{fwhm:.2f}\"", fontsize=8)
    fig.tight_layout()
    return fig


# --- scarlet2 models ----------------------------------------------------------

def save_scene(scene, path, id=0):
    """Save a scarlet2 scene (or any scarlet2 module) to the HDF5 file `path`, under group `id`.

    Like ``scarlet2.io.model_to_h5``, the model is pickled, but stored as a uint8 dataset
    ``<id>/model`` instead of an HDF5 attribute: attributes are limited to 64 kB, which a
    scene with many large sources exceeds. Overwrites an existing group `id` and keeps the
    rest of the file. Load it with `load_scene` (``scarlet2.io.model_from_h5`` cannot read it).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if str(id) in f:
            del f[str(id)]
        f.create_group(str(id)).create_dataset("model", data=np.frombuffer(pickle.dumps(scene), dtype=np.uint8))


def load_scene(path, id=0):
    """Load a scarlet2 model saved by `save_scene` from group `id` of the HDF5 file `path`."""
    with h5py.File(path, "r") as f:
        return pickle.loads(f[str(id)]["model"][()].tobytes())
