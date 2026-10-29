"""Shared helpers for the target-selection and cutout notebooks.

Catalog access, image search, and FITS cutouts for the OpenUniverse2024 Roman and
Rubin simulations hosted by IRSA (public S3 bucket, no credentials needed).
"""
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
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

# Arrays saved in each cutout: (name, HDU index in the source image, kind). The first is
# the science image; "error" is a per-pixel uncertainty or variance; "flags" is a
# quality bit mask, which must never be interpolated.
CUTOUT_LAYERS = {
    "roman": [("SCI", 1, "image"), ("ERR", 2, "error"), ("DQ", 3, "flags")],
    "rubin": [("IMAGE", 1, "image"), ("VARIANCE", 3, "error"), ("MASK", 2, "flags")],
}


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


def _write_cutout(arrays, wcs, job, src_uri, size_arcsec, north_up, center_on, layers):
    """Write the cutout layers to one FITS file; returns the NaN fraction of the image.

    The science image is the primary HDU (with the metadata keywords), followed by one
    image extension per remaining layer, all on the same pixel grid and WCS.
    """
    names = [name for name, _, _ in layers]
    wcs_hdr = wcs.to_header(relax=True)  # relax keeps the SIP distortion terms
    hdr = wcs_hdr.copy()
    hdr["EXTNAME"] = names[0]
    hdr["TARGETID"] = str(job["target_id"])
    hdr["BAND"] = job["band"]
    hdr["MJD"] = float(job["mjd"])
    hdr["OBSID"] = str(job["obs_id"])
    hdr["SRCIMG"] = src_uri
    hdr["TRA"], hdr["TDEC"] = float(job["target_ra"]), float(job["target_dec"])  # the transient
    hdr["CENTERON"] = center_on  # what the cutout is centred on: "transient" or "host"
    hdr["SIZEARCS"] = float(size_arcsec)
    hdr["NORTHUP"] = bool(north_up)
    hdr["LAYERS"] = ",".join(names)
    hdr["NANFRAC"] = float(np.isnan(arrays[0]).mean())
    hdus = [fits.PrimaryHDU(arrays[0], header=hdr)]
    hdus += [fits.ImageHDU(arr, header=wcs_hdr, name=name) for arr, name in zip(arrays[1:], names[1:])]
    Path(job["path"]).parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(job["path"], overwrite=True)
    return hdr["NANFRAC"]


def _is_current(path, size_arcsec, north_up, center_on, layers):
    """True if an existing cutout file was made with these settings and layers."""
    h = fits.getheader(path)
    return (h.get("SIZEARCS") == size_arcsec and h.get("NORTHUP") == north_up
            and h.get("CENTERON") == center_on
            and h.get("LAYERS") == ",".join(name for name, _, _ in layers))


def cutouts_from_image(src_uri, jobs, instrument, size_arcsec=10, north_up=False, center_on="transient"):
    """Read one image once and write a FITS cutout for each job that lies on it.

    Each cutout holds the science image plus the layers in ``CUTOUT_LAYERS[instrument]``
    (error/variance and quality flags for Roman and Rubin).

    ``jobs`` are dicts with ``target_id, ra, dec, target_ra, target_dec, band, obs_id, mjd,
    path``, where ``ra, dec`` is the cutout centre. Existing files made with the same
    settings are skipped, so interrupted runs can simply be repeated; files made with
    different settings are rewritten.
    Returns one manifest record (dict) per job; errors are recorded, not raised.
    """
    layers = CUTOUT_LAYERS[instrument]
    records, todo = [], []
    for job in jobs:
        rec = {"target_id": str(job["target_id"]), "band": job["band"], "obs_id": str(job["obs_id"]),
               "mjd": float(job["mjd"]), "path": str(job["path"]), "src_image": src_uri}
        if Path(job["path"]).exists() and _is_current(job["path"], size_arcsec, north_up, center_on, layers):
            rec.update(status="exists", nan_frac=fits.getheader(job["path"]).get("NANFRAC", np.nan))
            records.append(rec)
        else:
            todo.append((job, rec))
    if not todo:
        return records

    try:
        with fits.open(src_uri, fsspec_kwargs={"anon": True}, memmap=False) as hdul:
            wcs = WCS(hdul[layers[0][1]].header)
            data = [hdul[index].data for _, index, _ in layers]
            for job, rec in todo:
                try:
                    coord = SkyCoord(job["ra"], job["dec"], unit="deg")
                    cuts = [make_cutout(d, wcs, coord, size_arcsec, north_up, kind)
                            for d, (_, _, kind) in zip(data, layers)]
                    rec.update(status="ok", nan_frac=_write_cutout(
                        [arr for arr, _ in cuts], cuts[0][1], job, src_uri, size_arcsec, north_up,
                        center_on, layers))
                except Exception as err:  # e.g. NoOverlapError: record it and carry on
                    rec.update(status=f"error: {type(err).__name__}", nan_frac=np.nan)
                records.append(rec)
    except Exception as err:  # the image itself could not be read
        for job, rec in todo:
            rec.update(status=f"error: {type(err).__name__}", nan_frac=np.nan)
            records.append(rec)
    return records


def _map(fn, items, n_workers, desc):
    """Apply ``fn`` to ``items`` with a thread pool (the work is network-bound)."""
    with ThreadPoolExecutor(n_workers) as pool:
        return list(tqdm(pool.map(fn, items), total=len(items), desc=desc))


def run_cutouts(targets, instrument, bands, window_days=30, max_epochs=10, size_arcsec=10,
                north_up=False, center_on="transient", n_workers=4):
    """Find images for each target, then cut every image once for all targets on it.

    For each target and band, images observed within ``peak_mjd +/- window_days`` are
    searched, thinned to ``max_epochs`` evenly spaced epochs, and the cutouts (centred
    on the transient, or on its host galaxy if ``center_on="host"``) are
    written under ``DATA_DIR/cutouts/<instrument>/<band>/<target_id>/``.

    Returns this run's manifest (one row per cutout, plus one row for each target and
    band that had nothing to cut out). The same rows are merged into the cumulative
    ``DATA_DIR/cutouts/<instrument>/manifest.parquet``.
    """
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

    records = [r for recs in _map(lambda kv: cutouts_from_image(kv[0], kv[1], instrument, size_arcsec, north_up, center_on),
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
