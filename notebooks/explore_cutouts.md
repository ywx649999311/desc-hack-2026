---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python (desc-hack-2026)
  language: python
  name: desc-hack-2026
---

# Explore the saved cutouts

Open one Roman and one Rubin cutout written by `roman_cutouts.md` and `rubin_cutouts.md`, list what is inside each FITS file, and plot its science image, variance and PSF.

Each cutout is one FITS file with several HDUs (header-data units), all on the same pixel grid except the PSF stamp:

| HDU | Roman | Rubin |
|---|---|---|
| 0 (primary) | `SCI`: science image | `IMAGE`: science image |
| 1 | `VARIANCE`: per-pixel variance, computed by `roman_cutouts.md` | `VARIANCE`: per-pixel variance (sigma squared) |
| 2 | `DQ`: quality flags | `MASK`: quality flags |
| 3 | `BKGNOISE`: the background and noise in `SCI`, without the sources (see below) | `PSF`: PSF stamp at the cutout centre |
| 4 | `PSF`: PSF stamp at the cutout centre | |

`BKGNOISE` is the Roman source images' `ERR` plane, renamed because it is not an error map.
The section "What the Roman `BKGNOISE` plane is" shows what it is, and how `VARIANCE` is made from it.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

import ou_utils as ou    # this repo's helpers; ou.DATA_DIR is <repo>/data
```

## Parameters

```{code-cell} ipython3
TARGET_ID = None   # a target ID from the manifests, e.g. "30431441"; None = the first target with cutouts
EPOCH = 0          # which of that target's cutouts to open, in time order (0 = earliest)
```

The manifest of each instrument lists every cutout written so far, one row per file.
This helper picks one file from it.

```{code-cell} ipython3
def pick_cutout(instrument, target_id=TARGET_ID, epoch=EPOCH):
    """Path of one saved cutout of `instrument`, from its manifest."""
    manifest = pd.read_parquet(ou.DATA_DIR / "cutouts" / instrument / "manifest.parquet")
    saved = manifest[manifest["path"] != ""]          # rows with an empty path had no image to cut
    if target_id is None:
        target_id = saved["target_id"].iloc[0]
    rows = saved[saved["target_id"] == str(target_id)].sort_values("mjd")
    if rows.empty:
        raise ValueError(f"no {instrument} cutouts for target {target_id}")
    print(f"{instrument}: target {target_id}, {len(rows)} cutouts; opening number {epoch}")
    return rows["path"].iloc[epoch]
```

The plotting is the same for both instruments, so it is one function.
The image and the variance use a percentile stretch, so a few bright pixels do not wash out the rest; the PSF is shown on a log scale, which brings out its faint wings.
A cyan ring marks the transient, which is off-centre because the cutouts are centred on the host galaxy.

```{code-cell} ipython3
def show_planes(image, variance, psf, header, title, variance_label="Variance"):
    """Plot the science image, the variance and the PSF stamp side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Science image, with the transient's position (TRA, TDEC in the header) marked.
    lo, hi = np.nanpercentile(image, [5, 99.5])
    im = axes[0].imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    x, y = WCS(header).world_to_pixel(SkyCoord(header["TRA"], header["TDEC"], unit="deg"))
    axes[0].add_patch(plt.Circle((x, y), 0.06 * image.shape[0], fill=False, ec="#00bcd4", lw=1))
    axes[0].set_title("Science image")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # Variance: a variance that includes the source's own photon noise is higher on bright objects.
    lo, hi = np.nanpercentile(variance, [1, 99.5])
    im = axes[1].imshow(variance, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
    axes[1].set_title(variance_label)
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # PSF on a log scale; values below 1e-4 of the peak (and any small negatives) are clipped.
    im = axes[2].imshow(np.log10(np.clip(psf, psf.max() * 1e-4, None)), origin="lower", cmap="magma")
    axes[2].set_title(f"PSF, log10 ({psf.shape[0]}x{psf.shape[1]} pixels)")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x (pixels)")
    axes[0].set_ylabel("y (pixels)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
```

## Roman

### What is in the file

`hdul.info()` lists the HDUs: their names, types, dimensions and data types.

```{code-cell} ipython3
roman_path = pick_cutout("roman")
with fits.open(roman_path) as hdul:
    hdul.info()
```

A few of the header keywords the cutout notebook wrote, and those of the PSF extension:

```{code-cell} ipython3
with fits.open(roman_path) as hdul:
    hdr, psf_hdr = hdul[0].header, hdul["PSF"].header
    for key in ["TARGETID", "BAND", "MJD", "SRCIMG", "SRCX", "SRCY", "TRA", "TDEC", "CENTERON", "SIZEARCS", "LAYERS"]:
        print(f"{key:9s}= {hdr[key]}")
    print()
    for key in ["PSFMODEL", "SCA", "NWAVES", "PUPILBIN", "CHGDIFF", "PSFSED", "PSFSCALE"]:
        print(f"{key:9s}= {psf_hdr[key]}")
```

### Image, variance and PSF

```{code-cell} ipython3
with fits.open(roman_path) as hdul:
    image = hdul["SCI"].data
    variance = hdul["VARIANCE"].data   # computed by roman_cutouts.md; see the section below
    psf = hdul["PSF"].data
    header = hdul[0].header

show_planes(image, variance, psf, header,
            f"Roman {header['BAND']}, target {header['TARGETID']}, MJD {header['MJD']:.2f}")
print(f"PSF sum = {psf.sum():.4f}, peak at pixel {np.unravel_index(psf.argmax(), psf.shape)} "
      f"(the centre is {psf.shape[0] // 2})")
```

### What the Roman `BKGNOISE` plane is

In the source images this plane is called `ERR`, but its header does not say what it holds: no units or description, only a copy of the image header, which carries `SKY_MEAN` (the mean sky level).
The code that made these files does: roman_imsim v2.0 (`detector_physics.py`, `write_fits` and `finalize_sky_im`) writes into `ERR` the image's own background (sky plus thermal emission, with the same Poisson noise realization), plus the same dark-current and read-noise realizations that were added to `SCI`, converted to counts and rounded.
It is the noise and background of `SCI` without the astrophysical sources, not an uncertainty, so the cutouts save it as `BKGNOISE`.

If that is right, `SCI - BKGNOISE` is exactly 0 wherever there is no source, and is the sources alone elsewhere.

```{code-cell} ipython3
import galsim.roman   # only for the detector constants the simulation used

with fits.open(roman_path) as hdul:
    sci, bkg = hdul["SCI"].data.astype(float), hdul["BKGNOISE"].data.astype(float)

sources = sci - bkg
mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))   # robust standard deviation
print(f"median SCI {np.median(sci):.1f}, median BKGNOISE {np.median(bkg):.1f}  (the sky level)")
print(f"pixel-to-pixel scatter: SCI {mad(sci):.2f}, BKGNOISE {mad(bkg):.2f}, SCI - BKGNOISE {mad(sources):.2f}")
print(f"pixels where SCI - BKGNOISE is exactly 0: {(sources == 0).mean():.0%}")
print(f"expected sky noise, sqrt(sky + read_noise**2) = {np.sqrt(np.median(bkg) + galsim.roman.read_noise ** 2):.1f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, arr, title in zip(axes, [sci, bkg, sources],
                          ["SCI", "BKGNOISE (background + noise)", "SCI - BKGNOISE (sources only)"]):
    lo, hi = np.nanpercentile(arr, [5, 99.5])
    im = ax.imshow(arr, origin="lower", cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
plt.show()
```

`BKGNOISE` scatters by about the expected sky noise, and `SCI - BKGNOISE` shows the sources with no sky noise around them.
In a cutout centred on a large host, faint galaxy light covers most pixels, so few of them are exactly 0 and `SCI` scatters more than `BKGNOISE`.
On a whole Roman image (checked once for this notebook, Roman_TDS_simple_model_J129_29478_5), `SCI - BKGNOISE` is exactly 0 on 77% of the pixels, and on blank sky `SCI` and `BKGNOISE` have the same scatter, as predicted from the sky level and read noise.
So:

- `BKGNOISE` is not an uncertainty; do not use it as weights.
- `SCI - BKGNOISE` is the source counts and the mean of `BKGNOISE` the expected sky, which is how `VARIANCE` is made: `(max(SCI - BKGNOISE, 0) + mean(BKGNOISE)) / gain + (read_noise / gain)**2`. Neither part follows the noise in `SCI`, so the weights do not bias fits.
- `SCI - BKGNOISE` itself is a background-subtracted image with no sky noise at all. That is useful for tests, but it is not what a real observation looks like.

### Check: does `VARIANCE` match the noise on blank sky?

On pixels with no source, `(SCI - sky) / sqrt(VARIANCE)` should scatter by 1.

```{code-cell} ipython3
with fits.open(roman_path) as hdul:
    var_hdr = hdul["VARIANCE"].header
    variance = hdul["VARIANCE"].data.astype(float)

blank = sources == 0                                    # pixels with no source at all
pull = (sci - var_hdr["SKYEXP"]) / np.sqrt(variance)
print(f"blank-sky pixels: {blank.sum()}, scatter of the pull: {pull[blank].std():.3f}  (1 = variance correct)")
print(f"values used: GAIN {var_hdr['GAIN']}, RDNOISE {var_hdr['RDNOISE']}, SKYEXP {var_hdr['SKYEXP']:.1f}, "
      f"saturated pixels {var_hdr['NSAT']}")
```

## Rubin

### What is in the file

```{code-cell} ipython3
rubin_path = pick_cutout("rubin")
with fits.open(rubin_path) as hdul:
    hdul.info()
```

```{code-cell} ipython3
with fits.open(rubin_path) as hdul:
    hdr, psf_hdr = hdul[0].header, hdul["PSF"].header
    for key in ["TARGETID", "BAND", "MJD", "SRCIMG", "SRCX", "SRCY", "TRA", "TDEC", "CENTERON", "SIZEARCS", "LAYERS"]:
        print(f"{key:9s}= {hdr[key]}")
    print()
    for key in ["PSFMODEL", "PSFSCALE"]:
        print(f"{key:9s}= {psf_hdr[key]}")
```

### Image, variance and PSF

Rubin pixels are about twice the size of Roman's (0.2 against 0.11 arcsec) and the PSF is set by the atmosphere, so the Rubin PSF is much broader and has no diffraction spikes.
The Rubin variance is higher on the galaxy, because it includes the photon noise of the source as well as the sky.

```{code-cell} ipython3
with fits.open(rubin_path) as hdul:
    image = hdul["IMAGE"].data
    variance = hdul["VARIANCE"].data   # Rubin stores the variance directly
    psf = hdul["PSF"].data
    header = hdul[0].header

show_planes(image, variance, psf, header,
            f"Rubin {header['BAND']}, target {header['TARGETID']}, MJD {header['MJD']:.2f}")
print(f"PSF sum = {psf.sum():.4f}, peak at pixel {np.unravel_index(psf.argmax(), psf.shape)} "
      f"(the centre is {psf.shape[0] // 2})")
```

```{code-cell} ipython3

```
