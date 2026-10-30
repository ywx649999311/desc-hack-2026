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

# Rubin cutouts for the selected targets

Read the targets saved by `target_selection.md`, find the simulated Rubin `calexp` images that cover each one around its peak, and save a FITS cutout of every image to disk.

Rubin images are stored per visit and detector, not per sky position, so the image search (IRSA's Simple Image Access service) is run once per target.
Each large image is then opened only once and every selected target on it is cut out.
Each cutout is a FITS file whose header holds the WCS, the target ID, the band, the observation MJD, and the source image path.
The primary HDU is the science image (`IMAGE`); the extensions `VARIANCE` (per-pixel variance) and `MASK` (quality flags) cover exactly the same pixels, so they can be used for photometry.
Flags are never interpolated, even with `NORTH_UP = True`, and are 0 where the cutout falls off the detector (where `IMAGE` and `VARIANCE` are NaN).

With `PSF = True` each cutout also gets a `PSF` extension: the PSF model stored in the `calexp` (a PSFEx model), evaluated at the cutout centre with the LSST Science Pipelines.
The stack reads only local files, so each `calexp` (about 110 MB) is downloaded to `data/cache/`, cut out, and deleted again.
The stamp is the model's kernel image: on the `calexp` pixel grid, centred on its central pixel, and normalised to sum 1.
The model itself covers 41 x 41 pixels (8.2 arcsec, saved as `PSFMODSZ`), which is the default stamp size; a wider `PSF_SIZE_PIX` would pad it with zeros, since the model is not defined further out.
Stars in these images have essentially no light beyond 4 arcsec, so nothing is lost at this size.
Its header records where it was evaluated on the source image (`SRCX`, `SRCY`, 0-based pixels) and the pixel scale.
The stamp is not rotated, so `PSF = True` needs `NORTH_UP = False`.

```{code-cell} ipython3
import pandas as pd

import ou_utils as ou
```

## Parameters

```{code-cell} ipython3
BANDS = ["r"]         # Rubin filters to cut out: any of u, g, r, i, z, y
WINDOW_DAYS = 100      # keep images observed within peak_mjd +/- this many days
MAX_EPOCHS = 20        # per target and band, evenly spaced in time
CUTOUT_ARCSEC = 20    # cutout width on the sky
CENTER_ON = "host"    # "host" or "transient"; with "host" a ring marks the transient
NORTH_UP = False      # False keeps the native pixels; True resamples onto a north-up grid
PSF = True            # add a PSF extension at the cutout centre (needs NORTH_UP = False and the LSST stack)
PSF_SIZE_PIX = 41     # PSF stamp width in pixels (odd), 8.2 arcsec: the PSFEx model's own size
MAX_TARGETS = 10       # None = all targets. Start small: each Rubin image is large
```

Large hosts can be several arcseconds across, and a transient often sits well off its host's centre.
With `CENTER_ON = "host"` the cutout is centred on the host galaxy and is wide enough to include the transient; with `"transient"` it is centred on the transient.
The transient's position is always saved in the header (`TRA`, `TDEC`).

By default the cutouts are not rotated: the pixel grid is the detector's own and the saved WCS describes it exactly.
With `NORTH_UP = True` the cutout is resampled onto a north-up, east-left grid at the native pixel scale, and the saved WCS describes the new grid.

## 1. Load the targets

Rubin covers more sky than the Roman survey, so no footprint cut is needed.
Rubin visits are sparse in time, though, so some targets will have no image inside the window around their peak; those are reported below.

```{code-cell} ipython3
targets = pd.read_parquet(ou.TARGETS_PATH).head(MAX_TARGETS)
print(f"{len(targets)} targets")
```

## 2. Find the images and write the cutouts

Files that already exist with the same settings are skipped, so an interrupted run can simply be repeated.

```{code-cell} ipython3
manifest = ou.run_cutouts(targets, "rubin", BANDS, window_days=WINDOW_DAYS, max_epochs=MAX_EPOCHS,
                          size_arcsec=CUTOUT_ARCSEC, north_up=NORTH_UP, center_on=CENTER_ON,
                          psf=PSF, psf_size_pix=PSF_SIZE_PIX)
manifest["status"].value_counts()
```

Targets with no Rubin image of that band in the time window are listed in the manifest, not dropped silently.

```{code-cell} ipython3
manifest[manifest["path"] == ""][["target_id", "band", "status"]]
```

## 3. Look at the saved cutouts

These are read back from disk, so this also confirms the files are good.
The cumulative manifest of every Rubin run is `data/cutouts/rubin/manifest.parquet`.

```{code-cell} ipython3
saved = manifest[manifest["path"] != ""]
saved_unique = saved.drop_duplicates('target_id')
ou.show_cutouts(saved_unique["path"].head(len(saved_unique)));
```

The PSF of each of the same cutouts, on a log scale; the seeing changes from visit to visit.

```{code-cell} ipython3
if PSF:
    ou.show_psfs(saved["path"].head(8));
```

```{code-cell} ipython3

```
