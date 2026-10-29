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

# Roman cutouts for the selected targets

Read the targets saved by `target_selection.md`, find the simulated Roman Time Domain Survey images that cover each one around its peak, and save a FITS cutout of every image to disk.

Roman images are stored per exposure and detector, not per sky position, so the image search (IRSA's Simple Image Access service) is run once per target.
Each large image is then opened only once and every selected target on it is cut out.
Each cutout is a FITS file whose header holds the WCS, the target ID, the band, the observation MJD, and the source image path.
The primary HDU is the science image (`SCI`); the extensions `ERR` (per-pixel uncertainty) and `DQ` (quality flags) cover exactly the same pixels, so they can be used for photometry.
Flags are never interpolated, even with `NORTH_UP = True`, and are 0 where the cutout falls off the detector (where `SCI` and `ERR` are NaN).

```{code-cell} ipython3
import pandas as pd

import ou_utils as ou
```

## Parameters

```{code-cell} ipython3
BANDS = ["J129"]      # Roman filters to cut out, e.g. ["J129", "H158"]
WINDOW_DAYS = 30      # keep images observed within peak_mjd +/- this many days
MAX_EPOCHS = 5        # per target and band, evenly spaced in time
CUTOUT_ARCSEC = 20    # cutout width on the sky
CENTER_ON = "host"    # "host" or "transient"; with "host" a ring marks the transient
NORTH_UP = False      # False keeps the native pixels; True resamples onto a north-up grid
MAX_TARGETS = 5       # None = all targets. Start small: each Roman image is large
```

Large hosts can be several arcseconds across, and a transient often sits well off its host's centre.
With `CENTER_ON = "host"` the cutout is centred on the host galaxy and is wide enough to include the transient; with `"transient"` it is centred on the transient.
The transient's position is always saved in the header (`TRA`, `TDEC`).

By default the cutouts are not rotated: the pixel grid is the detector's own and the saved WCS describes it exactly.
With `NORTH_UP = True` the cutout is resampled onto a north-up, east-left grid at the native pixel scale, and the saved WCS describes the new grid.

## 1. Load the targets

Only transients that peak inside the Roman survey footprint and dates can have Roman images.

```{code-cell} ipython3
targets = pd.read_parquet(ou.TARGETS_PATH)
targets = targets[targets["in_tds"]].head(MAX_TARGETS)
print(f"{len(targets)} targets")
```

## 2. Find the images and write the cutouts

Files that already exist with the same settings are skipped, so an interrupted run can simply be repeated.

```{code-cell} ipython3
manifest = ou.run_cutouts(targets, "roman", BANDS, window_days=WINDOW_DAYS, max_epochs=MAX_EPOCHS,
                          size_arcsec=CUTOUT_ARCSEC, north_up=NORTH_UP, center_on=CENTER_ON)
manifest["status"].value_counts()
```

Targets with no Roman image of that band in the time window are listed in the manifest, not dropped silently.

```{code-cell} ipython3
manifest[manifest["path"] == ""][["target_id", "band", "status"]]
```

## 3. Look at the saved cutouts

These are read back from disk, so this also confirms the files are good.
The cumulative manifest of every Roman run is `data/cutouts/roman/manifest.parquet`.

```{code-cell} ipython3
saved = manifest[manifest["path"] != ""]
ou.show_cutouts(saved["path"].head(8));
```
