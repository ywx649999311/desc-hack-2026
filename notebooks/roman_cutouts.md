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
The HDUs, all covering exactly the same pixels, are in the same order as in the Rubin cutouts (image, variance, flags):

| HDU | Name | What it is |
|---|---|---|
| 0 | `SCI` | science image, in counts (gain 1, so counts = electrons); the sky is not subtracted |
| 1 | `VARIANCE` | per-pixel variance of `SCI`, in counts squared, computed here (see below) |
| 2 | `DQ` | quality flags |
| 3 | `BKGNOISE` | the source image's `ERR` plane, renamed: the background and noise of `SCI` without the sources, not an uncertainty (see `explore_cutouts.md`) |
| 4 | `PSF` | PSF stamp at the cutout centre (with `PSF = True`) |

`VARIANCE = (max(SCI - BKGNOISE, 0) + mean(BKGNOISE)) / gain + (read_noise / gain)**2`, with the simulation's gain (1) and read noise (8.5 electrons).
`SCI - BKGNOISE` is the source counts and the mean of `BKGNOISE` the expected sky, so this is the Poisson variance of sources plus sky, plus the read noise, without following the noise in `SCI` itself.
On blank sky it predicts the actual scatter of `SCI` to within 1%.
Saturated pixels (`SCI` within 5 read-noise sigma of 100,000) get an infinite variance, i.e. zero weight; `DQ` does not flag them in these simulations.
The `VARIANCE` header records the values used (`GAIN`, `RDNOISE`, `SKYEXP`, `SATLEVEL`, `NSAT`).

Flags are never interpolated, even with `NORTH_UP = True`, and are 0 where the cutout falls off the detector (where `SCI`, `VARIANCE` and `BKGNOISE` are NaN).

With `PSF = True` each cutout also gets a `PSF` extension: the PSF at the cutout centre, drawn with `galsim.roman.getPSF` the way the simulation drew its objects (the image's detector, band and WCS, 5 wavelengths across the band, the simulation's charge diffusion and the pixel response).
The stamp is on the cutout's own pixel grid, centred on its central pixel, and sums to 1.
Its header records where it was evaluated on the source image (`SRCX`, `SRCY`, 0-based pixels), the pixel scale and the source SED.
The SED is a placeholder, flat in photons; a redder source has a slightly broader PSF.
The stamp is not rotated, so `PSF = True` needs `NORTH_UP = False`.

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
PSF = True            # add a PSF extension at the cutout centre (needs NORTH_UP = False)
PSF_SIZE_PIX = 51     # PSF stamp width in pixels (odd), about 5.5 arcsec
MAX_TARGETS = 1       # None = all targets. Start small: each Roman image is large
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
                          size_arcsec=CUTOUT_ARCSEC, north_up=NORTH_UP, center_on=CENTER_ON,
                          psf=PSF, psf_size_pix=PSF_SIZE_PIX)
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

The PSF of each of the same cutouts, on a log scale.
The core is only about one pixel wide (J129 FWHM is about 0.13 arcsec), so the FWHM estimate in the titles is coarse; the diffraction pattern of the struts shows in the wings.

```{code-cell} ipython3
if PSF:
    ou.show_psfs(saved["path"].head(8));
```

```{code-cell} ipython3

```
