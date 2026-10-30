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

# Fit and Save a Roman and Rubin Scene

This notebook fits a transient and its host galaxy jointly in one Roman image and all Rubin images of one target, with [_scarlet2_](https://github.com/pmelchior/scarlet2), including the astrometric offsets between the images, and saves the best-fit model to an HDF5 file.
The companion notebook `scarlet2_sample_roman_rubin.md` loads that model and samples from the posterior, to get uncertainties on the transient's fluxes.

The fit is the one of `scarlet2_roman_rubin_joint_fit.md`, which combines two _scarlet2_ how-to guides: the [transient tutorial](https://scarlet2.readthedocs.io/en/latest/howto/timedomain.html), which models a static host and a variable point source across epochs, and the [stacking tutorial](https://scarlet2.readthedocs.io/en/latest/howto/stacking.html), which aligns observations from different instruments into a single observation, like HST and HSC there.
Here the astrometric offsets are always fitted, and saving the model follows the [real-world example](https://scarlet2.readthedocs.io/en/latest/0-example.html).

The inputs are the cutouts written by `roman_cutouts.md` and `rubin_cutouts.md`, read from the local files in `data/cutouts/`.
The Rubin cutouts must carry their flux calibration (`CALIB` in the header), which `rubin_cutouts.md` writes; rerun it if your Rubin cutouts predate that.

```{code-cell} ipython3
# Import Packages and setup
import json
from functools import partial



import astropy.io.fits as fits
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from numpyro.distributions import constraints

import ou_utils as ou
import scarlet2 as sc2
import scarlet2.io

sc2.set_validation(False)
```

```{code-cell} ipython3
def zoom(fig, center, half_width, label_size=14):
    """Show only the window of +-half_width model pixels around center (y, x) in every panel."""
    y, x = np.asarray(center)
    for ax in fig.axes:
        ax.set_xlim(x - half_width, x + half_width)
        ax.set_ylim(y - half_width, y + half_width)
        ax.tick_params(axis='both', labelsize=label_size)
    return fig
```

## Load Data

We pick a target from the list written by `target_selection.md` and find its cutouts in the manifests of the two cutout notebooks.
The best-fit model is saved under `data/scarlet2/`.

```{code-cell} ipython3
#TARGET_ID = 30403008  # a target ID from data/targets.parquet; None = the first target
TARGET_ID = 110001775
ROMAN_BAND = "J129"   # Roman band of the cutouts
ROMAN_EPOCH = 0       # which Roman cutout to use, in time order (0 = earliest)
RUBIN_BAND = "r"      # Rubin band of the cutouts

targets = pd.read_parquet(ou.TARGETS_PATH)
target = targets.iloc[0] if TARGET_ID is None else targets[targets["id"] == TARGET_ID].iloc[0]

MODEL_DIR = ou.DATA_DIR / "scarlet2"
MODEL_FILE = f"{target['id']}_roman_rubin.h5"


def cutout_rows(instrument, band):
    """Rows of the manifest of `instrument` for this target and band, in time order."""
    manifest = pd.read_parquet(ou.DATA_DIR / "cutouts" / instrument / "manifest.parquet")
    return manifest[(manifest["target_id"] == str(target["id"])) & (manifest["band"] == band)
                    & (manifest["path"] != "")].sort_values("mjd")


roman_path = cutout_rows("roman", ROMAN_BAND)["path"].iloc[ROMAN_EPOCH]
rubin_paths = cutout_rows("rubin", RUBIN_BAND)["path"].tolist()

print(f"target {target['id']} ({target['model_name']}), active from MJD {target['start_mjd']:.1f}, "
      f"peak at MJD {target['peak_mjd']:.1f}")
print(f"1 Roman {ROMAN_BAND} image and {len(rubin_paths)} Rubin {RUBIN_BAND} images")
```

All images together form one sequence of epochs in time order, and as in the transient tutorial, the channel of every image is a combined (band, epoch) identifier.
We first collect the times, so that every image gets its epoch number:

```{code-cell} ipython3
images = [("roman", ROMAN_BAND, roman_path)] + [("rubin", RUBIN_BAND, path) for path in rubin_paths]
images = sorted(images, key=lambda image: fits.getheader(image[2])["MJD"])
mjds = [fits.getheader(path)["MJD"] for _, _, path in images]

# the transient is active from start_mjd in the transient catalog
active_epochs = [epoch for epoch, mjd in enumerate(mjds) if mjd >= target["start_mjd"]]
for epoch, ((instrument, band, _), mjd) in enumerate(zip(images, mjds)):
    print(f"epoch {epoch}: {instrument:5s} {band:4s} MJD {mjd:.2f}"
          f"{'  (transient active)' if epoch in active_epochs else ''}")
```

Each cutout holds the science image, its variance, quality flags, and the PSF at the cutout centre on the cutout's own pixel grid.
The two instruments need slightly different preparation:

* Roman (`SCI`, `VARIANCE`, `DQ`): the science image still contains the sky, which _scarlet2_ does not model, so we subtract the median of `BKGNOISE`, the background of that exposure. Fluxes stay in counts.
* Rubin (`IMAGE`, `VARIANCE`, `MASK`): the `calexp` images are already sky-subtracted, but their counts are on a different scale in every visit (airmass, sky transparency). We convert them to nJy with the calibration `CALIB` (nJy per count) saved in the header, so that the host has the same flux in every visit.

Flagged pixels and pixels with an infinite variance (saturated) get zero weight.
For Rubin we ignore the `DETECTED` bits of the mask, which only mark sources.

```{note}
The Rubin mask bits are the LSST stack defaults (`BAD`=1, `SAT`=2, `INTRP`=4, `CR`=8, `EDGE`=16, `DETECTED`=32, `DETECTED_NEGATIVE`=64, `SUSPECT`=128, `NO_DATA`=256), since the cutouts do not carry the mask plane definitions.
```

The two bands have their own amplitudes in the model, so it does not matter that Roman is in counts and Rubin in nJy.
The sampling notebook prepares the data the same way, so it has a copy of these two functions.

```{code-cell} ipython3
RUBIN_BAD_BITS = 1 | 2 | 4 | 8 | 16 | 128 | 256   # BAD, SAT, INTRP, CR, EDGE, SUSPECT, NO_DATA


def load_roman(path):
    """Sky-subtracted Roman image (counts), weights, PSF and header of one cutout."""
    with fits.open(path) as hdul:
        sky = np.median(hdul["BKGNOISE"].data)
        data = np.nan_to_num(hdul["SCI"].data - sky)          # NaN off the detector
        var = hdul["VARIANCE"].data
        good = np.isfinite(var) & (var > 0) & (hdul["DQ"].data == 0)
        return data, var, good, hdul["PSF"].data, hdul["SCI"].header


def load_rubin(path):
    """Calibrated Rubin image (nJy), weights, PSF and header of one cutout."""
    with fits.open(path) as hdul:
        header = hdul["IMAGE"].header
        calib = header["CALIB"]                              # nJy per count
        data = np.nan_to_num(hdul["IMAGE"].data * calib)
        var = hdul["VARIANCE"].data * calib**2
        good = np.isfinite(var) & (var > 0) & ((hdul["MASK"].data & RUBIN_BAD_BITS) == 0)
        return data, var, good, hdul["PSF"].data, header


observations = []
for epoch, (instrument, band, path) in enumerate(images):
    data, var, good, psf, header = (load_roman if instrument == "roman" else load_rubin)(path)
    weights = np.where(good, 1 / np.where(good, var, 1), 0)

    # channel: combined band and epoch identifier
    obs = sc2.Observation(
        data[None].astype(np.float32),
        weights=weights[None].astype(np.float32),
        psf=psf[None].astype(np.float32),
        wcs=WCS(header),
        channels=[(band, epoch)],
        name=f"{instrument} {epoch}",
    )
    observations.append(obs)

roman_id = [instrument for instrument, _, _ in images].index("roman")
```

Let's look at the images. Note the different pixel sizes: about 0.11 arcsec for Roman and 0.2 arcsec for Rubin.

```{code-cell} ipython3
for obs in observations:
    norm = sc2.plot.AsinhAutomaticNorm(obs)
    sc2.plot.observation(obs, norm=norm, show_psf=True);
```

## Define Model Frame and Stack Observations

The model frame defines the common pixel grid for all images.
We use the Roman image as the reference (`reference_id`), so the model has Roman's resolution, and keep only the area covered by all images (`coverage="intersection"`), which avoids padding the stack with empty pixels.

The Roman PSF is barely sampled (its FWHM is about 1.1 pixels), so, as for ZTF in the transient tutorial, we reduce the internal model PSF to a narrow Gaussian:

```{code-cell} ipython3
model_psf = sc2.GaussianPSF(sigma=0.5)
model_frame = sc2.Frame.from_observations(
    observations,
    model_psf=model_psf,
    reference_id=roman_id,
    coverage="intersection",
)
```

{py:func}`~scarlet2.stack_observations` resamples every Rubin image onto the Roman pixel grid and concatenates all images as channels of a single observation.
Because resampling correlates the noise of neighboring pixels, the result is a {py:class}`~scarlet2.CorrelatedObservation`:

```{code-cell} ipython3
obs_stacked = sc2.stack_observations(observations, model_frame)
print(type(obs_stacked).__name__, obs_stacked.frame.channels)
```

## Define Transient Scene

As in the transient tutorial, we treat channels as (band, epoch) pairs and need lookup functions for either part:

```{code-cell} ipython3
# separate channel information into band and epoch: 0 and 1 element
band_selector = lambda channel: channel[0]
epoch_selector = lambda channel: channel[1]

# band list in a fixed order: Roman first, then Rubin
bands = [ROMAN_BAND, RUBIN_BAND]

host_coord = SkyCoord(target["host_ra"], target["host_dec"], unit="deg")
transient_coord = SkyCoord(target["ra"], target["dec"], unit="deg")
```

The cutouts are large enough to contain neighbors of the host, so we detect all sources instead of placing only the host.
As in the stacking tutorial, we detect them in the image with the higher resolution, the Roman image: it sits on the model grid, so the morphologies come out correctly sized and oriented.
Because the Roman PSF is narrow, we set `image_type="space"`, and because the host can have several detected peaks (e.g. star-forming knots), we keep them overlapping with `split_peaks=False`.
Deep Roman images show galaxy light out to large radii, so we also double the box of every detected source.

All detected sources are static: their spectrum is a {py:class}`~scarlet2.StaticArraySpectrum` with one amplitude per band, shared by all epochs.
The transient is a {py:class}`~scarlet2.PointSource` at its catalog position, with a {py:class}`~scarlet2.TransientArraySpectrum` that has free amplitudes only in the active epochs, and is fixed to zero before `start_mjd`.
We first give all of them placeholder amplitudes, and set the actual amplitudes below.

```{code-cell} ipython3
with sc2.Scene(model_frame) as scene:
    # 1) Static sources (host and neighbors), detected in the Roman image
    infos = sc2.init.hierarchical_sources(
        observations[roman_id],
        K=3,
        scales=[2, 3, 4],
        min_area=16,
        strict=True,
        split_peaks=False,
        image_type="space",
    )
    for sinfo in infos:
        # double the box, keeping the source centered
        morph = jnp.pad(sinfo.morphology, [(n // 2, n // 2) for n in sinfo.morphology.shape],
                        constant_values=1e-6)
        sc2.Source(
            sinfo.center,
            sc2.StaticArraySpectrum(jnp.ones(len(bands)), bands=bands, band_selector=band_selector),
            morph,
        )

    # 2) Point source for the transient at its catalog position
    n_active = sum(epoch_selector(c) in active_epochs for c in model_frame.channels)
    sc2.PointSource(
        transient_coord,
        sc2.TransientArraySpectrum(jnp.ones(n_active), epochs=active_epochs, epoch_selector=epoch_selector),
    )

transient_id = len(scene.sources) - 1
```

So far, the Rubin amplitudes are unknown. As in the stacking tutorial, we use the Roman-defined morphologies to perform forced photometry ({py:func}`scarlet2.measure.forced_photometry`) in every channel of the stacked observation.
It gives every source an amplitude per channel. For a static source, we average them over the channels of each band where the transient is not active, and for the transient, we keep its amplitudes in the active channels.
The forced photometry of the transient and the host near it is a mixture of their light, to be corrected by the fitting procedure.

```{code-cell} ipython3
forced_spectra = sc2.measure.forced_photometry(scene, obs_stacked)

channels = model_frame.channels
quiet = jnp.array([epoch_selector(c) not in active_epochs for c in channels])
active = ~quiet
for i in range(len(scene.sources)):
    spectrum = forced_spectra[i]
    if i == transient_id:
        new_data = spectrum[active]
    else:
        # average the transient-free channels of each band; a band without any such channel
        # averages all of its channels, and the fit has to sort out host and transient
        new_data = []
        for band in bands:
            in_band = jnp.array([band_selector(c) == band for c in channels])
            use = in_band & quiet if (in_band & quiet).any() else in_band
            new_data.append(spectrum[use].mean())
        new_data = jnp.array(new_data)
    # replace only the amplitudes: creating a new spectrum object would need the Scene context,
    # while the existing one already knows how its amplitudes map onto the channels
    src = scene.sources[i]
    scene.sources[i] = src.replace("spectrum", src.spectrum.replace("data", new_data))

print(scene.sources)
```

The host is the static source closest to the host position in the catalog:

```{code-cell} ipython3
host_pixel = model_frame.get_pixel(host_coord)
distance = [float(jnp.hypot(*(jnp.asarray(src.center) - host_pixel))) for src in scene.sources[:transient_id]]
host_id = int(np.argmin(distance))
print(f"host: source {host_id}, {distance[host_id]:.1f} pixels from the catalog position; transient: source {transient_id}")
```

We can have a quick look at the initial model. We use `split_channels=True`, so that we see every epoch separately instead of a (very false) color image:

```{code-cell} ipython3
# fig = sc2.plot.scene(
#     scene,
#     observation=obs_stacked,
#     show_rendered=True,
#     show_observed=True,
#     show_residual=True,
#     add_labels=False,
#     add_boxes=False,
#     box_kwargs={"edgecolor": "red", "facecolor": "none"},
#     label_kwargs={"color": "red"},
#     split_channels=True,
# );

# zoom(fig, host_pixel, half_width=50)   # 41 x 41 pixels, about 4.5" for Roman
# plt.show();
```

## Fitting

We fit in two stages, as in the stacking tutorial: first the scene alone, on the WCS alignment of the images, and then the scene together with the astrometric offsets between the images.
The first stage gets the sources close to their best fit, so that the offsets are fitted against a good model rather than absorbing its initial errors.

### Fit the Scene

The static sources have free-form morphologies (arrays) and a static spectrum, whose free parameter is its `.data` array.
For the transient, the free parameters are the amplitudes in the active epochs and its position.
Its amplitudes are not constrained to be positive, because they can be zero.

```{code-cell} ipython3
# best step size for parameters with a relative "uncertainty":
# big/bright sources adjust more quickly
spec_step = partial(sc2.relative_step, factor=0.05)
morph_step = 0.2
pos_step = 1e-2

with sc2.Parameters(scene):
    for i in range(transient_id):
        # because we want the spectrum parameter to be constrained,
        # we need to ensure that their initialization satisfies the constraint
        tiny = 1e-6
        spectrum = scene.sources[i].spectrum
        if (spectrum.data <= 0).any():
            scene.sources[i] = scene.sources[i].replace(
                "spectrum", spectrum.replace("data", jnp.maximum(spectrum.data, tiny))
            )
        sc2.Parameter(
            scene.sources[i].spectrum.data,
            name=f"spectrum:{i}",
            constraint=constraints.positive,
            stepsize=spec_step,
        )
        # same for [0,1] constrained morphology
        morph = scene.sources[i].morphology
        if ((morph <= 0) | (morph >= 1)).any():
            scene.sources[i] = scene.sources[i].replace("morphology", jnp.clip(morph, tiny, 1 - tiny))
        sc2.Parameter(
            scene.sources[i].morphology,
            name=f"morph:{i}",
            constraint=constraints.unit_interval,
            stepsize=morph_step,
        )

    # Transient point source parameters:
    # no positive constraint on spectrum because it can be zero
    sc2.Parameter(
        scene.sources[transient_id].spectrum.data,
        name="spectrum:transient",
        stepsize=spec_step,
    )
    sc2.Parameter(
        scene.sources[transient_id].center,
        name="center:transient",
        stepsize=pos_step,
    )
```

```{code-cell} ipython3
# Fit the scene
scene_ = scene.fit(obs_stacked, max_iter=3000, e_rel=1e-4, progress_bar=True)
print(f"goodness of fit without astrometric offsets: {obs_stacked.goodness_of_fit(scene_()):.2f}")
```

### Fit Astrometric Offsets

The WCS of each image places the sources slightly differently, more so between two instruments with their own astrometric solutions, and a single model cannot match all of them: the misalignment shows up as dipoles in the residuals.
So we always fit an astrometric offset for every image.

As in the [stacking tutorial](https://scarlet2.readthedocs.io/en/latest/howto/stacking.html), the stacked observation has one renderer whose last transformation is the convolution, with an attribute `.shift`, which we define as a {py:class}`~scarlet2.Parameter`.
The shift moves the whole rendered model of an image, so all sources in it move together.
In our stacked observation every channel is one image, so a per-channel shift is exactly a per-image shift, and we do not need the per-observation shift of the tutorial.

If shifts are permitted in all channels, nothing remains to define the astrometric reference.
So the Roman image, whose pixels define the model frame, is the reference: its shift is held at zero by a prior of 1e-4 pixel width, and the Rubin images move with respect to it, with a prior of 1 pixel (about 0.11 arcsec) width.
Shifts are in model pixels.

```{code-cell} ipython3
# replace the 2D shift with a (C,2) shift: one (y, x) shift per image
C = obs_stacked.frame.C
shift = jnp.zeros((C, 2))
object.__setattr__(obs_stacked.renderer[-1], "shift", shift)

ref_channel = roman_id   # the Roman image is the astrometric reference
ref_width = 1e-3         # prior width of the reference shift, in pixels: effectively fixed at zero
other_width = 2.0        # prior width of the other images' shifts, in pixels
with sc2.Parameters(obs_stacked):
    # define a Gaussian prior with zero mean and narrow width in ref_channel
    scale = other_width * jnp.ones((C, 2))
    scale = scale.at[ref_channel].set(ref_width)
    prior = dist.Normal(shift, scale=scale).to_event(2)  # last 2 dims are per-event

    # declare new shift as parameter, with milli-pixel stepsize
    sc2.Parameter(obs_stacked.renderer[-1].shift, name="shift", prior=prior, stepsize=1e-3)

# the more flexible fitter optimizes scene and observation and returns both
scene_, obs_stacked_ = sc2.infer.fit(scene_, obs_stacked, e_rel=1e-4, max_iter=3000)

# get the astrometry shift
shift = np.asarray(obs_stacked_.get("shift"))
gof = float(obs_stacked_.goodness_of_fit(scene_()))
print(f"goodness of fit with astrometric offsets: {gof:.2f}")
```

The model pixels are those of the Roman image, about 0.11 arcsec on a side:

```{code-cell} ipython3
pixel_arcsec = np.mean(proj_plane_pixel_scales(model_frame.wcs)) * 3600
for epoch, ((instrument, _, _), (dy, dx)) in enumerate(zip(images, shift)):
    print(f"epoch {epoch} ({instrument}): shift ({dy:+.3f}, {dx:+.3f}) pixels = "
          f"({dy * pixel_arcsec * 1000:+.1f}, {dx * pixel_arcsec * 1000:+.1f}) milli-arcsec")
```

## Inspect Result

```{code-cell} ipython3
# Plot the model, for each epoch
fig = sc2.plot.scene(
    scene_,
    observation=obs_stacked_,
    show_rendered=True,
    show_observed=True,
    show_residual=True,
    add_labels=False,
    add_boxes=False,
    title_kwargs={'fontsize':16},
    box_kwargs={"edgecolor": "red", "facecolor": "none"},
    label_kwargs={"color": "red"},
    split_channels=True,
);

zoom(fig, host_pixel, half_width=50, label_size=13)   # 41 x 41 pixels, about 4.5" for Roman
plt.savefig(f'figs/fit_{target['id']}.pdf')
plt.show();
```

Here are the best-fitting fluxes of the host and the transient in every epoch, in counts for Roman and nJy for Rubin:

```{code-cell} ipython3
print("----------------- {}".format(model_frame.channels))
for k in (host_id, transient_id):
    print(f"Source {k}, Fluxes: {sc2.measure.flux(scene_.sources[k])}")
```

Note that the host galaxy has the same flux in every epoch of a band, while the transient has zero flux in the epochs where we forced it to be 'off'.
The transient's Rubin fluxes are calibrated (nJy), so we can also express them as AB magnitudes; their uncertainties come from the sampling notebook.

```{code-cell} ipython3
transient_flux = np.asarray(sc2.measure.flux(scene_.sources[transient_id]))
is_rubin = np.array([band_selector(c) == RUBIN_BAND for c in model_frame.channels])
rubin_mjd = np.array(mjds)[is_rubin]
rubin_flux = transient_flux[is_rubin]                    # nJy

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(rubin_mjd, rubin_flux, "o", color="#2a78d6", label=f"Rubin {RUBIN_BAND}")
ax.axvline(target["start_mjd"], color="#52514e", lw=0.8, ls="--", label="start (catalog)")
ax.axvline(target["peak_mjd"], color="#52514e", lw=0.8, ls=":", label="peak (catalog)")
ax.set_xlabel("MJD")
ax.set_ylabel(f"Transient flux in {RUBIN_BAND} (nJy)")
ax.set_title(f"Target {target['id']}")
ax.legend(frameon=False)
# ax.set_yscale('log')
plt.show()

# for mjd, flux in zip(rubin_mjd, rubin_flux):
#     mag = 31.4 - 2.5 * np.log10(flux) if flux > 0 else np.nan     # AB magnitude of a flux in nJy
#     print(f"MJD {mjd:.3f}: {flux:10.1f} nJy = {mag:.2f} AB mag")
```

## Save the Best-Fit Model

As in the [real-world example](https://scarlet2.readthedocs.io/en/latest/0-example.html), we store the best-fit scene with {py:func}`scarlet2.io.model_to_h5`.
It contains the model frame, every source, and their parameter values.

The astrometric offsets belong to the observation, not the scene, so we store them next to it, together with what the sampling notebook needs to rebuild the same stacked observation: the cutout files in epoch order, the data settings, the source indices, and the channels.
They go into a group `fit` of the same file.

```{code-cell} ipython3
# scarlet2.io.model_to_h5(scene_, MODEL_FILE, id=0, path=MODEL_DIR, overwrite=True)
ou.save_scene(scene_, MODEL_DIR / MODEL_FILE, id=0)

with h5py.File(MODEL_DIR / MODEL_FILE, "a") as f:
    if "fit" in f:
        del f["fit"]
    g = f.create_group("fit")
    g.create_dataset("shift", data=shift)                 # (C, 2) per-image shift, model pixels (y, x)
    g.create_dataset("mjd", data=np.array(mjds))
    g.create_dataset("transient_flux", data=transient_flux)
    g.create_dataset("host_flux", data=np.asarray(sc2.measure.flux(scene_.sources[host_id])))
    g.attrs["target_id"] = int(target["id"])
    g.attrs["images"] = json.dumps([[instrument, band, str(path)] for instrument, band, path in images])
    g.attrs["channels"] = json.dumps([list(c) for c in model_frame.channels])
    g.attrs["bands"] = json.dumps(bands)
    g.attrs["active_epochs"] = json.dumps(active_epochs)
    g.attrs["roman_id"] = roman_id
    g.attrs["host_id"] = host_id
    g.attrs["transient_id"] = transient_id
    g.attrs["rubin_bad_bits"] = RUBIN_BAD_BITS
    g.attrs["goodness_of_fit"] = gof

print(f"saved to {MODEL_DIR / MODEL_FILE}")
```

Let's check that the saved scene renders the same model:

```{code-cell} ipython3
scene_loaded = ou.load_scene(MODEL_DIR / MODEL_FILE, id=0)
print("saved model identical:", bool(jnp.allclose(scene_loaded(), scene_())))
```

```{code-cell} ipython3

```
