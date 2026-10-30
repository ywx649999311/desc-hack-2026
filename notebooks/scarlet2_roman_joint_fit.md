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

# Joint Fit of Roman Epochs

This notebook fits a transient and its host galaxy jointly in all Roman Time Domain Survey epochs of one target, with [_scarlet2_](https://github.com/pmelchior/scarlet2).
It combines two of the _scarlet2_ how-to guides: the [transient tutorial](https://scarlet2.readthedocs.io/en/latest/howto/timedomain.html), which models a static host and a variable point source across epochs, and the [stacking tutorial](https://scarlet2.readthedocs.io/en/latest/howto/stacking.html), which aligns observations with different pixel grids into a single observation for a fast fit.

The input are the cutouts written by `roman_cutouts.md`, read from the local files in `data/cutouts/roman/`. Every epoch of a target was taken on a different detector, so each cutout has its own WCS and its own PSF.

```{code-cell} ipython3
# Import Packages and setup
from functools import partial

import jax
jax.config.update("jax_enable_x64", False)
import astropy.io.fits as fits
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from numpyro.distributions import constraints

import ou_utils as ou
import scarlet2 as sc2

sc2.set_validation(False)
```

## Load Data

We pick a target from the list written by `target_selection.md` and find its Roman cutouts in the manifest of `roman_cutouts.md`.

```{code-cell} ipython3
TARGET_ID = 30403008   # a target ID from data/targets.parquet; None = the first target
BAND = "J129"      # Roman band of the cutouts

targets = pd.read_parquet(ou.TARGETS_PATH)
target = targets.iloc[0] if TARGET_ID is None else targets[targets["id"] == TARGET_ID].iloc[0]
manifest = pd.read_parquet(ou.DATA_DIR / "cutouts" / "roman" / "manifest.parquet")
rows = manifest[(manifest["target_id"] == str(target["id"])) & (manifest["band"] == BAND)
                & (manifest["path"] != "")].sort_values("mjd")

print(f"target {target['id']} ({target['model_name']}), active from MJD {target['start_mjd']:.1f}, "
      f"peak at MJD {target['peak_mjd']:.1f}")
print(f"{len(rows)} {BAND} epochs at MJD {rows['mjd'].round(2).tolist()}")
```

Each cutout file holds the science image (`SCI`), its variance (`VARIANCE`), quality flags (`DQ`), the background without sources (`BKGNOISE`), and the PSF at the cutout centre (`PSF`, on the cutout's own pixel grid).
The science images still contain the sky, which _scarlet2_ does not model, so we subtract it: the median of `BKGNOISE` is the sky level of that exposure.
Flagged pixels and pixels with an infinite variance (saturated) get zero weight.

We store every epoch in its own {py:class}`~scarlet2.Observation`. As in the transient tutorial, the channel is a combined (band, epoch) identifier:

```{code-cell} ipython3
observations, mjds = [], []
n_epoch = 5
epoch_1 = 4
for epoch, path in enumerate(rows["path"][epoch_1:epoch_1+n_epoch]):
    with fits.open(path) as hdul:
        header = hdul["SCI"].header
        sky = np.median(hdul["BKGNOISE"].data)
        data = np.nan_to_num(hdul["SCI"].data - sky)          # NaN off the detector
        var = hdul["VARIANCE"].data
        good = np.isfinite(var) & (var > 0) & (hdul["DQ"].data == 0)
        weights = np.where(good, 1 / np.where(good, var, 1), 0)
        psf = hdul["PSF"].data

    # channel: combined band and epoch identifier
    obs = sc2.Observation(
        data[None].astype(np.float32),
        weights=weights[None].astype(np.float32),
        psf=psf[None].astype(np.float32),
        wcs=WCS(header),
        channels=[(BAND, epoch)],
        name=f"epoch {epoch}",
    )
    observations.append(obs)
    mjds.append(header["MJD"])
    print(f"epoch {epoch}: MJD {header['MJD']:.2f}, {data.shape[0]}x{data.shape[1]} pixels, sky {sky:.0f} counts")
```

All epochs have the same exposure time and zero point, so the fluxes of a static source are the same in every epoch.

Let's look at the epochs. The transient catalog tells us when the transient is active: epochs before `start_mjd` show the host alone.

```{code-cell} ipython3
active_epochs = [epoch for epoch, mjd in enumerate(mjds) if mjd >= target["start_mjd"]]
print("epochs with the transient:", active_epochs)

for obs in observations:
    norm = sc2.plot.AsinhAutomaticNorm(obs)
    sc2.plot.observation(obs, norm=norm, show_psf=True);
```

## Define Model Frame and Stack Observations

The model frame defines the common pixel grid for all epochs.
We use the first epoch as the reference (`reference_id=0`), so its pixels are the model pixels, and keep only the area covered by all epochs (`coverage="intersection"`), which avoids padding the stack with empty pixels.

The Roman PSF is barely sampled (its FWHM is about 1.1 pixels), so, as for ZTF in the transient tutorial, we reduce the internal model PSF to a narrow Gaussian:

```{code-cell} ipython3
model_psf = sc2.GaussianPSF(sigma=0.5)
model_frame = sc2.Frame.from_observations(
    observations,
    model_psf=model_psf,
    reference_id=0,
    coverage="intersection",
)
```

The epochs come from different detectors, so their pixel grids are rotated and shifted with respect to each other.
{py:func}`~scarlet2.stack_observations` resamples every epoch that is not on the model grid and concatenates all of them as channels of a single observation.
Because resampling correlates the noise of neighboring pixels, the result is a {py:class}`~scarlet2.CorrelatedObservation`:

```{code-cell} ipython3
obs_stacked = sc2.stack_observations(observations, model_frame)
print(type(obs_stacked).__name__, obs_stacked.frame.channels)
```

```{note}
The alternative is to fit the list of individual observations, which resamples the model into every epoch during the fit. That avoids the correlated noise, but is considerably slower (see the [multi-resolution tutorial](https://scarlet2.readthedocs.io/en/latest/howto/multiresolution.html)).
```

## Define Transient Scene

As in the transient tutorial, we treat channels as (band, epoch) pairs and need lookup functions for either part:

```{code-cell} ipython3
# separate channel information into band and epoch: 0 and 1 element
band_selector = lambda channel: channel[0]
epoch_selector = lambda channel: channel[1]

# get the band list (set avoids duplicates)
bands = list(set([band_selector(channel) for channel in model_frame.channels]))

host_coord = SkyCoord(target["host_ra"], target["host_dec"], unit="deg")
transient_coord = SkyCoord(target["ra"], target["dec"], unit="deg")
```

The cutouts are large enough to contain neighbors of the host, so we detect all sources instead of placing only the host.
We use the first epoch for the detection: it sits on the model grid, so the morphologies come out correctly sized and oriented, and if it was taken before the transient appeared, the transient light does not end up in the host or its neighbors.
Because the Roman PSF is narrow, we set `image_type="space"`, and because the host can have several detected peaks (e.g. star-forming knots), we keep them overlapping with `split_peaks=False`.

All detected sources are static: their spectrum is a {py:class}`~scarlet2.StaticArraySpectrum` with one amplitude per band, shared by all epochs.
We initialize it from the epochs without the transient.
Deep Roman images show galaxy light out to large radii, so we also double the box of every detected source.

The morphologies of the static sources are free-form, i.e. not a parametric profile: a 2D array of pixel values, as in the [stacking tutorial](https://scarlet2.readthedocs.io/en/latest/howto/stacking.html).

```{note}
An alternative, commented out below (search for "Starlet morphology"), is a {py:class}`~scarlet2.StarletMorphology`: the same image described by its starlet (wavelet) coefficients, one plane per spatial scale, with a prior that is much narrower on the fine scales than on the coarse ones, which keeps the fit from growing fine structure out of the noise. It follows the DDP3 scarlet2 demo (`other_repo/DDP3_Scarlet2_demo`, `SN_RubinEuclidDP1Q1.py`). To use it, uncomment the three marked blocks and comment out the pixel-array morphology parameter.
```

```{code-cell} ipython3
# --- Starlet morphology (disabled) ---
# MORPH_PRIOR_FINE = 1e-4   # starlet prior width on the finest scale; it grows to 1 on the coarsest
```

The transient is a {py:class}`~scarlet2.PointSource` at its catalog position, with a {py:class}`~scarlet2.TransientArraySpectrum` that has free amplitudes only in the active epochs, and is fixed to zero before `start_mjd`:

```{code-cell} ipython3
# channels without the transient, used to initialize the static amplitudes
quiet = jnp.array([epoch_selector(c) not in active_epochs for c in model_frame.channels])

with sc2.Scene(model_frame) as scene:
    # 1) Static sources (host and neighbors), detected in the first epoch
    infos = sc2.init.hierarchical_sources(
        observations[0],
        K=3,
        scales=[2, 3, 4],
        min_area=16,
        strict=True,
        split_peaks=False,
        image_type="space",
    )
    for sinfo in infos:
        # average the transient-free channels of each band; if the transient is active in
        # every epoch, this averages all channels and the fit has to sort out host and transient
        mask = quiet if quiet.any() else jnp.ones(len(quiet), dtype=bool)
        static_spectrum = jnp.array([
            sinfo.spectrum[mask & jnp.array([band_selector(c) == band for c in model_frame.channels])].mean()
            for band in bands
        ])
        # double the box, keeping the source centered
        morph = jnp.pad(sinfo.morphology, [(n // 2, n // 2) for n in sinfo.morphology.shape],
                        constant_values=1e-6)
        # --- Starlet morphology (disabled) ---
        # starlet coefficients of the same image, kept inside the (0, 1) constraint of the fit
        # morph = sc2.StarletMorphology.from_image(morph, min_value=1e-6, max_value=1 - 1e-6)
        sc2.Source(
            sinfo.center,
            sc2.StaticArraySpectrum(static_spectrum, bands=bands, band_selector=band_selector),
            morph,
        )

    # 2) Point source for the transient at its catalog position.
    # The flux at that position is a mixture of host and transient light,
    # to be corrected by the fitting procedure.
    pixel_spectrum = sc2.init.pixel_spectrum(obs_stacked, transient_coord)
    # TransientArraySpectrum only takes the free parameters (the active channels)
    transient_spectrum = pixel_spectrum[
        jnp.array([epoch_selector(c) in active_epochs for c in model_frame.channels])
    ]
    sc2.PointSource(
        transient_coord,
        sc2.TransientArraySpectrum(transient_spectrum, epochs=active_epochs, epoch_selector=epoch_selector),
    )

transient_id = len(scene.sources) - 1
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
# sc2.plot.scene(
#     scene,
#     observation=obs_stacked,
#     show_rendered=True,
#     show_observed=True,
#     show_residual=True,
#     add_labels=True,
#     add_boxes=True,
#     box_kwargs={"edgecolor": "red", "facecolor": "none"},
#     label_kwargs={"color": "red"},
#     split_channels=True,
# );
```

## Fitting

The static sources have free-form morphologies (arrays), whose pixel values are constrained to the interval (0, 1), and a static spectrum, whose free parameter is its `.data` array.
(With the disabled starlet morphology, the free parameters would be the starlet coefficients `.coeffs`, also in (0, 1), with a Cauchy prior whose width grows geometrically from `MORPH_PRIOR_FINE` on the finest scale to 1 on the coarsest.)
For the transient, the free parameters are the amplitudes in the active epochs and its position.
Its amplitudes are not constrained to be positive, because they can be zero.

```{code-cell} ipython3
# best step size for parameters with a relative "uncertainty":
# big/bright sources adjust more quickly
spec_step = partial(sc2.relative_step, factor=3e-3)
morph_step = 0.008
pos_step = 1e-3

# --- Starlet morphology (disabled) ---
# starlet_step = partial(sc2.relative_step, factor=1e-3)   # as in the DDP3 demo
# import numpyro.distributions as dist

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

        # --- Starlet morphology (disabled) ---
        # Replaces the pixel-array parameter above: [0,1] constrained starlet coefficients,
        # with a scale-dependent Cauchy prior (coarser scales get a broader prior).
        # coeffs = scene.sources[i].morphology.coeffs
        # coeffs = jnp.clip(jnp.nan_to_num(coeffs, nan=tiny), tiny, 1 - tiny)
        # scene.sources[i] = scene.sources[i].replace(
        #     "morphology", scene.sources[i].morphology.replace("coeffs", coeffs)
        # )
        # n_scales = coeffs.shape[0]
        # scale = jnp.geomspace(MORPH_PRIOR_FINE, 1, n_scales)[:, None, None] * jnp.ones_like(coeffs)
        # prior = dist.Cauchy(scale=scale).to_event(coeffs.ndim)
        # sc2.Parameter(
        #     scene.sources[i].morphology.coeffs,
        #     name=f"morph:{i}",
        #     prior=prior,
        #     constraint=constraints.unit_interval,
        #     stepsize=starlet_step,
        # )

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
scene_ = scene.fit(obs_stacked, max_iter=5000, e_rel=1e-4, progress_bar=True)
```

## Inspect Result

```{code-cell} ipython3
# Plot the model, for each epoch
sc2.plot.scene(
    scene_,
    observation=obs_stacked,
    show_rendered=True,
    show_observed=True,
    show_residual=True,
    add_labels=True,
    add_boxes=True,
    box_kwargs={"edgecolor": "red", "facecolor": "none"},
    label_kwargs={"color": "red"},
    split_channels=True,
);
```

The goodness of fit summarizes the residuals in one number, which should be close to 1 if the model describes the data to within the noise:

```{code-cell} ipython3
print(f"goodness of fit: {obs_stacked.goodness_of_fit(scene_()):.2f}")
```

Look for dipoles in the residuals, pairs of positive and negative residuals next to each other at the positions of sources. They are addressed in the next section.

## Fit Astrometric Offsets

Dipoles in the residuals are the typical sign that the epochs are not perfectly aligned: the WCS of each epoch places the sources slightly differently, and a single model cannot match all of them.
Because the host and its neighbors are shared by all epochs, a misaligned epoch also leaves dipoles in the others, including the reference epoch.

As in the [stacking tutorial](https://scarlet2.readthedocs.io/en/latest/howto/stacking.html), the stacked observation has one renderer whose last transformation is the convolution, with an attribute `.shift`, which we can define as a {py:class}`~scarlet2.Parameter` and refit.
In our stacked observation every channel is one epoch, so a per-channel shift is exactly a per-epoch shift, and we do not need the per-observation shift of the tutorial.
If shifts are permitted in all channels, nothing remains to define the astrometric reference, so the prior for the first epoch (the reference grid of the model frame) has essentially zero width.

```{code-cell} ipython3
# define shift parameter: unconstrained, with milli-arcsec stepsize
import astropy.units as u
import numpyro.distributions as dist

# replace the 2D shift with a (C,2) shift: one (y, x) shift per epoch
C = obs_stacked.frame.C
shift = jnp.zeros((C, 2))
object.__setattr__(obs_stacked.renderer[-1], "shift", shift)

ref_channel = 0
with sc2.Parameters(obs_stacked):
    # define a Gaussian prior with zero mean and narrow width in ref_channel
    scale = jnp.ones((C, 2))
    scale = scale.at[ref_channel].set(1e-6)
    scale = 0.1 * scale * u.arcsec
    prior = dist.Normal(shift, scale=scale).to_event(2)  # last 2 dims are per-event

    # declare new shift as parameter
    sc2.Parameter(obs_stacked.renderer[-1].shift, name="shift", prior=prior, stepsize=1e-3 * u.arcsec)

# the more flexible fitter optimizes scene and observation and returns both
scene_, obs_stacked_ = sc2.infer.fit(scene_, obs_stacked, e_rel=1e-4, max_iter=2000)

# get the astrometry shift
shift = obs_stacked_.get("shift")
print("\nShift per epoch in model pixels (y, x):")
print(shift)
```

The model pixels are those of the first epoch, about 0.11 arcsec on a side:

```{code-cell} ipython3
# from astropy.wcs.utils import proj_plane_pixel_scales

# pixel_arcsec = np.mean(proj_plane_pixel_scales(model_frame.wcs)) * 3600
# for epoch, (dy, dx) in enumerate(np.asarray(shift)):
#     print(f"epoch {epoch}: shift ({dy:+.3f}, {dx:+.3f}) pixels = "
#           f"({dy * pixel_arcsec * 1000:+.1f}, {dx * pixel_arcsec * 1000:+.1f}) milli-arcsec")
```

Let's compare the residuals with those before the shifts were fitted:

```{code-cell} ipython3
print(f"goodness of fit: {obs_stacked_.goodness_of_fit(scene_()):.2f}")
```

```{code-cell} ipython3
sc2.plot.scene(
    scene_,
    observation=obs_stacked_,
    show_rendered=True,
    show_observed=True,
    show_residual=True,
    add_labels=True,
    add_boxes=True,
    box_kwargs={"edgecolor": "red", "facecolor": "none"},
    label_kwargs={"color": "red"},
    split_channels=True,
);
print(f"goodness of fit: {obs_stacked_.goodness_of_fit(scene_()):.2f}")
```

If the dipoles are gone, the epochs were misaligned. If they remain, especially in the reference epoch, they more likely come from a mismatch between the PSF in the cutout files and the PSF in the images.

## Light Curve

All results below come from the fit with the astrometric offsets.
Here are the best-fitting fluxes, in counts, of the host and the transient in every epoch:

```{code-cell} ipython3
print("----------------- {}".format(model_frame.channels))
for k in (host_id, transient_id):
    print(f"Source {k}, Fluxes: {sc2.measure.flux(scene_.sources[k])}")
```

Note that the host galaxy has the same flux in every epoch, while the transient has zero flux in the epochs where we forced it to be 'off'.
The transient's flux per epoch is its light curve:

```{code-cell} ipython3
transient_flux = np.asarray(sc2.measure.flux(scene_.sources[transient_id]))

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(mjds, transient_flux, "o-", color="#2a78d6")
ax.axvline(target["start_mjd"], color="#52514e", lw=0.8, ls="--", label="start (catalog)")
ax.axvline(target["peak_mjd"], color="#52514e", lw=0.8, ls=":", label="peak (catalog)")
ax.set_xlabel("MJD")
ax.set_ylabel(f"Transient flux in {BAND} (counts)")
ax.set_title(f"Target {target['id']}")
ax.legend(frameon=False)
plt.show()
```

The transient's fitted position, compared to its catalog position:

```{code-cell} ipython3
offset = jnp.asarray(scene_.sources[transient_id].center) - model_frame.get_pixel(transient_coord)
print(f"position offset (y, x) in model pixels: {offset}")
```

```{code-cell} ipython3

```
