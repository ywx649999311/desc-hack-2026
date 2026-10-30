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

# Sample Transient Fluxes from the Posterior

This notebook takes the best-fit Roman and Rubin model of one target, saved by `scarlet2_fit_roman_rubin.md`, and samples from the posterior distribution of the transient's fluxes, to get their uncertainties.
It follows the _scarlet2_ [sampling tutorial](https://scarlet2.readthedocs.io/en/latest/howto/sampling.html), which uses the Hamiltonian Monte Carlo sampler from numpyro, starting from the best-fit model.

```{code-cell} ipython3
# Import Packages and setup
import json

import jax
jax.config.update("jax_enable_x64", False)

import astropy.io.fits as fits
import astropy.units as u
import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np
import numpyro.distributions as dist
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

import ou_utils as ou
import scarlet2 as sc2

sc2.set_validation(False)
```

## Load Model

We load the best-fit scene and the information saved with it: the astrometric offsets, the cutout files in epoch order, and the indices of the host and the transient.

```{code-cell} ipython3
TARGET_ID = 30403008   # must have been fitted by scarlet2_fit_roman_rubin.md

MODEL_DIR = ou.DATA_DIR / "scarlet2"
MODEL_FILE = f"{TARGET_ID}_roman_rubin.h5"

scene = ou.load_scene(MODEL_DIR / MODEL_FILE, id=0)   # saved by ou.save_scene

with h5py.File(MODEL_DIR / MODEL_FILE, "r") as f:
    g = f["fit"]
    shift = g["shift"][()]
    mjds = g["mjd"][()]
    best_transient_flux = g["transient_flux"][()]
    images = json.loads(g.attrs["images"])
    channels = [tuple(c) for c in json.loads(g.attrs["channels"])]
    bands = json.loads(g.attrs["bands"])
    active_epochs = json.loads(g.attrs["active_epochs"])
    host_id = int(g.attrs["host_id"])
    transient_id = int(g.attrs["transient_id"])
    RUBIN_BAD_BITS = int(g.attrs["rubin_bad_bits"])
    saved_gof = float(g.attrs["goodness_of_fit"])

targets = pd.read_parquet(ou.TARGETS_PATH)
target = targets[targets["id"] == TARGET_ID].iloc[0]
print(f"target {TARGET_ID}: {len(scene.sources)} sources, host = source {host_id}, transient = source {transient_id}")
print("channels:", channels)
```

## Create Observation

We need the observation because it contains the {py:func}`~scarlet2.Observation.log_likelihood` method of the posterior.
We prepare the same cutouts in the same way as `scarlet2_fit_roman_rubin.md` (sky-subtracted Roman counts, Rubin calibrated to nJy), with the same functions:

```{code-cell} ipython3
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
for (instrument, band, path), channel in zip(images, channels):
    data, var, good, psf, header = (load_roman if instrument == "roman" else load_rubin)(path)
    weights = np.where(good, 1 / np.where(good, var, 1), 0)
    obs = sc2.Observation(
        data[None].astype(np.float32),
        weights=weights[None].astype(np.float32),
        psf=psf[None].astype(np.float32),
        wcs=WCS(header),
        channels=[channel],
        name=f"{instrument} {channel[1]}",
    )
    observations.append(obs)
```

The saved scene carries its model frame, so we stack the observations onto exactly the grid of the fit, and apply the fitted astrometric offsets.
They are fixed during the sampling: the stacked observation has no parameters of its own here.

```{code-cell} ipython3
obs_stacked = sc2.stack_observations(observations, scene.frame)
# the renderer is only created when the observation is first matched to a frame (the fit did this
# implicitly); create it now, so that we can set the shifts on its last (convolution) step
obs_stacked.check_set_renderer(scene.frame)
object.__setattr__(obs_stacked.renderer[-1], "shift", jnp.asarray(shift))

gof = float(obs_stacked.goodness_of_fit(scene()))
print(f"goodness of fit: {gof:.2f} (saved: {saved_gof:.2f})")
```

If the two numbers agree, the data and the model are those of the fit. Let's have a look:

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

## Define Parameters with Prior

We sample the parameters that the transient's fluxes depend on most: the transient's amplitudes in the active epochs, its position, and the amplitudes of the host, which competes with the transient for the light at the same position.
As in the fit, the transient stays fixed to zero before the catalog's `start_mjd`, so those epochs are not sampled.
Sampling more parameters, e.g. all morphologies, would take much longer; it makes sense to sample simultaneously those parameters that might be correlated.

The cell below makes sure of that whatever the saved model is (a fit that freed every epoch would have saved a transient with an amplitude in all of them): the transient gets free amplitudes only in the active epochs, from its best fit there.
A new spectrum has to be created within the context of a scene, so we create it in a temporary one with the same frame and put it into the transient.

```{code-cell} ipython3
# free transient amplitudes only in the active epochs; fixed to zero before start_mjd
transient = scene.sources[transient_id]
amplitudes = jnp.asarray(transient.spectrum())     # best fit, one value per channel
is_active = np.array([c[1] in active_epochs for c in channels])
with sc2.Scene(scene.frame):
    spectrum_active = sc2.TransientArraySpectrum(
        amplitudes[is_active], epochs=active_epochs, epoch_selector=lambda channel: channel[1]
    )
scene.sources[transient_id] = transient.replace("spectrum", spectrum_active)
print("free transient amplitudes:", np.asarray(scene.sources[transient_id].spectrum.data))
```

To run the sampler, every one of these parameters needs a `prior`. As the tutorial warns, priors should describe what we know before looking at the fit:

* Transient amplitudes, in every active epoch: flat between minus once and plus twice the total flux in that image, i.e. the sum of all pixels. No single source can be brighter (for a point source the amplitude is its flux), and the transient may be fainter than the noise, so negative amplitudes are allowed.
* Host amplitudes: flat between zero and the largest total flux in any image of that band.
* Transient position: the catalog position, with 0.1 arcsec uncertainty, as in the tutorial.

```{code-cell} ipython3
# total flux in every image, as a generous upper bound on the amplitude of any single source
# (abs: a nearly empty image could sum to a negative value from noise alone)
total_flux = np.abs(np.asarray(obs_stacked.data.sum(axis=(-2, -1))))
band_max = np.array([total_flux[[c[0] == band for c in channels]].max() for band in bands])
catalog_pixel = scene.frame.get_pixel(SkyCoord(target["ra"], target["dec"], unit="deg"))
position_sigma = scene.frame.u_to_pixel(0.1 * u.arcsec)   # 0.1 arcsec in model pixels

with sc2.Parameters(scene):
    # .to_event(1) declares the last dimension to be the dimension of the parameter,
    # so we get one probability evaluation, not one per channel
    prior_transient = dist.Uniform(low=-total_flux[is_active], high=2 * total_flux[is_active]).to_event(1)

    # names must be valid Python identifiers (scarlet2 replaces other characters with "_"),
    # and the samples come back under these names
    sc2.Parameter(scene.sources[transient_id].spectrum.data, name="spectrum_transient", prior=prior_transient)

    prior_center = dist.Normal(catalog_pixel, scale=position_sigma).to_event(1)
    sc2.Parameter(scene.sources[transient_id].center, name="center_transient", prior=prior_center)

    prior_host = dist.Uniform(low=jnp.zeros(len(bands)), high=band_max*2).to_event(1)
    sc2.Parameter(scene.sources[host_id].spectrum.data, name="spectrum_host", prior=prior_host)
```

```{warning}
The uncertainties are conditional on everything that is not sampled: the morphologies of all sources, the amplitudes of the neighbors, and the astrometric offsets stay at their best-fit values.
They are therefore lower limits, especially where the host's morphology could trade light with the transient.
Also, the noise model of the stacked observation (the per-pixel variances, correlated by resampling) sets the widths: if the variances are off, so are the error bars.
```

## Run Sampler

Then we run numpyro's {py:class}`~numpyro.infer.hmc.NUTS` sampler with {py:func}`~scarlet2.Scene.sample`, which starts from the current (best-fit) values:

```{code-cell} ipython3
NUM_WARMUP = 200       # samples to tune the sampler, then discarded
NUM_SAMPLES = 500     # posterior samples to keep
SEED = 0

mcmc = scene.sample(
    obs_stacked,
    num_warmup=NUM_WARMUP,
    num_samples=NUM_SAMPLES,
    seed=SEED,
    progress_bar=True,
    # init_strategy=numpyro.infer.init_to_median
)
mcmc.print_summary()
```

Check `r_hat` (close to 1) and `n_eff` (a good fraction of `NUM_SAMPLES`) in the summary: if they are poor, increase `NUM_WARMUP` and `NUM_SAMPLES`.

## Access Samples

The samples are listed as arrays under the names chosen above for the respective `Parameter`.
To get fluxes, we create a version of the scene for every sample with {py:func}`scarlet2.Module.set` and measure the fluxes of the host and the transient, in counts for Roman and nJy for Rubin:

```{code-cell} ipython3
samples = mcmc.get_samples()
names = list(samples.keys())

transient_flux, host_flux = [], []
for s in range(NUM_SAMPLES):
    scene_s = scene.set({name: samples[name][s] for name in names})
    transient_flux.append(np.asarray(sc2.measure.flux(scene_s.sources[transient_id])))
    host_flux.append(np.asarray(sc2.measure.flux(scene_s.sources[host_id])))
transient_flux = np.array(transient_flux)   # (samples, channels)
host_flux = np.array(host_flux)

# median and 68% interval of every channel
lo, med, hi = np.percentile(transient_flux, [16, 50, 84], axis=0)
```

Here are the transient's fluxes with their uncertainties, next to the best fit.
For the Rubin epochs, which are calibrated, we also give AB magnitudes; the magnitude interval comes from the flux interval.

```{code-cell} ipython3
def ab_mag(flux_njy):
    """AB magnitude of a flux in nJy (NaN for non-positive fluxes)."""
    flux_njy = np.asarray(flux_njy, dtype=float)
    return np.where(flux_njy > 0, 31.4 - 2.5 * np.log10(np.where(flux_njy > 0, flux_njy, 1)), np.nan)


print(f"{'epoch':>5s} {'band':>5s} {'MJD':>10s} {'best fit':>11s} {'median':>11s} {'-1 sigma':>9s} {'+1 sigma':>9s}   AB mag")
for k, (c, mjd) in enumerate(zip(channels, mjds)):
    if c[1] not in active_epochs:                  # fixed to zero, not sampled
        continue
    line = (f"{c[1]:5d} {c[0]:>5s} {mjd:10.3f} {best_transient_flux[k]:11.1f} {med[k]:11.1f} "
            f"{med[k] - lo[k]:9.1f} {hi[k] - med[k]:9.1f}")
    if c[0] != bands[0]:                           # Rubin: nJy
        m, m_bright, m_faint = ab_mag(med[k]), ab_mag(hi[k]), ab_mag(lo[k])
        line += f"   {m:.3f} -{m - m_bright:.3f} +{m_faint - m:.3f}"
    else:
        line += "   (Roman, counts)"
    print(line)

# the host is static: its flux is the same in every channel of a band, so take the first of each
for band in bands:
    k = [c[0] for c in channels].index(band)
    h_lo, h_med, h_hi = np.percentile(host_flux[:, k], [16, 50, 84])
    unit = "counts" if band == bands[0] else "nJy"
    print(f"host in {band}: {h_med:.1f} -{h_med - h_lo:.1f} +{h_hi - h_med:.1f} {unit}")
```

The Rubin light curve with its uncertainties, in flux (plotted in μJy, i.e. nJy / 1000): the median and 68% interval of every epoch.
Fluxes, unlike magnitudes, need no special treatment for faint epochs: an interval that reaches zero or below simply shows that the transient is not significantly detected there.
A magnitude version, with 1-sigma upper limits for such epochs, is commented out below the flux plot.

```{code-cell} ipython3
is_rubin = np.array([c[0] == bands[1] for c in channels])
show = is_rubin #& is_active     # pre-explosion epochs are fixed to zero, not measured

to_ujy = 1e-3                   # plot in μJy (nJy / 1000), so the tick labels stay short
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(mjds[show], med[show] * to_ujy,
            yerr=[5 * (med[show] - lo[show]) * to_ujy, 5 * (hi[show] - med[show]) * to_ujy],
            fmt="o", color="#2a78d6", capsize=3, label=f"Rubin {bands[1]}")
# ax.plot(mjds[show], best_transient_flux[show] * to_ujy, "x", color="#52514e", label="best fit")
ax.axhline(0, color="#52514e", lw=0.6)
# ax.axvline(target["start_mjd"], color="#52514e", lw=0.8, ls="--", label="start (catalog)")
# ax.axvline(target["peak_mjd"], color="#52514e", lw=0.8, ls=":", label="peak (catalog)")
ax.set_xlabel("MJD", fontsize=13)
ax.set_ylabel(f"Transient flux in {bands[1]} (μJy)", fontsize=13)
ax.legend(frameon=True, fontsize=12)
ax.tick_params(axis='both', labelsize=11, direction='in')
plt.savefig(f'figs/lc_{TARGET_ID}.pdf')
plt.show();


# --- Magnitude light curve (disabled) ---
# The magnitude interval comes from the 68% flux interval (magnitude is monotonic in flux).
# Where the lower end of the flux interval is not positive, there is no fainter magnitude bound,
# so the epoch is shown as a 1-sigma upper limit on the brightness: an open downward triangle at
# the magnitude of the upper end of the flux interval.
#
# mag_med, mag_bright, mag_faint = ab_mag(med), ab_mag(hi), ab_mag(lo)
# detected = show & (lo > 0)
# limit = show & ~(lo > 0) & (hi > 0)

# fig, ax = plt.subplots(figsize=(6, 4))
# ax.errorbar(mjds[detected], mag_med[detected],
#             yerr=[mag_med[detected] - mag_bright[detected], mag_faint[detected] - mag_med[detected]],
#             fmt="o", color="#2a78d6", capsize=3, label=f"Rubin {bands[1]} (median, 68%)")
# if limit.any():
#     # a downward triangle: the transient is at most this bright (fainter = down on this axis)
#     ax.plot(mjds[limit], mag_bright[limit], "v", color="#2a78d6", ms=8, mfc="none", label="1-sigma limit")
# ax.plot(mjds[show], ab_mag(best_transient_flux)[show], "x", color="#52514e", label="best fit")
# ax.axvline(target["start_mjd"], color="#52514e", lw=0.8, ls="--", label="start (catalog)")
# ax.axvline(target["peak_mjd"], color="#52514e", lw=0.8, ls=":", label="peak (catalog)")
# ax.invert_yaxis()                                  # brighter is up
# ax.set_xlabel("MJD")
# ax.set_ylabel(f"Transient {bands[1]} (AB mag)")
# ax.set_title(f"Target {TARGET_ID}")
# ax.legend(frameon=False)
# plt.show()
```

## Visualize Posterior

The posterior of the transient's flux in every active epoch:

```{code-cell} ipython3
active_idx = np.flatnonzero(is_active)
ncols = min(4, len(active_idx))
nrows = -(-len(active_idx) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for ax, k in zip(axes.ravel(), active_idx):
    ax.axis("on")
    ax.hist(transient_flux[:, k], bins=40, color="#2a78d6", alpha=0.8)
    ax.axvline(best_transient_flux[k], color="#52514e", lw=1, ls="--")
    unit = "counts" if channels[k][0] == bands[0] else "nJy"
    ax.set_title(f"epoch {channels[k][1]} ({channels[k][0]})", fontsize=9)
    ax.set_xlabel(f"flux ({unit})", fontsize=8)
    ax.set_yticks([])
fig.tight_layout()
plt.show()
```

The dashed line is the best fit; it should sit within the bulk of each histogram.
The samples of the transient's position show how well it is located:

```{code-cell} ipython3
center = np.asarray(samples["center_transient"]) - np.asarray(catalog_pixel)
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(center[:, 1], center[:, 0], ".", ms=2, alpha=0.3, color="#2a78d6")
ax.plot(0, 0, "+", color="#52514e", ms=12, label="catalog position")
ax.set_xlabel("x offset (model pixels)")
ax.set_ylabel("y offset (model pixels)")
ax.set_aspect("equal")
ax.legend(frameon=False)
plt.show()
print(f"position: ({center[:, 0].mean():+.3f} +- {center[:, 0].std():.3f}, "
      f"{center[:, 1].mean():+.3f} +- {center[:, 1].std():.3f}) model pixels from the catalog")
```

## Save Samples

We add the samples and the flux intervals to the same file, in a group `samples`, so that the light curve and its uncertainties can be used without rerunning the sampler:

```{code-cell} ipython3
with h5py.File(MODEL_DIR / MODEL_FILE, "a") as f:
    if "samples" in f:
        del f["samples"]
    g = f.create_group("samples")
    for name in names:
        g.create_dataset(name, data=np.asarray(samples[name]))
    g.create_dataset("transient_flux", data=transient_flux)      # (samples, channels)
    g.create_dataset("host_flux", data=host_flux)
    g.create_dataset("transient_flux_percentiles", data=np.array([lo, med, hi]))   # 16, 50, 84
    g.attrs["num_warmup"] = NUM_WARMUP
    g.attrs["num_samples"] = NUM_SAMPLES
    g.attrs["seed"] = SEED

print(f"saved samples to {MODEL_DIR / MODEL_FILE}")
```

```{code-cell} ipython3

```
