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

# Light Curves with Laplace Uncertainties

This notebook loads the best-fit Roman and Rubin models saved by `scarlet2_fit_roman_rubin.md` and computes the transients' light curves with uncertainties, in seconds per target instead of the minutes of the posterior sampling in `scarlet2_sample_roman_rubin.md`.

It uses the Laplace approximation of _scarlet2_'s [real-world example](https://scarlet2.readthedocs.io/en/latest/0-example.html): around the best fit, the posterior is approximated by a normal distribution whose width follows from the curvature (Hessian) of the loss, and {py:func}`scarlet2.infer.get_scene_samples` draws scenes from it, on which we measure the fluxes.

```{note}
The real-world example gets the widths from {py:func}`scarlet2.infer.uncertainty`, which estimates the diagonal of the Hessian from random probes (Hutchinson's method). In our scenes that estimate fails for the transient: its curvature is tiny compared to the curvature of the thousands of morphology pixels and to its correlation with the host, so the probe noise swamps it, and the estimate comes out negative.
Instead, we compute the Hessian exactly for the parameters we care about, the amplitudes of the host and the transient, which is cheap because there are only a few of them.
```

```{code-cell} ipython3
# Import Packages and setup
import json
from pathlib import Path

import astropy.io.fits as fits
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import WCS

import ou_utils as ou
import scarlet2 as sc2

sc2.set_validation(False)
```

## Parameters

```{code-cell} ipython3
TARGET_IDS = None    # target IDs fitted by scarlet2_fit_roman_rubin.md; None = every saved model
NUM_SAMPLES = 1000   # scenes drawn from the Laplace approximation per target
SEED = 0

MODEL_DIR = ou.DATA_DIR / "scarlet2"
if TARGET_IDS is None:
    TARGET_IDS = sorted(int(p.name.split("_")[0]) for p in MODEL_DIR.glob("*_roman_rubin.h5"))
targets = pd.read_parquet(ou.TARGETS_PATH).set_index("id")
print("targets:", TARGET_IDS)
```

## Load Models and Observations

For every target we load the best-fit scene and the information saved with it (the astrometric offsets, the cutout files in epoch order, and the indices of the host and the transient).
To evaluate the likelihood we need the observation as well: we prepare the same cutouts in the same way as `scarlet2_fit_roman_rubin.md` (sky-subtracted Roman counts, Rubin calibrated to nJy), stack them onto the grid of the saved scene, and apply the fitted offsets.

```{code-cell} ipython3
def load_model(target_id):
    """Best-fit scene and the information saved with it by scarlet2_fit_roman_rubin.md."""
    path = MODEL_DIR / f"{target_id}_roman_rubin.h5"
    scene = ou.load_scene(path, id=0)
    with h5py.File(path, "r") as f:
        g = f["fit"]
        meta = dict(
            path=path,
            shift=g["shift"][()],
            mjd=g["mjd"][()],
            images=json.loads(g.attrs["images"]),
            channels=[tuple(c) for c in json.loads(g.attrs["channels"])],
            bands=json.loads(g.attrs["bands"]),
            active_epochs=json.loads(g.attrs["active_epochs"]),
            host_id=int(g.attrs["host_id"]),
            transient_id=int(g.attrs["transient_id"]),
            rubin_bad_bits=int(g.attrs["rubin_bad_bits"]),
            goodness_of_fit=float(g.attrs["goodness_of_fit"]),
        )
        # posterior samples from scarlet2_sample_roman_rubin.md, if it was run, for comparison
        meta["nuts_percentiles"] = f["samples/transient_flux_percentiles"][()] if "samples" in f else None
    return scene, meta


def load_cutout(instrument, path, rubin_bad_bits):
    """Image, weights, PSF and header of one cutout, prepared as in the fit."""
    with fits.open(path) as hdul:
        if instrument == "roman":
            sky = np.median(hdul["BKGNOISE"].data)
            data = np.nan_to_num(hdul["SCI"].data - sky)          # sky-subtracted counts
            var = hdul["VARIANCE"].data
            good = np.isfinite(var) & (var > 0) & (hdul["DQ"].data == 0)
            header = hdul["SCI"].header
        else:
            header = hdul["IMAGE"].header
            calib = header["CALIB"]                               # nJy per count
            data = np.nan_to_num(hdul["IMAGE"].data * calib)
            var = hdul["VARIANCE"].data * calib**2
            good = np.isfinite(var) & (var > 0) & ((hdul["MASK"].data & rubin_bad_bits) == 0)
        weights = np.where(good, 1 / np.where(good, var, 1), 0)
        return data, weights, hdul["PSF"].data, header


def load_observation(scene, meta):
    """The stacked observation of the fit, on the saved scene's grid, with the fitted offsets."""
    observations = []
    for (instrument, band, path), channel in zip(meta["images"], meta["channels"]):
        data, weights, psf, header = load_cutout(instrument, path, meta["rubin_bad_bits"])
        observations.append(sc2.Observation(
            data[None].astype(np.float32),
            weights=weights[None].astype(np.float32),
            psf=psf[None].astype(np.float32),
            wcs=WCS(header),
            channels=[channel],
            name=f"{instrument} {channel[1]}",
        ))
    obs = sc2.stack_observations(observations, scene.frame)
    obs.check_set_renderer(scene.frame)   # create the renderer, to set the shifts on its last step
    object.__setattr__(obs.renderer[-1], "shift", jnp.asarray(meta["shift"]))
    return obs


models = {}
for target_id in TARGET_IDS:
    scene, meta = load_model(target_id)
    obs = load_observation(scene, meta)
    gof = float(obs.goodness_of_fit(scene()))
    print(f"target {target_id}: {len(scene.sources)} sources, {len(meta['channels'])} epochs, "
          f"goodness of fit {gof:.3f} (saved {meta['goodness_of_fit']:.3f})")
    models[target_id] = (scene, obs, meta)
```

If the two goodness-of-fit numbers agree, the data and the model are those of the fit.

## Compute the Amplitude Uncertainties

The free amplitudes of the host (one per band) and of the transient (one per active epoch) are the parameters the transient's fluxes depend on most, and they are correlated: where the two overlap, light can move from one to the other.
We compute the Hessian of the negative log-likelihood with respect to all of these amplitudes jointly and exactly, with `jax.hessian`, at the best fit.
Its inverse is the covariance matrix of the Laplace approximation, whose diagonal gives the marginal variance of every amplitude, including the effect of the host-transient correlation.

```{code-cell} ipython3
def amplitude_covariance(scene, obs, source_ids):
    """Covariance of the spectrum amplitudes of `source_ids`, from the exact Hessian of the likelihood.

    Returns the covariance matrix (for all amplitudes, in the order of `source_ids`), and the
    marginal 1-sigma of each source's amplitudes.
    """
    data0 = [scene.sources[i].spectrum.data for i in source_ids]
    splits = np.cumsum([d.shape[0] for d in data0])[:-1]

    def loss(x):
        sources = list(scene.sources)
        for i, part in zip(source_ids, jnp.split(x, splits)):
            sources[i] = sources[i].replace("spectrum", sources[i].spectrum.replace("data", part))
        return -obs.log_likelihood(sum(scene.evaluate_source(s) for s in sources))

    hessian = np.asarray(jax.hessian(loss)(jnp.concatenate(data0)), dtype=np.float64)
    smallest = np.linalg.eigvalsh(hessian).min()
    if smallest <= 0:
        print(f"  warning: the Hessian is not positive definite (smallest eigenvalue {smallest:.3g}): "
              "the model is not at a minimum of the loss")
    cov = np.linalg.inv(hessian)
    sigma = np.sqrt(np.diag(cov))
    return cov, dict(zip(source_ids, np.split(sigma, splits)))


covariances, sigmas = {}, {}
for target_id, (scene, obs, meta) in models.items():
    ids = [meta["host_id"], meta["transient_id"]]
    covariances[target_id], sigmas[target_id] = amplitude_covariance(scene, obs, ids)
    print(f"target {target_id}: host amplitude sigma {np.round(sigmas[target_id][ids[0]], 2)}, "
          f"transient amplitude sigma {np.round(sigmas[target_id][ids[1]], 1)}")
```

## Draw Scenes and Measure Fluxes

{py:func}`scarlet2.infer.get_scene_samples` treats every parameter it is given as normally distributed, with the best-fit value as the mean and the given width, draws `NUM_SAMPLES` joint samples, and returns one scene per sample.
It only draws parameters that the scene declares as {py:class}`~scarlet2.Parameter`; a scene loaded from disk has none, so we declare the two amplitudes first (without priors or step sizes, which are not needed here).
On every drawn scene we measure the fluxes of the host and the transient, in counts for Roman and nJy for Rubin.

```{code-cell} ipython3
results = {}
for k, (target_id, (scene, obs, meta)) in enumerate(models.items()):
    host_id, transient_id = meta["host_id"], meta["transient_id"]
    with sc2.Parameters(scene):
        sc2.Parameter(scene.sources[host_id].spectrum.data, name="spectrum_host")
        sc2.Parameter(scene.sources[transient_id].spectrum.data, name="spectrum_transient")
    widths = {"spectrum_host": jnp.asarray(sigmas[target_id][host_id]),
              "spectrum_transient": jnp.asarray(sigmas[target_id][transient_id])}

    scenes = sc2.infer.get_scene_samples(scene, widths, samples=NUM_SAMPLES, key=jax.random.PRNGKey(SEED + k))
    transient_flux = np.array([np.asarray(sc2.measure.flux(s.sources[transient_id])) for s in scenes])
    host_flux = np.array([np.asarray(sc2.measure.flux(s.sources[host_id])) for s in scenes])

    results[target_id] = dict(
        best=np.asarray(sc2.measure.flux(scene.sources[transient_id])),     # (channels,)
        mean=transient_flux.mean(axis=0),
        std=transient_flux.std(axis=0),
        host_best=np.asarray(sc2.measure.flux(scene.sources[host_id])),
        host_std=host_flux.std(axis=0),
    )
```

Here are the transient's fluxes with their uncertainties, and for the calibrated Rubin epochs the AB magnitudes.
If `scarlet2_sample_roman_rubin.md` was run for a target, the last column shows the half-width of its 68% posterior interval for comparison; the two should be similar if the posterior is close to normal.

```{code-cell} ipython3
def ab_mag(flux_njy):
    """AB magnitude of a flux in nJy (NaN for non-positive fluxes)."""
    flux_njy = np.asarray(flux_njy, dtype=float)
    return np.where(flux_njy > 0, 31.4 - 2.5 * np.log10(np.where(flux_njy > 0, flux_njy, 1)), np.nan)


for target_id, (scene, obs, meta) in models.items():
    r = results[target_id]
    nuts = meta["nuts_percentiles"]
    print(f"\ntarget {target_id} ({targets.loc[target_id, 'model_name']})")
    print(f"{'epoch':>5s} {'band':>5s} {'MJD':>10s} {'flux':>11s} {'sigma':>9s} {'AB mag':>14s} {'NUTS 68%/2':>11s}")
    for k, (c, mjd) in enumerate(zip(meta["channels"], meta["mjd"])):
        if c[1] not in meta["active_epochs"]:
            continue                                   # fixed to zero before start_mjd
        line = f"{c[1]:5d} {c[0]:>5s} {mjd:10.3f} {r['best'][k]:11.1f} {r['std'][k]:9.1f}"
        if c[0] != meta["bands"][0]:                   # Rubin: nJy
            m = ab_mag(r["best"][k])
            dm = 2.5 / np.log(10) * r["std"][k] / r["best"][k] if r["best"][k] > 0 else np.nan
            line += f" {m:8.3f}+-{dm:.3f}"
        else:
            line += f" {'(counts)':>14s}"
        if nuts is not None:
            line += f" {(nuts[2][k] - nuts[0][k]) / 2:11.1f}"
        print(line)
    for b in meta["bands"]:
        k = [c[0] for c in meta["channels"]].index(b)
        print(f"host in {b}: {r['host_best'][k]:.1f} +- {r['host_std'][k]:.1f}")
```

## Light Curves

The Rubin light curve of a target with its Laplace uncertainties (in μJy, i.e. nJy / 1000), at every Rubin epoch, in the same style as the sampling notebook's light curve.
Before the catalog's `start_mjd` the transient is fixed to zero, not measured, so those points sit at zero without error bars.
The Roman image is in counts, not μJy, so it is listed in the table above but not plotted.

`plot_light_curve` plots one target and saves the figure as `figs/lc_laplace_<target_id>.pdf` next to this notebook (the sampling notebook's figures are `figs/lc_<target_id>.pdf`).
The transients are bright, so their errors are a fraction of a percent of the flux; as in the sampling notebook, the error bars are enlarged by `error_scale` (5 by default) to make them visible.
With `compare_samples=True`, the posterior medians and 68% intervals of the sampling notebook are shown too, where it was run, slightly offset in time.

```{code-cell} ipython3
FIG_DIR = Path(ou.__file__).parent / "figs"   # notebooks/figs, whatever the working directory


def plot_light_curve(target_id, error_scale=5, compare_samples=False, save=True):
    """Plot the Rubin light curve of `target_id` with Laplace uncertainties; save it to FIG_DIR.

    Same style as the light curve of scarlet2_sample_roman_rubin.md. `error_scale` multiplies the
    error bars, to make them visible next to the bright transient's flux (1 = 1-sigma).
    `compare_samples` adds the posterior medians and 68% intervals of the sampling notebook,
    if it was run for this target.
    """
    scene, obs, meta = models[target_id]
    r = results[target_id]
    band = meta["bands"][1]
    to_ujy = 1e-3                   # plot in μJy (nJy / 1000), so the tick labels stay short
    show = np.array([c[0] == band for c in meta["channels"]])   # all Rubin epochs (zero before start_mjd)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(meta["mjd"][show], r["best"][show] * to_ujy,
                yerr=error_scale * r["std"][show] * to_ujy,
                fmt="o", color="#2a78d6", capsize=3, label=f"Rubin {band}")
    nuts = meta["nuts_percentiles"]
    if compare_samples and nuts is not None:
        lo, med, hi = (p[show] * to_ujy for p in nuts)
        ax.errorbar(meta["mjd"][show] + 0.5, med, yerr=[error_scale * (med - lo), error_scale * (hi - med)],
                    fmt="s", mfc="none", color="#52514e", capsize=3, label="posterior samples")
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_xlabel("MJD", fontsize=13)
    ax.set_ylabel(f"Transient flux in {band} (μJy)", fontsize=13)
    ax.legend(frameon=True, fontsize=12)
    ax.tick_params(axis="both", labelsize=11, direction="in")
    if save:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"lc_laplace_{target_id}.pdf")
    return fig


for target_id in models:
    plot_light_curve(target_id)
    plt.show()
```

```{warning}
These uncertainties are conditional on everything that is held fixed: the morphologies of all sources, the amplitudes of the neighbors, the transient's position, and the astrometric offsets stay at their best-fit values, so the uncertainties are lower limits.
The Laplace approximation also assumes that the posterior is normal around the best fit; the comparison with the posterior samples tests that.
```

## Save Results

We add the fluxes, their uncertainties, and the amplitude covariance to the same file, in a group `laplace`:

```{code-cell} ipython3
for target_id, (scene, obs, meta) in models.items():
    r = results[target_id]
    with h5py.File(meta["path"], "a") as f:
        if "laplace" in f:
            del f["laplace"]
        g = f.create_group("laplace")
        g.create_dataset("transient_flux", data=r["best"])          # (channels,), best fit
        g.create_dataset("transient_flux_std", data=r["std"])
        g.create_dataset("host_flux", data=r["host_best"])
        g.create_dataset("host_flux_std", data=r["host_std"])
        g.create_dataset("amplitude_covariance", data=covariances[target_id])   # host bands, then transient epochs
        g.attrs["num_samples"] = NUM_SAMPLES
        g.attrs["seed"] = SEED
    print(f"saved to {meta['path']}")
```

```{code-cell} ipython3

```
