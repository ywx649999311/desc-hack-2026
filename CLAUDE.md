# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Notebooks that select transients and their host galaxies from the OpenUniverse2024 simulated Roman and Rubin surveys, cut out their images, and measure light curves by modelling each scene jointly across epochs and instruments with scarlet2. Data are read anonymously from `s3://nasa-irsa-simulations/` and IRSA's simulated-data SIA service.

- `NOTEBOOKS.md`: what each notebook reads/writes and the run order.
- `setup_on_mac.md`: building the environment.
- `memory.md`: running project notes (decisions, data gotchas, state of the work, open items). Read it before non-trivial work and update it when decisions or findings change.

## Environment

One conda env, `desc-hack-2026` (Python 3.13), built on the LSST stack (`stackvana`), with the project's packages solved against the stack's pins (pandas < 3, numpy 2.4). Activate with `conda activate desc-hack-2026`.

- `lsst.afw` is needed for Rubin PSFs/calibration. The Jupyter kernel `Python (desc-hack-2026)` starts through a wrapper script (`$CONDA_PREFIX/bin/desc-hack-kernel`) so eups sets up the stack; rerunning `ipykernel install` overwrites `kernel.json` and breaks this (see `setup_on_mac.md`).
- Add packages with `mamba install -c conda-forge` where possible; dry-run any `pip install` so it does not replace a conda package the stack depends on.
- `scarlet2` is an editable install of `other_repo/scarlet2`. `other_repo/` is the user's untracked clones (`scarlet2`, `DDP3_Scarlet2_demo`); never commit it.
- `requirements-lock.txt` is from an old Python 3.12 venv, not this env; don't regenerate it with a blind `pip freeze`.

## Commands

```bash
./run_notebooks.sh                                  # execute every notebook -> executed/<name>.ipynb
./run_notebooks.sh notebooks/target_selection.md    # execute one notebook
jupytext --sync notebooks/*.md                      # create/refresh the paired .ipynb twins
jupytext --sync executed/<name>.ipynb               # write .ipynb edits back to the .md
pip install --no-deps -e .                          # makes notebooks/ou_utils.py importable everywhere
```

There is no test suite or linter. "DM stack could not be deactivated!" from `run_notebooks.sh` is harmless.

## Notebook format

- `notebooks/*.md` (MyST markdown via jupytext) are the source of truth. `jupytext.toml` pairs each with `executed/<name>.ipynb`, which is gitignored and holds outputs. `jupytext --sync` keeps whichever file is newer, so sync before editing the other.
- Notebooks run from the repo root (`jupytext --execute`), which is why `ou_utils` is pip-installed rather than imported by relative path.
- Each notebook starts with a parameters cell (e.g. `MAX_TARGETS`, selection cuts). The user has asked for line-by-line explanatory comments in the notebooks.
- Everything written goes to `data/` (gitignored); figures go to `notebooks/figs/`.

## Architecture

Two groups of notebooks (see `NOTEBOOKS.md` for the full table):

1. **IRSA tutorials** (`quickstart`, `firefly_visualization`, `roman_tds_supernova`, `TDE_light_curve*`, `SED_fit`): copied from Caltech-IPAC/irsa-tutorials with small local fixes; standalone.
2. **Project pipeline**, run in order, all sharing `notebooks/ou_utils.py`:
   - `target_selection` → `data/targets.parquet` (+ `targets.json` with the parameters used). Cuts on peak mag, peak-minus-host mag, host half-light radius, then a Roman+Rubin coverage check around peak.
   - `roman_cutouts` / `rubin_cutouts` → `data/cutouts/<instrument>/<band>/<target_id>/*.fits` + `manifest.parquet`. Both go through `ou.run_cutouts(...)`.
   - `scarlet2_fit_roman_rubin` → `data/scarlet2/<target>_roman_rubin.h5` (best-fit scene). Then either `scarlet2_sample_roman_rubin` (NUTS) or `scarlet2_laplace_roman_rubin` (Laplace approximation, all saved models) adds uncertainties to the same `.h5`.
   - `scarlet2_roman_joint_fit`, `scarlet2_roman_rubin_joint_fit`, `scarlet2_roman_rubin_joint_fit_starlet` (starlet morphologies with a noise-scaled Cauchy prior): exploratory fits that write nothing.

### Cutout file contract

Each cutout FITS has the same HDU order for both instruments: image, variance, flags, then extras. Roman: `SCI, VARIANCE, DQ, BKGNOISE, PSF`; Rubin: `IMAGE, VARIANCE, MASK, PSF` (`ou.CUTOUT_ORDER`, `ou.CUTOUT_LAYERS`). Rubin headers carry `CALIB` (nJy per count), needed to put visits on one flux scale. Cutouts are host-centred, native pixel orientation (`NORTH_UP = False`; `psf=True` with north-up raises). Existing cutouts are skipped only if size, centring, rotation, layers and PSF size match (`_is_current`).

### Non-obvious data facts

- Roman `ERR` is **not** an uncertainty; it is background + noise without sources. It is saved as `BKGNOISE`, and `VARIANCE` is computed by `ou.roman_variance` (saturated pixels → inf).
- Roman PSF: `galsim.roman.getPSF` + charge diffusion, drawn with a flat-SED placeholder, 51×51. Rubin PSF: PSFEx kernel image from the calexp via `lsst.afw`, 41×41 (the model's own size). Calexps are cached in `data/cache/` then deleted.

### scarlet2 conventions and gotchas

- The transient is a point source fixed to zero flux before the catalog's `start_mjd`; only active epochs are fitted/sampled. Sampling pre-explosion flux was tried and is too slow; don't retry without a new idea.
- Save/load scenes with `ou.save_scene` / `ou.load_scene`, not `scarlet2.io.model_to_h5` (HDF5 attribute 64 kB limit).
- Spectrum objects can only be created inside `with sc2.Scene(...)`; elsewhere use `.replace("data", ...)`.
- Parameter names are sanitized (`"center:transient"` → `"center_transient"`).
- A freshly stacked observation needs `check_set_renderer(frame)` before setting `renderer[-1].shift`.
- More detail and open items are in `memory.md` ("scarlet2 modelling").

## Git

`notebooks/figs/` PDFs: ask before committing new ones. Never commit `other_repo/`, `data/`, or `executed/`.
