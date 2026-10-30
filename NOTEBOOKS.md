# Notebooks and scripts

What every notebook and script in this repository does, what it reads and writes, and in which order to run them.
For setting up the environment, see [`setup_on_mac.md`](setup_on_mac.md).

All notebooks are in `notebooks/`, as [MyST markdown](https://jupytext.readthedocs.io/en/latest/formats-markdown.html) files (see [Working with the notebooks](#working-with-the-notebooks)).
Everything they write goes to `data/` (gitignored) unless noted.

The notebooks come from two sources:

* [**From IRSA**](#from-irsa-the-openuniverse2024-tutorials): the OpenUniverse2024 tutorials written by IRSA, copied here so they can be rerun on their own.
* [**Written for this project**](#written-for-this-project): the target selection and cutout pipeline, and the scarlet2 scene modelling built on it.

## From IRSA: the OpenUniverse2024 tutorials

Written by IRSA and taken from [Caltech-IPAC/irsa-tutorials](https://github.com/Caltech-IPAC/irsa-tutorials) (`tutorials/simulated-data/OpenUniverse2024/`, commit `2f8e0295`), with only small local fixes (e.g. for pandas 3 in `roman_tds_supernova.md`) and the `openuniverse2024_` prefix dropped from the file names. Each stands on its own and reads its data from S3.

| Notebook | What it does |
|---|---|
| `quickstart.md` | Start here: the S3 layout, the parquet catalogs, and the image search (IRSA's simulated-data SIA service) that the other notebooks build on. |
| `firefly_visualization.md` | Compares Roman and Rubin images of one position in Firefly. |
| `roman_tds_supernova.md` | Follows a simulated SN Ia through the Roman Time Domain Survey and animates it. |
| `TDE_light_curve.md` | Finds a simulated TDE and builds a Roman J129/H158 light curve with aperture photometry. |
| `TDE_light_curve_Rubin.md` | Work in progress: the TDE notebook extended with Rubin r-band images (no Rubin light-curve plot yet). |
| `SED_fit.md` | Fits the SEDs of supernova host galaxies with Prospector. By default (`RUN_FITS = False`) it loads saved fit results; new fits need FSPS (see [below](#running-new-sed-fits)). |

## Written for this project

These were written here, building on the access patterns of the IRSA tutorials. They share the helper module `ou_utils.py`.

### Target selection and cutouts

These turn the transient catalog into cutout files on disk. Run them in this order.

| Notebook | What it does | Reads | Writes |
|---|---|---|---|
| `target_selection.md` | Selects transients by peak brightness, peak-minus-host magnitude and host half-light radius, and keeps those with Roman and Rubin images around the peak. | catalogs on S3 | `data/targets.parquet`, `data/targets.json` (the parameters used) |
| `roman_cutouts.md` | Finds the Roman TDS images around each target's peak and saves a FITS cutout of each, with variance, quality flags, background (`BKGNOISE`) and a PSF from `galsim.roman.getPSF`. | `data/targets.parquet`, images on S3 | `data/cutouts/roman/<band>/<target_id>/*.fits`, `data/cutouts/roman/manifest.parquet` |
| `rubin_cutouts.md` | The same for Rubin `calexp` images, with the image's own PSF model and flux calibration (`CALIB`, nJy per count), both read with the LSST Science Pipelines. | `data/targets.parquet`, images on S3 | `data/cutouts/rubin/...` |
| `explore_cutouts.md` | Opens one saved Roman and one Rubin cutout, lists its contents and plots the image, variance and PSF. | `data/cutouts/` | nothing |

Each notebook starts with a parameters cell; the cutout notebooks default to a small number of targets (`MAX_TARGETS`), so raise it once a first run looks right.
Cutouts keep the detector's native pixels (`NORTH_UP = False`), which the PSF stamps require.
Targets with no image in the time window are listed in the manifest rather than dropped.

### scarlet2 scene modelling

These model the transient and its host galaxy with [scarlet2](https://github.com/pmelchior/scarlet2), following its how-to guides (the transient and stacking tutorials): the host and its neighbors are static sources, the transient is a point source whose flux varies from epoch to epoch and is fixed to zero before the catalog's `start_mjd`, and all images are stacked onto one pixel grid.
They read the local cutouts in `data/cutouts/` only.
They need a recent scarlet2 (for `stack_observations`), installed from its GitHub repository, e.g. `pip install -e /path/to/scarlet2`.

| Notebook | What it does | Writes |
|---|---|---|
| `scarlet2_roman_joint_fit.md` | Joint fit of all Roman epochs of one target, with per-epoch astrometric offsets relative to the first epoch. Pixel-array morphologies; a starlet morphology with a scale-dependent prior is included but commented out. | nothing |
| `scarlet2_roman_rubin_joint_fit.md` | Joint fit of one Roman image (the reference grid and the source shapes) and all Rubin epochs (calibrated to nJy), with astrometric offsets of the Rubin images relative to Roman. | nothing |
| `scarlet2_fit_roman_rubin.md` | The Roman and Rubin fit as a first step of two: always fits the astrometric offsets, and saves the best-fit model. | `data/scarlet2/<target>_roman_rubin.h5` (scene, offsets, best-fit fluxes), `notebooks/figs/fit_<target>.pdf` |
| `scarlet2_sample_roman_rubin.md` | The second step: loads that model and samples the posterior with NUTS (as in the scarlet2 sampling tutorial) for the transient's fluxes and position and the host's fluxes, to get uncertainties; plots the light curve with error bars. | samples added to the same `.h5` file, `notebooks/figs/lc_<target>.pdf` |

The sampled uncertainties are conditional on what is not sampled (the morphologies, the neighbors and the astrometric offsets stay at their best fit), so they are lower limits.

## Scripts and configuration

All written for this project.

| File | What it does |
|---|---|
| `notebooks/ou_utils.py` | Shared helpers: catalog reading, image search, cutouts with variance, flags and PSF, the coverage check, plotting, and saving/loading scarlet2 scenes to HDF5 (`save_scene`, `load_scene`). |
| `run_notebooks.sh` | Executes notebooks from the command line and writes the executed copies to `executed/`: `./run_notebooks.sh` for all of them, or `./run_notebooks.sh notebooks/<name>.md` for one. |
| `pyproject.toml` | Makes `ou_utils.py` installable (`pip install -e .`), so every notebook can import it from any working directory. |
| `jupytext.toml` | Pairs each `notebooks/<name>.md` with `executed/<name>.ipynb` (see below). |
| `requirements.txt`, `requirements-lock.txt` | Python dependencies: lower bounds, and exact versions of a Python 3.12 environment without the LSST stack. |
| `setup_on_mac.md` | How to build the environment: one conda environment with the LSST stack (needed for the Rubin PSFs and calibration), or a plain venv without it. |
| `memory.md` | Running project notes: decisions, data gotchas, the state of the work and open items. |

## Working with the notebooks

### Running

Interactively: `jupyter lab`, then open a notebook from `notebooks/` (with `jupytext` installed, right-click → *Open With* → *Notebook*).
From the command line: `./run_notebooks.sh` as above. Most notebooks take a few minutes, mostly spent reading from S3.

### Editing in VS Code

`jupytext.toml` pairs each `notebooks/<name>.md` with `executed/<name>.ipynb`.
The `.ipynb` files are local only (gitignored); the `.md` files are the source of truth.

```bash
jupytext --sync notebooks/*.md          # create or refresh the .ipynb twins
# ... edit executed/<name>.ipynb in VS Code with the "Python (desc-hack-2026)" kernel ...
jupytext --sync executed/<name>.ipynb   # write your changes back to the .md
```

`jupytext --sync` keeps whichever of the two files is newer, so sync before editing the other one.
Do not edit the `.md` files through the *vscode-jupytext* extension's notebook view: it can open them but cannot save them.

### Running new SED fits

`SED_fit.md` with `RUN_FITS = True` needs FSPS, which compiles Fortran code and needs a few GB of data:

```bash
git clone --depth=1 https://github.com/cconroy20/fsps
export SPS_HOME="$PWD/fsps"
pip install fsps
```
