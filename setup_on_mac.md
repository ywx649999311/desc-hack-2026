# Recreating the environment on a Mac (Apple silicon, outside the container)

These steps rebuild the `desc-hack-2026` environment natively on macOS, without Docker.

**Status:** not yet tried on a Mac. What was checked (2026-10-29, from the Linux ARM container): pip resolved
all 177 pinned packages in `requirements-lock.txt` against Apple-silicon wheels for Python 3.12, without
installing anything. Treat a failure at any step as news worth reporting back.

## Requirements

- Apple-silicon Mac running **macOS 14 (Sonoma) or newer**. The pinned numpy, scipy, pyarrow and galsim
  wheels are only published from the macOS 14 tag; below that, some would have to be built from source.
- Python 3.12 (Homebrew, python.org, conda or uv all work) and git.
- No compiler, FFTW or other system libraries are needed. Everything has a prebuilt wheel except
  `PIMS==0.7`, which is pure Python (no compiled files), so pip builds it without a compiler.
  It is pulled in by `dask-image`.

## Steps

```bash
git clone https://github.com/ywx649999311/desc-hack-2026.git
cd desc-hack-2026

python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install -r requirements-lock.txt    # exact versions from the working environment
pip install -e .                        # makes notebooks/ou_utils.py importable from any directory

# Register the Jupyter kernel the notebooks are set to use
python -m ipykernel install --user --name desc-hack-2026 --display-name "Python (desc-hack-2026)"
```

With conda instead of venv: `conda create -n desc-hack-2026 python=3.12`, `conda activate desc-hack-2026`,
then the same `pip` and `ipykernel` commands.

## Check it worked

```bash
python -c "import galsim, photutils, hpgeom, firefly_client, reproject, ou_utils; print('imports ok')"
```

Then regenerate the files that are not in git (`data/` and `executed/` are gitignored):

```bash
jupytext --sync notebooks/*.md                         # optional: recreates the paired executed/*.ipynb twins
./run_notebooks.sh notebooks/target_selection.md       # run this first; writes data/targets.parquet
./run_notebooks.sh notebooks/roman_cutouts.md
./run_notebooks.sh notebooks/rubin_cutouts.md
```

`target_selection` takes about 5 minutes (most of it the Roman and Rubin coverage check). The cutout notebooks
default to 5 targets; each Roman or Rubin image is large, so raise `MAX_TARGETS` once the first run looks right.

## Differences from the container

- **Not in the lock file: `scarlet2`.** It was a separate editable install in the container
  (`pip install -e /path/to/scarlet2`, which also installs jax). Install it yourself if you need it.
- **Galsim needs no special build on the Mac.** In the Linux ARM container it had to be compiled against FFTW;
  `requirements.txt` has those commands in a comment if you ever need them on Linux ARM.
- **If a pinned version will not install**, fall back to `pip install -r requirements.txt` (lower bounds only),
  then `pip install -e .` as above.
- **The optional SED-fit step** (running new Prospector fits with FSPS) is separate; see the README.

## Rubin PSFs: the LSST stack goes in its own environment

Evaluating the Rubin PSF stored in each `calexp` file needs the LSST Science Pipelines (`lsst.afw`), which is
not on PyPI or conda-forge for Linux ARM (hence not in the container) and is not part of this venv.

- On the Mac, install it natively with LSST's `lsstinstall`, which has a macOS Apple-silicon prebuilt. Not tried yet.
- Keep it in its own environment; do not mix it with the `desc-hack-2026` venv.
- A container on a Mac is a Linux guest, so the macOS prebuilt cannot run inside it.
- The PSF-extraction script (read each Rubin image from S3, evaluate `exposure.getPsf().computeImage(pos)` at
  the target, save a `.npy` stamp next to each cutout) would run in that environment. It has not been written yet.

See `memory.md` for the wider project notes and open items.
