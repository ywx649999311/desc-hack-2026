# Setting up the environment on a Mac (Apple silicon)

These steps build one conda environment, `desc-hack-2026`, that holds both the LSST Science Pipelines
(needed for the Rubin PSFs) and everything the notebooks use. It starts from the stack and adds the
project's packages on top, so the stack's own versions are never changed.

**Status:** done and checked on 2026-10-29 (macOS 27.0.1, Apple silicon, mambaforge with conda 22.9 / mamba 1.0).
All imports work, including `lsst.afw` and `lsst.meas.extensions.psfex`, both in an activated shell and
in a Jupyter kernel. `target_selection`, `roman_cutouts` and `rubin_cutouts` (with PSFs, 1 target) have run
in it; the other notebooks have not yet.

## Requirements

- Apple-silicon Mac and a conda install that uses conda-forge (mambaforge, miniforge or similar) and git.
- About 7 GB of disk for the environment (1 GB download).
- No compiler or system libraries are needed: everything comes prebuilt from conda-forge or pip.

## Steps

```bash
git clone https://github.com/ywx649999311/desc-hack-2026.git
cd desc-hack-2026

# 1. The LSST stack (stackvana = lsst_distrib as conda packages) plus the project's packages, in one solve
#    so conda picks versions that work with the stack's pins.
mamba create -y -n desc-hack-2026 -c conda-forge --override-channels \
  python=3.13 "stackvana=0.2026.39" \
  numpy scipy matplotlib pandas astropy astroquery "pyvo>=1.5" pyarrow s3fs fsspec hpgeom \
  "photutils>=2.0" "firefly-client>=3.4.0" reproject h5py seaborn gdown "dynesty<2.0.0" galsim \
  "jupytext>=1.19.4" jupyterlab jupyterlab-myst ipykernel ipython nbclient pytest nbval
conda activate desc-hack-2026

# 2. What conda-forge does not have, from pip. None of these change a conda package.
pip install --no-deps "astroquery>=0.4.10"   # the stack pins 0.4.9; pure Python, its dependencies are already there
pip install jupyter-firefly-extensions astro-sedpy astro-prospector
pip install --no-deps -e .                   # makes notebooks/ou_utils.py importable from any directory

# 3. Register the Jupyter kernel the notebooks are set to use
python -m ipykernel install --user --name desc-hack-2026 --display-name "Python (desc-hack-2026)"
```

### Make the kernel load the stack

The stack is set up by eups in the environment's activation scripts (about 100 `PYTHONPATH` entries plus
`DYLD_LIBRARY_PATH`). Jupyter starts a kernel's Python directly, without activating the environment, so
`import lsst.afw` fails in notebooks unless the kernel is started through an activation wrapper:

```bash
CONDA_BASE=$(conda info --base)
cat > "$CONDA_PREFIX/bin/desc-hack-kernel" <<EOF
#!/bin/bash
# Start Python with the conda env fully activated, so the LSST stack (set up by eups
# in the env's activation scripts) is importable from Jupyter kernels.
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate desc-hack-2026
exec python "\$@"
EOF
chmod +x "$CONDA_PREFIX/bin/desc-hack-kernel"

# Point the kernel at the wrapper instead of python
python - <<'EOF'
import json, os
path = os.path.expanduser("~/Library/Jupyter/kernels/desc-hack-2026/kernel.json")
k = json.load(open(path))
k["argv"][0] = os.path.join(os.environ["CONDA_PREFIX"], "bin", "desc-hack-kernel")
json.dump(k, open(path, "w"), indent=1)
EOF
```

Rerunning `ipykernel install` overwrites `kernel.json`, so redo this step after it.

## Check it worked

```bash
conda activate desc-hack-2026
python -c "import galsim, photutils, hpgeom, firefly_client, reproject, ou_utils, prospect, sedpy; print('imports ok')"
python -c "import lsst.afw.image, lsst.meas.extensions.psfex; print('stack ok')"

# The same through the kernel's launcher, from a bare shell (as Jupyter starts it)
env -i HOME=$HOME PATH=/usr/bin:/bin "$CONDA_PREFIX/bin/desc-hack-kernel" -c "import lsst.afw.image; print('kernel ok')"
```

`pip check` reports a few problems in the stack's own packages (idds, pgcli, sphinx-prompt, ws4py, maturin).
They come with stackvana, are not used here and can be ignored.

Then regenerate the files that are not in git (`data/` and `executed/` are gitignored):

```bash
jupytext --sync notebooks/*.md                         # optional: recreates the paired executed/*.ipynb twins
./run_notebooks.sh notebooks/target_selection.md       # run this first; writes data/targets.parquet
./run_notebooks.sh notebooks/roman_cutouts.md
./run_notebooks.sh notebooks/rubin_cutouts.md
```

`target_selection` takes about 5 minutes (most of it the Roman and Rubin coverage check). The cutout notebooks
default to 1 target; each Roman or Rubin image is large, so raise `MAX_TARGETS` once the first run looks right.

## Versions and differences from `requirements-lock.txt`

The lock file is for Python 3.12 and is not used here. Versions follow `requirements.txt` and the stack's pins.
As installed on 2026-10-29:

| Package | Here | Lock file |
|---|---|---|
| Python | 3.13.18 | 3.12 |
| numpy | 2.4.6 | 2.5.4 |
| pandas | 2.3.3 (the stack keeps it below 3) | 3.1.0 |
| astropy | 8.0.2 | 8.0.2 |
| astroquery | 0.4.11 (pip, over the stack's 0.4.9) | 0.4.11 |
| galsim | 2.8.5 (conda-forge, prebuilt) | 2.8.5 |
| photutils | 3.1.1 | 3.1.1 |
| dynesty | 1.2.3 | 1.2.3 |

- **pandas 2, not 3, and numpy 2.4, not 2.5.** The `roman_tds_supernova` fix for pandas 3 (`Table(...).to_pandas()`) also works with pandas 2.
- **Not included: `scarlet2`.** It was a separate editable install in the container
  (`pip install -e /path/to/scarlet2`, which also installs jax). Install it yourself if you need it; check
  `pip install --dry-run` first so it does not upgrade numpy or other packages the stack depends on.
- **Adding packages later:** prefer `mamba install -c conda-forge <pkg>`, so the solver respects the stack's
  pins. With pip, run `pip install --dry-run` first and check that it does not replace a conda package.
- **The optional SED-fit step** (running new Prospector fits with FSPS) is separate; see the README.

## Alternative: a plain venv without the stack

If you only need the notebooks and not the Rubin PSFs, the original route still applies:
`python3.12 -m venv .venv`, `pip install -r requirements-lock.txt`, `pip install -e .`, then the
`ipykernel install` command above (no wrapper is needed). That was checked with a pip dry run only.

See `memory.md` for the wider project notes and open items.
