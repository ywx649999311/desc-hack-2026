# desc-hack-2026

Tutorial notebooks for the [OpenUniverse2024](https://doi.org/10.1093/mnras/staf1833)
simulated Roman and Rubin images and catalogs that IRSA hosts. They were taken from
[Caltech-IPAC/irsa-tutorials](https://github.com/Caltech-IPAC/irsa-tutorials)
(`tutorials/simulated-data/OpenUniverse2024/`, commit `2f8e0295`) so they can be rerun on their own.

All data are read directly from the public `s3://nasa-irsa-simulations/` bucket and
IRSA's simulated-data SIA service, so no AWS account or local data download is needed.

## Notebooks

| Notebook | What it does |
|---|---|
| `quickstart.md` | Start here: S3 layout, parquet catalogs, and the image search the others build on |
| `firefly_visualization.md` | Compare Roman and Rubin images of one position in Firefly |
| `roman_tds_supernova.md` | Follow a simulated SN Ia through the Roman Time Domain Survey and animate it |
| `TDE_light_curve.md` | Find a simulated TDE and build a Roman J129/H158 light curve with aperture photometry |
| `TDE_light_curve_Rubin.md` | Work in progress: the TDE notebook extended with Rubin r-band images |
| `SED_fit.md` | Fit SEDs of supernova host galaxies with Prospector |

### Target selection and cutouts pipeline

Three short notebooks, run in this order, turn the transient catalog into cutout files on disk.
They share the helper module `notebooks/ou_utils.py`.

| Notebook | What it does | Writes |
|---|---|---|
| `target_selection.md` | Select transients by peak-minus-host magnitude and host half-light radius | `data/targets.parquet` (+ `targets.json` with the parameters used) |
| `roman_cutouts.md` | Find Roman TDS images around each target's peak and save FITS cutouts | `data/cutouts/roman/<band>/<target_id>/*.fits`, `data/cutouts/roman/manifest.parquet` |
| `rubin_cutouts.md` | The same for Rubin `calexp` images | `data/cutouts/rubin/...` |

Each notebook starts with a parameters cell. The cutout notebooks default to `MAX_TARGETS = 5`;
raise it (or set `None`) once the first run looks right. Cutouts keep the detector's native
pixels by default (`NORTH_UP = False`); each FITS header carries the WCS, target ID, band and MJD.
Every cutout also stores the error/variance and quality-flag arrays as extra extensions (Roman `ERR`, `DQ`; Rubin `VARIANCE`, `MASK`).
Targets with no image in the time window are listed in the manifest, not dropped silently.
Rubin visits are sparse in time, so some targets will have none. `data/` is gitignored.

The notebooks are [MyST markdown](https://jupytext.readthedocs.io/en/latest/formats-markdown.html)
files. With `jupytext` installed, JupyterLab opens them directly as notebooks
(right-click → *Open With* → *Notebook*).

### Editing in VS Code

`jupytext.toml` pairs each `notebooks/<name>.md` with `executed/<name>.ipynb`.
The `.ipynb` files are local only (gitignored); the `.md` files are the source of
truth. `run_notebooks.sh` writes to the same `executed/` paths, so the twin is the
executed notebook with its outputs.

```bash
source .venv/bin/activate
jupytext --sync notebooks/*.md          # create or refresh the .ipynb twins
# ... edit executed/<name>.ipynb in VS Code with the "Python (desc-hack-2026)" kernel ...
jupytext --sync executed/<name>.ipynb   # write your changes back to the .md
```

Do not edit the `.md` files through the *vscode-jupytext* extension's notebook
view: it can open them but cannot save them (`Format 'percent' is not associated
to extension '.md'`). Also turn off format-on-save for notebooks if you plan to
send changes upstream, or the whole file gets reformatted.

## Set up the environment

Python 3.12 is the tested version (upstream tests 3.10–3.13).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Exact versions known to work:
pip install -r requirements-lock.txt
# ...or the latest compatible versions:
# pip install -r requirements.txt

# Make the shared helper module (notebooks/ou_utils.py) importable from any notebook
pip install -e .

# Register the Jupyter kernel the notebooks are set to use
python -m ipykernel install --user --name desc-hack-2026 --display-name "Python (desc-hack-2026)"
```

With conda instead of venv, run `conda create -n desc-hack-2026 python=3.12` and
`conda activate desc-hack-2026`, then follow the same `pip` and `ipykernel` steps.

### Optional: simulating the Roman PSF

`galsim` (in `requirements.txt`) provides `galsim.roman.getPSF(SCA, bandpass)`, the model the
Roman simulation used. The Roman PSF is chromatic, so drawing it needs a source SED. On Linux
ARM there is no prebuilt wheel and pip compiles it, which needs the FFTW headers; the exact
commands are in the comment in `requirements.txt`.

The Rubin PSF is stored in each `calexp` image as `PsfexPsf` extensions. Evaluating it needs the
LSST Science Pipelines (`lsst.afw`), which is not available as a conda or pip package on Linux ARM.

### Optional: running new SED fits

`SED_fit.md` loads saved fit results by default (`RUN_FITS = False`).
Running new fits with `RUN_FITS = True` also needs FSPS, which compiles Fortran code
and needs a few GB of data:

```bash
git clone --depth=1 https://github.com/cconroy20/fsps
export SPS_HOME="$PWD/fsps"
pip install fsps
```

## Run the notebooks

Interactively: `jupyter lab`, then open a notebook from `notebooks/`.

From the command line (executed copies are written to `executed/`):

```bash
source .venv/bin/activate
./run_notebooks.sh                             # all notebooks
./run_notebooks.sh notebooks/quickstart.md   # just one
```

Each notebook takes a few minutes, mostly spent reading from S3.

## License

BSD 3-Clause, from the original repository. See `LICENSE`.
