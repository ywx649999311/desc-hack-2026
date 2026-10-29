# Project memory: desc-hack-2026

Running notes from the working sessions on 2026-10-28 and 2026-10-29, so the next session can pick up
without re-deriving things. Update it as the project changes.

## What this repo is

A standalone copy of the OpenUniverse2024 tutorial notebooks from Caltech-IPAC/irsa-tutorials
(commit `2f8e0295`, plus local edits), so they can be rerun on their own, plus three new notebooks that
select transient targets and cut out their Roman and Rubin images.

Data are read anonymously from `s3://nasa-irsa-simulations/` and IRSA's simulated-data SIA service.

## Environment

**Since 2026-10-29 the work runs natively on the Mac** (macOS 27.0.1, Apple silicon), no longer in the Linux
container. Full steps are in `setup_on_mac.md`.

- **One conda env, `desc-hack-2026`, built on the LSST stack** (user's choice, 2026-10-29: start from the stack
  env and install the project's packages into it, rather than two separate envs). It lives in the user's
  mambaforge base (`/Users/weixiang/mambaforge`, conda 22.9 / mamba 1.0, conda-forge only).
  - The stack is `stackvana=0.2026.39` (weekly 2026.39 of `lsst_distrib` as conda-forge packages), Python 3.13.
  - The project's packages were solved together with the stack, so their versions follow the stack's pins:
    numpy 2.4.6, pandas 2.3.3 (stack keeps pandas < 3), astropy 8.0.2, galsim 2.8.5 (conda-forge), photutils 3.1.1,
    dynesty 1.2.3.
  - From pip: astroquery 0.4.11 with `--no-deps` (the stack pins 0.4.9; the notebooks want >= 0.4.10),
    `jupyter-firefly-extensions`, `astro-sedpy`, `astro-prospector` (none of these changed a conda package),
    and `pip install --no-deps -e .`.
  - **Kernel gotcha:** eups sets up the stack in the env's activation scripts (~100 `PYTHONPATH` entries plus
    `DYLD_LIBRARY_PATH`), so a kernel started as plain `python` cannot import `lsst`. The kernel `desc-hack-2026`
    therefore starts through `~/mambaforge/envs/desc-hack-2026/bin/desc-hack-kernel`, which activates the env
    and then runs python. Rerunning `ipykernel install` overwrites `kernel.json`; redo the wrapper step after it.
  - `pip check` complaints (idds, pgcli, sphinx-prompt, ws4py, maturin) come with the stack and are harmless here.
  - Install new packages with `mamba install -c conda-forge` where possible; dry-run any pip install so it does
    not replace a conda package the stack depends on.
- `requirements.txt` (lower bounds) still describes the project's needs. `requirements-lock.txt` (exact versions,
  Python 3.12) is from the container venv and is **not** what the Mac env uses; it is kept for the plain-venv route.
  It never included the user's editable `scarlet2` and its jax stack, so do not regenerate it with a blind `pip freeze`.
  `scarlet2` is not installed in the Mac env yet.
- `pip install -e .` makes `notebooks/ou_utils.py` importable from any working directory, including the
  paired notebooks in `executed/`.
- `jupytext --execute` runs from the repo root, not the notebook folder, which is why the helper module is
  installed (`pip install -e .`) rather than imported by relative path.
- Old container setup (no longer used): Linux ARM guest, Python 3.12 venv in `.venv`, galsim built from source
  against FFTW (commands in `requirements.txt`).

## Notebooks (`notebooks/`, MyST markdown, paired with `executed/<name>.ipynb`)

Originals (prefix `openuniverse2024_` removed): `quickstart`, `firefly_visualization`,
`roman_tds_supernova` (patched to build its DataFrame via `Table(...).to_pandas()` because pandas 3
rejects big-endian FITS buffers), `TDE_light_curve`, `TDE_light_curve_Rubin` (draft, see below), `SED_fit`.
Only `quickstart` and `roman_tds_supernova` were verified end to end, and only in the old container.

New pipeline, run in order:

| Notebook | Writes |
|---|---|
| `target_selection.md` | `data/targets.parquet` + `targets.json` (parameters used) |
| `roman_cutouts.md` | `data/cutouts/roman/<band>/<target_id>/*.fits` + `manifest.parquet` |
| `rubin_cutouts.md` | `data/cutouts/rubin/...` |

Shared helpers are in `notebooks/ou_utils.py`. Current parameter choices (all in each notebook's first cell):

- Selection: catalog band `i`, `PEAK_MAG_MAX = 23`, no contrast or size cut, ranked by host half-light
  radius (larger of disk and spheroid), largest first, `REQUIRE_IN_TDS = True`. Coverage required: Roman J129
  and Rubin r images within peak MJD ± 30 days. Stops at 20 targets. Only the TDS-centre HEALPix region is
  searched; add regions via `REGIONS`.
- Cutouts: 20 arcsec wide, **centred on the host** (`CENTER_ON = "host"`), native orientation
  (`NORTH_UP = False`), `MAX_TARGETS = 5`, up to 5 epochs per target. A cyan ring marks the transient in the montage.
- Each cutout FITS file: primary HDU is the science image, extensions hold the error and quality arrays
  (Roman `SCI`/`ERR`/`DQ`, Rubin `IMAGE`/`VARIANCE`/`MASK`), same pixel grid and WCS (with SIP). Header keys:
  `TARGETID BAND MJD OBSID SRCIMG TRA TDEC CENTERON SIZEARCS NORTHUP LAYERS NANFRAC`.
  Flags are never interpolated (nearest neighbour when north-up); flags are 0 off the detector.
- Resume: existing cutouts are skipped only if size, centring, rotation and `LAYERS` all match; otherwise rewritten.
  Targets with no image in the window are listed in the manifest as `no images in window`.

## Data gotchas found

- `snana` already has `peak_mag_{g,i,F}` and `host_mag_{g,i,F}`. `F` is closest to Roman F184 in the flux catalog.
- `host_mag_* = 999` is a placeholder. `FIXMAG` rows (peak exactly 15.005) are a calibration model.
  Some host IDs are missing from the galaxy catalog. Host size is only in the galaxy catalog.
- Rubin visits are sparse in time, so many targets have no image within ±30 days of peak. Roman needs the
  `TDS_simple_model` images and only covers the survey field and dates.
- The transient is often several arcsec off its host (0.4 to 6.7 arcsec for the largest hosts). That is why
  cutouts are host-centred and wide; transient-centred cutouts looked "off-centre" to the user.
- Roman cutouts come out 93 or 95 pixels square (detector pixel scales differ slightly); Rubin 101. Realised
  widths are within about 2% of the requested size (Roman 19.96 to 20.14, Rubin about 20.2 arcsec).

## PSFs

- The simulated Roman images are convolved with GalSim's Roman PSF (`galsim.roman.getPSF`, 5 wavelengths,
  plus charge diffusion; see `tds.yaml` in the bucket). It is chromatic, so drawing it needs a source SED.
  Tested with a flat photon SED as a placeholder. Roman files contain no PSF.
- Rubin `calexp` files carry the PSF as two `PsfexPsf` extensions (11 and 12) and `PSF_ID` in the header.
  The model is a PSFEx polynomial basis; the basis array has 41,334 values, which would fit 6 components of
  83x83 pixels (inferred from the length, not decoded). Evaluating it needs the LSST Science Pipelines (`lsst.afw`).
- **The LSST stack cannot be installed in this container**: conda-forge has only `lsst-sphgeom`, `lsst-utils`
  and `lsst-resources` for Linux ARM, nothing on PyPI, and the macOS Apple-silicon prebuilt does not run in a
  Linux guest. Options offered: run the stack natively on the Mac and save PSF stamps (recommended),
  run an x86 Linux container under emulation, or decode `PsfexPsf` in numpy (unverified against the stack).
  **Decided 2026-10-29: run the stack natively on the Mac.** The first plan was a separate stack environment
  exchanging only files with the desc venv, because the lock file pins newer numpy, pandas etc. than the stack.
  **Superseded the same day:** the user preferred one env built on the stack, with the project's packages
  solved against the stack's pins (see Environment). That is now installed and `lsst.afw` imports in it.
- How `/home/jovyan/work/DDP3_Scarlet2_demo` handles it: reads the image with the LSST stack,
  `exposure.getPsf().computeImage(pos)` at the cutout centre (default) or patch centre, normalised to sum 1 and
  passed to scarlet2 as `ArrayPSF` per band. The model frame uses a separate narrower Gaussian PSF
  (0.2 arcsec sigma in Rubin mode) or the Euclid PSF. Each band is rendered through the kernel
  `obs PSF / model PSF`. Things to check there: PSF stamp centring (scarlet2 wants odd size, centred), and the
  Gaussian fallback (FWHM 0.7 arcsec) used silently when the stack is missing. In the per-visit SN script,
  later epochs are evaluated at the image centre, with x and y swapped, not at the target.

### Planned: a PSF image in every cutout (agreed 2026-10-29, NOT implemented yet)

Two stages that pass files to each other; one PSF per cutout (per epoch), evaluated at the cutout centre (the host).

1. desc env: make the cutouts as now, and also record each target's pixel position on the source image
   (header and manifest), which both PSF models need.
2. desc env: Roman PSFs with galsim. Detector (SCA) and band come from the image name, e.g.
   `Roman_TDS_simple_model_J129_29063_9` is pointing 29063, SCA 9. Draw on the cutout's native pixel grid,
   odd size, centred, sum 1, with the source SED.
3. desc env: write a plain CSV of Rubin requests (source image, position, output name) and download the needed
   `calexp` files to a local cache (a few GB for large target lists; deletable afterwards). The stack env then
   needs no S3 access and no libraries beyond numpy.
4. stack env, native on the Mac: a small standalone script reads the CSV, calls
   `ExposureF.readFits(file).getPsf().computeImage(pos)`, centres and normalises the stamp on an odd grid and
   saves one `.npy` per cutout. It can only be syntax-checked in the container; run it first on one cutout and
   check the stamp.
5. desc env: append a `PSF` extension to each cutout FITS (header keys: source, position, stamp scale, SED),
   skipping cutouts that already have one, so each cutout stays one self-contained file.

**With the single env, steps 3 and 4 can be simpler** (not yet discussed with the user): the cutout code can
call the stack directly, so the request CSV, the separate script and the `.npy` hand-off are no longer needed.
The calexp files still have to be read locally (`ExposureF.readFits` takes a path), so a download cache stays.

Defaults agreed: everything runs natively on the Mac sharing one repo folder; PSF stored as a FITS extension
(not separate `.npy` files); flat photon SED as a placeholder for Roman (a redder per-target SED broadens the
PSF slightly, so revisit it if the PSF is used in fits). Keep cutouts in native orientation, because the PSF
comes out on the detector pixel grid; for `NORTH_UP = True` the PSF would also have to be rotated.

## State of the work

- Git: `origin` is `https://github.com/ywx649999311/desc-hack-2026.git`, branch `main`, and `gh` is logged in as
  `ywx649999311`. As of the push on 2026-10-29 everything is committed and pushed (see `git log`). In the container
  the commit author was `astro.flyfish@gmail.com`, set per repo; the Mac clone commits as the GitHub noreply
  address `11515291+ywx649999311@users.noreply.github.com`.
- `notebooks/TDE_light_curve.md` carries a cosmetic front-matter change (a jupytext 1.16.6 stamp) that was not
  made by me; it was committed along with everything else.
- `setup_on_mac.md` has the steps used to build the single stack + project env on the Mac (done and import-checked
  2026-10-29; a one-cell test notebook ran through the kernel and imported `lsst.afw`).
- The Mac clone is fresh: no `data/` or `executed/` yet, and **none of the repo's notebooks have been run in the
  Mac env**. So the cutout notebooks have also not been rerun since the error/quality layers were added (the
  user had asked not to run them yet). The layer code was tested on synthetic images only.
- `jupytext.toml` excludes `widgets` from synced metadata, after tqdm progress-bar state once bloated two notebooks.

## Open items

1. Run the notebooks in the Mac env (pandas 2.3 and Python 3.13 there, versus pandas 3 and 3.12 in the container):
   `target_selection` first, then `roman_cutouts` and `rubin_cutouts`, checking the new `ERR`/`DQ` and
   `VARIANCE`/`MASK` extensions on real data.
2. Implement the PSF plan above: record pixel positions, galsim Roman PSFs, Rubin PSFs from the stack (decide with
   the user whether to drop the CSV/script hand-off now that it is one env), and a `PSF` extension in each cutout.
3. Pick a better SED than the flat placeholder for Roman PSFs of real targets, if the PSF is used in fits.
4. Optional: a minimum-image-count requirement in the coverage check; matching Roman and Rubin cutouts to an
   exactly common pixel grid if needed.
5. Install `scarlet2` in the Mac env if needed (dry-run first; it brings jax).
6. `TDE_light_curve_Rubin.md` is still a draft: shared `aperture_radius_pix` bug, scratch cell, no Rubin light-curve plot.
7. The README's setup section still describes the venv and says the stack is unavailable; update it to point at
   `setup_on_mac.md` if wanted.
