# Project memory: desc-hack-2026

Running notes from the working sessions on 2026-10-28 and 2026-10-29, so the next session can pick up
without re-deriving things. Update it as the project changes.

## What this repo is

A standalone copy of the OpenUniverse2024 tutorial notebooks from Caltech-IPAC/irsa-tutorials
(commit `2f8e0295`, plus local edits), so they can be rerun on their own, plus three new notebooks that
select transient targets and cut out their Roman and Rubin images.

Data are read anonymously from `s3://nasa-irsa-simulations/` and IRSA's simulated-data SIA service.

## Environment

- Python 3.12 venv in `.venv` (gitignored; the user said never to commit it). Jupyter kernel `desc-hack-2026`.
- `requirements.txt` (lower bounds) and `requirements-lock.txt` (exact versions). The lock file was
  built from a clean install; only galsim and its dependencies were added since. The venv also holds the
  user's editable `scarlet2` and its jax stack. Those are not in the lock file, so do not regenerate it
  with a blind `pip freeze`.
- `pip install -e .` makes `notebooks/ou_utils.py` importable from any working directory, including the
  paired notebooks in `executed/`.
- The container is a Linux guest (`Linux ... aarch64`, linuxkit kernel) on a Mac with Apple silicon.
- **galsim 2.8.5** is installed. There is no prebuilt wheel for Linux ARM, so it was built from source against
  FFTW in `/home/jovyan/.local/fftw-prefix` (conda-forge). Build commands are in `requirements.txt`.
- **`gh` 2.103.3** is installed in the base conda env but the user has not logged in. No git remote yet.
- `jupytext --execute` runs from the repo root, not the notebook folder, which is why the helper module is
  installed (`pip install -e .`) rather than imported by relative path.

## Notebooks (`notebooks/`, MyST markdown, paired with `executed/<name>.ipynb`)

Originals (prefix `openuniverse2024_` removed): `quickstart`, `firefly_visualization`,
`roman_tds_supernova` (patched to build its DataFrame via `Table(...).to_pandas()` because pandas 3
rejects big-endian FITS buffers), `TDE_light_curve`, `TDE_light_curve_Rubin` (draft, see below), `SED_fit`.
Only `quickstart` and `roman_tds_supernova` were verified end to end in this environment.

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
  **No decision yet.**
- How `/home/jovyan/work/DDP3_Scarlet2_demo` handles it: reads the image with the LSST stack,
  `exposure.getPsf().computeImage(pos)` at the cutout centre (default) or patch centre, normalised to sum 1 and
  passed to scarlet2 as `ArrayPSF` per band. The model frame uses a separate narrower Gaussian PSF
  (0.2 arcsec sigma in Rubin mode) or the Euclid PSF. Each band is rendered through the kernel
  `obs PSF / model PSF`. Things to check there: PSF stamp centring (scarlet2 wants odd size, centred), and the
  Gaussian fallback (FWHM 0.7 arcsec) used silently when the stack is missing. In the per-visit SN script,
  later epochs are evaluated at the image centre, with x and y swapped, not at the target.

## State of the work

- Git: one commit, `9e92469` ("Add standalone OpenUniverse2024 tutorial notebooks"), author set per repo to
  `astro.flyfish@gmail.com` (the global git email is still the old Drexel one).
- **Uncommitted**: `ou_utils.py`, `target_selection.md`, `roman_cutouts.md`, `rubin_cutouts.md`, `pyproject.toml`,
  and edits to `README.md`, `.gitignore`, `jupytext.toml`, `requirements*.txt`. `notebooks/TDE_light_curve.md`
  also has a cosmetic front-matter change not made by me.
- **Not yet run**: the cutout notebooks have not been rerun since the error/quality layers were added (the
  user asked not to run them yet). The layer code was tested on synthetic images only. The cutouts now on disk
  are the earlier host-centred 20 arcsec ones without those layers; the next run rewrites them.
- `jupytext.toml` excludes `widgets` from synced metadata, after tqdm progress-bar state once bloated two notebooks.

## Open items

1. Rerun `roman_cutouts` and `rubin_cutouts` and check the new `ERR`/`DQ` and `VARIANCE`/`MASK` extensions on real data.
2. Decide how to get Rubin PSFs (see above); then write a script that saves a PSF stamp next to each cutout.
3. Pick an SED for drawing Roman PSFs of real targets.
4. Optional: a minimum-image-count requirement in the coverage check; matching Roman and Rubin cutouts to an
   exactly common pixel grid if needed.
5. Log in to `gh` and push to GitHub when ready; commit the new work.
6. `TDE_light_curve_Rubin.md` is still a draft: shared `aperture_radius_pix` bug, scratch cell, no Rubin light-curve plot.
