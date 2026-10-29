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

- Selection (changed 2026-10-29 at the user's request): catalog band `i`, `PEAK_MAG_MAX = 23`,
  `HOST_SIZE_ARCSEC = (1.0, 3.0)` (half-light radius, larger of disk and spheroid), `DELTA_MAG = (None, 2.0)`
  (peak - host at most +2), ranked by host size, largest first, `REQUIRE_IN_TDS = True`. Coverage required: Roman
  J129 and Rubin r images within peak MJD ± 30 days. Stops at 20 targets. Only the TDS-centre HEALPix region is
  searched (user's choice); the 12 regions overlapping the TDS are 10051, 10052, 10178, 10179, 10180, 10306,
  10307, 10308, 10430, 10431, 10550, 10551. Range limits may now have an open end (`None`).
  - Result: 711 after peak/TDS cuts, 644 with peak - host <= 2, 23 with host 1-3 arcsec, **7 selected** after the
    coverage check (Rubin coverage removes most). First target 50084332 (CC SN, host 2.32 arcsec, delta +0.85).
  - Why not `delta < 0` (the user first asked for it): large hosts are bright (median delta +2.3 mag for 1-3 arcsec
    hosts), so only 2 candidates in the centre region (9 in all 12 regions) pass both cuts, before coverage.
    Also explained to the user: outshining the whole host is not needed for detection, since a 1-3 arcsec host's
    light is spread over many PSF areas; `PEAK_MAG_MAX` sets detectability. Counts with host 1-3 arcsec across all
    12 regions: delta < 0: 9, < 0.5: 21, < 1: 44, < 2: 110.
  - The notebook was rewritten with line-by-line comments for the user, who wanted to understand each cell.
- Cutouts: 20 arcsec wide, **centred on the host** (`CENTER_ON = "host"`), native orientation
  (`NORTH_UP = False`), `PSF = True`, `MAX_TARGETS = 1` (user asked for 1 target while testing), up to 5 epochs
  per target. A cyan ring marks the transient in the montage.
- Each cutout FITS file, same HDU order for both instruments (user's choice, 2026-10-29): image, variance, flags.
  Roman `SCI, VARIANCE, DQ, BKGNOISE, PSF`; Rubin `IMAGE, VARIANCE, MASK, PSF` (order in `ou.CUTOUT_ORDER`; what
  is read from the source image in `ou.CUTOUT_LAYERS`). Same pixel grid and WCS (with SIP). Header keys:
  `TARGETID BAND MJD OBSID SRCIMG SRCX SRCY TRA TDEC CENTERON SIZEARCS NORTHUP LAYERS NANFRAC`
  (`SRCX SRCY` = cutout centre on the source image, 0-based pixels; also `src_x`, `src_y` in the manifest).
  Flags are never interpolated (nearest neighbour when north-up); flags are 0 off the detector.
- Resume: existing cutouts are skipped only if size, centring, rotation and `LAYERS` (which includes `PSF`) all
  match and `SRCX` is present; otherwise rewritten.
  Targets with no image in the window are listed in the manifest as `no images in window`.

## Data gotchas found

- `snana` already has `peak_mag_{g,i,F}` and `host_mag_{g,i,F}`. `F` is closest to Roman F184 in the flux catalog.
- `host_mag_* = 999` is a placeholder. `FIXMAG` rows (peak exactly 15.005) are a calibration model.
  Some host IDs are missing from the galaxy catalog. Host size is only in the galaxy catalog.
- Rubin visits are sparse in time, so many targets have no image within ±30 days of peak. Roman needs the
  `TDS_simple_model` images and only covers the survey field and dates.
- The transient is often several arcsec off its host (0.4 to 6.7 arcsec for the largest hosts). That is why
  cutouts are host-centred and wide; transient-centred cutouts looked "off-centre" to the user.
- At 20 arcsec, Roman cutouts come out 185 to 189 pixels square (about 0.108 arcsec/pixel; detector pixel scales
  differ slightly), Rubin 101 (0.2 arcsec/pixel). An earlier note here said Roman 93 or 95, which matches
  10 arcsec, not 20.
- **Roman `ERR` is not an uncertainty: it is the background and noise of `SCI` without the sources**
  (resolved 2026-10-29). roman_imsim v2.0 `detector_physics.py`: `write_fits(..., img, sky_noise, dq, ...)` puts
  `sky_noise` into `ERR`; `finalize_sky_im` builds it from `self.sky` (sky + thermal, with Poisson noise already
  added in `setup_sky`, the same realization added to the image in `add_background`) + `im_dark` + `im_read`
  (the same dark and read-noise realizations added to `SCI`), /gain, rounded. Checked on a full image: `SCI - ERR`
  is exactly 0 on 77% of pixels; SCI and ERR robust scatter both 14.83 ≈ sqrt(sky + 8.5**2) = 14.4. Headers have
  no units; `SKY_MEAN` in the image header ≈ median ERR. Gain 1, read noise 8.5 e-, sky not subtracted in SCI.
  (The 14.83 was a robust scatter over all pixels, sources included; on blank pixels only the prediction is exact.)
  Rubin `VARIANCE` behaves as expected (higher on sources).
- **Roman cutouts now have a real `VARIANCE` and save `ERR` as `BKGNOISE`** (implemented 2026-10-29, user agreed
  the plan, the name `BKGNOISE` and inf for saturated pixels). `ou.roman_variance`:
  `VARIANCE = (max(SCI - BKGNOISE, 0) + mean(BKGNOISE)) / gain + (read_noise / gain)**2` (galsim.roman gain 1,
  read noise 8.5), computed from the cut arrays; one sky per cutout is enough (sky flat across a detector,
  134.8 to 135.2 per 256x256 block). Chosen over `SCI + 8.5**2` because weights that follow the pixel noise bias
  fits. Saturated: `SCI >= (1e5 - 5 * 8.5) / gain` = 99957.5 -> inf (pixels clip at 1e5 e- before read noise, so
  saturated values are 99960 to 100025; ~100 per image; DQ is all zero in these simple-model files).
  VARIANCE header: `BUNIT GAIN RDNOISE SKYEXP SATLEVEL NSAT` + formula comments; BKGNOISE header explains itself.
  Checks done before implementing, on 2 full images: blank-sky pull std 1.000 and 1.001 (every 256-px block
  0.99 to 1.01), no neighbour correlation. Source term: an empirical test with a "smooth neighbours" cut was
  biased low (rejects noisy pixels) and a curvature-corrected one was dominated by structure; settled from code
  instead: roman_imsim v2.0 `stamp.py` draws a Poisson realized flux and photon-shoots it with unit photons, or
  draws with FFT and adds `PoissonNoise` (line 313), so source pixels are Poisson. After implementing: values
  match an independent recomputation to 5e-8; blank-sky pull 1.01 in a cutout; saturation/NaN/clip unit-tested.

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

### Implemented 2026-10-29: a PSF image in every cutout

One PSF per cutout (per epoch), evaluated at the cutout centre (the host), written as a `PSF` extension in
the same FITS file, on the native pixel grid, odd size, centred on the central pixel, sum 1. Done inside
`ou.run_cutouts(..., psf=True)` in the single env (the user chose this over the old CSV/script/`.npy` hand-off).
`psf=True` with `north_up=True` raises, because the stamp is not rotated.

- Roman (`ou.roman_psf`): `galsim.roman.getPSF(SCA_NUM from the primary header, band, SCA_pos, wcs=local WCS
  from the SCI header via GSFitsWCS, pupil_bin=4, n_waves=5)`, convolved with roman_imsim's charge diffusion
  (3 Gaussians in pixels: weights 0.17519/0.53146/0.29335, sigmas 0.4522/0.8050/1.4329 x 0.3279 px, from
  `roman_imsim/photonOps/_charge_diffusion.py` at DukeCosmology/roman_imsim HEAD, which may differ from the 2023
  run) and the pixel response, flat-photon SED. **51x51 (5.5 arcsec)** by default (`ou.PSF_SIZE_PIX`, notebook
  `PSF_SIZE_PIX`). History: 51, then doubled to 101 at the user's request, then halved back to 51 (also the
  user's request). 1.2% of the PSF's light lies outside the central 51x51 (measured on a 101 px stamp), so the
  51 px stamp, renormalised to sum 1, puts ~1% too much light in the core.
  Drawing is behind a lock because cutouts run in threads. Header: `PSFMODEL SCA NWAVES PUPILBIN CHGDIFF PSFSED
  SRCX SRCY PSFSCALE`.
- Rubin (`ou.rubin_psf`): each calexp (~110 MB) is downloaded to `data/cache/`, used for both the cutouts and
  `ExposureFitsReader(path).readPsf().computeKernelImage(Point2D(x + x0, y + y0))`, then deleted. Kernel image
  (centred), not `computeImage` (which is shifted to the sub-pixel position). The PSFEx model only covers
  41x41 pixels, and the stamp is now **41x41 (8.2 arcsec)**, the model's own size. History: the user first asked
  to double it to 81 (zero-padded outside the model), worried about the padding, and settled on 41.
  `rubin_psf(size_pix=...)` still pads with zeros if larger, crops if smaller. Checked on 43 bright unflagged
  stars in one calexp: light at 4-8 arcsec is 0.2% +/- 1% of the total (consistent with zero), and the model's
  r <= 2 px fraction matches the stars (0.45 vs 0.44), so 41 px loses nothing. The model falls to ~1e-3 of the
  peak by 2 arcsec and to noise (~1e-4) by 3 arcsec. Small negative values (~1e-4 of the peak) remain from PSFEx;
  not clipped. Header: `PSFMODEL PSFMODSZ (= 41, model size) SRCX SRCY PSFSCALE`.
- Primary header `PSFSIZE` = stamp width; the resume check rewrites cutouts whose `PSFSIZE` differs.
- **Tried and rejected (2026-10-29): a Gaussian Roman PSF** with the galsim PSF's FWHM (0.126 arcsec in J129,
  measured with `calculateFWHM` before pixelisation). It matched the width but not the shape: central pixel
  0.48 vs 0.28 of the light (J129), because the real PSF has 30-45% of its light in the wings (J129 to F184),
  and in a 101 px stamp it looked like a dot. The user had it reverted to the full galsim PSF; the code was
  removed, not kept as an option.
- The user's earlier request "make the PSF 10x the FWHM" would have *shrunk* the Roman stamp (FWHM ~1.1 px);
  after that was pointed out they chose to double both stamps instead.
- Checked on real data (1 target, 5 epochs each): all stamps sum to 1 with the peak on the central pixel.
  Rubin FWHM 0.68 to 1.03 arcsec across visits. Roman core FWHM about 0.10 to 0.12 arcsec. Roman orientation
  checked against 12 unsaturated stars (peak 3000 to 40000; ~1e5 means saturated) in two images: the model as
  drawn matches the wings best on average (r about 0.85), transposed or x-flipped clearly worse in one image;
  central 5x5 flux fraction 0.80-0.83 (model) vs 0.81-0.85 (stars), consistent with the flat-SED placeholder.

Still a placeholder: the flat photon SED for Roman (a redder per-target SED broadens the PSF slightly, so
revisit it if the PSF is used in fits).

## State of the work

- Git: `origin` is `https://github.com/ywx649999311/desc-hack-2026.git`, branch `main`, and `gh` is logged in as
  `ywx649999311`. As of the push on 2026-10-29 everything is committed and pushed (see `git log`). In the container
  the commit author was `astro.flyfish@gmail.com`, set per repo; the Mac clone commits as the GitHub noreply
  address `11515291+ywx649999311@users.noreply.github.com`.
- `notebooks/TDE_light_curve.md` carries a cosmetic front-matter change (a jupytext 1.16.6 stamp) that was not
  made by me; it was committed along with everything else.
- `setup_on_mac.md` has the steps used to build the single stack + project env on the Mac (done and import-checked
  2026-10-29; a one-cell test notebook ran through the kernel and imported `lsst.afw`).
- Run in the Mac env on 2026-10-29: `target_selection` with the old cuts (20 targets; jupytext refused to write
  the executed copy once because the `.md` timestamp changed mid-run, probably an editor), then `roman_cutouts`
  and `rubin_cutouts` with `MAX_TARGETS = 1` (target 30431441, 5 epochs each, all `ok`, no NaNs, all layers plus
  `PSF`). Then `target_selection` was rerun with the new cuts (7 targets, above), the old cutouts and manifests
  were deleted at the user's request, and the cutout notebooks were rerun for the new first target 50084332
  (4 Roman J129 epochs, 5 Rubin r epochs) with the doubled galsim/PSFEx stamps. Other notebooks not run on the Mac yet.
- The user added `notebooks/explore_cutouts.md` (opens one Roman and one Rubin cutout, plots image, variance and
  PSF) and a note in `roman_cutouts.md` that Roman `ERR` is not a 1-sigma uncertainty (it behaves like background
  plus noise without sources; they suggest `SCI + 8.5**2` as a variance). Not yet checked by me.
- `run_notebooks.sh` prints "DM stack could not be deactivated!" from the stack's conda scripts; harmless.
- `jupytext.toml` excludes `widgets` from synced metadata, after tqdm progress-bar state once bloated two notebooks.

## Open items

1. Raise `MAX_TARGETS` in `roman_cutouts` and `rubin_cutouts` once the 1-target run is accepted, and
   run the other notebooks in the Mac env (pandas 2.3 and Python 3.13 there, versus pandas 3 and 3.12 in the container).
   If 7 targets are too few, add TDS regions to `REGIONS` (about 6x more candidates) or relax `DELTA_MAG`.
2. Optional PSF follow-ups: a PSF for `NORTH_UP = True` (rotate/resample the stamp); a Roman PSF check by
   fitting stars rather than correlating; keeping calexps cached across runs if reruns get slow.
3. Pick a better SED than the flat placeholder for Roman PSFs of real targets, if the PSF is used in fits.
4. Optional: a minimum-image-count requirement in the coverage check; matching Roman and Rubin cutouts to an
   exactly common pixel grid if needed.
5. Install `scarlet2` in the Mac env if needed (dry-run first; it brings jax).
6. `TDE_light_curve_Rubin.md` is still a draft: shared `aperture_radius_pix` bug, scratch cell, no Rubin light-curve plot.
