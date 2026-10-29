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

# Select transient targets

Choose simulated transients by how bright their peak is, how bright it is relative to its host galaxy, and how large that host is, and require that Roman and Rubin images exist around the peak.
The selection is saved to `data/targets.parquet` for the Roman and Rubin cutout notebooks.

The transient catalog (`snana_{region}.parquet`) already lists each transient's peak magnitude and its host's magnitude in the Rubin `g` and `i` bands and one Roman band, `F` (closest to `roman_flux_F184` in the flux catalog).
Host size comes from the galaxy catalog, which has a disk and a spheroid half-light radius; we use the larger of the two.

The notebook runs as a funnel: each step removes transients, and the table at the end of step 4 shows how many are left after each one.

```{code-cell} ipython3
import json                      # to save the selection parameters next to the targets

import hpgeom                    # HEALPix maths: the catalogs are split into HEALPix sky regions
import matplotlib.pyplot as plt
import pandas as pd

import ou_utils as ou            # this repo's helpers (catalog reading, image search), notebooks/ou_utils.py
```

## Selection parameters

All the choices are in this one cell.
`delta_mag = peak_mag - host_mag` compares the transient at its peak with the total light of its host galaxy.
Magnitudes are backwards (smaller = brighter), so a negative value means the transient outshines the whole galaxy, and `+2` means it is about 6 times fainter than the whole galaxy (each magnitude is a factor of 2.5 in flux).
For a large host that is not a detection problem: the galaxy's light is spread over an area many times the size of the PSF, while the transient's light is concentrated in one PSF, so it still stands out locally.
Whether the transient is bright enough to detect at all is set by `PEAK_MAG_MAX`.

A range is written `(low, high)`; either end can be `None` to leave it open, and the whole range can be `None` to skip that cut.
Candidates are ranked by `SORT_BY`; with these settings the largest hosts come first.

```{code-cell} ipython3
# HEALPix regions (nside=32, RING ordering) to search. Each catalog file covers one region of about
# 3.4 square degrees. The default is the single region at the centre of the Roman TDS (RA 9.45, Dec -44.02);
# 12 regions overlap the TDS, so add more here to search a larger area.
REGIONS = [int(hpgeom.angle_to_pixel(32, 9.45, -44.02, lonlat=True, nest=False))]

MODEL_NAMES = None            # None = every type except FIXMAG; or e.g. ["SALT3.NIR_WAVEEXT"] (SN Ia)
BAND = "i"                    # catalog band for the magnitudes: "g", "i" or "F"
DELTA_MAG = (None, 2.0)       # allowed peak_mag - host_mag: here at most 2 mag fainter than the whole host
PEAK_MAG_MAX = 23.0           # peak must be at least this bright (smaller number = brighter)
HOST_SIZE_ARCSEC = (1.0, 3.0) # allowed host half-light radius in arcsec (larger of disk and spheroid)
REQUIRE_IN_TDS = True         # keep only transients that peak inside the Roman TDS footprint and dates

SORT_BY, DESCENDING = "host_size_arcsec", True    # rank candidates; largest hosts first

# Each instrument: band that must have images within peak_mjd +/- WINDOW_DAYS ({} skips the check).
COVERAGE = {"roman": "J129", "rubin": "r"}
WINDOW_DAYS = 30
N_TARGETS = 20                # stop once this many covered targets are found
```

## 1. Load and clean the transient catalog

Only the needed columns are read.
Two cleaning steps matter: the `FIXMAG` rows are a fixed-magnitude calibration model (every one peaks at 15.005), and some hosts carry a placeholder magnitude of 999.

```{code-cell} ipython3
# Column names of the two magnitudes in the chosen band, e.g. "peak_mag_i" and "host_mag_i".
peak, host = f"peak_mag_{BAND}", f"host_mag_{BAND}"
# Read only these columns: the catalog has many more, and reading fewer is much faster.
columns = ["id", "model_name", "ra", "dec", "host_id", "host_ra", "host_dec", "z_CMB",
           "peak_mjd", "start_mjd", "end_mjd", peak, host]

# Read each region's transient catalog from S3, note which region each row came from
# (needed later to find the host in the matching galaxy catalog), and stack them into one table.
sn = pd.concat([ou.read_catalog("snana", r, columns=columns).assign(region=r) for r in REGIONS],
               ignore_index=True)
steps = {"all transients": len(sn)}   # the funnel: number of transients left after each step

# Keep the requested transient types, or every type except the FIXMAG calibration model.
sn = sn[sn["model_name"].isin(MODEL_NAMES)] if MODEL_NAMES else sn[sn["model_name"] != "FIXMAG"]
steps["transient type"] = len(sn)

sn = sn[sn[host] < 99]  # drop the 999 placeholder (host magnitude unknown)
steps["valid host magnitude"] = len(sn)

# The Roman TDS covers a fixed patch of sky over a fixed range of dates; read those limits from the
# survey's observation sequence. A transient is "in the TDS" if it peaks inside both.
fp = ou.tds_footprint()
sn["in_tds"] = (sn["peak_mjd"].between(*fp["mjd"]) & sn["ra"].between(*fp["ra"])
                & sn["dec"].between(*fp["dec"]))
if REQUIRE_IN_TDS:
    sn = sn[sn["in_tds"]]
steps["inside Roman TDS"] = len(sn)

# Keep transients whose peak is at least PEAK_MAG_MAX bright.
sn = sn[sn[peak] <= PEAK_MAG_MAX]
steps[f"peak {BAND} <= {PEAK_MAG_MAX}"] = len(sn)
```

## 2. Add the host galaxy size

Each transient's `host_id` is the `galaxy_id` in the galaxy catalog.
A few hosts are missing from it; the inner join drops those transients.

```{code-cell} ipython3
# The galaxy catalog gives two half-light radii, one for the disk and one for the bulge (spheroid).
radii = ["diskHalfLightRadiusArcsec", "spheroidHalfLightRadiusArcsec"]
# For each region, read only the rows of the galaxies that host one of our transients
# (the filter is applied while reading, so the large galaxy catalog is never loaded whole).
galaxies = pd.concat([
    ou.read_catalog("galaxy", r, columns=["galaxy_id", *radii],
                    filters=[("galaxy_id", "in", grp["host_id"].astype("int64").tolist())])
    for r, grp in sn.groupby("region")
])
# One size per galaxy: the larger of the two radii.
galaxies["host_size_arcsec"] = galaxies[radii].max(axis=1)

# Attach each host's size to its transient. merge() is an inner join by default, so transients whose
# host is missing from the galaxy catalog are dropped here.
sn = sn.merge(galaxies[["galaxy_id", "host_size_arcsec"]], left_on="host_id", right_on="galaxy_id")
# Give the magnitude columns band-independent names, so the cutout notebooks need not know the band ...
sn = sn.drop(columns="galaxy_id").rename(columns={peak: "peak_mag", host: "host_mag"})
sn["mag_band"] = BAND                                  # ... and record which band they are in.
sn["delta_mag"] = sn["peak_mag"] - sn["host_mag"]      # transient-to-host contrast (negative = brighter than host)
steps["host found in galaxy catalog"] = len(sn)
```

## 3. Apply the brightness-contrast and size filters

```{code-cell} ipython3
def in_range(column, limits):
    """True for rows whose `column` lies within `limits` = (low, high), ends included.

    Either end can be None (open), and limits=None keeps every row.
    """
    if limits is None:
        return pd.Series(True, index=sn.index)
    low, high = limits
    keep = pd.Series(True, index=sn.index)
    if low is not None:
        keep &= sn[column] >= low
    if high is not None:
        keep &= sn[column] <= high
    return keep

in_delta = in_range("delta_mag", DELTA_MAG)            # True/False per transient
in_size = in_range("host_size_arcsec", HOST_SIZE_ARCSEC)
steps["peak - host in range"] = int(in_delta.sum())
steps["host size in range"] = int((in_delta & in_size).sum())

# Candidates pass both cuts; rank them (largest host first with the defaults) and renumber the rows 0, 1, 2, ...
candidates = (sn[in_delta & in_size].sort_values(SORT_BY, ascending=not DESCENDING)
              .reset_index(drop=True))
```

## 4. Require Roman and Rubin coverage

Whether images exist around the peak needs one image search per target and instrument, so candidates are checked in batches, best-ranked first, until `N_TARGETS` pass.
The same search is used by the cutout notebooks, so a target that passes here gets cutouts there.

```{code-cell} ipython3
# Column names that check_coverage adds: "n_roman_images" and "n_rubin_images" (images found per target).
n_images = [f"n_{inst}_images" for inst in COVERAGE]
covered, checked = [], 0   # tables of covered targets per batch; number of candidates checked so far
while sum(map(len, covered)) < N_TARGETS and checked < len(candidates):
    # Search the next batch of N_TARGETS candidates for images within peak_mjd +/- WINDOW_DAYS.
    batch = ou.check_coverage(candidates.iloc[checked:checked + N_TARGETS], COVERAGE, WINDOW_DAYS)
    checked += len(batch)
    # Keep those with at least one image from every instrument. A failed search returns -1, so it fails too.
    covered.append(batch[batch[n_images].ge(1).all(axis=1)])

# Join the batches; the ranking is kept, so head() takes the best-ranked targets.
selected = pd.concat(covered).head(N_TARGETS).reset_index(drop=True)
steps["checked for coverage"] = checked
steps["with Roman and Rubin images (selected)"] = len(selected)

# Show the funnel: how many transients were left after each step.
pd.Series(steps, name="transients remaining").to_frame()
```

The plot below shows where the selection sits among the candidates, to help tune the limits.
Candidates are the transients that passed every cut before the coverage check, so with size and contrast cuts set they all sit inside the dashed limits; the grey points not selected either lacked Roman or Rubin images or were not checked because enough targets had already been found.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
# Grey: every candidate. Blue: the selected targets, drawn on top.
ax.scatter(candidates["host_size_arcsec"], candidates["delta_mag"], s=3, alpha=0.3, color="#9b9a93",
           linewidths=0, label=f"candidates ({len(candidates)})")
ax.scatter(selected["host_size_arcsec"], selected["delta_mag"], s=6, color="#2a78d6", linewidths=0,
           label=f"selected ({len(selected)})")
# Dashed lines mark the cut limits (open ends are not drawn).
for v in HOST_SIZE_ARCSEC or []:
    if v is not None:
        ax.axvline(v, color="#52514e", lw=0.8, ls="--")
for v in DELTA_MAG or []:
    if v is not None:
        ax.axhline(v, color="#52514e", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xlabel("Host half-light radius, larger of disk and spheroid (arcsec)")
ax.set_ylabel(f"Peak - host magnitude ({BAND} band)")
ax.legend(frameon=False, loc="upper right", markerscale=3)
ax.grid(alpha=0.15)
plt.show()
```

## 5. Save the targets

```{code-cell} ipython3
ou.DATA_DIR.mkdir(exist_ok=True)                   # <repo>/data, created on the first run
selected.to_parquet(ou.TARGETS_PATH, index=False)  # data/targets.parquet, read by the cutout notebooks

# Save the parameters used next to the targets (data/targets.json), so the selection can be reproduced.
params = dict(REGIONS=REGIONS, MODEL_NAMES=MODEL_NAMES, BAND=BAND, DELTA_MAG=DELTA_MAG,
              PEAK_MAG_MAX=PEAK_MAG_MAX, HOST_SIZE_ARCSEC=HOST_SIZE_ARCSEC, REQUIRE_IN_TDS=REQUIRE_IN_TDS,
              SORT_BY=SORT_BY, DESCENDING=DESCENDING, COVERAGE=COVERAGE, WINDOW_DAYS=WINDOW_DAYS,
              N_TARGETS=N_TARGETS)
ou.TARGETS_PATH.with_suffix(".json").write_text(json.dumps(params, indent=2))

print(f"Saved {len(selected)} targets to {ou.TARGETS_PATH}")
selected.head()   # the first five targets
```

The targets keep the ranking, so the cutout notebooks' `MAX_TARGETS` takes the first ones.
