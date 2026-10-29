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

```{code-cell} ipython3
import json

import hpgeom
import matplotlib.pyplot as plt
import pandas as pd

import ou_utils as ou
```

## Selection parameters

The values below are placeholders to tune.
`delta_mag = peak_mag - host_mag`, so a negative value means the transient peaks brighter than its whole host galaxy.
Set a range to `None` to skip that cut.
Candidates are ranked by `SORT_BY`; with the defaults the largest hosts come first.

```{code-cell} ipython3
# HEALPix regions (nside=32, RING) to search; the default is the one at the Roman TDS centre.
REGIONS = [int(hpgeom.angle_to_pixel(32, 9.45, -44.02, lonlat=True, nest=False))]

MODEL_NAMES = None            # None = every type except FIXMAG; or e.g. ["SALT3.NIR_WAVEEXT"] (SN Ia)
BAND = "i"                    # catalog band for the magnitudes: "g", "i" or "F"
DELTA_MAG = None              # allowed peak_mag - host_mag, e.g. (-2.0, 0.0); None = no cut
PEAK_MAG_MAX = 23.0           # peak must be at least this bright (smaller number = brighter)
HOST_SIZE_ARCSEC = None       # allowed host half-light radius in arcsec, e.g. (0.3, 1.5); None = no cut
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
peak, host = f"peak_mag_{BAND}", f"host_mag_{BAND}"
columns = ["id", "model_name", "ra", "dec", "host_id", "host_ra", "host_dec", "z_CMB",
           "peak_mjd", "start_mjd", "end_mjd", peak, host]

sn = pd.concat([ou.read_catalog("snana", r, columns=columns).assign(region=r) for r in REGIONS],
               ignore_index=True)
steps = {"all transients": len(sn)}

sn = sn[sn["model_name"].isin(MODEL_NAMES)] if MODEL_NAMES else sn[sn["model_name"] != "FIXMAG"]
steps["transient type"] = len(sn)

sn = sn[sn[host] < 99]  # drop the 999 placeholder
steps["valid host magnitude"] = len(sn)

fp = ou.tds_footprint()
sn["in_tds"] = (sn["peak_mjd"].between(*fp["mjd"]) & sn["ra"].between(*fp["ra"])
                & sn["dec"].between(*fp["dec"]))
if REQUIRE_IN_TDS:
    sn = sn[sn["in_tds"]]
steps["inside Roman TDS"] = len(sn)

sn = sn[sn[peak] <= PEAK_MAG_MAX]
steps[f"peak {BAND} <= {PEAK_MAG_MAX}"] = len(sn)
```

## 2. Add the host galaxy size

Each transient's `host_id` is the `galaxy_id` in the galaxy catalog.
A few hosts are missing from it; the inner join drops those transients.

```{code-cell} ipython3
radii = ["diskHalfLightRadiusArcsec", "spheroidHalfLightRadiusArcsec"]
galaxies = pd.concat([
    ou.read_catalog("galaxy", r, columns=["galaxy_id", *radii],
                    filters=[("galaxy_id", "in", grp["host_id"].astype("int64").tolist())])
    for r, grp in sn.groupby("region")
])
galaxies["host_size_arcsec"] = galaxies[radii].max(axis=1)

sn = sn.merge(galaxies[["galaxy_id", "host_size_arcsec"]], left_on="host_id", right_on="galaxy_id")
sn = sn.drop(columns="galaxy_id").rename(columns={peak: "peak_mag", host: "host_mag"})
sn["mag_band"] = BAND
sn["delta_mag"] = sn["peak_mag"] - sn["host_mag"]
steps["host found in galaxy catalog"] = len(sn)
```

## 3. Apply the brightness-contrast and size filters

```{code-cell} ipython3
def in_range(column, limits):
    return sn[column].between(*limits) if limits else pd.Series(True, index=sn.index)

in_delta = in_range("delta_mag", DELTA_MAG)
in_size = in_range("host_size_arcsec", HOST_SIZE_ARCSEC)
steps["peak - host in range"] = int(in_delta.sum())
steps["host size in range"] = int((in_delta & in_size).sum())

candidates = (sn[in_delta & in_size].sort_values(SORT_BY, ascending=not DESCENDING)
              .reset_index(drop=True))
```

## 4. Require Roman and Rubin coverage

Whether images exist around the peak needs one image search per target and instrument, so candidates are checked in batches, best-ranked first, until `N_TARGETS` pass.
The same search is used by the cutout notebooks, so a target that passes here gets cutouts there.

```{code-cell} ipython3
n_images = [f"n_{inst}_images" for inst in COVERAGE]
covered, checked = [], 0
while sum(map(len, covered)) < N_TARGETS and checked < len(candidates):
    batch = ou.check_coverage(candidates.iloc[checked:checked + N_TARGETS], COVERAGE, WINDOW_DAYS)
    checked += len(batch)
    covered.append(batch[batch[n_images].ge(1).all(axis=1)])

selected = pd.concat(covered).head(N_TARGETS).reset_index(drop=True)
steps["checked for coverage"] = checked
steps["with Roman and Rubin images (selected)"] = len(selected)

pd.Series(steps, name="transients remaining").to_frame()
```

The plot below shows where the selection sits among the candidates, to help tune the limits.
Candidates are the transients that passed every cut before the coverage check.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(candidates["host_size_arcsec"], candidates["delta_mag"], s=3, alpha=0.3, color="#9b9a93",
           linewidths=0, label=f"candidates ({len(candidates)})")
ax.scatter(selected["host_size_arcsec"], selected["delta_mag"], s=6, color="#2a78d6", linewidths=0,
           label=f"selected ({len(selected)})")
for v in HOST_SIZE_ARCSEC or []:
    ax.axvline(v, color="#52514e", lw=0.8, ls="--")
for v in DELTA_MAG or []:
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
ou.DATA_DIR.mkdir(exist_ok=True)
selected.to_parquet(ou.TARGETS_PATH, index=False)

params = dict(REGIONS=REGIONS, MODEL_NAMES=MODEL_NAMES, BAND=BAND, DELTA_MAG=DELTA_MAG,
              PEAK_MAG_MAX=PEAK_MAG_MAX, HOST_SIZE_ARCSEC=HOST_SIZE_ARCSEC, REQUIRE_IN_TDS=REQUIRE_IN_TDS,
              SORT_BY=SORT_BY, DESCENDING=DESCENDING, COVERAGE=COVERAGE, WINDOW_DAYS=WINDOW_DAYS,
              N_TARGETS=N_TARGETS)
ou.TARGETS_PATH.with_suffix(".json").write_text(json.dumps(params, indent=2))

print(f"Saved {len(selected)} targets to {ou.TARGETS_PATH}")
selected.head()
```

The targets keep the ranking, so the cutout notebooks' `MAX_TARGETS` takes the first ones.
