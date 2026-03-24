# nuGUNDAM

nuGUNDAM is a package for fast two-point correlation functions in galaxy surveys.

It combines highly optimized Fortran/OpenMP counting cores with a modern, modular
Python interface. The current package exposes clean, typed APIs for:

- **angular auto/cross correlations**
- **projected auto/cross correlations**
- **marked angular/projected auto/cross correlations**
- pair-count runs in angular and projected space
- bootstrap and jackknife uncertainty workflows
- generic input tables (Astropy, pandas, PyArrow, NumPy structured arrays, mappings)
- native result I/O, ASCII export, and plotting routines in 1D/2D
- split-random acceleration for LS autocorrelations

## What is included

### Angular API

- `acf(data, random, config)`
- `accf(data1, data2, config, *, random1=None, random2=None)`
- `macf(data, random, config, *, mark=...)`
- `maccf(data1, data2, config, *, mark=..., random1=None, random2=None)`
- `ang_auto_counts(data, config)`
- `ang_cross_counts(data1, data2, config)`

### Projected API

- `pcf(data, random, config)`
- `pccf(data1, data2, config, *, random1=None, random2=None)`
- `mpcf(data, random, config, *, mark=...)`
- `mpccf(data1, data2, config, *, mark=..., random1=None, random2=None)`
- `proj_auto_counts(data, config)`
- `proj_cross_counts(data1, data2, config)`

### Marked-correlation API

Marked correlations are implemented as a thin layer on top of the existing weighted
correlation machinery. nuGUNDAM runs:

1. an ordinary unweighted branch, and
2. a mark-weighted branch where the data objects carry the processed mark values,

then combines both into the final marked statistic.

The public mark specification objects are:

- `AutoMarkSpec`
- `CrossMarkSpec`

The public marked result classes are:

- `MarkedAngularCorrelationResult`
- `MarkedProjectedCorrelationResult`

For angular auto-correlations, the default marked statistic is

\[
M(\theta)=\frac{1+w_{\mathrm{marked}}(\theta)}{1+w(\theta)}.
\]

For projected auto-correlations, the default marked statistic is

\[
M(r_p)=\frac{1+w_{p,\mathrm{marked}}(r_p)/r_p}{1+w_p(r_p)/r_p}.
\]

Only the **data sample** needs a mark column. **Random catalogs remain unweighted**,
as their role is to encode the survey geometry and selection function.

### Estimators

Currently implemented for both angular and projected auto/cross workflows:

- `NAT`: natural estimator
- `DP`: David-Peebles
- `LS`: Landy-Szalay

### Result utilities

- native round-trip result I/O:
  - `write_result`
  - `read_result`
- instance methods on result/count objects:
  - `.write(...)`
  - `.save(...)`
  - `.to_ascii(...)`
  - `.plot(...)`
  - `.plot2d(...)` for projected 2D views
  - `.plot_cov_matrix(...)`
  - `.plot_corr_matrix(...)`
- marked result helpers:
  - `.plain_wtheta`, `.weighted_wtheta` on marked angular results
  - `.plain_wp`, `.weighted_wp` on marked projected results

### Plotting routines

- `plotcf`
- `plot_result`
- `plotcf2d`
- `plot_result2d`
- `plot_compare_ratio`
- `plot_cov_matrix`
- `plot_corr_matrix`
- `plot_jk_regions`

## Installation

Editable install:

```bash
pip install -e .
```

Developer install with tests:

```bash
pip install -e .[dev]
pytest
```

## Build notes

The package uses:

- `scikit-build-core`
- `numpy.f2py`
- CMake
- a working Fortran compiler
- OpenMP when available

The compiled extension is built from:

- `src/nugundam/cflibfor.f90`
- `src/nugundam/cflibfor.pyf`

## Supported input catalogs

The public APIs accept several tabular backends as long as the required columns are present:

- Astropy tables
- pandas DataFrames
- PyArrow tables
- NumPy structured arrays / recarrays
- plain mappings of column names to arrays

The Python preparation layer extracts the requested columns once and then hands plain
NumPy arrays to the compiled counters.

## Public configuration objects

nuGUNDAM uses dataclass-based configs to handle the various parameters and options.

Angular:

- `CatalogColumns`
- `AngularBinning`
- `AngularGridSpec`
- `WeightSpec`
- `BootstrapSpec`
- `JackknifeSpec`
- `ProgressSpec`
- `SplitRandomSpec`
- `AngularAutoConfig`
- `AngularCrossConfig`
- `AngularAutoCountsConfig`
- `AngularCrossCountsConfig`

Projected:

- `ProjectedCatalogColumns`
- `ProjectedBinning`
- `ProjectedGridSpec`
- `DistanceSpec`
- `ProjectedAutoConfig`
- `ProjectedCrossConfig`
- `ProjectedAutoCountsConfig`
- `ProjectedCrossCountsConfig`

Marked correlations:

- `AutoMarkSpec`
- `CrossMarkSpec`

A convenient way to inspect config schemas is:

```python
from nugundam import AngularAutoConfig, ProjectedAutoConfig

AngularAutoConfig.describe(recursive=True)
ProjectedAutoConfig.describe(recursive=True)
```

This prints a readable summary of nested options and defaults.

For the binning classes themselves, `AngularBinning.describe()` and
`ProjectedBinning.describe()` document the available named constructors and the
resolved inspection helpers such as `edges`, `centers`, `widths`, `sepmax`,
`rp_edges`, and `pi_edges`.

## Binning construction and inspection

The binning classes support **two explicit ways** to define bins:

1. **`from_binsize(...)`**: define the lower bound, number of bins, and bin size.
2. **`from_limits(...)`**: define the lower and upper bounds, and let nuGUNDAM derive the step.

This applies to both angular and projected workflows.

### Angular binning from a bin size

```python
from nugundam import AngularBinning

binning = AngularBinning.from_binsize(
    nsep=21,
    sepmin=0.000277778,
    dsep=0.17,
    logsep=True,
)

print(binning)
print(binning.table())

edges = binning.edges
centers = binning.centers
sepmax = binning.sepmax
```

### Angular binning from explicit limits

```python
from nugundam import AngularBinning

binning = AngularBinning.from_limits(
    nsep=21,
    sepmin=0.000277778,
    sepmax=1.0,
    logsep=True,
)
```

### Projected binning from a bin size

```python
from nugundam import ProjectedBinning

binning = ProjectedBinning.from_binsize(
    nsepp=16,
    seppmin=0.1,
    dsepp=0.15,
    logsepp=True,
    nsepv=20,
    dsepv=2.0,
)

print(binning)
print(binning.table("rp"))
print(binning.table("pi"))

rp_edges = binning.rp_edges
pi_edges = binning.pi_edges
seppmax = binning.seppmax
sepvmax = binning.sepvmax
```

### Projected binning from explicit limits

```python
from nugundam import ProjectedBinning

binning = ProjectedBinning.from_limits(
    nsepp=16,
    seppmin=0.1,
    seppmax=10.0,
    logsepp=True,
    nsepv=20,
    dsepv=2.0,
)
```

## Quick start: angular auto-correlation

```python
from nugundam import (
    AngularAutoConfig,
    CatalogColumns,
    AngularBinning,
    AngularGridSpec,
    WeightSpec,
    BootstrapSpec,
    acf,
)

cfg = AngularAutoConfig(
    estimator="LS",
    columns_data=CatalogColumns(ra="ra", dec="dec", weight="wei"),
    columns_random=CatalogColumns(ra="ra", dec="dec"),
    binning=AngularBinning.from_binsize(
        nsep=21,
        sepmin=0.000277778,
        dsep=0.17,
        logsep=True,
    ),
    grid=AngularGridSpec(
        autogrid=True,
        pxorder="cell-dec",
    ),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

res = acf(data, random, cfg)
```

The returned object is an `AngularCorrelationResult` with fields such as:

- `theta_edges`
- `theta_centers`
- `wtheta`
- `wtheta_err`
- `estimator`
- `counts`
- `metadata`

## Quick start: angular cross-correlation

```python
from nugundam import (
    AngularCrossConfig,
    CatalogColumns,
    AngularBinning,
    AngularGridSpec,
    WeightSpec,
    BootstrapSpec,
    accf,
)

cfg = AngularCrossConfig(
    estimator="LS",
    columns_data1=CatalogColumns(ra="ra_gal", dec="dec_gal", weight="wei_gal"),
    columns_random1=CatalogColumns(ra="ra_r1", dec="dec_r1"),
    columns_data2=CatalogColumns(ra="ra_qso", dec="dec_qso", weight="wei_qso"),
    columns_random2=CatalogColumns(ra="ra_r2", dec="dec_r2"),
    binning=AngularBinning.from_binsize(nsep=21, sepmin=0.000277778, dsep=0.17, logsep=True),
    grid=AngularGridSpec(autogrid="legacy", pxorder="natural"),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

res = accf(data1, data2, cfg, random1=random1, random2=random2)
```

Notes:

- For `NAT` and `LS`, provide both `random1` and `random2`.
- For `DP`, the required random catalog depends on `config.bootstrap.primary`
  (`"data1"` by default).

## Quick start: projected auto-correlation

```python
from nugundam import (
    ProjectedAutoConfig,
    ProjectedCatalogColumns,
    ProjectedBinning,
    ProjectedGridSpec,
    DistanceSpec,
    WeightSpec,
    BootstrapSpec,
    pcf,
)

cfg = ProjectedAutoConfig(
    estimator="LS",
    columns_data=ProjectedCatalogColumns(
        ra="ra",
        dec="dec",
        redshift="z",
        weight="wei",
    ),
    columns_random=ProjectedCatalogColumns(
        ra="ra",
        dec="dec",
        redshift="z",
    ),
    binning=ProjectedBinning.from_binsize(
        nsepp=16,
        seppmin=0.1,
        dsepp=0.15,
        logsepp=True,
        nsepv=20,
        dsepv=2.0,
    ),
    grid=ProjectedGridSpec(
        autogrid=True,
        pxorder="natural",
    ),
    distance=DistanceSpec(
        calcdist=True,
        h0=100.0,
        omegam=0.3,
        omegal=0.7,
    ),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

res = pcf(data, random, cfg)
```

The returned object is a `ProjectedCorrelationResult` with fields such as:

- `rp_edges`
- `rp_centers`
- `wp`
- `wp_err`
- `estimator`
- `counts`
- `metadata`

If you already have comoving distances in the catalog, set `DistanceSpec(calcdist=False)`
and point `ProjectedCatalogColumns.distance` to that column.

## Quick start: projected cross-correlation

```python
from nugundam import (
    ProjectedCrossConfig,
    ProjectedCatalogColumns,
    ProjectedBinning,
    ProjectedGridSpec,
    DistanceSpec,
    WeightSpec,
    BootstrapSpec,
    pccf,
)

cfg = ProjectedCrossConfig(
    estimator="LS",
    columns_data1=ProjectedCatalogColumns(ra="ra1", dec="dec1", redshift="z1", weight="w1"),
    columns_random1=ProjectedCatalogColumns(ra="rar1", dec="decr1", redshift="zr1"),
    columns_data2=ProjectedCatalogColumns(ra="ra2", dec="dec2", redshift="z2", weight="w2"),
    columns_random2=ProjectedCatalogColumns(ra="rar2", dec="decr2", redshift="zr2"),
    binning=ProjectedBinning.from_binsize(nsepp=16, seppmin=0.1, dsepp=0.15, logsepp=True, nsepv=20, dsepv=2.0),
    grid=ProjectedGridSpec(autogrid=True, pxorder="natural"),
    distance=DistanceSpec(calcdist=True),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

res = pccf(data1, data2, cfg, random1=random1, random2=random2)
```

## Quick start: marked projected auto-correlation

```python
from nugundam import (
    ProjectedAutoConfig,
    ProjectedCatalogColumns,
    ProjectedBinning,
    ProjectedGridSpec,
    DistanceSpec,
    WeightSpec,
    BootstrapSpec,
    AutoMarkSpec,
    mpcf,
)

cfg = ProjectedAutoConfig(
    estimator="LS",
    columns_data=ProjectedCatalogColumns(ra="ra", dec="dec", redshift="z"),
    columns_random=ProjectedCatalogColumns(ra="ra", dec="dec", redshift="z"),
    binning=ProjectedBinning.from_binsize(
        nsepp=28,
        seppmin=0.02,
        dsepp=0.12,
        logsepp=True,
        nsepv=4,
        dsepv=10.0,
    ),
    grid=ProjectedGridSpec(autogrid=True, pxorder="natural"),
    weights=WeightSpec(weight_mode="unweighted"),
    bootstrap=BootstrapSpec(enabled=True, nbts=50, bseed=12345),
    distance=DistanceSpec(calcdist=True, h0=100.0, omegam=0.25, omegal=0.75),
    nthreads=4,
)

mres = mpcf(
    data,
    random,
    cfg,
    mark=AutoMarkSpec(column="GMR", normalize="mean"),
)
```

The returned object is a `MarkedProjectedCorrelationResult` with fields such as:

- `rp_edges`
- `rp_centers`
- `mrp`
- `mrp_err`
- `plain`
- `weighted`
- `metadata`

Useful convenience properties include:

- `mres.plain_wp`
- `mres.weighted_wp`

The same pattern applies to `macf`, `maccf`, and `mpccf`.

## Mark specifications

### `AutoMarkSpec`

`AutoMarkSpec` controls how a per-object mark is read and preprocessed before it
is used as the weight in the marked branch.

Main options:

- `column`: data-table column containing the mark
- `normalize`: one of `"mean"`, `"median"`, or `"none"`
- `transform`: one of `"identity"` or `"rank"`
- `clip`: optional `(lo, hi)` clipping interval
- `missing`: either `"raise"` or `"drop"`

Typical example:

```python
mark = AutoMarkSpec(
    column="D4000",
    normalize="mean",
    transform="identity",
    clip=None,
    missing="raise",
)
```

### `CrossMarkSpec`

For cross-correlations, `CrossMarkSpec` lets you mark sample 1, sample 2, or both.

Main options:

- `column1`, `column2`
- `mark_on`: `"data1"`, `"data2"`, or `"both"`
- `normalize`, `transform`, `clip`, `missing`

## Count-only APIs

The count-only APIs use dedicated configs so you need to specify only the required columns.

### Angular count-only

```python
from nugundam import (
    AngularAutoCountsConfig,
    AngularCrossCountsConfig,
    CatalogColumns,
    AngularBinning,
    AngularGridSpec,
    WeightSpec,
    BootstrapSpec,
    ang_auto_counts,
    ang_cross_counts,
)

cfg_dd = AngularAutoCountsConfig(
    columns=CatalogColumns(ra="ra", dec="dec", weight="wei"),
    binning=AngularBinning.from_binsize(nsep=21, sepmin=0.000277778, dsep=0.17, logsep=True),
    grid=AngularGridSpec(autogrid="legacy", pxorder="cell-dec"),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

dd = ang_auto_counts(data, cfg_dd)

cfg_x = AngularCrossCountsConfig(
    columns1=CatalogColumns(ra="ra1", dec="dec1", weight="w1"),
    columns2=CatalogColumns(ra="ra2", dec="dec2", weight="w2"),
    binning=AngularBinning.from_binsize(nsep=21, sepmin=0.000277778, dsep=0.17, logsep=True),
    grid=AngularGridSpec(autogrid="legacy", pxorder="cell-dec"),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

x = ang_cross_counts(data1, data2, cfg_x)
```

### Projected count-only

```python
from nugundam import (
    ProjectedAutoCountsConfig,
    ProjectedCrossCountsConfig,
    ProjectedCatalogColumns,
    ProjectedBinning,
    ProjectedGridSpec,
    DistanceSpec,
    WeightSpec,
    BootstrapSpec,
    proj_auto_counts,
    proj_cross_counts,
)

cfg_dd = ProjectedAutoCountsConfig(
    columns=ProjectedCatalogColumns(ra="ra", dec="dec", redshift="z", weight="wei"),
    binning=ProjectedBinning.from_binsize(nsepp=16, seppmin=0.1, dsepp=0.15, logsepp=True, nsepv=20, dsepv=2.0),
    grid=ProjectedGridSpec(autogrid=True, pxorder="natural"),
    distance=DistanceSpec(calcdist=True),
    weights=WeightSpec(weight_mode="auto"),
    bootstrap=BootstrapSpec(enabled=False),
    nthreads=8,
)

dd = proj_auto_counts(data, cfg_dd)
```

Projected count objects also keep the underlying 2D `(r_p, \pi)` count grids and the
integrated `intpi_*` arrays used for projected estimators.

## Weight handling

The current weighting model is:

- real data catalogs may carry object weights
- random catalogs remain unweighted
- `weight_mode` may be:
  - `"auto"`
  - `"weighted"`
  - `"unweighted"`

`"auto"` uses a slightly faster unweighted counter when all relevant weights are unity.

For marked correlations, the wrapper manages the weighted branch internally, so the
mark itself should be passed through `AutoMarkSpec` or `CrossMarkSpec`, not through the
regular `weight=` column definition.

## Bootstrap and jackknife

### Bootstrap

Bootstrap is supported in the refactored correlation and count pipelines.

Key options live in `BootstrapSpec`:

- `enabled`
- `nbts`
- `bseed`
- `mode`
- `primary`

Cross-correlation bootstrap defaults to a primary-sample scheme.

For marked correlations, nuGUNDAM computes the marked bootstrap realizations from
**matched plain and weighted resamples**, rather than combining already-compressed
error bars afterward.

### Jackknife

Jackknife is currently supported for the full correlation APIs:

- `acf`
- `accf`
- `pcf`
- `pccf`
- `macf`
- `maccf`
- `mpcf`
- `mpccf`

If no explicit region column is supplied, the package can generate sky regions
automatically using a k-means-based partitioner. Jackknife-aware results can carry:

- covariance matrices
- leave-one-region-out realizations
- metadata describing the region source and fast-path usage

Marked jackknife covariances are built from matched marked realizations.

## Jackknife plotting routines

- `plot_cov_matrix`
- `plot_corr_matrix`
- `plot_jk_regions`

Examples:

```python
ax = res.plot_cov_matrix()
ax = res.plot_corr_matrix()
```

To visualize the generated jackknife sky regions:

```python
from nugundam import plot_jk_regions

ax = plot_jk_regions(data=data, random=random, config=cfg, catalog="data")
```

## Split-random RR acceleration

nuGUNDAM includes split-random support for the LS estimator in autocorrelation in both
angular and projected space. This feature is configured through `SplitRandomSpec` and
is intended to reduce the cost of the RR term by splitting the random catalog into
shuffled chunks, counting RR pairs in each, accumulating and properly normalizing these
counts before combining them into the total estimator.

Modes:

- `match_data`
- `nchunks`
- `chunk_size`

Example:

```python
from nugundam import SplitRandomSpec

cfg.split_random = SplitRandomSpec(
    enabled=True,
    mode="match_data",
    seed=12345,
)
```

By default, split-random operates in `match_data` mode, creating as many RR chunks as
needed, each roughly the size of the data sample. Split-random is not available together
with jackknife.

## Grid selection and preparatory ordering

### Angular grid modes

`AngularGridSpec.autogrid` accepts:

- `True` or `"legacy"` for the original nuGUNDAM heuristic
- `False` for fully manual `mxh1/mxh2`
- `"adaptive"` for the newer runtime-aware count-box probe

### Angular `pxorder`

Supported values:

- `"cell-dec"`
- `"natural"`
- `"none"`

`"cell-dec"` adds an intra-cell declination sort on top of the counter-cell ordering.

### Projected grid modes

Projected space currently keeps the legacy boolean `autogrid` choice:

- `True`
- `False`

Projected `pxorder` currently supports:

- `"natural"`
- `"none"`

## Progress reporting

Progress behavior is controlled through `ProgressSpec`.

Typical defaults:

- compact notebook status updates
- direct streaming output in terminal sessions

Example:

```python
cfg.progress.enabled = True
cfg.progress.progress_file = None
cfg.progress.poll_interval = 0.2
```

To keep a progress file:

```python
cfg.progress.progress_file = "run.progress"
```

To silence progress:

```python
cfg.progress.enabled = False
```

## Result persistence

The native on-disk format is a single compressed file containing arrays and embedded
JSON metadata/config/provenance.

Top-level auxiliary functions:

```python
from nugundam import write_result, read_result

write_result(res, "run.gres")
res2 = read_result("run.gres")
```

Instance methods:

```python
res.save("run.gres")
res.write("run.gres")
res2 = type(res).read_result("run.gres")
```

Round-tripped results preserve:

- the result class
- nested count objects
- numeric arrays
- stored metadata
- original run configuration
- provenance information

This includes marked result classes and their nested plain/weighted branches.

## ASCII export

Result and count objects can be exported to plain ASCII:

```python
res.to_ascii("acf.txt")
```

The default exported columns are chosen from the object type and estimator. To control
which columns are exported, customize the `cols` keyword:

```python
res.to_ascii(
    "acf_custom.txt",
    cols=["theta_centers", "dd", "rr", "dr", "wtheta", "wtheta_err"],
)
```

Marked results can likewise be exported with their default marked columns or with an
explicit custom column list.

## Plotting

### 1D plotting from result objects

Result objects can be plotted directly using their attached `plot()` method, either on a
new figure or inserted into a pre-existing axis:

```python
res.plot(color="r", label="sample A", errors="bar")
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
res.plot(ax=ax, color="r", label="sample A", errors="bar")
ax.legend()
```

Supported uncertainty styles:

- `errors="bar"`
- `errors="band"`
- `errors="none"`

Marked results follow the same interface:

```python
mres.plot(label="marked", errors="bar")
```

### 1D plotting from raw arrays

```python
from nugundam import plotcf

ax = plotcf(theta, wtheta, yerr=werr, errors="band")
```

### Overlaying several curves

```python
fig, ax = plt.subplots()

res1.plot(ax=ax, label="sample 1", color="tab:blue")
res2.plot(ax=ax, label="sample 2", errors="band", color="tab:orange",
          band_kwargs={"alpha": 0.15})

ax.legend()
```

### Projected 2D plotting

Projected results and projected count objects can be visualized in 2D. They have options
to overlay contours and smooth the resulting map:

```python
ax = res.plot2d(which="xi")
```

Low-level auxiliary functions are also available:

- `plotcf2d`
- `plot_result2d`

### Ratio comparison plotting

The function `plot_compare_ratio` builds a two-panel comparison plot with shared axes,
which supports several correlation curves on top and multiple curve ratios below.

## Package layout

```text
src/nugundam/
├── angular/
├── projected/
├── core/
├── io.py
├── ascii_io.py
├── plotting.py
├── angular_public.py
├── projected_public.py
├── marked.py
├── result_meta.py
├── cflibfor.f90
└── cflibfor.pyf
```
