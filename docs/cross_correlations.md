# Cross-correlations

Cross-correlations use two data samples and estimator-dependent random catalogs. νGundam keeps separate column mappings for the two sides and does not infer which random belongs to which sample.

## Angular interface

```python
result = ng.accf(
    data1,
    data2,
    config,
    random1=random1,
    random2=random2,
)
```

## Projected interface

```python
result = ng.pccf(
    data1,
    data2,
    config,
    random1=random1,
    random2=random2,
)
```

## Random-catalog contract

| Estimator | `bootstrap.primary` | Required random arguments | Mixed denominator |
|---|---|---|---|
| `NAT` | either | `random1`, `random2` | R1R2 |
| `LS` | either | `random1`, `random2` | both mixed terms plus R1R2 |
| `DP` | `"data1"` | `random2` | D1R2 |
| `DP` | `"data2"` | `random1` | D2R1 after internal role swap |

`bootstrap.primary` identifies the primary sample for the cross-bootstrap design and establishes the role ordering of Davis--Peebles. With `primary="data2"`, νGundam swaps the data/random pairs internally before counting.

!!! warning
    `random1` must describe sample 1's selection and `random2` sample 2's selection. Matching column names do not make the two random catalogs interchangeable.

## Angular configuration template

```python
import nugundam as ng

cfg = ng.AngularCrossConfig(
    estimator="LS",
    columns_data1=ng.CatalogColumns(ra="ra", dec="dec", weight="w1"),
    columns_random1=ng.CatalogColumns(ra="ra", dec="dec"),
    columns_data2=ng.CatalogColumns(ra="ra", dec="dec", weight="w2"),
    columns_random2=ng.CatalogColumns(ra="ra", dec="dec"),
    binning=ng.AngularBinning.from_binsize(
        nsep=20,
        sepmin=0.001,
        dsep=0.15,
        logsep=True,
    ),
    grid=ng.AngularGridSpec(autogrid=True, pxorder="natural"),
    weights=ng.WeightSpec(weight_mode="auto"),
    bootstrap=ng.BootstrapSpec(enabled=False, mode="primary", primary="data1"),
    nthreads=8,
)

result = ng.accf(data1, data2, cfg, random1=random1, random2=random2)
```

## Projected Davis--Peebles template

```python
cfg = ng.ProjectedCrossConfig(
    estimator="DP",
    columns_data1=ng.ProjectedCatalogColumns(
        ra="ra", dec="dec", redshift="z", weight="w1"
    ),
    columns_random1=ng.ProjectedCatalogColumns(
        ra="ra", dec="dec", redshift="z"
    ),
    columns_data2=ng.ProjectedCatalogColumns(
        ra="ra", dec="dec", redshift="z", weight="w2"
    ),
    columns_random2=ng.ProjectedCatalogColumns(
        ra="ra", dec="dec", redshift="z"
    ),
    binning=ng.ProjectedBinning.from_binsize(
        nsepp=16,
        seppmin=0.3,
        dsepp=0.11,
        logsepp=True,
        nsepv=12,
        dsepv=8.0,
    ),
    distance=ng.DistanceSpec(calcdist=True),
    bootstrap=ng.BootstrapSpec(enabled=False, mode="primary", primary="data1"),
    nthreads=8,
)

result = ng.pccf(data1, data2, cfg, random2=random2)
```

Only `random2` is required in this example because data 1 is primary.

## PDF-aware cross-correlations

### Deterministic modes

Supply a PDF source for each data side:

```python
cfg.pdf.enabled = True
cfg.pdf.kind = "quantile_chi"     # or "gmm_chi", "grid_chi_exact"
cfg.pdf.nquant = 16

cfg.pdf_source.enabled = True
cfg.pdf_source.chi_grid = chi_grid
cfg.pdf_source.grid_kind = "edges"
cfg.pdf_source.pdf_data1 = ng.PDFSourceSpec(matrix=pdf_matrix1)
cfg.pdf_source.pdf_data2 = ng.PDFSourceSpec(matrix=pdf_matrix2)
```

The two empirical libraries use the common grid configured in `ProjectedPdfSourceSpec`. For `quantile_chi`, both sides must use quantile mode; mixing quantile and a different prepared representation in one native cross counter is not implemented.

Deterministic random catalogs inherit PDF indices separately from their associated data side. Explicit deterministic random-PDF libraries are currently reserved and rejected by the preparation layer.

### Monte Carlo

```python
cfg.mc_pdf.enabled = True
cfg.mc_pdf.z_grid = z_grid
cfg.mc_pdf.grid_kind = "edges"
cfg.mc_pdf.pdf_data1 = ng.PDFSourceSpec(matrix=pdf_matrix1)
cfg.mc_pdf.pdf_data2 = ng.PDFSourceSpec(matrix=pdf_matrix2)
```

Monte Carlo can also accept `pdf_random1` and `pdf_random2`. When omitted, the selected `random_mode` generates radial random realizations from the corresponding data-side information.

### Spectroscopic--photometric designs

The deterministic pair-count interface expects a radial representation on both sides. A spectroscopic side can be encoded by very narrow or effectively delta-like PDFs on the common grid, while the photometric side uses its full empirical PDFs. For Monte Carlo, the same effect can be obtained with narrow rows whose draws are effectively fixed. Document the chosen approximation because it defines how the spectroscopic radial information enters the PDF-aware calculation.

## Marked cross-correlations

```python
mark = ng.CrossMarkSpec(
    column1="mass",
    mark_on="data1",
    normalize="mean",
)

result = ng.maccf(
    data1,
    data2,
    angular_cfg,
    mark=mark,
    random1=random1,
    random2=random2,
)
```

Use `mpccf` for projected marked cross-correlations. The same random-catalog rules apply.
