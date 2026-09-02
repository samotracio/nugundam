# Example 3: PDF-aware correlations

This page is a configuration-first placeholder for a future worked PDF example. It assumes:

- `data` and `random` are the projected catalogs;
- `base_cfg` is a valid point-redshift `ProjectedAutoConfig`;
- `z_grid` contains common PDF-bin edges;
- `pdf_matrix` has shape `(len(data), len(z_grid) - 1)` and follows the data row order.

<div class="placeholder-block">
<strong>To add later:</strong> sample provenance, PDF normalization diagnostics, chosen radial grid, object-level PDFs, convergence plots, timing/memory comparisons, and final $w_p$ ratios.
</div>

!!! note "Mutually exclusive mode blocks"
    Deterministic modes use `cfg.pdf` plus `cfg.pdf_source`. Monte Carlo uses `cfg.mc_pdf`. Do not enable both for the same run.

## Common empirical-PDF source

```python
from copy import deepcopy

import nugundam as ng

source = ng.PDFSourceSpec(
    kind="external_matrix",
    matrix=pdf_matrix,
)
```

For large arrays, `PDFSourceSpec(path="pdfs.npy")` can load the matrix from disk instead of embedding it in the configuration.

## 16quant

```python
cfg_q16 = deepcopy(base_cfg)

cfg_q16.pdf.enabled = True
cfg_q16.pdf.kind = "quantile_chi"
cfg_q16.pdf.nquant = 16
cfg_q16.pdf.quantile_storage = "float32"
cfg_q16.pdf.quantile_positions = "midpoint"
cfg_q16.pdf.random_pdf_policy = "inherit"

cfg_q16.pdf_source.enabled = True
cfg_q16.pdf_source.z_grid = z_grid
cfg_q16.pdf_source.grid_kind = "edges"
cfg_q16.pdf_source.pdf_data = source

q16_result = ng.pcf(data, random, cfg_q16)
```

Each object is represented by `nquant` equal-probability midpoint nodes in comoving distance. The Cartesian product of the two objects' nodes is tested with the same $(r_p,\pi)$ geometry as the projected pair counter.

`grid_kind="edges"` is preferred when the input PDFs are histograms: the inverse CDF is interpolated inside each input bin, reducing artificial distance quantization.

## Monte Carlo

```python
cfg_mc = deepcopy(base_cfg)

cfg_mc.mc_pdf.enabled = True
cfg_mc.mc_pdf.nreal = 25
cfg_mc.mc_pdf.seed = 12345
cfg_mc.mc_pdf.z_grid = z_grid
cfg_mc.mc_pdf.grid_kind = "edges"
cfg_mc.mc_pdf.pdf_data = source
cfg_mc.mc_pdf.random_mode = "inherit_realization"
cfg_mc.mc_pdf.sample_within_bin = True
cfg_mc.mc_pdf.store_realizations = True

mc_result = ng.pcf(data, random, cfg_mc)
```

νGundam averages the DD, DR, and RR count fields across realizations **before** applying the estimator. `mc_realizations` is optional diagnostic output; it is not the quantity averaged to define the main result.

## GMM compression

```python
cfg_gmm = deepcopy(base_cfg)

cfg_gmm.pdf.enabled = True
cfg_gmm.pdf.kind = "gmm_chi"
cfg_gmm.pdf.k = 3
cfg_gmm.pdf.random_pdf_policy = "inherit"

cfg_gmm.pdf_source.enabled = True
cfg_gmm.pdf_source.z_grid = z_grid
cfg_gmm.pdf_source.grid_kind = "edges"
cfg_gmm.pdf_source.edge_moments = True
cfg_gmm.pdf_source.pdf_data = source

gmm_result = ng.pcf(data, random, cfg_gmm)
```

The default compressor divides each empirical CDF into approximately equal-mass segments and stores the mass, conditional mean, and variance of each segment in $\chi$ space.

## Exact empirical grid

```python
cfg_exact = deepcopy(base_cfg)

cfg_exact.pdf.enabled = True
cfg_exact.pdf.kind = "grid_chi_exact"
cfg_exact.pdf.prob_floor = 0.0

cfg_exact.pdf_source.enabled = True
cfg_exact.pdf_source.z_grid = z_grid
cfg_exact.pdf_source.grid_kind = "edges"
cfg_exact.pdf_source.edge_refine = 2
cfg_exact.pdf_source.pdf_data = source

exact_result = ng.pcf(data, random, cfg_exact)
```

For precision validation, `prob_floor=0.0` avoids deliberate support truncation. Increasing `edge_refine` treats each histogram bin as several equal-width sub-bins in $\chi$; this improves radial resolution but raises runtime and memory use.

## Compare methods

```python
fig, axes = ng.plot_compare_ratio(
    {
        "point z": point_result,
        "16quant": q16_result,
        "MC": mc_result,
        "GMM": gmm_result,
        "exact": exact_result,
    },
    ratios=[
        {
            "numerator": "16quant",
            "denominator": "exact",
            "label": "16quant / exact",
        },
        {
            "numerator": "MC",
            "denominator": "exact",
            "label": "MC / exact",
        },
    ],
)
```

Inspect the full LOS structure as well as the integrated curve:

```python
q16_result.plot2d(which="xi")
print(q16_result.counts.pi_edges)
print(ng.summarize_pair_diagnostics(q16_result))
```

Detailed 16quant hot-kernel counters are disabled by default. Enable them only for a diagnostic run:

```python
cfg_q16.pdf.diagnostics = True
q16_diagnostic = ng.pcf(data, random, cfg_q16)
ng.print_pair_diagnostics(q16_diagnostic)
```
