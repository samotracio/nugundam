# Resampling and uncertainties

νGundam supports bootstrap and region-based delete-one jackknife workflows. Enable only one in a given configuration.

## Bootstrap

```python
import nugundam as ng

bootstrap = ng.BootstrapSpec(
    enabled=True,
    nbts=100,
    bseed=12345,
    mode="none",
)
```

For cross-correlations, use a primary-sample scheme:

```python
bootstrap = ng.BootstrapSpec(
    enabled=True,
    nbts=100,
    bseed=12345,
    mode="primary",
    primary="data1",
)
```

`primary` is also used to define the Davis--Peebles role ordering. The diagonal uncertainty (`wtheta_err` or `wp_err`) is computed from the resulting realizations. Projected results may store `bootstrap_realizations`.

## Jackknife

```python
jackknife = ng.JackknifeSpec(
    enabled=True,
    nregions=30,
    generator="kmeans",
    geometry_from="auto",
    seed=12345,
    return_cov=True,
    return_realizations=False,
)
```

When no region column is mapped, νGundam generates shared sky regions. If `nregions=None`, it chooses a practical default from the output binning.

For $N_{\mathrm{JK}}$ leave-one-region-out vectors $\mathbf{x}^{(k)}$,

$$
\mathbf{C}_{\mathrm{JK}}=
\frac{N_{\mathrm{JK}}-1}{N_{\mathrm{JK}}}
\sum_{k=1}^{N_{\mathrm{JK}}}
\left(\mathbf{x}^{(k)}-\bar{\mathbf{x}}\right)
\left(\mathbf{x}^{(k)}-\bar{\mathbf{x}}\right)^{\mathsf T}.
$$

The one-dimensional errors are the square roots of the covariance diagonal.

## Fast touch-count jackknife

Native counters can store, for every region, the count contributed by pairs touching that region. Leave-one-region-out counts are then reconstructed by subtraction rather than a complete recount. Metadata records whether the fast touch path was used or available.

A policy that changes the prepared random PDFs in every leave-one-out sample cannot reuse fixed touch fields and therefore requires rerun behavior.

## PDF-aware resampling

### Monte Carlo

The full-sample MC resolution and resampling resolution are separate:

```python
cfg.mc_pdf.nreal = 25
cfg.mc_pdf.resampling_nreal = 5
cfg.mc_pdf.resampling_backend = "fast"
cfg.mc_pdf.resampling_random_policy = "fixed"
```

- `resampling_backend="auto"` selects an available compatible backend;
- `"fast"` averages point-resampling counts across MC draws;
- `"rerun"` rebuilds the requested resample explicitly.

When `random_mode="inherit_realization"`, the fast backend requires `resampling_random_policy="fixed"`. Incompatible explicit requests raise an error rather than silently changing the policy.

### Deterministic exact-grid and GMM

Exact-grid and GMM modes expose native bootstrap and jackknife paths when their compiled kernels are available. With inherited random PDFs:

```python
cfg.pdf.jk_random_policy = "fixed"       # permits touch reconstruction
```

or

```python
cfg.pdf.jk_random_policy = "reinherit"   # conservative rerun behavior
```

Result metadata such as `pdf_bootstrap_backend` and `jk_touch_fast` records the selected behavior.

### 16quant

Main 16quant auto- and cross-counts support ordinary weighted or unweighted data. Native fast 16quant bootstrap and jackknife-touch kernels in 0.7.1 are **unweighted only**.

```python
cfg.weights.weight_mode = "unweighted"
cfg.pdf.kind = "quantile_chi"
cfg.pdf.jk_random_policy = "fixed"
```

For a weighted 16quant measurement with resampling, use/allow the rerun path rather than requiring the native fast path. If a compiled-extension build lacks the requested quantile resampling kernel, νGundam reports that explicitly.

## Marked resampling

Marked uncertainties are not obtained by independently propagating the diagonal errors of the plain and marked branches. νGundam pairs matched plain/weighted realizations and evaluates $M$ in each realization, preserving numerator--denominator covariance.

## Restrictions in 0.7.1

- Bootstrap and jackknife cannot both be enabled.
- Count-only APIs do not expose full jackknife estimation; use `acf`, `accf`, `pcf`, or `pccf`.
- Split-random autocorrelation cannot be combined with jackknife.
- Deterministic `jk_random_policy="reinherit"` requires rerun behavior.
- Native fast 16quant resampling is unweighted only.
- Some MC random/resampling policy combinations are intentionally incompatible.

## Inspect uncertainty products

```python
print(result.wp_err)
print(result.cov)
print(result.realizations)
print(result.bootstrap_realizations)
print(result.metadata)

result.plot_cov_matrix()
result.plot_corr_matrix()
```

View the spatial region assignment with:

```python
ng.plot_jk_regions(
    data=data,
    random=random,
    config=cfg,
    catalog="data",
)
```
