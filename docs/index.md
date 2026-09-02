# νGundam: fast correlation functions for galaxy surveys with PDF-aware methods

nugGundam is a Python package for high-performance two-point correlation functions in galaxy surveys. A modern Python configuration and result layer drives state-of-the-art Fortran/OpenMP pair-counting kernels, linked-cell searches, automatic resampling, plotting, and result I/O.

Version **0.7.1** provides:

- auto/cross-correlations in angular and projected space
- PDF-aware projected clustering with exact-grid, gaussian mixtures, Monte Carlo, and 16quant modes
- marked correlation functions
- NAT, Davis--Peebles, and Landy--Szalay estimators
- fully automatic bootstrap and delete-one jackknife uncertainties
- split-random acceleration for Landy--Szalay autocorrelations
- resolved $(r_p,\pi)$ count fields, $\xi(r_p,\pi)$ views, and integrated $w_p(r_p)$
- ordinary and data-weighted pair counts
- native result files, ASCII export, and plotting utilities

<div class="hero-note">
The public package name and import namespace are <code>nugundam</code>.
</div>

## Main entry points

| Measurement | Auto-correlation | Cross-correlation |
|---|---|---|
| Angular | `acf(data, random, config)` | `accf(data1, data2, config, ...)` |
| Projected | `pcf(data, random, config)` | `pccf(data1, data2, config, ...)` |
| Marked angular | `macf(...)` | `maccf(...)` |
| Marked projected | `mpcf(...)` | `mpccf(...)` |
| Count-only angular | `ang_auto_counts(...)` | `ang_cross_counts(...)` |
| Count-only projected | `proj_auto_counts(...)` | `proj_cross_counts(...)` |

All four PDF representations use the normal `pcf` and `pccf` interfaces. The selected mode is controlled by the nested `pdf`, `pdf_source`, or `mc_pdf` configuration blocks.

## Typical workflow

```python
import nugundam as ng

cfg = ng.ProjectedAutoConfig(
    estimator="LS",
    binning=ng.ProjectedBinning.from_binsize(
        nsepp=16,
        seppmin=0.3,
        dsepp=0.11,
        logsepp=True,
        nsepv=12,
        dsepv=8.0,
    ),
    nthreads=8,
)

result = ng.pcf(data, random, cfg)
result.plot()
result.save("projected_result.gres")
```

## PDF-aware projected clustering

For a pair of objects with radial PDFs $p_i(\chi)$ and $p_j(\chi)$, νGundam targets the expected contribution to a cell of the $(r_p,\pi)$ grid as

$$
W_{ij}^{(bv)} = \omega_i\omega_j
\int \mathrm{d}\chi_1\int \mathrm{d}\chi_2\,
p_i(\chi_1)p_j(\chi_2)
\Theta_{bv}(\theta_{ij},\chi_1,\chi_2),
$$

where $\omega_i$ and $\omega_j$ are optional data-object weights and $\Theta_{bv}$ is the projected/line-of-sight bin indicator. The four modes differ in how they approximate this same PDF-marginalized pair-count target:

| Mode | Configuration | Character |
|---|---|---|
| Exact empirical grid | `pdf.kind="grid_chi_exact"` | deterministic reference on the shared grid |
| Gaussian mixture | `pdf.kind="gmm_chi"` | deterministic, compact analytic LOS probabilities |
| 16quant | `pdf.kind="quantile_chi"` | deterministic equal-probability quantile nodes |
| Monte Carlo | `mc_pdf.enabled=True` | stochastic average of point-distance count fields |

See [PDF-aware methods](pdf_methods.md) for the mathematical definitions and the exact 0.7.1 configuration contract.

## Choose the right guide

- Start with [Installation](installation.md), then use one of the four example templates.
- Read [Correlation-function basics](correlation_basics.md) for estimator and projected-geometry definitions.
- Read [Cross-correlations](cross_correlations.md) before supplying estimator-dependent random arguments.
- Read [PDF-aware methods](pdf_methods.md) before enabling full redshift PDFs.
- Read [Resampling and uncertainties](resampling.md) before interpreting error bars.
- Use the [API reference](api.md) for signatures, nested dataclass fields, and source docstrings.

!!! note "Scope"
    These pages document the **0.7.1 source distribution**. Public behavior, defaults, restrictions, and examples were checked against that source rather than inferred from an older release.
