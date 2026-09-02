# Results, plotting, and I/O

## Main result classes

- `AngularCorrelationResult`;
- `ProjectedCorrelationResult`;
- `MarkedAngularCorrelationResult`;
- `MarkedProjectedCorrelationResult`;
- angular and projected count-result classes.

### Angular result fields

```python
result.theta_edges
result.theta_centers
result.wtheta
result.wtheta_err
result.estimator
result.counts
result.cov
result.realizations
result.metadata
```

### Projected result fields

```python
result.rp_edges
result.rp_centers
result.wp
result.wp_err
result.estimator
result.counts
result.cov
result.realizations
result.mc_wp_std
result.mc_realizations
result.bootstrap_realizations
result.metadata
```

The nested projected count object stores `rp_edges`, `pi_edges`, and two-dimensional pair-count arrays with shape `(n_rp, n_pi)`. Depending on estimator and run type, fields include `dd`, `dr`, `rr`, or `d1d2`, `d1r2`, `r1d2`, and `r1r2`.

## Configuration and provenance

`store_config` controls the configuration snapshot in metadata:

- `"none"`;
- `"compact"` (default);
- `"full"`.

Large in-memory PDF matrices are represented compactly unless full storage is requested. Metadata also records catalog sizes, estimator details, PDF representation, random policy, backends, preparation choices, and selected diagnostics.

## Native result files

```python
import nugundam as ng

ng.write_result(result, "run.gres")
loaded = ng.read_result("run.gres")
```

Equivalent object methods are:

```python
result.save("run.gres")
result.write("run.gres")
loaded = type(result).read_result("run.gres")
```

The native format is a compressed NumPy container with JSON metadata and reconstructs nested dataclass/count structures.

## ASCII export

```python
result.to_ascii("result.txt")
```

Select fields explicitly when needed:

```python
result.to_ascii(
    "result_custom.txt",
    cols=["rp_centers", "wp", "wp_err"],
)
```

## One-dimensional plotting

```python
ax = result.plot(label="sample", errors="bar")
ax.legend()
```

Uncertainty styles are `"bar"`, `"band"`, and `"none"`. Overlay results by passing an existing axis.

## Two-dimensional projected views

```python
result.plot2d(which="xi")
```

The result-aware helper reconstructs the requested field from the stored count terms. Low-level alternatives are `plotcf2d` and `plot_result2d`.

## Covariance and correlation matrices

```python
result.plot_cov_matrix()
result.plot_corr_matrix()
```

The top-level functions also accept raw square matrices and optional bin coordinates.

## Comparison and ratio panels

```python
fig, axes = ng.plot_compare_ratio(
    {"point": point_result, "16quant": q16_result, "exact": exact_result},
    ratios=[
        {
            "numerator": "16quant",
            "denominator": "exact",
            "label": "16quant / exact",
        },
    ],
)
```

Curve keys are arbitrary labels used by the explicit ratio definitions.

## Pair diagnostics

Every projected run records lightweight pair-count timing and relevant preparation metadata. For 16quant, detailed hot-kernel acceptance/pruning counters are collected only when `cfg.pdf.diagnostics=True`.

```python
print(ng.summarize_pair_diagnostics(result))
ng.print_pair_diagnostics(result)
```

The summary may include pair wall times, quantile candidate/acceptance information, quantile-library memory, and split-random chunk details.

## Reintegrate a stored projected run

A resolved projected result can be integrated to a smaller LOS limit that coincides with a stored edge without rerunning pair counts.

```python
import numpy as np
from nugundam.projected.estimators import compute_auto_xi2d

counts = result.counts
xi2d = compute_auto_xi2d(
    counts,
    estimator=result.estimator,
)

pi_edges = np.asarray(counts.pi_edges)
pi_widths = np.diff(pi_edges)
pi_max_new = 88.0

keep = pi_edges[1:] <= pi_max_new + 1.0e-12
wp_new = 2.0 * np.sum(
    xi2d[:, keep] * pi_widths[keep][None, :],
    axis=1,
)
rp_centers = np.asarray(result.rp_centers)
```

For weighted runs, pass the stored data-weight sums needed by `compute_auto_xi2d`. For cross-correlations, use `compute_cross_xi2d`.

!!! warning "A limit inside a stored bin is unresolved"
    If stored edges are separated by 8, a requested limit of 90 lies inside a bin. The saved count field contains no information about the within-bin distribution. Use a complete edge such as 88 or 96, state a partial-bin approximation, or rerun with an edge at the desired limit.

## Cumulative projected curves

The same shell structure can be used to compute all cumulative stored edges:

```python
cumulative_wp = 2.0 * np.cumsum(
    xi2d * pi_widths[None, :],
    axis=1,
)
pi_limits = pi_edges[1:]
```

These curves are especially useful in PDF-aware runs for diagnosing LOS convergence and leakage.
