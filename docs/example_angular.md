# Example 1: angular correlations

This page is a structured template for a future survey-specific example. The objects named `data` and `random` are intentionally not created here.

## Required inputs

Prepare two table-like objects:

- `data`: the science sample with right ascension and declination;
- `random`: points sampling the same angular selection function.

<div class="placeholder-block">
<strong>To add later:</strong> sample definition, masks, random-catalog construction, quality cuts, catalog sizes, timing, and final plots.
</div>

## Configuration template

```python
import nugundam as ng

cfg = ng.AngularAutoConfig(
    estimator="LS",
    columns_data=ng.CatalogColumns(
        ra="ra",
        dec="dec",
        weight="weight",
    ),
    columns_random=ng.CatalogColumns(
        ra="ra",
        dec="dec",
    ),
    binning=ng.AngularBinning.from_binsize(
        nsep=20,
        sepmin=0.001,   # degrees
        dsep=0.15,      # dex because logsep=True
        logsep=True,
    ),
    grid=ng.AngularGridSpec(
        autogrid=True,
        pxorder="natural",
    ),
    weights=ng.WeightSpec(weight_mode="unweighted"),
    bootstrap=ng.BootstrapSpec(enabled=False),
    jackknife=ng.JackknifeSpec(enabled=False),
    progress=ng.ProgressSpec(enabled=True),
    split_random=ng.SplitRandomSpec(enabled=False),
    nthreads=8,
    description="Placeholder angular auto-correlation example",
)

result = ng.acf(data, random, cfg)
```

Use `weight_mode="auto"` when the configured data weight column may contain non-unit values. Random catalogs remain unweighted.

## Inspect the result

```python
print(result.theta_edges)
print(result.theta_centers)
print(result.wtheta)
print(result.wtheta_err)
print(result.estimator)
print(result.metadata)
```

## Plot and save

```python
ax = result.plot(label="sample", errors="bar")
ax.legend()

result.save("angular_result.gres")
result.to_ascii("angular_result.txt")
```

## Common variations

=== "Davis--Peebles"

    ```python
    cfg.estimator = "DP"
    result_dp = ng.acf(data, random, cfg)
    ```

=== "Natural"

    ```python
    cfg.estimator = "NAT"
    result_nat = ng.acf(data, random, cfg)
    ```

=== "Jackknife"

    ```python
    cfg.bootstrap.enabled = False
    cfg.jackknife.enabled = True
    cfg.jackknife.nregions = 25
    cfg.jackknife.return_cov = True

    result_jk = ng.acf(data, random, cfg)
    result_jk.plot_cov_matrix()
    ```

=== "Split random"

    ```python
    cfg.estimator = "LS"
    cfg.jackknife.enabled = False
    cfg.split_random.enabled = True
    cfg.split_random.mode = "match_data"

    result_split = ng.acf(data, random, cfg)
    ```

Split-random counting is restricted to Landy--Szalay autocorrelations and cannot be combined with jackknife in 0.7.1.
