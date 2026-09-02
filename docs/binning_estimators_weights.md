# Binning, estimators, and weights

## Angular binning

`AngularBinning` is created with a named constructor.

```python
import nugundam as ng

bins = ng.AngularBinning.from_binsize(
    nsep=21,
    sepmin=0.000277778,
    dsep=0.17,
    logsep=True,
)
```

For logarithmic bins, `dsep` is a step in dex. Alternatively:

```python
bins = ng.AngularBinning.from_limits(
    nsep=21,
    sepmin=0.000277778,
    sepmax=1.0,
    logsep=True,
)
```

Inspect the resolved arrays with `edges`, `centers`, `widths`, `sepmax`, and `table()`.

## Projected binning

```python
bins = ng.ProjectedBinning.from_binsize(
    nsepp=16,
    seppmin=0.3,
    dsepp=0.11,
    logsepp=True,
    nsepv=12,
    dsepv=8.0,
)
```

The transverse bins may be specified from a size or limits. The LOS axis consists of `nsepv` equal-width bins of width `dsepv`:

```python
print(bins.rp_edges)
print(bins.rp_centers)
print(bins.pi_edges)
print(bins.pi_centers)
print(bins.sepvmax)
print(bins.table("rp"))
print(bins.table("pi"))
```

`bins.sepvmax` is the outer LOS integration edge.

!!! tip "Use resolved LOS bins when possible"
    Resolved bins preserve $\xi(r_p,\pi)$ and allow reintegration to any stored edge. A single wide bin is faster and can be useful as a coarse projected-window measurement, but it does not preserve the shell structure and is not guaranteed to match a resolved measurement with the same outer edge.

## Estimator requirements

| Estimator | Auto terms | Cross terms | Random requirement |
|---|---|---|---|
| `NAT` | DD, RR | D1D2, R1R2 | auto random; both cross randoms |
| `DP` | DD, DR | D1D2 plus one mixed term | auto random; random for the non-primary cross side |
| `LS` | DD, DR, RR | all four terms | auto random; both cross randoms |

The scientific estimator choice depends on the survey and random-catalog design. νGundam does not silently substitute one estimator for another.

## Weight modes

```python
weights = ng.WeightSpec(weight_mode="auto")
```

- `"unweighted"`: force the unit-weight path;
- `"weighted"`: require and use configured data weights;
- `"auto"`: inspect relevant data weights and select the path.

Randoms remain unweighted. Cross samples may use different data-weight columns through the two catalog mappings or `data1_col`/`data2_col`.

## Marks are not ordinary weights

Pass marks through `AutoMarkSpec` or `CrossMarkSpec`, not through the ordinary weight configuration. The marked wrapper builds a plain unweighted branch and a branch weighted by processed positive marks.

## Linked-cell grids

The grid organizes the candidate-pair search; it does not define scientific output bins.

### Angular

`AngularGridSpec.autogrid` accepts:

- `True` or `"legacy"`: original heuristic;
- `"adaptive"`: runtime-aware footprint probe;
- `False`: explicit `mxh1`, `mxh2`.

Angular `pxorder` accepts `"natural"`, `"cell-dec"`, or `"none"`.

### Projected

`ProjectedGridSpec.autogrid` is boolean in 0.7.1. With `False`, set `mxh1`, `mxh2`, and `mxh3`. Projected `pxorder` accepts `"natural"` or `"none"`.

Start with automatic grid selection and benchmark alternatives only on representative catalog sizes and geometry.
