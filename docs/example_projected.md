# Example 2: projected correlations

This page provides the skeleton of a point-redshift projected-correlation analysis. The sample and plots are intentionally left for a later revision.

## Required inputs

Prepare:

- `data`: angular positions plus redshift or comoving distance;
- `random`: angular positions plus a radial selection matching the data.

<div class="placeholder-block">
<strong>To add later:</strong> sample selection, cosmology, random-redshift assignment, the scientific choice of $\pi_{\max}$, timings, and plots of $\xi(r_p,\pi)$ and $w_p(r_p)$.
</div>

## Redshift-based template

```python
import nugundam as ng

cfg = ng.ProjectedAutoConfig(
    estimator="LS",
    columns_data=ng.ProjectedCatalogColumns(
        ra="ra",
        dec="dec",
        redshift="z",
        weight="weight",
    ),
    columns_random=ng.ProjectedCatalogColumns(
        ra="ra",
        dec="dec",
        redshift="z",
    ),
    binning=ng.ProjectedBinning.from_binsize(
        nsepp=16,
        seppmin=0.3,
        dsepp=0.11,
        logsepp=True,
        nsepv=12,
        dsepv=8.0,
    ),
    grid=ng.ProjectedGridSpec(
        autogrid=True,
        pxorder="natural",
    ),
    distance=ng.DistanceSpec(
        calcdist=True,
        h0=100.0,
        omegam=0.3,
        omegal=0.7,
    ),
    weights=ng.WeightSpec(weight_mode="unweighted"),
    bootstrap=ng.BootstrapSpec(enabled=False),
    jackknife=ng.JackknifeSpec(enabled=False),
    nthreads=8,
    description="Placeholder projected auto-correlation example",
)

result = ng.pcf(data, random, cfg)
```

The configured LOS limit is

$$
\pi_{\max}=N_\pi\,\Delta\pi=12\times8=96.
$$

The numerical unit must match the comoving-distance convention used during catalog preparation.

## Use precomputed distances

```python
cfg.distance.calcdist = False
cfg.columns_data.distance = "comoving_distance"
cfg.columns_random.distance = "comoving_distance"

result = ng.pcf(data, random, cfg)
```

## Inspect and plot

```python
print(result.rp_centers)
print(result.wp)
print(result.wp_err)
print(result.counts.pi_edges)

result.plot(label="sample", errors="bar")
result.plot2d(which="xi")
```

For resolved LOS bins, νGundam evaluates the estimator in each cell and then integrates

$$
w_p(r_{p,b};\pi_{\max})
\simeq 2\sum_{v=0}^{N_\pi-1}\xi_{bv}\,\Delta\pi_v.
$$

!!! important "One wide LOS bin is not the same discretization"
    A single bin spanning $0\leq\pi<\pi_{\max}$ applies the nonlinear estimator after counts have already been accumulated over the full window. A resolved multi-$\pi$ run applies it shell by shell and then sums $\xi$. Equal outer limits therefore do not guarantee numerically identical results.

## Save the result

```python
result.save("projected_result.gres")
result.to_ascii("projected_result.txt")
```

The native result preserves the count grids, configuration snapshot, metadata, and any stored covariance or realizations.
