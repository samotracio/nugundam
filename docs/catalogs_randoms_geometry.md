# Catalogs, randoms, and geometry

## Supported table types

The public APIs accept table-like inputs with named one-dimensional columns:

- Astropy `Table`;
- pandas `DataFrame`;
- PyArrow table;
- NumPy structured array or record array;
- mapping such as `dict[str, array]`.

The preparation layer extracts requested columns once, converts numerical arrays to the forms expected by the compiled counters, and applies the configured preparatory ordering.

## Angular columns

```python
import nugundam as ng

columns = ng.CatalogColumns(
    ra="RA",
    dec="DEC",
    weight="systematic_weight",
    region="jk_region",
)
```

| Field | Purpose |
|---|---|
| `ra` | right ascension in degrees |
| `dec` | declination in degrees |
| `weight` | optional data-object weight |
| `region` | optional integer jackknife label |

Use a consistent RA convention, normally $[0,360)$, and physical declinations in $[-90,90]$.

## Projected columns

```python
columns = ng.ProjectedCatalogColumns(
    ra="RA",
    dec="DEC",
    redshift="Z",
    distance="CHI",
    weight="weight",
    region="jk_region",
)
```

`redshift` is read when `DistanceSpec.calcdist=True`; `distance` is read when `calcdist=False`.

## Distance convention and units

When `calcdist=True`, νGundam constructs an Astropy `LambdaCDM` cosmology from `h0`, `omegam`, and `omegal`, then uses the numerical value of the comoving distance. Projected bin edges, supplied `chi_grid` arrays, and precomputed distance columns must use the same numerical unit.

Many analyses choose `h0=100 km s^-1 Mpc^-1` and label the resulting numbers $h^{-1}\,\mathrm{Mpc}$. νGundam does not apply an additional hidden factor of $h$. State the adopted convention and keep all inputs consistent.

## Random catalogs

Randoms represent the survey selection in the absence of clustering, not merely a large uniform point set. Angular randoms should reproduce the footprint, masks, holes, and relevant completeness variation. Projected randoms must also reproduce the radial selection.

!!! warning "Randoms are unweighted"
    In 0.7.1, the public weight model applies ordinary weights and marks to data objects. Random catalogs remain unweighted.

For deterministic PDF modes, randoms normally inherit PDF-library indices from their associated data sample. For Monte Carlo, explicit random PDF matrices are supported, or radial realizations can be generated through the configured `random_mode`.

## Empirical PDF alignment

An external PDF matrix has shape

```text
(n_catalog_objects, n_pdf_bins)
```

unless the input uses a vector-valued or multi-column table representation. Rows must follow the catalog order unless IDs are configured:

```python
source = ng.PDFSourceSpec(
    path="pdfs.parquet",
    column="pdf",
    id_column="object_id",
    catalog_id_column="object_id",
)
```

Supported source forms are:

- `kind="external_matrix"`: in-memory matrix or `.npy`, `.npz`, or compatible path;
- `kind="vector_column"`: one vector-valued table column;
- `kind="columns"`: many scalar columns, explicitly listed or discovered by prefix.

The number of PDF values per row must match the common grid:

- centers: `n_pdf_bins == len(grid)`;
- edges: `n_pdf_bins == len(grid) - 1`.

## Center grids versus edge grids

For histogram-like PDFs, `grid_kind="edges"` is usually preferable:

- MC can sample continuously within the selected bin;
- GMM can include finite-bin moments;
- exact-grid can refine bins in $\chi$;
- 16quant can interpolate the inverse CDF inside the bin.

This avoids treating every PDF bin as a point mass at its center.

## PDF normalization

PDF rows are normalized during preparation. Before a production run, check for:

- negative or non-finite values;
- rows with zero total probability;
- inconsistent grid shape;
- duplicated or missing IDs after alignment;
- support that extends beyond the selected cosmological/redshift range.

`pdf_source.eps` may add a small floor before normalization, but it changes the represented PDF and should not be used as a substitute for data validation.

## Jackknife regions

A configured region column supplies integer region labels. Otherwise νGundam can generate regions automatically from the sky geometry:

```python
cfg.jackknife.enabled = True
cfg.jackknife.nregions = 30
cfg.jackknife.generator = "kmeans"
cfg.jackknife.geometry_from = "auto"
```

Cross-correlations require a common spatial partition across all relevant catalogs. The preparation layer maps the generated geometry consistently.

## Input checklist

Before a long run, confirm:

1. RA, Dec, redshift, distance, weight, and region mappings;
2. finite values after sample cuts;
3. the same distance unit in catalogs, PDF grids, and binning;
4. random catalogs reproduce angular and radial selection;
5. PDF rows align with the intended catalog rows;
6. the estimator's random-catalog contract is satisfied;
7. the selected $r_p$ and $\pi$ ranges are scientifically and computationally sensible.
