# Example 4: marked correlations

A marked measurement compares an ordinary unweighted correlation with a second branch in which data objects carry processed mark values. νGundam runs both branches with matched settings and combines their matched resampling realizations when errors are requested.

<div class="placeholder-block">
<strong>To add later:</strong> physical definition of the mark, sample selection, normalization rationale, mark distribution, timing, and final plots.
</div>

## Projected auto-correlation template

Start from a valid point-redshift projected configuration `cfg`:

```python
import nugundam as ng

mark = ng.AutoMarkSpec(
    column="mark",
    normalize="mean",
    transform="identity",
    clip=None,
    missing="raise",
)

marked_result = ng.mpcf(
    data,
    random,
    cfg,
    mark=mark,
)
```

The default projected statistic is

$$
M(r_p)=
\frac{1+w_{p,\mathrm{marked}}(r_p)/r_p}
     {1+w_p(r_p)/r_p}.
$$

Only the data catalog requires a mark column. Random catalogs remain unweighted.

## Angular auto-correlation template

```python
marked_angular = ng.macf(
    data,
    random,
    angular_cfg,
    mark=ng.AutoMarkSpec(column="mark", normalize="mean"),
)
```

The angular statistic is

$$
M(\theta)=\frac{1+w_{\mathrm{marked}}(\theta)}{1+w(\theta)}.
$$

## Mark preprocessing

=== "Rank transform"

    ```python
    rank_mark = ng.AutoMarkSpec(
        column="environment",
        transform="rank",
        normalize="mean",
    )
    ```

    Ranks reduce dependence on units and long tails but change the scientific definition of the mark.

=== "Clipping"

    ```python
    clipped_mark = ng.AutoMarkSpec(
        column="stellar_mass_mark",
        clip=(0.2, 5.0),
        normalize="mean",
    )
    ```

=== "Drop non-finite values"

    ```python
    tolerant_mark = ng.AutoMarkSpec(
        column="mark",
        missing="drop",
    )
    ```

Processed marks must be strictly positive. Mark values are not ordinary systematic weights and should not be configured as the regular `weight` column.

## Inspect the result

```python
print(marked_result.mrp)
print(marked_result.mrp_err)
print(marked_result.plain_wp)
print(marked_result.weighted_wp)

marked_result.plot(label="marked sample")
marked_result.save("marked_projected.gres")
```

## Marked cross-correlation

```python
cross_mark = ng.CrossMarkSpec(
    column1="mark_sample_1",
    column2=None,
    mark_on="data1",
    normalize="mean",
)

marked_cross = ng.mpccf(
    data1,
    data2,
    cross_cfg,
    mark=cross_mark,
    random1=random1,
    random2=random2,
)
```

`mark_on` may be `"data1"`, `"data2"`, or `"both"`. The estimator-specific random-catalog contract is unchanged from ordinary cross-correlation.
