# PDF-aware methods

νGundam 0.7.1 implements four ways of propagating per-object radial PDFs through projected pair counts:

1. exact empirical-grid integration (`grid_chi_exact`);
2. deterministic Gaussian-mixture compression (`gmm_chi`);
3. Monte Carlo PDF sampling (`mc_pdf`);
4. deterministic quantile compression, **16quant** (`quantile_chi`).

All four produce expected or realization-averaged DD, DR, and RR fields on the same $(r_p,\pi)$ grid. The selected NAT, DP, or LS estimator is then applied cell by cell, and the final $w_p(r_p)$ is obtained by LOS integration.

## Common statistical target

Define the bin indicator

$$
\Theta_{bv}(\theta_{ij},\chi_1,\chi_2)=
\begin{cases}
1, & r_{p,b}\leq \theta_{ij}\sqrt{\chi_1\chi_2}<r_{p,b+1},\\
   & \pi_v\leq |\chi_1-\chi_2|<\pi_{v+1},\\
0, & \text{otherwise}.
\end{cases}
$$

For object PDFs $p_i(\chi)$ and $p_j(\chi)$, the expected pair contribution is

$$
W_{ij}^{(bv)}=\omega_i\omega_j
\int\mathrm{d}\chi_1\int\mathrm{d}\chi_2\,
p_i(\chi_1)p_j(\chi_2)
\Theta_{bv}(\theta_{ij},\chi_1,\chi_2).
$$

The count field is

$$
C_{bv}=\sum_{\langle i,j\rangle}W_{ij}^{(bv)}.
$$

In the spectroscopic limit $p_i(\chi)=\delta_D(\chi-\chi_i)$, the pair reduces to the usual hard assignment to one cell.

## Configuration overview

| Mode | Enable | Primary resolution parameter | Random PDF behavior |
|---|---|---|---|
| Exact grid | `pdf.enabled=True`, `pdf.kind="grid_chi_exact"` | grid and `edge_refine` | inherit data PDF indices |
| GMM | `pdf.enabled=True`, `pdf.kind="gmm_chi"` | `pdf.k` | inherit data GMM indices |
| 16quant | `pdf.enabled=True`, `pdf.kind="quantile_chi"` | `pdf.nquant` | inherit data quantile indices |
| Monte Carlo | `mc_pdf.enabled=True` | `mc_pdf.nreal` | explicit random PDFs or `random_mode` |

`pdf.enabled` and `mc_pdf.enabled` are mutually exclusive.

## Empirical PDF inputs

### External matrix

```python
source = ng.PDFSourceSpec(
    kind="external_matrix",
    matrix=pdf_matrix,
)
```

The matrix can also be loaded from `.npy`, `.npz`, or supported tabular storage:

```python
source = ng.PDFSourceSpec(
    kind="external_matrix",
    path="pdfs.npz",
    array_key="pdf",
)
```

### Vector column

```python
source = ng.PDFSourceSpec(
    kind="vector_column",
    column="pdf",
)
```

### Scalar columns

```python
source = ng.PDFSourceSpec(
    kind="columns",
    columns=["pdf_000", "pdf_001", "pdf_002"],
)
```

A `prefix` can be used instead of an explicit list when the columns follow a stable naming convention.

### Shared grid

The grid may be supplied in redshift or directly in comoving distance:

```python
cfg.pdf_source.z_grid = z_edges
cfg.pdf_source.grid_kind = "edges"
```

or

```python
cfg.pdf_source.chi_grid = chi_edges
cfg.pdf_source.grid_kind = "edges"
```

`chi_grid` overrides `z_grid`. Redshift grids are converted with `cfg.distance` before pair probabilities are evaluated.

## Exact empirical-grid mode

For a shared discrete grid $\{\chi_m\}$ with row probabilities $p_{im}$,

$$
W_{ij,\mathrm{ePDF}}^{(bv)}
=\omega_i\omega_j\sum_m\sum_n
p_{im}p_{jn}\Theta_{bv}(\theta_{ij},\chi_m,\chi_n).
$$

The implementation avoids a naive independent double sum for every LOS shell by evaluating cumulative interval masses and differencing them at adjacent $\pi$ edges. This mode propagates radial support through both $\pi$ and $r_p$ and is the deterministic reference for the adopted grid representation.

```python
cfg.pdf.enabled = True
cfg.pdf.kind = "grid_chi_exact"
cfg.pdf.prob_floor = 0.0

cfg.pdf_source.enabled = True
cfg.pdf_source.chi_grid = chi_edges
cfg.pdf_source.grid_kind = "edges"
cfg.pdf_source.edge_refine = 2
cfg.pdf_source.pdf_data = source
```

### Edge refinement

With edge-grid histograms, `edge_refine > 1` divides each input bin into equal-width sub-bins in $\chi$ and distributes its probability uniformly among them. This reduces radial quantization when requested LOS bins are comparable to the PDF-grid spacing.

### Probability floor

The exact-grid active support is pruned using `pdf.prob_floor`. A nonzero floor can reduce work but changes the retained support and is therefore approximate. Use `0.0` for precision validation when feasible.

## GMM mode

The empirical PDF is compressed into $K$ Gaussian components in comoving distance,

$$
p_i(\chi)\simeq\sum_{a=1}^{K}
\alpha_{ia}\,\mathcal{N}(\chi\mid\mu_{ia},\sigma_{ia}^2),
\qquad \sum_a\alpha_{ia}=1.
$$

The default `segments_equal_mass` compressor divides the empirical CDF into approximately equal-probability segments and records each segment's mass, conditional mean, and conditional variance. It is a deterministic moment compression, not an expectation-maximization fit.

For component pair $(a,c)$, $X=\chi_i-\chi_j$ is Gaussian with

$$
m_{iacj}=\mu_{ia}-\mu_{jc},
\qquad
s_{iacj}^2=\sigma_{ia}^2+\sigma_{jc}^2.
$$

The probability in LOS shell $[\pi_v,\pi_{v+1})$ is obtained from the folded-normal CDF,

$$
P_{ia,cj}^{(v)}=
F_{|X|}(\pi_{v+1};m_{iacj},s_{iacj})-
F_{|X|}(\pi_v;m_{iacj},s_{iacj}).
$$

The compiled contribution is

$$
W_{ij,\mathrm{GMM}}^{(bv)}
=\omega_i\omega_j\sum_a\sum_c
\alpha_{ia}\alpha_{jc}
B_{b,ia,cj}(\theta_{ij})P_{ia,cj}^{(v)},
$$

where $B_b$ assigns $r_p$ from the component-pair means. Thus GMM integrates LOS uncertainty analytically, but its current $r_p$ treatment is an approximation relative to the full two-dimensional empirical-grid integral.

```python
cfg.pdf.enabled = True
cfg.pdf.kind = "gmm_chi"
cfg.pdf.k = 3
cfg.pdf.rv_search_nsigma = 4.0
cfg.pdf.prob_floor = 1.0e-10

cfg.pdf_source.enabled = True
cfg.pdf_source.z_grid = z_edges
cfg.pdf_source.grid_kind = "edges"
cfg.pdf_source.edge_moments = True
cfg.pdf_source.pdf_data = source
```

`edge_moments=True` treats mass as uniform inside each input bin and includes finite-bin width in the compressed moments.

Precomputed GMM columns are also supported through `alpha_cols`, `mu_cols`, and `sigma_cols`, or their prefix equivalents.

## Monte Carlo mode

For each realization $r=1,\ldots,R$, one scalar distance is drawn per object,

$$
\chi_i^{(r)}\sim p_i(\chi),
$$

and the normal point-distance projected counter is run. Each count term is averaged before estimator evaluation:

$$
\bar C_{bv}=\frac{1}{R}\sum_{r=1}^{R}C_{bv}^{(r)}.
$$

MC samples migration in both $\pi$ and $r_p$. Its finite-realization fluctuations decrease approximately as $R^{-1/2}$ at fixed data and binning.

```python
cfg.mc_pdf.enabled = True
cfg.mc_pdf.nreal = 25
cfg.mc_pdf.seed = 12345
cfg.mc_pdf.z_grid = z_edges
cfg.mc_pdf.grid_kind = "edges"
cfg.mc_pdf.pdf_data = source
cfg.mc_pdf.sample_within_bin = True
```

### Random modes

When explicit `pdf_random` is absent:

- `fixed_global`: draw one random radial realization from the global mean data PDF and reuse it;
- `rerun_global`: draw a fresh global random realization for every MC realization;
- `inherit_realization`: sample random distances with replacement from the current realized data distances.

For cross-correlations, the policy is applied separately to each side.

### Within-bin sampling

`sample_within_bin=True` samples continuously within a selected PDF bin. For edge grids, the actual edges are used; for center grids, pseudo-edges are inferred. This is the MC counterpart of edge-aware GMM compression and exact-grid refinement.

### Stored MC diagnostics

```python
cfg.mc_pdf.store_realizations = True
```

This stores full-sample per-realization $w_p$ curves and `mc_wp_std`. The main estimate still comes from averaged count fields, not from averaging the stored final curves.

## 16quant mode

16quant replaces each PDF by $N_q$ equal-probability quantile nodes in comoving distance. Let $Q_i(u)=F_i^{-1}(u)$. Midpoint nodes are

$$
u_a=\frac{a-1/2}{N_q},
\qquad
\chi_{ia}=Q_i(u_a),
\qquad a=1,\ldots,N_q.
$$

The pair contribution is the equally weighted Cartesian product

$$
W_{ij,Q}^{(bv)}=
\frac{\omega_i\omega_j}{N_q^2}
\sum_{a=1}^{N_q}\sum_{c=1}^{N_q}
\Theta_{bv}(\theta_{ij},\chi_{ia},\chi_{jc}).
$$

This deterministic representation:

- has no realization noise;
- makes no Gaussian assumption;
- propagates finite-node support through both $\pi$ and $r_p$;
- costs approximately as a quantile-node Cartesian product for candidate pairs;
- is controlled primarily by `nquant`.

```python
cfg.pdf.enabled = True
cfg.pdf.kind = "quantile_chi"
cfg.pdf.nquant = 16
cfg.pdf.quantile_positions = "midpoint"
cfg.pdf.quantile_storage = "float32"

cfg.pdf_source.enabled = True
cfg.pdf_source.z_grid = z_edges
cfg.pdf_source.grid_kind = "edges"
cfg.pdf_source.pdf_data = source
```

In 0.7.1, only midpoint positions are supported. `quantile_storage` accepts `"float32"` and `"float64"`; `"uint16"` is reserved for a future backend. Precomputed quantile columns are not accepted by this branch: 16quant requires empirical `pdf_source` inputs.

!!! note "Current implementation status"
    The 0.7.1 source describes `quantile_chi` as an experimental CPU-first mode, but it is fully connected to projected auto/cross main counts, resolved multi-$\pi$ binning, inherited random assignments, and native unweighted resampling kernels. Treat performance and convergence as analysis settings to benchmark on the intended catalog.

## Deterministic random inheritance

For exact-grid, GMM, and 16quant, random catalogs inherit prepared PDF-library indices from the associated data sample with replacement:

```python
cfg.pdf.random_pdf_policy = "inherit"
```

This is the only deterministic random-PDF policy implemented in 0.7.1. It avoids storing an independent large PDF library for randoms. In cross-correlation, inheritance is performed separately on each side.

`pdf.jk_random_policy` controls resampling:

- `"fixed"`: keep full-run inherited assignments and enable fast touch jackknife where supported;
- `"reinherit"`: redraw from each surviving data sample and use rerun behavior.

## Estimating an LOS scale from PDFs

```python
estimate = ng.estimate_pi_max_from_pdfs(
    pdf_matrix,
    z_grid=z_edges,
    grid_kind="edges",
    distance=cfg.distance,
    statistic="median_variance",
    multiplier=2.5,
)

print(estimate.pi_max_guess)
print(estimate.sigma_pw_eff)
```

Conceptually the helper returns

$$
\pi_{\max}^{\mathrm{guess}}=m_\pi\,\sigma_{\mathrm{pair,eff}}.
$$

The **code default** is `multiplier=2.5`. This is a data-driven starting point, not part of the estimator. A production analysis should inspect cumulative $w_p(r_p;\Pi)$, simulations, or spectroscopic validation to balance signal recovery against foreground--background leakage.

For a photo--photo cross-correlation, pass `pdfs2`; for a spectroscopic--photometric width estimate, use `sample2_kind="spec"` as documented by the API.

## Inspecting GMM compression

```python
compressed = ng.compress_pdfs_to_gmm(
    pdf_matrix,
    z_grid=z_edges,
    grid_kind="edges",
    distance=cfg.distance,
    k=3,
)

ng.plot_gmm_for_object(
    0,
    pdfs=pdf_matrix,
    z_grid=z_edges,
    grid_kind="edges",
    distance=cfg.distance,
    compressed=compressed,
)
```

## Selecting a mode

| Goal | Good starting point |
|---|---|
| Deterministic reference on a manageable sample | exact grid, zero floor, enough edge refinement |
| Fast deterministic representation without a Gaussian assumption | 16quant with `nquant=16` |
| Compact analytic LOS representation | GMM with `k=2` or `3` |
| Reuse of ordinary point-distance counters and direct convergence test | MC with `nreal≈25` |
| Sensitivity study | compare at least two modes and inspect resolved/cumulative LOS behavior |

No single internal resolution is universally sufficient. Validate `nquant`, `k`, `nreal`, `edge_refine`, and $\pi_{\max}$ for the actual PDF widths, grid spacing, sample selection, and required accuracy.
