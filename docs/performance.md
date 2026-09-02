# Performance and large catalogs

νGundam's runtime is controlled by candidate-pair density, output ranges, catalog sizes, random-catalog size, PDF representation, resampling, and memory layout. Benchmark with representative geometry and PDFs; small synthetic tests can rank configurations incorrectly.

## Compile for the target machine

For performance-sensitive production, prefer a local source build. The optional `-march=native` flags described in [Installation](installation.md) allow the compiler to target the current CPU, at the cost of binary portability.

Always verify that the compiled extension is active:

```python
from nugundam.cflibfor import compiled_available
assert compiled_available
```

## Threads

```python
cfg.nthreads = 16
```

More threads do not guarantee proportional speedup. Pair density, memory bandwidth, NUMA placement, and scheduler allocation can dominate. Test a small set of thread counts on the intended node.

## Automatic linked-cell grids

Angular:

```python
cfg.grid.autogrid = "adaptive"   # or True / "legacy"
cfg.grid.pxorder = "natural"
```

Projected:

```python
cfg.grid.autogrid = True
cfg.grid.pxorder = "natural"
```

The grid and preparatory ordering reduce the number and cost of candidate pairs. They should not change scientific bin definitions. Explicit grid dimensions are advanced tuning parameters.

## Split-random LS autocorrelation

For large random catalogs, RR can dominate Landy--Szalay autocorrelation. Split-random divides the prepared random catalog into smaller chunks and combines properly normalized contributions.

```python
cfg.estimator = "LS"
cfg.split_random.enabled = True
cfg.split_random.mode = "match_data"
```

Alternative modes are `"nchunks"` and `"chunk_size"` with the corresponding field. Split-random is restricted to LS autocorrelation and is not available together with jackknife.

## LOS and transverse ranges

Candidate search volume grows with the outer $r_p$ and $\pi$ limits. Broad photometric PDFs can further enlarge conservative radial support. Avoid selecting a very large $\pi_{\max}$ solely because it is affordable in a small pilot sample.

Resolved LOS bins add output resolution and permit later reintegration. They also store larger count fields, though candidate search is primarily controlled by the outer limit rather than the number of shells.

## PDF-mode trade-offs

### Exact grid

- deterministic reference for the adopted grid;
- propagates support through both $r_p$ and $\pi$;
- usually the highest memory/runtime cost;
- `edge_refine` increases radial resolution and cost;
- `prob_floor` can prune support, but nonzero values are approximate.

### GMM

- cost grows with component-pair combinations, approximately $K^2$ per accepted object pair;
- small `k` gives compact memory and analytic LOS shells;
- `edge_moments=True` improves compression of edge-grid histograms;
- current $r_p$ assignment uses component-pair means rather than full component support.

### Monte Carlo

- cost grows roughly with `nreal` times the point-distance workflow;
- different realizations are conceptually independent, although the current public workflow manages them internally;
- `nreal` controls stochastic convergence;
- `resampling_nreal` can be smaller than the full-sample value to control covariance cost;
- `fixed_global` randoms reduce random-term variation, while other policies target different coupling choices.

### 16quant

- deterministic and non-Gaussian;
- each accepted candidate pair can enter an $N_q^2$ quantile-product loop;
- support summaries prune many candidates before that loop;
- `nquant=16` is the default practical starting point;
- `float32` quantile-library storage reduces memory, while the compiled bridge converts data as required by the kernels;
- detailed pair-pruning diagnostics add hot-loop bookkeeping and should remain off in normal production.

A rough per-data-object quantile-library payload is proportional to

$$
N_q\times \mathrm{sizeof}(\mathrm{storage\ dtype}),
$$

plus object indices and support summaries. Actual preparation metadata is available in the pair diagnostic summary.

## 16quant diagnostics

Production:

```python
cfg.pdf.diagnostics = False
result = ng.pcf(data, random, cfg)
print(ng.summarize_pair_diagnostics(result))
```

Diagnostic run:

```python
cfg.pdf.diagnostics = True
result_diag = ng.pcf(data, random, cfg)
ng.print_pair_diagnostics(result_diag)
```

Use detailed counters to determine whether candidate rejection is effective and whether the quantile-product kernel dominates. Disable them for final timing.

## PDF resolution convergence

Treat internal resolution as part of validation:

- exact: grid spacing and `edge_refine`;
- GMM: `k`;
- MC: `nreal`;
- 16quant: `nquant`.

Compare integrated curves, resolved $\xi(r_p,\pi)$, cumulative $w_p(r_p;\Pi)$, and runtime. A mode can agree in LOS response yet differ in final $w_p$ because of $r_p$ migration or estimator normalization.

## Resampling cost

Jackknife-touch and native bootstrap kernels can avoid repeated full preparation/counting. Constraints differ by PDF mode and random policy. In particular, native fast 16quant resampling is unweighted in 0.7.1; weighted resampling must use a rerun path.

Check result metadata rather than assuming the backend:

```python
for key in (
    "pdf_bootstrap_backend",
    "mc_resampling_backend",
    "jk_touch_fast",
):
    print(key, result.metadata.get(key))
```

## Memory checklist

Before a survey-scale run, account for:

- prepared data and random coordinate arrays;
- linked-list/skip-grid arrays;
- two-dimensional DD/DR/RR fields;
- bootstrap arrays and jackknife touch fields;
- empirical PDF matrices;
- exact-grid probabilities/CDFs and active-support indices;
- GMM component libraries;
- 16quant node libraries and support summaries;
- stored MC or resampling realizations;
- split-random chunk work arrays.

Avoid `store_config="full"` with huge in-memory PDF matrices unless the full payload is genuinely required in the result file.

## Reproducibility

Record:

- package version and source-build flags;
- compiler and OpenMP environment;
- thread count;
- grid/autogrid and ordering settings;
- estimator and random policy;
- PDF mode and internal resolution;
- all RNG seeds;
- $r_p$ and $\pi$ edges;
- selected resampling backend;
- relevant pair diagnostics and wall times.
