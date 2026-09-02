# Correlation-function basics

## Angular correlation

The angular two-point correlation function $w(\theta)$ describes the excess probability of finding two objects separated by an angle $\theta$ relative to an unclustered catalog with the same selection function:

$$
\mathrm{d}P=n^2[1+w(\theta)]\,\mathrm{d}\Omega_1\mathrm{d}\Omega_2.
$$

νGundam bins unordered pairs for autocorrelations and pairs between two distinct input catalogs for cross-correlations.

## Projected geometry

For angular separation $\theta_{ij}$ and comoving radial distances $\chi_1$ and $\chi_2$, the projected kernels use the small-angle definitions

$$
\pi=|\chi_1-\chi_2|,
\qquad
r_p\simeq \theta_{ij}\sqrt{\chi_1\chi_2}.
$$

The angular separation is evaluated from the three-dimensional unit vectors of the two sky directions. The geometric-mean distance makes the transverse definition symmetric under exchange of the two objects.

The two-dimensional estimator is evaluated on bins $(b,v)$ of $(r_p,\pi)$. The projected statistic is

$$
w_p(r_p)=2\int_0^{\pi_{\max}}\xi(r_p,\pi)\,\mathrm{d}\pi,
$$

and in the resolved implementation,

$$
w_p(r_{p,b};\pi_{\max})
\simeq 2\sum_{v=0}^{N_\pi-1}\xi_{bv}\,\Delta\pi_v.
$$

Cumulative measurements at each stored LOS edge are therefore

$$
w_p(r_{p,b};\Pi_q)
\simeq 2\sum_{v=0}^{q-1}\xi_{bv}\,\Delta\pi_v,
\qquad \Pi_q=\pi_q.
$$

!!! important "Resolved and one-bin projected measurements"
    With several LOS bins, the estimator is evaluated in every $(r_p,\pi)$ shell and the shell values are integrated. With one wide LOS bin, the pair counts are first combined over the entire window and the estimator is applied once. Because NAT, DP, and LS are nonlinear ratios of normalized counts, the two discretizations can differ even when they share the same outer $\pi_{\max}$.

## Normalized pair-count notation

The formulas below use normalized count fields. For an unweighted auto-correlation with $N_D$ data and $N_R$ random objects,

$$
DD_n=\frac{DD}{N_D(N_D-1)/2},\qquad
DR_n=\frac{DR}{N_DN_R},\qquad
RR_n=\frac{RR}{N_R(N_R-1)/2}.
$$

For cross-correlations,

$$
D_1D_2{}_n=\frac{D_1D_2}{N_1N_2},
$$

with analogous normalizations for $D_1R_2$, $R_1D_2$, and $R_1R_2$.

The same formulas apply to angular bins and cell-by-cell on the projected $(r_p,\pi)$ grid.

## Autocorrelation estimators

### Natural

$$
\xi_{\mathrm{NAT}}=\frac{DD_n}{RR_n}-1.
$$

Required terms: `DD`, `RR`.

### Davis--Peebles

$$
\xi_{\mathrm{DP}}=\frac{DD_n}{DR_n}-1.
$$

Required terms: `DD`, `DR`.

### Landy--Szalay

$$
\xi_{\mathrm{LS}}
=\frac{DD_n-2DR_n+RR_n}{RR_n}.
$$

Required terms: `DD`, `DR`, `RR`.

## Cross-correlation estimators

### Natural

$$
\xi_{12,\mathrm{NAT}}
=\frac{D_1D_2{}_n}{R_1R_2{}_n}-1.
$$

### Davis--Peebles

For `bootstrap.primary="data1"`, νGundam uses

$$
\xi_{12,\mathrm{DP}}
=\frac{D_1D_2{}_n}{D_1R_2{}_n}-1.
$$

Only `random2` is required. With `primary="data2"`, the sample roles are swapped internally and only `random1` is required.

### Landy--Szalay

$$
\xi_{12,\mathrm{LS}}
=\frac{D_1D_2{}_n-D_1R_2{}_n-R_1D_2{}_n+R_1R_2{}_n}
       {R_1R_2{}_n}.
$$

Both random catalogs are required.

## Weighted pair counts

νGundam's public weight model applies object weights to data samples while keeping randoms unweighted. For auto-correlations with data weights $w_i$,

$$
N_{DD}^{(w)}=\frac{1}{2}
\left[\left(\sum_i w_i\right)^2-\sum_i w_i^2\right],
\qquad
N_{DR}^{(w)}=N_R\sum_i w_i.
$$

For cross-correlations,

$$
N_{D_1D_2}^{(w)}=
\left(\sum_i w_i^{(1)}\right)
\left(\sum_j w_j^{(2)}\right).
$$

These normalization sums are stored in result metadata when needed to reconstruct estimators from saved weighted counts.

## PDF-aware pair counts

A point-redshift pair contributes to one $(r_p,\pi)$ cell. With radial PDFs $p_i(\chi)$ and $p_j(\chi)$, its expected contribution is distributed over all cells allowed by the PDF support:

$$
W_{ij}^{(bv)}=\omega_i\omega_j
\int\mathrm{d}\chi_1\int\mathrm{d}\chi_2\,
p_i(\chi_1)p_j(\chi_2)
\Theta_{bv}(\theta_{ij},\chi_1,\chi_2).
$$

The expected count field is the sum over catalog pairs,

$$
C_{bv}=\sum_{\langle i,j\rangle}W_{ij}^{(bv)}.
$$

After DD, DR, and RR are accumulated, estimator algebra and the LOS integration are unchanged. The four PDF modes differ only in how they evaluate or approximate $W_{ij}^{(bv)}$.

## Marked statistics

The marked wrapper computes a plain branch and a mark-weighted branch with matched configuration. For angular measurements,

$$
M(\theta)=\frac{1+w_{\mathrm{marked}}(\theta)}{1+w(\theta)},
$$

and for projected measurements,

$$
M(r_p)=
\frac{1+w_{p,\mathrm{marked}}(r_p)/r_p}
     {1+w_p(r_p)/r_p}.
$$

When resampling is enabled, νGundam evaluates the marked ratio in matched plain/weighted realizations rather than propagating two independent diagonal errors.
