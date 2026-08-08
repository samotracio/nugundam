import numpy as np

from nugundam.projected.fixed_rp_diagnostic import (
    count_auto_fixed_rp,
    count_cross_fixed_rp,
)
from nugundam.projected.prepare import _prepared_projected_from_arrays


def _prepared(rng, n, *, sbound, rp_pi_edges, mxh3):
    ra = rng.uniform(0.0, 10.0, n)
    dec = rng.uniform(-4.0, 4.0, n)
    dist_pi = rng.uniform(900.0, 1100.0, n)
    dist_rp = rng.uniform(900.0, 1100.0, n)
    weights = rng.uniform(0.5, 1.5, n)
    return _prepared_projected_from_arrays(
        ra=ra,
        dec=dec,
        dist=dist_pi,
        dcang=dist_rp,
        weights=weights,
        sbound=sbound,
        mxh1=8,
        mxh2=32,
        mxh3=mxh3,
        pi_edges=rp_pi_edges,
        grid_meta={"pxorder": "natural"},
        sort_rows=True,
    )


def _brute_auto(sample, rp_edges, pi_edges):
    out = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.float64)
    dpi = pi_edges[1] - pi_edges[0]
    for i in range(sample.nrows):
        for j in range(i + 1, sample.nrows):
            pi = abs(sample.dist[i] - sample.dist[j])
            if pi >= pi_edges[-1]:
                continue
            ang2 = (
                (sample.x[i] - sample.x[j]) ** 2
                + (sample.y[i] - sample.y[j]) ** 2
                + (sample.z[i] - sample.z[j]) ** 2
            )
            rp = np.sqrt(sample.dcang[i] * sample.dcang[j] * ang2)
            irp = np.searchsorted(rp_edges, rp, side="left") - 1
            ipi = int(pi / dpi)
            if 0 <= irp < out.shape[0] and 0 <= ipi < out.shape[1]:
                out[irp, ipi] += sample.weights[i] * sample.weights[j]
    return out


def _brute_cross(left, right, rp_edges, pi_edges):
    out = np.zeros((len(rp_edges) - 1, len(pi_edges) - 1), dtype=np.float64)
    dpi = pi_edges[1] - pi_edges[0]
    for i in range(left.nrows):
        for j in range(right.nrows):
            pi = abs(left.dist[i] - right.dist[j])
            if pi >= pi_edges[-1]:
                continue
            ang2 = (
                (left.x[i] - right.x[j]) ** 2
                + (left.y[i] - right.y[j]) ** 2
                + (left.z[i] - right.z[j]) ** 2
            )
            rp = np.sqrt(left.dcang[i] * right.dcang[j] * ang2)
            irp = np.searchsorted(rp_edges, rp, side="left") - 1
            ipi = int(pi / dpi)
            if 0 <= irp < out.shape[0] and 0 <= ipi < out.shape[1]:
                out[irp, ipi] += left.weights[i] * right.weights[j]
    return out


def test_fixed_rp_numba_counts_match_brute_force():
    rng = np.random.default_rng(3)
    sbound = (0.0, 360.0, -5.0, 5.0, 850.0, 1150.0)
    rp_edges = np.array([0.1, 1.0, 3.0, 10.0, 30.0])
    pi_edges = np.arange(0.0, 61.0, 10.0)

    # The production Fortran SK builder uses nc3=int(distance_span/pi_max)=5.
    # Use that same value in source-tree tests where the Python fallback SK
    # builder otherwise uses mxh3 literally.
    left = _prepared(rng, 80, sbound=sbound, rp_pi_edges=pi_edges, mxh3=5)
    right = _prepared(rng, 60, sbound=sbound, rp_pi_edges=pi_edges, mxh3=5)

    auto = count_auto_fixed_rp(left, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=2)
    cross = count_cross_fixed_rp(left, right, rp_edges=rp_edges, pi_edges=pi_edges, nthreads=2)

    np.testing.assert_allclose(auto, _brute_auto(left, rp_edges, pi_edges), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(cross, _brute_cross(left, right, rp_edges, pi_edges), rtol=2e-6, atol=2e-6)


def test_paired_random_hybrid_preserves_reference_mc_joint_for_inherit():
    from nugundam.projected.fixed_rp_diagnostic import _sample_paired_random_hybrid

    rng = np.random.default_rng(17)
    data_ref = np.array([100.0, 200.0, 300.0, 400.0])
    data_mc = np.array([110.0, 220.0, 330.0, 440.0])
    # CDF is unused by inherit_realization but supplied for API completeness.
    cdf = np.ones((4, 2), dtype=np.float64)
    rmc, rref, donor = _sample_paired_random_hybrid(
        nrand=100,
        data_cdf=cdf,
        data_dist=data_mc,
        data_rp_distance=data_ref,
        chi_grid=np.array([100.0, 200.0]),
        random_mode="inherit_realization",
        rng=rng,
        sample_edges_chi=None,
    )
    np.testing.assert_allclose(rmc, data_mc[donor])
    np.testing.assert_allclose(rref, data_ref[donor])
    np.testing.assert_allclose(rmc / rref, 1.1)


def test_paired_random_hybrid_zero_error_has_equal_pi_and_rp_distances():
    from nugundam.projected.fixed_rp_diagnostic import _sample_paired_random_hybrid

    rng = np.random.default_rng(23)
    data_ref = np.array([100.0, 200.0, 300.0, 400.0])
    cdf = np.ones((4, 2), dtype=np.float64)
    rmc, rref, _ = _sample_paired_random_hybrid(
        nrand=200,
        data_cdf=cdf,
        data_dist=data_ref,
        data_rp_distance=data_ref,
        chi_grid=np.array([100.0, 200.0]),
        random_mode="inherit_realization",
        rng=rng,
        sample_edges_chi=None,
    )
    np.testing.assert_array_equal(rmc, rref)
