import numpy as np
import pytest
from astropy.table import Table

import nugundam.angular.fortran_bridge as fb
from nugundam import AngularAutoConfig, AngularBinning, AngularGridSpec, ProgressSpec, acf
from nugundam.angular.prepare import prepare_angular_auto

# th_A_grp and th_C_grp walk each DEC strip as contiguous blocks of particles
# instead of following the linked list, and must give exactly the same counts as
# th_A and th_C. The fields cover the RA=0 wrap and the pole, where a search
# window spans a whole DEC strip; pxorder="none" exercises the fallback.

FIELDS = {
    "small": (178.0, 182.0, -2.0, 2.0, 0.5),
    "ra_wrap": (350.0, 370.0, -5.0, 5.0, 1.0),
    "pole": (0.0, 360.0, -89.5, -80.0, 2.0),
}


def _uniform(rng, n, ra0, ra1, dec0, dec1):
    ra = rng.uniform(ra0, ra1, n) % 360.0
    s0, s1 = np.sin(np.radians(dec0)), np.sin(np.radians(dec1))
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    return Table({"ra": ra, "dec": dec})


def _prepared(field, pxorder):
    ra0, ra1, dec0, dec1, theta_max = FIELDS[field]
    rng = np.random.default_rng(7)
    data = _uniform(rng, 2500, ra0, ra1, dec0, dec1)
    random = _uniform(rng, 6000, ra0, ra1, dec0, dec1)
    cfg = AngularAutoConfig(
        estimator="LS",
        binning=AngularBinning.from_limits(nsep=12, sepmin=0.01, sepmax=theta_max),
        grid=AngularGridSpec(pxorder=pxorder),
        progress=ProgressSpec(enabled=False),
        nthreads=4,
    )
    return prepare_angular_auto(data, random, cfg)


def _counts(data_p, rand_p, meta):
    common = dict(theta_edges=meta["theta_edges"], nthreads=4, weight_mode="unweighted",
                  doboot=False, nbts=0, bseed=0, progress_file=None)
    dd = fb.run_theta_auto_counts(data_p, cntid="DD", **common)[0]
    rr = fb.run_theta_auto_counts(rand_p, cntid="RR", **common)[0]
    dr = fb.run_theta_cross_counts(data_p, rand_p, cntid="DR", **common)[0]
    return {"dd": np.asarray(dd), "rr": np.asarray(rr), "dr": np.asarray(dr)}


def test_block_kernels_are_the_default(monkeypatch):
    monkeypatch.delenv("NUGUNDAM_LEGACY_KERNELS", raising=False)
    assert fb._use_block_kernels("th_A_grp") and fb._use_block_kernels("th_C_grp")
    monkeypatch.setenv("NUGUNDAM_LEGACY_KERNELS", "1")
    assert not fb._use_block_kernels("th_A_grp") and not fb._use_block_kernels("th_C_grp")


@pytest.mark.parametrize("pxorder", ["natural", "cell-dec", "none"])
@pytest.mark.parametrize("field", list(FIELDS))
def test_block_kernels_match_linked_list_counts(monkeypatch, field, pxorder):
    data_p, rand_p, meta = _prepared(field, pxorder)
    monkeypatch.setenv("NUGUNDAM_LEGACY_KERNELS", "1")
    legacy = _counts(data_p, rand_p, meta)
    monkeypatch.delenv("NUGUNDAM_LEGACY_KERNELS")
    block = _counts(data_p, rand_p, meta)
    for name in legacy:
        assert legacy[name].sum() > 0, name
        np.testing.assert_array_equal(block[name], legacy[name], err_msg=name)


@pytest.mark.parametrize("group_size", [1, 7, 64])
def test_block_kernels_do_not_depend_on_group_size(monkeypatch, group_size):
    data_p, rand_p, meta = _prepared("small", "natural")
    monkeypatch.delenv("NUGUNDAM_LEGACY_KERNELS", raising=False)
    reference = _counts(data_p, rand_p, meta)
    monkeypatch.setattr(fb, "_GROUP_SIZE", group_size)
    counts = _counts(data_p, rand_p, meta)
    for name in reference:
        np.testing.assert_array_equal(counts[name], reference[name], err_msg=name)


def test_acf_is_unchanged_by_block_kernels(monkeypatch):
    rng = np.random.default_rng(11)
    data = _uniform(rng, 3000, 178.0, 182.0, -2.0, 2.0)
    random = _uniform(rng, 9000, 178.0, 182.0, -2.0, 2.0)
    cfg = AngularAutoConfig(
        estimator="LS",
        binning=AngularBinning.from_limits(nsep=10, sepmin=0.01, sepmax=1.0),
        progress=ProgressSpec(enabled=False),
        nthreads=4,
    )
    monkeypatch.setenv("NUGUNDAM_LEGACY_KERNELS", "1")
    legacy = acf(data, random, cfg)
    monkeypatch.delenv("NUGUNDAM_LEGACY_KERNELS")
    block = acf(data, random, cfg)
    np.testing.assert_array_equal(block.wtheta, legacy.wtheta)
