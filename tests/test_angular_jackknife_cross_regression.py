import numpy as np
import pytest

from nugundam import AngularAutoConfig, AngularBinning, AngularCrossConfig, JackknifeSpec, WeightSpec, acf, accf

# The unweighted jackknife cross counter (th_Cjk) never initialized jq1m, so it
# only searched each particle's own DEC stripe and lost cross pairs (DR in acf;
# D1D2, D1R2, R1D2 and R1R2 in accf), increasingly so at large separations.
# R1R2 in accf is always counted unweighted, so weight_mode="weighted" did not
# avoid the bug there.

AUTO_FIELDS = ("dd", "dr", "rr")
CROSS_FIELDS = ("d1d2", "d1r2", "r1d2", "r1r2")


def _uniform_box(rng, n, half=2.0):
    ra = rng.uniform(180.0 - half, 180.0 + half, n)
    smax = np.sin(np.radians(half))
    dec = np.degrees(np.arcsin(rng.uniform(-smax, smax, n)))
    return {"ra": ra, "dec": dec, "wei": np.ones(n, dtype=float)}


@pytest.fixture(scope="module")
def catalogs():
    rng = np.random.default_rng(1)
    return [_uniform_box(rng, n) for n in (2000, 4000, 2000, 4000)]


def _run(kind, catalogs, *, jackknife, weight_mode):
    data1, random1, data2, random2 = catalogs
    common = dict(
        estimator="LS",
        binning=AngularBinning.from_limits(nsep=8, sepmin=0.01, sepmax=1.0),
        weights=WeightSpec(weight_mode=weight_mode),
        jackknife=JackknifeSpec(enabled=jackknife, nregions=9),
        nthreads=1,
    )
    if kind == "acf":
        cfg = AngularAutoConfig(**common)
        cfg.progress.enabled = False
        return acf(data1, random1, cfg).counts
    cfg = AngularCrossConfig(**common)
    cfg.progress.enabled = False
    return accf(data1, data2, cfg, random1=random1, random2=random2).counts


@pytest.mark.parametrize("weight_mode", ["unweighted", "weighted"])
@pytest.mark.parametrize("kind, fields", [("acf", AUTO_FIELDS), ("accf", CROSS_FIELDS)])
def test_jackknife_does_not_change_pair_counts(catalogs, kind, fields, weight_mode):
    plain = _run(kind, catalogs, jackknife=False, weight_mode=weight_mode)
    jk = _run(kind, catalogs, jackknife=True, weight_mode=weight_mode)
    for name in fields:
        np.testing.assert_array_equal(getattr(jk, name), getattr(plain, name), err_msg=name)


@pytest.mark.parametrize("kind, fields", [("acf", AUTO_FIELDS), ("accf", CROSS_FIELDS)])
def test_unweighted_jackknife_touches_match_unit_weights(catalogs, kind, fields):
    unweighted = _run(kind, catalogs, jackknife=True, weight_mode="unweighted")
    weighted = _run(kind, catalogs, jackknife=True, weight_mode="weighted")
    for name in fields:
        touch = f"{name}_jk_touch"
        assert getattr(unweighted, touch) is not None, touch
        np.testing.assert_array_equal(getattr(unweighted, touch), getattr(weighted, touch), err_msg=touch)
