import numpy as np

from nugundam import pcf
from nugundam.projected.models import ProjectedAutoConfig, ProjectedBinning


def test_projected_ls_single_pi_bin_larger_than_radial_span_keeps_dr_nonzero():
    # Radial span is only 20 Mpc/h, so dsepv=30 exceeds the full sample depth.
    # This used to zero DR because some projected counters computed nc3=int(span/rvmax)=0.
    data = {
        "ra": np.array([0.00, 0.03, 0.06], dtype=float),
        "dec": np.array([0.00, 0.00, 0.00], dtype=float),
        "distance": np.array([100.0, 110.0, 120.0], dtype=float),
        "wei": np.ones(3, dtype=float),
    }
    random = {
        "ra": np.array([0.01, 0.04, 0.07, 0.10], dtype=float),
        "dec": np.array([0.00, 0.00, 0.00, 0.00], dtype=float),
        "distance": np.array([101.0, 107.0, 114.0, 119.0], dtype=float),
        "wei": np.ones(4, dtype=float),
    }

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(
            nsepp=1,
            seppmin=0.0,
            dsepp=10.0,
            logsepp=False,
            nsepv=1,
            dsepv=30.0,
        ),
        nthreads=1,
    )
    cfg.progress.enabled = False
    cfg.distance.calcdist = False
    cfg.columns_data.distance = "distance"
    cfg.columns_random.distance = "distance"

    res = pcf(data, random, cfg)

    assert np.isfinite(res.counts.dd).all()
    assert np.isfinite(res.counts.rr).all()
    assert np.isfinite(res.counts.dr).all()
    assert float(np.sum(res.counts.rr)) > 0.0
    assert float(np.sum(res.counts.dr)) > 0.0
    assert np.isfinite(res.wp).all()
