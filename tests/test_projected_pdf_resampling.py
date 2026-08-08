import numpy as np

from nugundam.projected import api as projected_api
from nugundam.projected.models import (
    PreparedProjectedSample,
    ProjectedAutoConfig,
    ProjectedAutoCounts,
    ProjectedBinning,
    ProjectedCrossCounts,
)
from nugundam.projected.prepare import subset_prepared_projected_sample


def _prepared_pdf_sample(n=4, *, with_regions=False):
    ra = np.linspace(0.0, 3.0, n, dtype=np.float64)
    dec = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    dist = np.linspace(100.0, 103.0, n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float32)
    region_id = None if not with_regions else np.array([0, 0, 1, 1][:n], dtype=np.int32)
    alpha = np.full((3, n), 1.0 / 3.0, dtype=np.float64, order="F")
    mu = np.vstack([dist - 1.0, dist, dist + 1.0]).astype(np.float64, order="F")
    sig = np.full((3, n), 0.5, dtype=np.float64, order="F")
    return PreparedProjectedSample(
        table=None,
        ra=ra,
        dec=dec,
        dist=dist,
        weights=weights,
        x=np.ones(n, dtype=np.float64),
        y=np.zeros(n, dtype=np.float64),
        z=np.zeros(n, dtype=np.float64),
        sk=np.zeros((2, 2, 2), dtype=np.int32),
        ll=np.zeros(n, dtype=np.int32),
        wunit=True,
        sbound=(0.0, 360.0, -2.0, 2.0, 99.0, 104.0),
        mxh1=2,
        mxh2=2,
        mxh3=2,
        dcang=np.full(n, 95.0, dtype=np.float64),
        pdf_k=3,
        pdf_alpha_lib=alpha,
        pdf_mu_lib=mu,
        pdf_sig_lib=sig,
        pdf_idx=np.arange(1, n + 1, dtype=np.int32),
        region_id=region_id,
        grid_meta={"autogrid": True, "dens": None, "pxorder": "natural", "nsepv": 1, "dsepv": 20.0, "pi_edges_search": np.array([0.0, 20.0], dtype=np.float64)},
        nrows=n,
    )


def test_subset_prepared_projected_sample_preserves_pdf_payload():
    sample = _prepared_pdf_sample(n=4)
    sub = subset_prepared_projected_sample(sample, np.array([0, 2, 3]), pi_edges=np.array([0.0, 20.0], dtype=np.float64), regrid=False)
    assert sub.pdf_k == sample.pdf_k
    assert sub.pdf_alpha_lib is sample.pdf_alpha_lib
    assert sub.pdf_mu_lib is sample.pdf_mu_lib
    assert sub.pdf_sig_lib is sample.pdf_sig_lib
    np.testing.assert_array_equal(sub.pdf_idx, np.array([1, 3, 4], dtype=np.int32))


def test_pcf_pdf_bootstrap_uses_rerun_backend(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=False)
    rand_p = _prepared_pdf_sample(n=5, with_regions=False)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
    }
    calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_counts(data, random, **kwargs):
        calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[float(np.sum(data.weights))]], dtype=np.float64),
            rr=np.array([[10.0]], dtype=np.float64),
            dr=np.array([[5.0]], dtype=np.float64),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)
    monkeypatch.setattr(projected_api, "pdf_auto_bootstrap_fast_available", lambda *, weighted: False)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 3
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert calls[0] == (False, False)
    assert len(calls) == 4
    assert result.counts.dd_boot.shape == (1, 1, 3)
    assert result.counts.metadata["pdf_bootstrap_backend"] == "rerun"


def test_pcf_pdf_jackknife_reinherit_rebuilds_randoms(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=True)
    rand_p = _prepared_pdf_sample(n=4, with_regions=True)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
        "jk_nregions": 2,
        "jk_region_source": "user",
    }
    rebuild_calls = []
    count_calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_reinherit(random_sample, data_lib, config, *, pi_edges, seed=None, regrid=False):
        rebuild_calls.append((int(random_sample.nrows), int(data_lib.nrows), int(seed)))
        return random_sample

    def fake_counts(data, random, **kwargs):
        count_calls.append((int(data.nrows), int(random.nrows), bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[float(data.nrows)]], dtype=np.float64),
            rr=np.array([[10.0]], dtype=np.float64),
            dr=np.array([[5.0]], dtype=np.float64),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)
    monkeypatch.setattr(projected_api, "rebuild_pdf_random_inheritance_from_prepared", fake_reinherit)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.pdf.jk_random_policy = "reinherit"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert count_calls[0][2:] == (False, False)
    assert len(rebuild_calls) == 2
    assert result.realizations.shape == (2, 1)
    assert result.metadata["jk_touch_fast"] is False


def test_pcf_pdf_jackknife_fixed_uses_touch_backend(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=True)
    rand_p = _prepared_pdf_sample(n=4, with_regions=True)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
        "jk_nregions": 2,
        "jk_region_source": "user",
    }
    count_calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_counts(data, random, **kwargs):
        count_calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        touch = np.array([[[1.0, 1.0]]], dtype=np.float64)
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[4.0]], dtype=np.float64),
            rr=np.array([[10.0]], dtype=np.float64),
            dr=np.array([[5.0]], dtype=np.float64),
            dd_jk_touch=touch,
            rr_jk_touch=np.zeros_like(touch),
            dr_jk_touch=np.zeros_like(touch),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": True},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.pdf.jk_random_policy = "fixed"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert count_calls == [(False, True)]
    assert result.realizations.shape == (2, 1)
    assert result.metadata["jk_touch_fast"] is True


def test_pccf_pdf_jackknife_fixed_uses_touch_backend(monkeypatch):
    prep1 = _prepared_pdf_sample(n=4, with_regions=True)
    prep2 = _prepared_pdf_sample(n=4, with_regions=True)
    rand1 = _prepared_pdf_sample(n=4, with_regions=True)
    rand2 = _prepared_pdf_sample(n=4, with_regions=True)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
        "jk_nregions": 2,
        "jk_region_source": "user",
    }
    count_calls = []

    def fake_prepare(data1, random1_in, data2, random2_in, config):
        return prep1, rand1, prep2, rand2, meta

    def fake_counts(d1, r1, d2, r2, **kwargs):
        count_calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        touch = np.array([[[1.0, 1.0]]], dtype=np.float64)
        return ProjectedCrossCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            d1d2=np.array([[4.0]], dtype=np.float64),
            d1r2=np.array([[2.0]], dtype=np.float64),
            r1d2=np.array([[2.0]], dtype=np.float64),
            r1r2=np.array([[1.0]], dtype=np.float64),
            d1d2_jk_touch=touch,
            d1r2_jk_touch=np.zeros_like(touch),
            r1d2_jk_touch=np.zeros_like(touch),
            r1r2_jk_touch=np.zeros_like(touch),
            metadata={"n_data1": int(d1.nrows), "n_random1": int(r1.nrows), "n_data2": int(d2.nrows), "n_random2": int(r2.nrows), "primary": "data1", "jk_touch_available": True},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_cross", fake_prepare)
    monkeypatch.setattr(projected_api, "build_cross_counts", fake_counts)

    cfg = projected_api.ProjectedCrossConfig(estimator="LS", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.pdf.jk_random_policy = "fixed"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True
    cfg.progress.enabled = False

    result = projected_api.pccf(object(), object(), cfg, random1=object(), random2=object())
    assert count_calls == [(False, True)]
    assert result.realizations.shape == (2, 1)
    assert result.metadata["jk_touch_fast"] is True


def test_pcf_pdf_bootstrap_prefers_compiled_backend_when_available(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=False)
    rand_p = _prepared_pdf_sample(n=5, with_regions=False)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
    }
    calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_counts(data, random, **kwargs):
        calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[4.0]], dtype=np.float64),
            rr=np.array([[10.0]], dtype=np.float64),
            dr=np.array([[5.0]], dtype=np.float64),
            dd_boot=np.ones((1, 1, 2), dtype=np.float64),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)
    monkeypatch.setattr(projected_api, "pdf_auto_bootstrap_fast_available", lambda *, weighted: True)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 2
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert calls == [(True, False)]
    assert result.counts.dd_boot.shape == (1, 1, 2)
    assert result.counts.metadata["pdf_bootstrap_backend"] == "compiled"


def test_pccf_pdf_bootstrap_prefers_compiled_backend_when_available(monkeypatch):
    prep1 = _prepared_pdf_sample(n=4, with_regions=False)
    prep2 = _prepared_pdf_sample(n=4, with_regions=False)
    rand1 = _prepared_pdf_sample(n=4, with_regions=False)
    rand2 = _prepared_pdf_sample(n=4, with_regions=False)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([10.0], dtype=np.float64),
        "pi_delta": np.array([20.0], dtype=np.float64),
        "pdf_enabled": True,
    }
    calls = []

    def fake_prepare(data1, random1_in, data2, random2_in, config):
        return prep1, rand1, prep2, rand2, meta

    def fake_counts(d1, r1, d2, r2, **kwargs):
        calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk")), kwargs.get("primary")))
        return ProjectedCrossCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            d1d2=np.array([[4.0]], dtype=np.float64),
            d1r2=np.array([[2.0]], dtype=np.float64),
            r1d2=np.array([[2.0]], dtype=np.float64),
            r1r2=np.array([[1.0]], dtype=np.float64),
            d1d2_boot=np.ones((1, 1, 2), dtype=np.float64),
            d1r2_boot=np.ones((1, 1, 2), dtype=np.float64),
            metadata={"n_data1": int(d1.nrows), "n_random1": int(r1.nrows), "n_data2": int(d2.nrows), "n_random2": int(r2.nrows), "primary": kwargs.get("primary", "data1"), "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_cross", fake_prepare)
    monkeypatch.setattr(projected_api, "build_cross_counts", fake_counts)
    monkeypatch.setattr(projected_api, "pdf_cross_bootstrap_fast_available", lambda *, weighted: True)

    cfg = projected_api.ProjectedCrossConfig(estimator="LS", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=20.0))
    cfg.pdf.enabled = True
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 2
    cfg.bootstrap.primary = "data2"
    cfg.progress.enabled = False

    result = projected_api.pccf(object(), object(), cfg, random1=object(), random2=object())
    assert calls == [(True, False, "data2")]
    assert result.counts.d1d2_boot.shape == (1, 1, 2)
    assert result.counts.metadata["pdf_bootstrap_backend"] == "compiled"


def test_pcf_pdf_multi_pi_bootstrap_prefers_compiled_backend_when_available(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=False)
    rand_p = _prepared_pdf_sample(n=5, with_regions=False)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([5.0, 15.0], dtype=np.float64),
        "pi_delta": np.array([10.0, 10.0], dtype=np.float64),
        "pdf_enabled": True,
    }
    calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_counts(data, random, **kwargs):
        calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[2.0, 2.0]], dtype=np.float64),
            rr=np.array([[5.0, 5.0]], dtype=np.float64),
            dr=np.array([[2.5, 2.5]], dtype=np.float64),
            dd_boot=np.ones((1, 2, 2), dtype=np.float64),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)
    monkeypatch.setattr(projected_api, "pdf_auto_bootstrap_fast_available", lambda *, weighted: True)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=2, dsepv=10.0))
    cfg.pdf.enabled = True
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 2
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert calls == [(True, False)]
    assert result.counts.dd_boot.shape == (1, 2, 2)
    assert result.bootstrap_counts is None
    assert result.bootstrap_cumulative_realizations.shape == (2, 1, 2)
    np.testing.assert_allclose(
        result.bootstrap_cumulative_realizations[:, :, -1],
        result.bootstrap_realizations,
    )
    assert result.counts.metadata["pdf_bootstrap_backend"] == "compiled"


def test_pccf_pdf_multi_pi_bootstrap_prefers_compiled_backend_when_available(monkeypatch):
    prep1 = _prepared_pdf_sample(n=4, with_regions=False)
    prep2 = _prepared_pdf_sample(n=4, with_regions=False)
    rand1 = _prepared_pdf_sample(n=4, with_regions=False)
    rand2 = _prepared_pdf_sample(n=4, with_regions=False)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([5.0, 15.0], dtype=np.float64),
        "pi_delta": np.array([10.0, 10.0], dtype=np.float64),
        "pdf_enabled": True,
    }
    calls = []

    def fake_prepare(data1, random1_in, data2, random2_in, config):
        return prep1, rand1, prep2, rand2, meta

    def fake_counts(d1, r1, d2, r2, **kwargs):
        calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk")), kwargs.get("primary")))
        return ProjectedCrossCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            d1d2=np.array([[2.0, 2.0]], dtype=np.float64),
            d1r2=np.array([[1.0, 1.0]], dtype=np.float64),
            r1d2=np.array([[1.0, 1.0]], dtype=np.float64),
            r1r2=np.array([[1.0, 1.0]], dtype=np.float64),
            d1d2_boot=np.ones((1, 2, 2), dtype=np.float64),
            d1r2_boot=np.ones((1, 2, 2), dtype=np.float64),
            metadata={"n_data1": int(d1.nrows), "n_random1": int(r1.nrows), "n_data2": int(d2.nrows), "n_random2": int(r2.nrows), "primary": kwargs.get("primary", "data1"), "jk_touch_available": False},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_cross", fake_prepare)
    monkeypatch.setattr(projected_api, "build_cross_counts", fake_counts)
    monkeypatch.setattr(projected_api, "pdf_cross_bootstrap_fast_available", lambda *, weighted: True)

    cfg = projected_api.ProjectedCrossConfig(estimator="LS", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=2, dsepv=10.0))
    cfg.pdf.enabled = True
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 2
    cfg.bootstrap.primary = "data1"
    cfg.progress.enabled = False

    result = projected_api.pccf(object(), object(), cfg, random1=object(), random2=object())
    assert calls == [(True, False, "data1")]
    assert result.counts.d1d2_boot.shape == (1, 2, 2)
    assert result.bootstrap_counts is None
    assert result.bootstrap_cumulative_realizations.shape == (2, 1, 2)
    np.testing.assert_allclose(
        result.bootstrap_cumulative_realizations[:, :, -1],
        result.bootstrap_realizations,
    )
    assert result.counts.metadata["pdf_bootstrap_backend"] == "compiled"


def test_pcf_pdf_multi_pi_jackknife_fixed_uses_touch_backend(monkeypatch):
    data_p = _prepared_pdf_sample(n=4, with_regions=True)
    rand_p = _prepared_pdf_sample(n=4, with_regions=True)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([5.0, 15.0], dtype=np.float64),
        "pi_delta": np.array([10.0, 10.0], dtype=np.float64),
        "pdf_enabled": True,
        "jk_nregions": 2,
        "jk_region_source": "user",
    }
    count_calls = []

    def fake_prepare(data, random, config):
        return data_p, rand_p, meta

    def fake_counts(data, random, **kwargs):
        count_calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        touch = np.ones((1, 2, 2), dtype=np.float64)
        return ProjectedAutoCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            dd=np.array([[2.0, 2.0]], dtype=np.float64),
            rr=np.array([[5.0, 5.0]], dtype=np.float64),
            dr=np.array([[2.5, 2.5]], dtype=np.float64),
            dd_jk_touch=touch,
            rr_jk_touch=np.zeros_like(touch),
            dr_jk_touch=np.zeros_like(touch),
            metadata={"n_data": int(data.nrows), "n_random": int(random.nrows), "rr_norm_pairs": 10.0, "jk_touch_available": True},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_auto", fake_prepare)
    monkeypatch.setattr(projected_api, "build_auto_counts", fake_counts)

    cfg = ProjectedAutoConfig(estimator="NAT", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=2, dsepv=10.0))
    cfg.pdf.enabled = True
    cfg.pdf.jk_random_policy = "fixed"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True
    cfg.progress.enabled = False

    result = projected_api.pcf(object(), object(), cfg)
    assert count_calls == [(False, True)]
    assert result.realizations.shape == (2, 1)
    assert result.metadata["jk_touch_fast"] is True


def test_pccf_pdf_multi_pi_jackknife_fixed_uses_touch_backend(monkeypatch):
    prep1 = _prepared_pdf_sample(n=4, with_regions=True)
    prep2 = _prepared_pdf_sample(n=4, with_regions=True)
    rand1 = _prepared_pdf_sample(n=4, with_regions=True)
    rand2 = _prepared_pdf_sample(n=4, with_regions=True)
    meta = {
        "rp_edges": np.array([0.1, 1.0], dtype=np.float64),
        "rp_centers": np.array([0.31622777], dtype=np.float64),
        "pi_edges": np.array([0.0, 10.0, 20.0], dtype=np.float64),
        "pi_centers": np.array([5.0, 15.0], dtype=np.float64),
        "pi_delta": np.array([10.0, 10.0], dtype=np.float64),
        "pdf_enabled": True,
        "jk_nregions": 2,
        "jk_region_source": "user",
    }
    count_calls = []

    def fake_prepare(data1, random1_in, data2, random2_in, config):
        return prep1, rand1, prep2, rand2, meta

    def fake_counts(d1, r1, d2, r2, **kwargs):
        count_calls.append((bool(kwargs.get("doboot")), bool(kwargs.get("dojk"))))
        touch = np.ones((1, 2, 2), dtype=np.float64)
        return ProjectedCrossCounts(
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            d1d2=np.array([[2.0, 2.0]], dtype=np.float64),
            d1r2=np.array([[1.0, 1.0]], dtype=np.float64),
            r1d2=np.array([[1.0, 1.0]], dtype=np.float64),
            r1r2=np.array([[1.0, 1.0]], dtype=np.float64),
            d1d2_jk_touch=touch,
            d1r2_jk_touch=np.zeros_like(touch),
            r1d2_jk_touch=np.zeros_like(touch),
            r1r2_jk_touch=np.zeros_like(touch),
            metadata={"n_data1": int(d1.nrows), "n_random1": int(r1.nrows), "n_data2": int(d2.nrows), "n_random2": int(r2.nrows), "primary": "data1", "jk_touch_available": True},
        )

    monkeypatch.setattr(projected_api, "prepare_projected_cross", fake_prepare)
    monkeypatch.setattr(projected_api, "build_cross_counts", fake_counts)

    cfg = projected_api.ProjectedCrossConfig(estimator="LS", binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=2, dsepv=10.0))
    cfg.pdf.enabled = True
    cfg.pdf.jk_random_policy = "fixed"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True
    cfg.progress.enabled = False

    result = projected_api.pccf(object(), object(), cfg, random1=object(), random2=object())
    assert count_calls == [(False, True)]
    assert result.realizations.shape == (2, 1)
    assert result.metadata["jk_touch_fast"] is True
