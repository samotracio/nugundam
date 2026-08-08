from types import SimpleNamespace

import numpy as np

from nugundam.projected import api as projected_api
from nugundam.projected import mc_pdf as mc_pdf_mod
from nugundam.projected.models import PDFSourceSpec, ProjectedAutoConfig, ProjectedBinning, ProjectedCrossConfig


def test_pcf_mc_pdf_fixed_global_reuses_rr(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0, 2.0]),
        "dec": np.array([0.0, 0.5, 1.0]),
    }
    random = {
        "ra": np.array([0.5, 1.5]),
        "dec": np.array([-0.2, 0.2]),
    }
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    calls = {"full": 0, "dd": 0, "dr": 0}

    def fake_full(data_p, rand_p, **kwargs):
        calls["full"] += 1
        s = float(np.sum(data_p.dist))
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[s]], dtype=np.float64),
            rr=np.array([[100.0]], dtype=np.float64),
            dr=np.array([[10.0 + s]], dtype=np.float64),
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 100.0, "jk_touch_available": False},
        )

    def fake_dd(data_p, **kwargs):
        calls["dd"] += 1
        s = float(np.sum(data_p.dist))
        return SimpleNamespace(dd=np.array([[s]], dtype=np.float64), metadata={"n_data": int(data_p.nrows), "data_weighted": False})

    def fake_dr(data_p, rand_p, **kwargs):
        calls["dr"] += 1
        s = float(np.sum(data_p.dist))
        return SimpleNamespace(d1d2=np.array([[10.0 + s]], dtype=np.float64))

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)
    monkeypatch.setattr(mc_pdf_mod, "build_auto_count_result", fake_dd)
    monkeypatch.setattr(mc_pdf_mod, "build_cross_count_result", fake_dr)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 3
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "fixed_global"
    cfg.mc_pdf.store_realizations = True

    result = projected_api.pcf(data, random, cfg)

    assert calls == {"full": 1, "dd": 2, "dr": 2}
    np.testing.assert_allclose(result.counts.dd, np.array([[40.0]]))
    np.testing.assert_allclose(result.counts.rr, np.array([[100.0]]))
    np.testing.assert_allclose(result.counts.dr, np.array([[50.0]]))
    np.testing.assert_allclose(result.wp, np.array([-9.333333333333332]))
    assert result.metadata["mc_pdf"] is True
    assert result.metadata["mc_rr_fixed"] is True
    assert result.mc_realizations.shape == (3, 1)
    np.testing.assert_allclose(result.mc_wp_std, np.array([0.0]))


def test_pccf_mc_pdf_supports_one_pdf_side(monkeypatch):
    data1 = {
        "ra": np.array([0.0, 1.0]),
        "dec": np.array([0.0, 0.5]),
        "cdcom": np.array([50.0, 60.0]),
    }
    data2 = {
        "ra": np.array([1.0, 2.0]),
        "dec": np.array([0.2, 0.7]),
    }
    random2 = {
        "ra": np.array([0.5, 1.5]),
        "dec": np.array([-0.1, 0.1]),
    }
    matrix2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    calls = {"cross": 0}

    def fake_cross(prep1, prep_r1, prep2, prep_r2, **kwargs):
        calls["cross"] += 1
        s = float(np.sum(prep2.dist))
        return projected_api.ProjectedCrossCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            d1d2=np.array([[s]], dtype=np.float64),
            d1r2=np.array([[s + 10.0]], dtype=np.float64),
            r1d2=None,
            r1r2=None,
            metadata={"n_data1": int(prep1.nrows), "n_random1": 0, "n_data2": int(prep2.nrows), "n_random2": int(prep_r2.nrows), "primary": "data1", "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "build_cross_counts", fake_cross)

    cfg = ProjectedCrossConfig(
        estimator="DP",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.distance.calcdist = False
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 2
    cfg.mc_pdf.seed = 7
    cfg.mc_pdf.chi_grid = np.array([100.0, 200.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data2 = PDFSourceSpec(matrix=matrix2)
    cfg.mc_pdf.random_mode = "fixed_global"
    cfg.mc_pdf.store_realizations = True

    result = projected_api.pccf(data1, data2, cfg, random2=random2)

    assert calls["cross"] == 2
    np.testing.assert_allclose(result.counts.d1d2, np.array([[300.0]]))
    np.testing.assert_allclose(result.counts.d1r2, np.array([[310.0]]))
    np.testing.assert_allclose(result.wp, np.array([-0.12903225806451624]))
    assert result.metadata["mc_pdf"] is True
    assert result.mc_realizations.shape == (2, 1)




def test_pcf_mc_pdf_reuses_notebook_status_emitter(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0]),
        "dec": np.array([0.0, 0.5]),
    }
    random = {
        "ra": np.array([0.25, 1.25]),
        "dec": np.array([-0.1, 0.1]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    progress_calls = []
    created_emitters = []

    class DummyEmitter:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    def fake_create_status_emitter(*, notebook=None, min_update_interval=0.20, status_prefix=None):
        emitter = DummyEmitter()
        created_emitters.append((notebook, min_update_interval, status_prefix, emitter))
        return emitter

    def fake_run_with_progress(enabled, progress_file, poll_interval, target, *, status_prefix=None, status_emitter=None):
        progress_calls.append((enabled, progress_file, poll_interval, status_prefix, status_emitter))
        return target("wrapped.progress")

    def fake_full(data_p, rand_p, **kwargs):
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[1.0]], dtype=np.float64),
            rr=np.array([[1.0]], dtype=np.float64),
            dr=np.array([[1.0]], dtype=np.float64),
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 1.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "in_notebook", lambda: True)
    monkeypatch.setattr(mc_pdf_mod, "create_status_emitter", fake_create_status_emitter)
    monkeypatch.setattr(mc_pdf_mod, "run_with_progress", fake_run_with_progress)
    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = True
    cfg.progress.progress_file = None
    cfg.progress.poll_interval = 0.03
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 2
    cfg.mc_pdf.seed = 5
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "rerun_global"

    projected_api.pcf(data, random, cfg)

    assert len(created_emitters) == 1
    emitter = created_emitters[0][3]
    assert emitter.closed == 1
    assert len(progress_calls) == 2
    assert progress_calls[0][4] is emitter
    assert progress_calls[1][4] is emitter
    assert progress_calls[0][3] == "[pcf:mc_pdf] realization 1/2  "
    assert progress_calls[1][3] == "[pcf:mc_pdf] realization 2/2  "

def test_pcf_mc_pdf_uses_standard_progress_wrapper(monkeypatch, tmp_path):
    data = {
        "ra": np.array([0.0, 1.0]),
        "dec": np.array([0.0, 0.5]),
    }
    random = {
        "ra": np.array([0.25, 1.25]),
        "dec": np.array([-0.1, 0.1]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    progress_calls = []
    seen_progress_files = []

    def fake_run_with_progress(enabled, progress_file, poll_interval, target, *, status_prefix=None, status_emitter=None):
        progress_calls.append((enabled, progress_file, poll_interval, status_prefix))
        return target("wrapped.progress")

    def fake_full(data_p, rand_p, **kwargs):
        seen_progress_files.append(kwargs.get("progress_file"))
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[1.0]], dtype=np.float64),
            rr=np.array([[1.0]], dtype=np.float64),
            dr=np.array([[1.0]], dtype=np.float64),
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 1.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "run_with_progress", fake_run_with_progress)
    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = True
    cfg.progress.progress_file = str(tmp_path / "mc.progress")
    cfg.progress.poll_interval = 0.03
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.seed = 5
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "fixed_global"

    projected_api.pcf(data, random, cfg)

    assert len(progress_calls) == 1
    assert progress_calls[0][0] is True
    assert progress_calls[0][1] == str(tmp_path / "mc.progress")
    assert progress_calls[0][2] == 0.03
    assert progress_calls[0][3] == "[pcf:mc_pdf] realization 1/1  "
    assert seen_progress_files == ["wrapped.progress"]


def test_pcf_mc_pdf_sample_within_bin_uses_continuous_distances(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0]),
        "dec": np.array([0.0, 0.5]),
    }
    random = {
        "ra": np.array([0.25, 1.25]),
        "dec": np.array([-0.1, 0.1]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    seen = {}

    def fake_full(data_p, rand_p, **kwargs):
        seen["data_dist"] = np.asarray(data_p.dist, dtype=np.float64).copy()
        seen["rand_dist"] = np.asarray(rand_p.dist, dtype=np.float64).copy()
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[1.0]], dtype=np.float64),
            rr=np.array([[1.0]], dtype=np.float64),
            dr=np.array([[1.0]], dtype=np.float64),
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 1.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.seed = 123
    cfg.mc_pdf.chi_grid = np.array([0.0, 10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.grid_kind = "edges"
    cfg.mc_pdf.sample_within_bin = True
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "rerun_global"

    result = projected_api.pcf(data, random, cfg)

    assert result.metadata["mc_sample_within_bin"] is True
    assert result.metadata["mc_pdf_mode"] == "grid_sampler_within_bin"
    assert 0.0 <= seen["data_dist"][0] <= 10.0
    assert 10.0 <= seen["data_dist"][1] <= 20.0
    assert not np.allclose(seen["data_dist"], np.array([5.0, 15.0], dtype=np.float64))
    assert np.all((seen["rand_dist"] >= 0.0) & (seen["rand_dist"] <= 20.0))


def test_pcf_mc_pdf_bootstrap_rerun_backend(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0, 2.0]),
        "dec": np.array([0.0, 0.5, 1.0]),
    }
    random = {
        "ra": np.array([0.5, 1.5]),
        "dec": np.array([-0.2, 0.2]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    calls = {"auto": 0}

    def fake_full(data_p, rand_p, **kwargs):
        calls["auto"] += 1
        s = float(np.sum(data_p.dist))
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[s]], dtype=np.float64),
            rr=np.array([[1.0]], dtype=np.float64),
            dr=None,
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 1.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 2
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "inherit_realization"
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 3
    cfg.bootstrap.bseed = 222

    result = projected_api.pcf(data, random, cfg)

    assert calls["auto"] == 1 + 3 * 2
    assert result.bootstrap_realizations.shape == (3, 1)
    assert result.bootstrap_cumulative_realizations.shape == (3, 1, 1)
    np.testing.assert_allclose(
        result.bootstrap_cumulative_realizations[:, :, -1],
        result.bootstrap_realizations,
    )
    assert result.bootstrap_counts is None
    assert result.cov.shape == (1, 1)
    assert result.wp_err.shape == (1,)
    assert result.metadata["bootstrap_backend"] == "mc_pdf_rerun"
    assert result.metadata["mc_resampling_nreal"] == 2
    assert result.metadata["mc_resampling_random_policy"] == "reinherit"


def test_pcf_mc_pdf_jackknife_rerun_backend_with_user_regions(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0, 2.0, 3.0]),
        "dec": np.array([0.0, 0.5, 1.0, 1.5]),
        "reg": np.array([0, 0, 1, 1]),
    }
    random = {
        "ra": np.array([0.5, 1.5, 2.5, 3.5]),
        "dec": np.array([-0.2, 0.2, 0.6, 1.2]),
        "reg": np.array([0, 0, 1, 1]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    calls = {"auto": 0}

    def fake_full(data_p, rand_p, **kwargs):
        calls["auto"] += 1
        s = float(np.sum(data_p.dist))
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=np.array([[s]], dtype=np.float64),
            rr=np.array([[1.0]], dtype=np.float64),
            dr=None,
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 1.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.columns_data.region = "reg"
    cfg.columns_random.region = "reg"
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 2
    cfg.mc_pdf.seed = 13
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "inherit_realization"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True

    result = projected_api.pcf(data, random, cfg)

    assert calls["auto"] == 1 + 2 * 2
    assert result.realizations.shape == (2, 1)
    assert result.cov.shape == (1, 1)
    assert result.wp_err.shape == (1,)
    assert result.metadata["jackknife"] is True
    assert result.metadata["jk_touch_fast"] is False
    assert result.metadata["mc_resampling_nreal"] == 2


def test_pcf_mc_pdf_bootstrap_fast_backend_with_fixed_global(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0, 2.0]),
        "dec": np.array([0.0, 0.5, 1.0]),
    }
    random = {
        "ra": np.array([0.5, 1.5]),
        "dec": np.array([-0.2, 0.2]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    calls = {"auto": 0}

    def fake_full(data_p, rand_p, **kwargs):
        calls["auto"] += 1
        nbts = int(kwargs.get("nbts", 0))
        doboot = bool(kwargs.get("doboot", False))
        s = float(np.sum(data_p.dist))
        dd = np.array([[s]], dtype=np.float64)
        dd_boot = None
        if doboot:
            dd_boot = np.repeat(dd[:, :, None], nbts, axis=2)
            dd_boot += np.arange(nbts, dtype=np.float64)[None, None, :]
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=dd,
            rr=np.array([[100.0]], dtype=np.float64),
            dr=np.array([[10.0 + s]], dtype=np.float64),
            dd_boot=dd_boot,
            metadata={"n_data": int(data_p.nrows), "n_random": int(rand_p.nrows), "rr_norm_pairs": 100.0, "jk_touch_available": False},
        )

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 2
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "fixed_global"
    cfg.mc_pdf.resampling_backend = "fast"
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 3
    cfg.bootstrap.bseed = 222

    result = projected_api.pcf(data, random, cfg)

    assert calls["auto"] == 1 + 2
    assert result.bootstrap_realizations.shape == (3, 1)
    assert result.bootstrap_cumulative_realizations.shape == (3, 1, 1)
    np.testing.assert_allclose(
        result.bootstrap_cumulative_realizations[:, :, -1],
        result.bootstrap_realizations,
    )
    assert result.bootstrap_counts is not None
    assert result.bootstrap_counts.dd_boot.shape == (1, 1, 3)
    assert result.cov.shape == (1, 1)
    assert result.metadata["bootstrap_backend"] == "mc_pdf_fast"
    assert result.metadata["mc_resampling_backend"] == "fast"
    assert result.metadata["mc_fast_bootstrap_count_average"] is True


def test_pcf_mc_pdf_fast_bootstrap_rejects_reinherit_policy():
    data = {
        "ra": np.array([0.0, 1.0, 2.0]),
        "dec": np.array([0.0, 0.5, 1.0]),
    }
    random = {
        "ra": np.array([0.5, 1.5]),
        "dec": np.array([-0.2, 0.2]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 1
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "inherit_realization"
    cfg.mc_pdf.resampling_backend = "fast"
    cfg.mc_pdf.resampling_random_policy = "reinherit"
    cfg.bootstrap.enabled = True
    cfg.bootstrap.nbts = 2

    import pytest
    with pytest.raises(NotImplementedError, match="requires fixed random treatment"):
        projected_api.pcf(data, random, cfg)


def test_pcf_mc_pdf_jackknife_fast_backend_with_fixed_policy(monkeypatch):
    data = {
        "ra": np.array([0.0, 1.0, 2.0, 3.0]),
        "dec": np.array([0.0, 0.5, 1.0, 1.5]),
        "reg": np.array([0, 0, 1, 1], dtype=np.int32),
    }
    random = {
        "ra": np.array([0.5, 1.5, 2.5, 3.5]),
        "dec": np.array([-0.2, 0.2, 0.8, 1.2]),
        "reg": np.array([0, 0, 1, 1], dtype=np.int32),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    calls = {"auto": 0}

    def fake_full(data_p, rand_p, **kwargs):
        calls["auto"] += 1
        nreg = int(kwargs.get("nreg", 0))
        dojk = bool(kwargs.get("dojk", False))
        dd = np.array([[30.0]], dtype=np.float64)
        rr = np.array([[100.0]], dtype=np.float64)
        dr = np.array([[20.0]], dtype=np.float64)
        dd_touch = rr_touch = dr_touch = None
        if dojk:
            dd_touch = np.repeat(np.array([[[5.0]]], dtype=np.float64), nreg, axis=2)
            rr_touch = np.repeat(np.array([[[10.0]]], dtype=np.float64), nreg, axis=2)
            dr_touch = np.repeat(np.array([[[4.0]]], dtype=np.float64), nreg, axis=2)
        return projected_api.ProjectedAutoCounts(
            rp_edges=np.array([0.1, 1.0], dtype=np.float64),
            rp_centers=np.array([0.31622777], dtype=np.float64),
            pi_edges=np.array([0.0, 2.0], dtype=np.float64),
            pi_centers=np.array([1.0], dtype=np.float64),
            dd=dd,
            rr=rr,
            dr=dr,
            dd_jk_touch=dd_touch,
            rr_jk_touch=rr_touch,
            dr_jk_touch=dr_touch,
            metadata={
                "n_data": int(data_p.nrows),
                "n_random": int(rand_p.nrows),
                "rr_norm_pairs": 0.5 * int(rand_p.nrows) * max(int(rand_p.nrows) - 1, 0),
                "jk_touch_available": dojk,
                "jk_nregions": nreg if dojk else 0,
            },
        )

    monkeypatch.setattr(mc_pdf_mod, "build_auto_counts", fake_full)

    cfg = ProjectedAutoConfig(
        estimator="LS",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.columns_data.region = "reg"
    cfg.columns_random.region = "reg"
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 2
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "inherit_realization"
    cfg.mc_pdf.resampling_backend = "fast"
    cfg.mc_pdf.resampling_random_policy = "fixed"
    cfg.jackknife.enabled = True
    cfg.jackknife.return_realizations = True

    result = projected_api.pcf(data, random, cfg)

    assert calls["auto"] == 1 + 2
    assert result.realizations.shape == (2, 1)
    assert result.cov.shape == (1, 1)
    assert result.wp_err.shape == (1,)
    assert result.metadata["jackknife"] is True
    assert result.metadata["jk_touch_fast"] is True
    assert result.metadata["mc_resampling_backend"] == "fast"
    assert result.metadata["mc_fast_jackknife_count_average"] is True


def test_pcf_mc_pdf_fast_jackknife_rejects_reinherit_policy():
    data = {
        "ra": np.array([0.0, 1.0, 2.0, 3.0]),
        "dec": np.array([0.0, 0.5, 1.0, 1.5]),
    }
    random = {
        "ra": np.array([0.5, 1.5, 2.5, 3.5]),
        "dec": np.array([-0.2, 0.2, 0.8, 1.2]),
    }
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cfg = ProjectedAutoConfig(
        estimator="NAT",
        binning=ProjectedBinning.from_binsize(nsepp=1, seppmin=0.1, dsepp=1.0, logsepp=True, nsepv=1, dsepv=2.0),
    )
    cfg.progress.enabled = False
    cfg.mc_pdf.enabled = True
    cfg.mc_pdf.nreal = 1
    cfg.mc_pdf.resampling_nreal = 1
    cfg.mc_pdf.seed = 11
    cfg.mc_pdf.chi_grid = np.array([10.0, 20.0], dtype=np.float64)
    cfg.mc_pdf.pdf_data = PDFSourceSpec(matrix=matrix)
    cfg.mc_pdf.random_mode = "inherit_realization"
    cfg.mc_pdf.resampling_backend = "fast"
    cfg.mc_pdf.resampling_random_policy = "reinherit"
    cfg.jackknife.enabled = True
    cfg.jackknife.nregions = 2

    import pytest
    with pytest.raises(NotImplementedError, match="requires fixed random treatment"):
        projected_api.pcf(data, random, cfg)
