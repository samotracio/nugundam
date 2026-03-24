import numpy as np

from nugundam import (
    AutoMarkSpec,
    CrossMarkSpec,
    AngularAutoConfig,
    ProjectedAutoConfig,
    BootstrapSpec,
    JackknifeSpec,
    plot_result,
    read_result,
    save_result,
    macf,
    mpcf,
)
import nugundam.marked as marked_mod
from nugundam.angular.models import AngularAutoCounts, AngularCorrelationResult
from nugundam.projected.models import ProjectedAutoCounts, ProjectedCorrelationResult


def _angular_result(edges, centers, *, wtheta, counts, realizations=None):
    return AngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        wtheta=np.asarray(wtheta, dtype=float),
        wtheta_err=np.zeros_like(np.asarray(wtheta, dtype=float)),
        estimator="NAT",
        counts=counts,
        realizations=None if realizations is None else np.asarray(realizations, dtype=float),
        metadata={},
    )


def _projected_result(rp_edges, rp_centers, pi_edges, pi_centers, *, wp, counts, realizations=None):
    return ProjectedCorrelationResult(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        wp=np.asarray(wp, dtype=float),
        wp_err=np.zeros_like(np.asarray(wp, dtype=float)),
        estimator="NAT",
        counts=counts,
        realizations=None if realizations is None else np.asarray(realizations, dtype=float),
        metadata={},
    )


def test_macf_bootstrap_wrapper(monkeypatch, theta_grid):
    edges, centers = theta_grid
    plain_counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([9.0, 4.0]),
        rr=np.array([6.0, 2.0]),
        dr=np.array([1.0, 1.0]),
        dd_boot=np.array([[9.0, 12.0, 6.0], [4.0, 5.0, 3.0]]),
        metadata={"n_data": 4, "n_random": 4},
    )
    weighted_counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([11.0, 4.5]),
        rr=np.array([6.0, 2.0]),
        dr=np.array([1.0, 1.0]),
        dd_boot=np.array([[11.0, 13.0, 9.0], [4.5, 5.5, 3.5]]),
        norm_dd_boot=np.array([8.0, 9.0, 7.0]),
        metadata={"n_data": 4, "n_random": 4},
    )
    plain_result = _angular_result(edges, centers, wtheta=[0.5, 1.0], counts=plain_counts)
    weighted_result = _angular_result(edges, centers, wtheta=[0.8, 1.2], counts=weighted_counts)

    calls = []

    def fake_acf(data, random, config):
        calls.append(config)
        if config.weights.weight_mode == "unweighted":
            return plain_result
        assert config.columns_data.weight == "__nugundam_mark_weight__"
        return weighted_result

    monkeypatch.setattr(marked_mod, "acf", fake_acf)

    data = {
        "ra": np.array([0.0, 1.0, 2.0, 3.0]),
        "dec": np.array([0.0, 0.0, 1.0, 1.0]),
        "mark": np.array([1.0, 2.0, 1.0, 2.0]),
    }
    random = {"ra": np.array([0.0, 1.0]), "dec": np.array([0.0, 1.0])}
    cfg = AngularAutoConfig(estimator="NAT", bootstrap=BootstrapSpec(enabled=True, nbts=3))

    out = macf(data, random, cfg, mark=AutoMarkSpec(column="mark", normalize="none"))

    assert [cfg.weights.weight_mode for cfg in calls] == ["unweighted", "weighted"]
    np.testing.assert_allclose(out.mtheta, np.array([(1.0 + 0.8) / (1.0 + 0.5), (1.0 + 1.2) / (1.0 + 1.0)]))

    plain_boot = marked_mod._bootstrap_realizations_angular_auto(plain_counts, estimator="NAT")
    weighted_boot = marked_mod._bootstrap_realizations_angular_auto(weighted_counts, estimator="NAT", data_weights=np.array([1.0, 2.0, 1.0, 2.0]))
    expected_boot = (1.0 + weighted_boot) / (1.0 + plain_boot)
    np.testing.assert_allclose(out.mtheta_err, np.std(expected_boot, axis=0))
    assert out.metadata["config"]["mark"]["column"] == "mark"


def test_mpcf_jackknife_wrapper(monkeypatch, rp_pi_grid):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    counts = ProjectedAutoCounts(
        rp_edges=rp_edges,
        rp_centers=rp_centers,
        pi_edges=pi_edges,
        pi_centers=pi_centers,
        dd=np.ones((len(rp_centers), len(pi_centers))),
        rr=np.ones((len(rp_centers), len(pi_centers))),
        dr=np.ones((len(rp_centers), len(pi_centers))),
        metadata={"n_data": 4, "n_random": 4},
    )
    plain_real = np.array([[0.2, 0.4], [0.1, 0.3], [0.15, 0.35]])
    weighted_real = np.array([[0.5, 0.8], [0.4, 0.7], [0.45, 0.75]])
    plain_result = _projected_result(rp_edges, rp_centers, pi_edges, pi_centers, wp=[0.3, 0.5], counts=counts, realizations=plain_real)
    weighted_result = _projected_result(rp_edges, rp_centers, pi_edges, pi_centers, wp=[0.6, 0.9], counts=counts, realizations=weighted_real)

    calls = []

    def fake_pcf(data, random, config):
        calls.append(config)
        return plain_result if config.weights.weight_mode == "unweighted" else weighted_result

    monkeypatch.setattr(marked_mod, "pcf", fake_pcf)

    data = {
        "ra": np.array([0.0, 1.0, 2.0, 3.0]),
        "dec": np.array([0.0, 0.0, 1.0, 1.0]),
        "cdcom": np.array([100.0, 101.0, 102.0, 103.0]),
        "mark": np.array([1.0, 2.0, 1.5, 2.5]),
    }
    random = {"ra": np.array([0.0, 1.0]), "dec": np.array([0.0, 1.0]), "cdcom": np.array([100.0, 101.0])}
    cfg = ProjectedAutoConfig(
        estimator="NAT",
        jackknife=JackknifeSpec(enabled=True, return_cov=True, return_realizations=False),
    )

    out = mpcf(data, random, cfg, mark=AutoMarkSpec(column="mark", normalize="none"))

    assert calls[0].jackknife.return_realizations is True
    assert calls[1].jackknife.return_realizations is True
    expected_real = (1.0 + weighted_real / rp_centers[None, :]) / (1.0 + plain_real / rp_centers[None, :])
    expected_cov = np.cov(expected_real, rowvar=False, ddof=1) * (expected_real.shape[0] - 1)
    np.testing.assert_allclose(out.mrp, (1.0 + np.array([0.6, 0.9]) / rp_centers) / (1.0 + np.array([0.3, 0.5]) / rp_centers))
    np.testing.assert_allclose(out.cov, expected_cov)
    np.testing.assert_allclose(out.mrp_err, np.sqrt(np.diag(expected_cov)))
    assert out.realizations is None


def test_marked_roundtrip_ascii_and_plot(tmp_path, theta_grid):
    edges, centers = theta_grid
    counts = AngularAutoCounts(
        theta_edges=edges,
        theta_centers=centers,
        dd=np.array([1.0, 2.0]),
        rr=np.array([1.0, 2.0]),
        dr=np.array([1.0, 2.0]),
        metadata={"n_data": 3, "n_random": 3},
    )
    plain = _angular_result(edges, centers, wtheta=[0.1, 0.2], counts=counts)
    weighted = _angular_result(edges, centers, wtheta=[0.3, 0.4], counts=counts)
    result = marked_mod.MarkedAngularCorrelationResult(
        theta_edges=edges,
        theta_centers=centers,
        mtheta=np.array([1.1, 1.2]),
        mtheta_err=np.array([0.01, 0.02]),
        estimator="NAT",
        plain=plain,
        weighted=weighted,
        metadata={},
    )
    path = tmp_path / "marked.gres"
    txt = tmp_path / "marked.txt"
    save_result(result, path)
    loaded = read_result(path)
    np.testing.assert_allclose(loaded.mtheta, result.mtheta)
    loaded.to_ascii(txt)
    assert txt.exists()
    assert plot_result(loaded) is not None


def test_cross_mark_spec_exported():
    spec = CrossMarkSpec(column1="mass", mark_on="data1")
    assert spec.column1 == "mass"
