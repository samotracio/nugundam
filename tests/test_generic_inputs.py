import numpy as np
import pandas as pd
import pytest

import nugundam.cflibfor as cff
from nugundam.angular.models import AngularAutoConfig
from nugundam.angular.prepare import prepare_angular_auto, subset_prepared_angular_sample
from nugundam.core.catalogs import catalog_get_column, catalog_nrows
from nugundam.projected.models import ProjectedAutoConfig, PreparedProjectedSample
from nugundam.projected.prepare import prepare_projected_auto, subset_prepared_projected_sample


class _FakeArrowColumn:
    def __init__(self, values):
        self._values = np.asarray(values)

    def to_numpy(self, zero_copy_only=False):
        return np.asarray(self._values)


class _FakeArrowTable:
    def __init__(self, mapping):
        self._mapping = {str(k): np.asarray(v) for k, v in mapping.items()}
        self.column_names = list(self._mapping.keys())
        self.num_rows = len(next(iter(self._mapping.values()))) if self._mapping else 0

    def __getitem__(self, key):
        return _FakeArrowColumn(self._mapping[key])


def test_catalog_helpers_accept_pyarrow_duck_table():
    table = _FakeArrowTable({"ra": [1.0, 2.0], "dec": [-1.0, 0.5]})
    np.testing.assert_allclose(catalog_get_column(table, "ra", dtype=np.float64), [1.0, 2.0])
    assert catalog_nrows(table) == 2


def test_prepare_angular_auto_accepts_pandas_dataframe(monkeypatch):
    monkeypatch.setattr(
        cff,
        "mod",
        type("_FakeMod", (), {
            "skll2d": staticmethod(lambda mxh1, mxh2, npt, ra, dec, sbound: (
                np.zeros((int(mxh2), int(mxh1)), dtype=np.int32),
                np.zeros(int(npt), dtype=np.int32),
            ))
        })(),
    )

    data = pd.DataFrame({
        "ra": [3.0, 1.0, 2.0],
        "dec": [-1.0, 0.0, 1.0],
        "wei": [1.0, 2.0, 1.5],
    })
    random = pd.DataFrame({
        "ra": [4.0, 0.5, 2.5, 1.5],
        "dec": [-1.5, -0.5, 0.5, 1.5],
        "wei": [1.0, 1.0, 1.0, 1.0],
    })
    cfg = AngularAutoConfig(estimator="NAT")

    data_p, rand_p, meta = prepare_angular_auto(data, random, cfg)

    assert data_p.table is None
    assert rand_p.table is None
    assert data_p.nrows == len(data)
    assert rand_p.nrows == len(random)
    assert data_p.ra.shape == (len(data),)
    assert rand_p.ra.shape == (len(random),)
    assert meta["theta_edges"].ndim == 1


def test_prepare_angular_auto_accepts_structured_array(monkeypatch):
    monkeypatch.setattr(
        cff,
        "mod",
        type("_FakeMod", (), {
            "skll2d": staticmethod(lambda mxh1, mxh2, npt, ra, dec, sbound: (
                np.zeros((int(mxh2), int(mxh1)), dtype=np.int32),
                np.zeros(int(npt), dtype=np.int32),
            ))
        })(),
    )

    dtype = [("ra", "f8"), ("dec", "f8"), ("wei", "f4")]
    data = np.array([(2.0, -1.0, 1.0), (1.0, 0.0, 2.0)], dtype=dtype)
    random = np.array([(3.0, -1.5, 1.0), (0.5, 1.5, 1.0)], dtype=dtype)
    cfg = AngularAutoConfig(estimator="NAT")

    data_p, rand_p, _ = prepare_angular_auto(data, random, cfg)

    assert data_p.nrows == 2
    assert rand_p.nrows == 2
    np.testing.assert_allclose(np.sort(data_p.weights), [1.0, 2.0])


def test_prepare_projected_auto_accepts_mapping(monkeypatch):
    monkeypatch.setattr(
        "nugundam.projected.prepare._build_skll3d",
        lambda mxh1, mxh2, mxh3, ra, dec, dist, sbound, pi_edges: (
            np.zeros((int(mxh3), int(mxh2), int(mxh1)), dtype=np.int32),
            np.zeros(len(ra), dtype=np.int32),
        ),
    )

    data = {
        "ra": np.array([2.0, 1.0, 3.0]),
        "dec": np.array([-1.0, 0.0, 1.0]),
        "distance": np.array([100.0, 110.0, 120.0]),
        "wei": np.array([1.0, 2.0, 1.0]),
    }
    random = {
        "ra": np.array([0.5, 1.5, 2.5, 3.5]),
        "dec": np.array([-1.5, -0.5, 0.5, 1.5]),
        "distance": np.array([95.0, 105.0, 115.0, 125.0]),
        "wei": np.ones(4),
    }
    cfg = ProjectedAutoConfig(estimator="NAT")
    cfg.distance.calcdist = False
    cfg.columns_data.distance = "distance"
    cfg.columns_random.distance = "distance"

    data_p, rand_p, meta = prepare_projected_auto(data, random, cfg)

    assert data_p.table is None
    assert rand_p.table is None
    assert data_p.nrows == 3
    assert rand_p.nrows == 4
    assert meta["pi_edges"].ndim == 1
    np.testing.assert_allclose(np.sort(data_p.weights), [1.0, 1.0, 2.0])


def test_subset_prepared_helpers_do_not_require_backing_table(monkeypatch):
    monkeypatch.setattr(
        cff,
        "mod",
        type("_FakeMod", (), {
            "skll2d": staticmethod(lambda mxh1, mxh2, npt, ra, dec, sbound: (
                np.zeros((int(mxh2), int(mxh1)), dtype=np.int32),
                np.zeros(int(npt), dtype=np.int32),
            ))
        })(),
    )
    monkeypatch.setattr(
        "nugundam.projected.prepare._build_skll3d",
        lambda mxh1, mxh2, mxh3, ra, dec, dist, sbound, pi_edges: (
            np.zeros((int(mxh3), int(mxh2), int(mxh1)), dtype=np.int32),
            np.zeros(len(ra), dtype=np.int32),
        ),
    )

    ang = prepare_angular_auto(
        pd.DataFrame({"ra": [0.0, 1.0], "dec": [0.0, 1.0], "wei": [1.0, 1.0]}),
        pd.DataFrame({"ra": [0.0, 1.0], "dec": [0.0, 1.0], "wei": [1.0, 1.0]}),
        AngularAutoConfig(estimator="NAT"),
    )[0]
    ang.table = None
    ang_sub = subset_prepared_angular_sample(ang, np.array([True, False]))
    assert ang_sub.nrows == 1

    proj = PreparedProjectedSample(
        table=None,
        ra=np.array([0.0, 1.0]),
        dec=np.array([0.0, 1.0]),
        dist=np.array([100.0, 101.0]),
        weights=np.ones(2, dtype=np.float32),
        x=np.zeros(2),
        y=np.zeros(2),
        z=np.ones(2),
        sk=np.zeros((1, 1, 1), dtype=np.int32),
        ll=np.zeros(2, dtype=np.int32),
        wunit=True,
        sbound=(0.0, 360.0, -1.0, 1.0, 99.0, 102.0),
        mxh1=1,
        mxh2=1,
        mxh3=1,
    )
    proj_sub = subset_prepared_projected_sample(proj, np.array([True, False]), pi_edges=np.array([0.0, 1.0]))
    assert proj_sub.nrows == 1
