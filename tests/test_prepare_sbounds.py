import numpy as np

from nugundam.angular.models import AngularAutoConfig
from nugundam.angular.prepare import prepare_angular_auto


class _Column:
    def __init__(self, data):
        self.data = np.asarray(data)


class _MiniTable:
    def __init__(self, mapping):
        self._mapping = {k: np.asarray(v) for k, v in mapping.items()}
        self.colnames = list(mapping)

    def __len__(self):
        return len(next(iter(self._mapping.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return _Column(self._mapping[key])
        idx = np.asarray(key)
        return _MiniTable({k: v[idx] for k, v in self._mapping.items()})


def test_prepare_angular_auto_uses_shared_original_style_sbound(monkeypatch):
    data = _MiniTable(
        {
            "ra": np.array([10.0, 12.0, 14.0]),
            "dec": np.array([-1.0, 0.0, 1.0]),
            "wei": np.ones(3),
        }
    )
    random = _MiniTable(
        {
            "ra": np.array([200.0, 220.0, 240.0, 260.0]),
            "dec": np.array([-3.0, -2.0, 2.0, 3.0]),
            "wei": np.ones(4),
        }
    )

    seen_sbounds = []

    class _FakeMod:
        @staticmethod
        def skll2d(mxh1, mxh2, npt, ra, dec, sbound):
            seen_sbounds.append(tuple(float(v) for v in sbound))
            return np.zeros((mxh1, mxh2), dtype=np.int32), np.zeros(npt, dtype=np.int32)

    import nugundam.angular.prepare as prep_mod

    monkeypatch.setattr(prep_mod.cff, "mod", _FakeMod)

    config = AngularAutoConfig()
    config.grid.autogrid = False
    config.grid.mxh1 = 2
    config.grid.mxh2 = 3
    config.grid.pxorder = "none"

    data_prepared, random_prepared, meta = prepare_angular_auto(data, random, config)

    expected = (0.0, 360.0, -3.001, 3.001)
    assert data_prepared.sbound == expected
    assert random_prepared.sbound == expected
    assert seen_sbounds == [expected, expected]
    assert meta["grid_data"]["sbound"] == expected
    assert meta["grid_random"]["sbound"] == expected
