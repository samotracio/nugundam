import numpy as np

from nugundam.angular.models import AngularAutoConfig
from nugundam.angular.prepare import _grid_cell_indices, pixsort


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


def _toy_table():
    return _MiniTable(
        {
            "ra": np.array([8.2, 1.0, 3.2, 7.9, 3.8, 1.9], dtype=float),
            "dec": np.array([9.0, 1.0, 0.2, 8.0, 1.8, 0.9], dtype=float),
        }
    )


def test_config_describe_mentions_pxorder_modes_and_not_cust_ra_bound():
    text = str(AngularAutoConfig.describe(recursive=True))
    assert "cell-dec" in text
    assert "cust_ra_bound" not in text


def test_pxorder_none_preserves_input_order():
    table = _toy_table()
    sidx = pixsort(table, "ra", "dec", sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2, pxorder="none")
    assert np.array_equal(sidx, np.arange(len(table)))


def test_pxorder_natural_uses_exact_counter_cells():
    table = _toy_table()
    sidx = pixsort(table, "ra", "dec", sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2, pxorder="natural")
    ra = np.asarray(table["ra"].data)
    dec = np.asarray(table["dec"].data)
    qdec, qra = _grid_cell_indices(ra, dec, sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2)
    expected = np.lexsort((qra, qdec))
    assert np.array_equal(sidx, expected)


def test_pxorder_cell_dec_adds_intra_cell_declination_ordering():
    table = _toy_table()
    sidx = pixsort(table, "ra", "dec", sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2, pxorder="cell-dec")
    ra = np.asarray(table["ra"].data)
    dec = np.asarray(table["dec"].data)
    qdec, qra = _grid_cell_indices(ra, dec, sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2)
    expected = np.lexsort((ra, dec, qra, qdec))
    assert np.array_equal(sidx, expected)

    natural = pixsort(table, "ra", "dec", sbound=(0.0, 10.0, 0.0, 10.0), mxh1=2, mxh2=2, pxorder="natural")
    assert np.array_equal(qdec[sidx], qdec[natural])
    assert np.array_equal(qra[sidx], qra[natural])
    for cell in {(int(a), int(b)) for a, b in zip(qdec, qra)}:
        idx = np.where((qdec[sidx] == cell[0]) & (qra[sidx] == cell[1]))[0]
        if len(idx) > 1:
            assert np.all(np.diff(dec[sidx][idx]) >= 0.0)
