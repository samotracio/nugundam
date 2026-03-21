import numpy as np
import pytest

from nugundam.angular.models import BootstrapSpec, JackknifeSpec
from nugundam.core.jackknife import choose_default_nregions, jackknife_cov, normalize_region_labels, validate_resampling_choice


def test_choose_default_nregions_bounds():
    assert choose_default_nregions(5) == 36
    assert choose_default_nregions(30) == 60
    assert choose_default_nregions(80) == 100


def test_normalize_region_labels_preserves_partitioning():
    labels = np.array([10, 10, 7, 42, 7])
    out = normalize_region_labels(labels)
    assert out.shape == labels.shape
    assert out[0] == out[1]
    assert out[2] == out[4]
    assert len(np.unique(out)) == 3


def test_jackknife_cov_matches_standard_formula():
    real = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
    mean = real.mean(axis=0)
    expected = ((3 - 1) / 3) * ((real - mean).T @ (real - mean))
    np.testing.assert_allclose(jackknife_cov(real), expected)


def test_validate_resampling_choice_rejects_simultaneous_modes():
    with pytest.raises(ValueError):
        validate_resampling_choice(BootstrapSpec(enabled=True), JackknifeSpec(enabled=True))
