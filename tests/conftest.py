import numpy as np
import pytest


@pytest.fixture
def theta_grid():
    edges = np.array([0.1, 0.2, 0.4], dtype=float)
    centers = np.array([0.15, 0.30], dtype=float)
    return edges, centers


@pytest.fixture
def rp_pi_grid():
    rp_edges = np.array([0.1, 0.2, 0.4], dtype=float)
    rp_centers = np.array([0.15, 0.30], dtype=float)
    pi_edges = np.array([0.0, 1.0, 2.0], dtype=float)
    pi_centers = np.array([0.5, 1.5], dtype=float)
    return rp_edges, rp_centers, pi_edges, pi_centers
