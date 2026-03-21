"""Projected configuration, prepared-sample, and result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..angular.models import (
    BootstrapSpec,
    JackknifeSpec,
    CatalogColumns,
    ConfigDocMixin,
    Estimator,
    ProgressSpec,
    ResultIOMixin,
    SplitRandomSpec,
    WeightSpec,
)


@dataclass(slots=True)
class ProjectedCatalogColumns(ConfigDocMixin):
    """
    ProjectedCatalogColumns helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    ra: str = field(default="ra", metadata={"doc": "Right ascension column name."})
    dec: str = field(default="dec", metadata={"doc": "Declination column name."})
    redshift: str = field(default="z", metadata={"doc": "Redshift column name used when calcdist=True."})
    distance: str = field(default="cdcom", metadata={"doc": "Comoving-distance column name used when calcdist=False."})
    weight: str = field(default="wei", metadata={"doc": "Weight column name."})
    region: str | None = field(default=None, metadata={"doc": "Optional jackknife-region column name."})


@dataclass(slots=True)
class ProjectedBinning(ConfigDocMixin):
    """
    ProjectedBinning helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    nsepp: int = field(default=20, metadata={"doc": "Number of projected-separation bins."})
    seppmin: float = field(default=0.1, metadata={"doc": "Minimum projected separation in Mpc/h."})
    dsepp: float = field(default=0.1, metadata={"doc": "Projected-bin width; dex when logsepp=True, otherwise linear."})
    logsepp: bool = field(default=True, metadata={"doc": "Use logarithmic projected bins when True."})
    nsepv: int = field(default=20, metadata={"doc": "Number of line-of-sight bins."})
    dsepv: float = field(default=2.0, metadata={"doc": "Line-of-sight bin width in Mpc/h."})


@dataclass(slots=True)
class ProjectedGridSpec(ConfigDocMixin):
    """
    ProjectedGridSpec helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    autogrid: bool = field(default=True, metadata={"doc": "Automatically choose the 3D skip-grid size when True using the legacy projected heuristic."})
    mxh1: int = field(default=8, metadata={"doc": "Requested DEC grid size when autogrid=False."})
    mxh2: int = field(default=8, metadata={"doc": "Requested RA grid size when autogrid=False."})
    mxh3: int = field(default=4, metadata={"doc": "Requested comoving-distance grid size when autogrid=False."})
    dens: float | None = field(default=None, metadata={"doc": "Optional target particle density used by the automatic 3D autogrid."})
    pxorder: str = field(default="natural", metadata={"doc": "Preparatory memory ordering: 'none' keeps the input order, 'natural' sorts by 3D grid cell."})


@dataclass(slots=True)
class DistanceSpec(ConfigDocMixin):
    """
    DistanceSpec helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    calcdist: bool = field(default=True, metadata={"doc": "When True compute comoving distances from redshift; otherwise read the configured distance column."})
    h0: float = field(default=100.0, metadata={"doc": "Hubble constant in km/s/Mpc."})
    omegam: float = field(default=0.3, metadata={"doc": "Matter density parameter."})
    omegal: float = field(default=0.7, metadata={"doc": "Dark-energy density parameter."})


@dataclass(slots=True)
class ProjectedAutoConfig(ConfigDocMixin):
    """Configuration for projected auto-correlation measurements.

    This dataclass groups all options required by :func:`nugundam.pcf`, including
    projected binning, 3D gridding, distance handling, weighting, bootstrap,
    jackknife, progress reporting, and the optional split-random RR
    acceleration controlled through :class:`~nugundam.angular.models.SplitRandomSpec`.
    """
    estimator: Estimator = field(default="NAT", metadata={"doc": "Projected auto-correlation estimator: 'NAT', 'DP', or 'LS'."})
    columns_data: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the data catalog."})
    columns_random: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the random catalog."})
    binning: ProjectedBinning = field(default_factory=ProjectedBinning, metadata={"doc": "Projected separation binning specification."})
    grid: ProjectedGridSpec = field(default_factory=ProjectedGridSpec, metadata={"doc": "3D grid / linked-list preparation options."})
    distance: DistanceSpec = field(default_factory=DistanceSpec, metadata={"doc": "Comoving-distance handling options."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec, metadata={"doc": "Bootstrap uncertainty options."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    split_random: SplitRandomSpec = field(default_factory=SplitRandomSpec, metadata={"doc": "Optional split-random RR acceleration for LS auto-correlations."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class ProjectedCrossConfig(ConfigDocMixin):
    """Configuration for projected cross-correlation measurements.

    Separate column mappings are provided for the two data samples and their
    associated random catalogs. Estimator-specific random requirements are
    enforced at API runtime.
    """
    estimator: Estimator = field(default="DP", metadata={"doc": "Projected cross-correlation estimator: 'NAT', 'DP', or 'LS'."})
    columns_data1: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the first data catalog."})
    columns_random1: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the first random catalog when that catalog is required by the chosen estimator."})
    columns_data2: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the second data catalog."})
    columns_random2: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the second random catalog when that catalog is required by the chosen estimator."})
    binning: ProjectedBinning = field(default_factory=ProjectedBinning, metadata={"doc": "Projected separation binning specification."})
    grid: ProjectedGridSpec = field(default_factory=ProjectedGridSpec, metadata={"doc": "3D grid / linked-list preparation options."})
    distance: DistanceSpec = field(default_factory=DistanceSpec, metadata={"doc": "Comoving-distance handling options."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=lambda: BootstrapSpec(mode="primary"), metadata={"doc": "Bootstrap uncertainty options."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class ProjectedAutoCountsConfig(ConfigDocMixin):
    """Configuration for count-only projected auto-pair runs.

    This helper mirrors the counting-related subset of
    :class:`ProjectedAutoConfig` and is used by :func:`nugundam.proj_auto_counts`.
    It documents the same preparation and pair-counting options used by the
    projected auto-correlation pipeline.
    """
    columns: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the input catalog used by count-only projected auto-pair runs."})
    binning: ProjectedBinning = field(default_factory=ProjectedBinning, metadata={"doc": "Projected separation binning specification."})
    grid: ProjectedGridSpec = field(default_factory=ProjectedGridSpec, metadata={"doc": "3D grid / linked-list preparation options."})
    distance: DistanceSpec = field(default_factory=DistanceSpec, metadata={"doc": "Comoving-distance handling options."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec, metadata={"doc": "Bootstrap resampling options for DD counts."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class ProjectedCrossCountsConfig(ConfigDocMixin):
    """
    ProjectedCrossCountsConfig helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    columns1: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the first input catalog used by count-only projected cross-pair runs."})
    columns2: ProjectedCatalogColumns = field(default_factory=ProjectedCatalogColumns, metadata={"doc": "Column names for the second input catalog used by count-only projected cross-pair runs."})
    binning: ProjectedBinning = field(default_factory=ProjectedBinning, metadata={"doc": "Projected separation binning specification."})
    grid: ProjectedGridSpec = field(default_factory=ProjectedGridSpec, metadata={"doc": "3D grid / linked-list preparation options."})
    distance: DistanceSpec = field(default_factory=DistanceSpec, metadata={"doc": "Comoving-distance handling options."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=lambda: BootstrapSpec(mode="primary"), metadata={"doc": "Bootstrap resampling options for D1D2 counts."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class PreparedProjectedSample:
    """Prepared projected sample ready for the compiled counters.

    The arrays stored here are sorted, converted to ``float64``, and augmented
    with the linked-list work arrays expected by the projected Fortran kernels.
    """
    table: object
    ra: np.ndarray
    dec: np.ndarray
    dist: np.ndarray
    weights: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sk: np.ndarray
    ll: np.ndarray
    wunit: bool
    sbound: tuple[float, float, float, float, float, float]
    mxh1: int
    mxh2: int
    mxh3: int
    region_id: np.ndarray | None = None
    grid_meta: dict = field(default_factory=dict)

    nrows: int | None = None

    def __post_init__(self) -> None:
        if self.nrows is None:
            self.nrows = int(len(self.ra))

    def __len__(self) -> int:
        return int(self.nrows)


@dataclass(slots=True)
class ProjectedAutoCounts(ResultIOMixin):
    """
    ProjectedAutoCounts helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    rp_edges: np.ndarray
    rp_centers: np.ndarray
    pi_edges: np.ndarray
    pi_centers: np.ndarray
    dd: np.ndarray
    rr: np.ndarray | None = None
    dr: np.ndarray | None = None
    dd_boot: np.ndarray | None = None
    norm_dd_boot: np.ndarray | None = None
    sum_w_data_boot: np.ndarray | None = None
    dd_jk_touch: np.ndarray | None = None
    rr_jk_touch: np.ndarray | None = None
    dr_jk_touch: np.ndarray | None = None
    intpi_dd: np.ndarray | None = None
    intpi_rr: np.ndarray | None = None
    intpi_dr: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedCrossCounts(ResultIOMixin):
    """
    ProjectedCrossCounts helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    rp_edges: np.ndarray
    rp_centers: np.ndarray
    pi_edges: np.ndarray
    pi_centers: np.ndarray
    d1d2: np.ndarray
    d1r2: np.ndarray | None = None
    r1d2: np.ndarray | None = None
    r1r2: np.ndarray | None = None
    d1d2_boot: np.ndarray | None = None
    d1r2_boot: np.ndarray | None = None
    d1d2_jk_touch: np.ndarray | None = None
    d1r2_jk_touch: np.ndarray | None = None
    r1d2_jk_touch: np.ndarray | None = None
    r1r2_jk_touch: np.ndarray | None = None
    intpi_d1d2: np.ndarray | None = None
    intpi_d1r2: np.ndarray | None = None
    intpi_r1d2: np.ndarray | None = None
    intpi_r1r2: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedAutoCountsResult(ResultIOMixin):
    """
    ProjectedAutoCountsResult helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    rp_edges: np.ndarray
    rp_centers: np.ndarray
    pi_edges: np.ndarray
    pi_centers: np.ndarray
    dd: np.ndarray
    dd_boot: np.ndarray | None = None
    norm_dd_boot: np.ndarray | None = None
    sum_w_data_boot: np.ndarray | None = None
    dd_jk_touch: np.ndarray | None = None
    rr_jk_touch: np.ndarray | None = None
    dr_jk_touch: np.ndarray | None = None
    intpi_dd: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedCrossCountsResult(ResultIOMixin):
    """
    ProjectedCrossCountsResult helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    rp_edges: np.ndarray
    rp_centers: np.ndarray
    pi_edges: np.ndarray
    pi_centers: np.ndarray
    d1d2: np.ndarray
    d1d2_boot: np.ndarray | None = None
    intpi_d1d2: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class ProjectedCorrelationResult(ResultIOMixin):
    """Final projected correlation-function result returned by :func:`nugundam.pcf` or :func:`nugundam.pccf`.

    The result stores the integrated projected statistic ``wp`` together with
    the underlying count terms so that ``xi(r_p, pi)`` can be reconstructed for
    plotting and I/O. When jackknife resampling is enabled, the full covariance
    matrix is stored in ``cov`` and the optional leave-one-region-out
    realizations are stored in ``realizations``.
    """
    rp_edges: np.ndarray
    rp_centers: np.ndarray
    wp: np.ndarray
    wp_err: np.ndarray
    estimator: Estimator
    counts: ProjectedAutoCounts | ProjectedCrossCounts
    cov: np.ndarray | None = None
    realizations: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
