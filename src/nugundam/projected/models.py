"""Projected configuration, prepared-sample, and result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.common import makebins

from ..angular.models import (
    BootstrapSpec,
    JackknifeSpec,
    CatalogColumns,
    ConfigDescription,
    ConfigDocMixin,
    _fmt_bin_value,
    _format_bin_table,
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


@dataclass(init=False, slots=True)
class ProjectedBinning(ConfigDocMixin):
    """Projected-separation binning specification.

    Instances are created with the explicit named constructors
    :meth:`from_binsize` or :meth:`from_limits`. The projected separation
    ``r_p`` supports both binsize-based and min/max-based construction, while
    the line-of-sight separation ``pi`` keeps the existing ``(nsepv, dsepv)``
    specification used by the projected counters.

    Attributes
    ----------
    nsepp : int
        Number of projected-separation bins.
    seppmin : float
        Lower edge of the first projected-separation bin.
    dsepp : float
        Linear projected-bin width or logarithmic projected-bin step, depending
        on ``logsepp``.
    logsepp : bool
        If True, build logarithmically spaced ``r_p`` bins.
    nsepv : int
        Number of line-of-sight bins.
    dsepv : float
        Line-of-sight bin width in Mpc/h.
    """
    nsepp: int = field(default=20, metadata={"doc": "Number of projected-separation bins."})
    seppmin: float = field(default=0.1, metadata={"doc": "Lower edge of the first projected-separation bin in Mpc/h."})
    dsepp: float = field(default=0.1, metadata={"doc": "Resolved projected-bin width; dex when logsepp=True, otherwise linear."})
    logsepp: bool = field(default=True, metadata={"doc": "Use logarithmic projected bins if True."})
    nsepv: int = field(default=20, metadata={"doc": "Number of line-of-sight bins."})
    dsepv: float = field(default=2.0, metadata={"doc": "Line-of-sight bin width in Mpc/h."})

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "ProjectedBinning instances must be created with ProjectedBinning.from_binsize() "
            "or ProjectedBinning.from_limits()."
        )

    @classmethod
    def _build(
        cls,
        *,
        nsepp: int,
        seppmin: float,
        dsepp: float,
        logsepp: bool,
        nsepv: int,
        dsepv: float,
    ) -> "ProjectedBinning":
        obj = cls.__new__(cls)
        obj.nsepp = int(nsepp)
        obj.seppmin = float(seppmin)
        obj.dsepp = float(dsepp)
        obj.logsepp = bool(logsepp)
        obj.nsepv = int(nsepv)
        obj.dsepv = float(dsepv)
        return obj

    @classmethod
    def from_binsize(
        cls,
        nsepp: int = 20,
        seppmin: float = 0.1,
        dsepp: float = 0.1,
        logsepp: bool = True,
        nsepv: int = 20,
        dsepv: float = 2.0,
    ) -> "ProjectedBinning":
        """Create projected bins from lower edges, counts, and bin sizes.

        Parameters
        ----------
        nsepp : int, default=20
            Number of projected-separation bins.
        seppmin : float, default=0.1
            Lower edge of the first projected-separation bin in Mpc/h.
        dsepp : float, default=0.1
            Projected-bin width; dex when ``logsepp=True``, otherwise linear.
        logsepp : bool, default=True
            If True, use logarithmic spacing for ``r_p``.
        nsepv : int, default=20
            Number of line-of-sight bins.
        dsepv : float, default=2.0
            Line-of-sight bin width in Mpc/h.

        Returns
        -------
        ProjectedBinning
            Binning instance with the resolved internal representation used by
            the projected preparation and counting layer.
        """
        nsepp = int(nsepp)
        seppmin = float(seppmin)
        dsepp = float(dsepp)
        logsepp = bool(logsepp)
        nsepv = int(nsepv)
        dsepv = float(dsepv)
        if nsepp <= 0:
            raise ValueError("nsepp must be positive.")
        if nsepv <= 0:
            raise ValueError("nsepv must be positive.")
        if dsepp <= 0.0:
            raise ValueError("dsepp must be positive.")
        if dsepv <= 0.0:
            raise ValueError("dsepv must be positive.")
        if logsepp and seppmin <= 0.0:
            raise ValueError("seppmin must be positive when logsepp=True.")
        return cls._build(
            nsepp=nsepp, seppmin=seppmin, dsepp=dsepp, logsepp=logsepp, nsepv=nsepv, dsepv=dsepv
        )

    @classmethod
    def from_limits(
        cls,
        nsepp: int = 20,
        seppmin: float = 0.1,
        seppmax: float = 10.0,
        logsepp: bool = True,
        nsepv: int = 20,
        dsepv: float = 2.0,
    ) -> "ProjectedBinning":
        """Create projected bins from projected lower and upper separation limits.

        Parameters
        ----------
        nsepp : int, default=20
            Number of projected-separation bins.
        seppmin : float, default=0.1
            Lower edge of the first projected-separation bin in Mpc/h.
        seppmax : float, default=10.0
            Upper edge of the last projected-separation bin in Mpc/h.
        logsepp : bool, default=True
            If True, use logarithmic spacing for ``r_p`` and derive ``dsepp``.
        nsepv : int, default=20
            Number of line-of-sight bins.
        dsepv : float, default=2.0
            Line-of-sight bin width in Mpc/h.

        Returns
        -------
        ProjectedBinning
            Binning instance whose internal projected step matches the
            requested ``r_p`` range.
        """
        nsepp = int(nsepp)
        seppmin = float(seppmin)
        seppmax = float(seppmax)
        logsepp = bool(logsepp)
        nsepv = int(nsepv)
        dsepv = float(dsepv)
        if nsepp <= 0:
            raise ValueError("nsepp must be positive.")
        if nsepv <= 0:
            raise ValueError("nsepv must be positive.")
        if dsepv <= 0.0:
            raise ValueError("dsepv must be positive.")
        if logsepp:
            if seppmin <= 0.0 or seppmax <= 0.0:
                raise ValueError("seppmin and seppmax must be positive when logsepp=True.")
            if seppmax <= seppmin:
                raise ValueError("seppmax must be larger than seppmin.")
            dsepp = np.log10(seppmax / seppmin) / float(nsepp)
        else:
            if seppmax <= seppmin:
                raise ValueError("seppmax must be larger than seppmin.")
            dsepp = (seppmax - seppmin) / float(nsepp)
        return cls._build(
            nsepp=nsepp, seppmin=seppmin, dsepp=dsepp, logsepp=logsepp, nsepv=nsepv, dsepv=dsepv
        )

    @property
    def rp_edges(self) -> np.ndarray:
        """Resolved projected-separation bin edges."""
        return makebins(self.nsepp, self.seppmin, self.dsepp, self.logsepp)[0]

    @property
    def rp_centers(self) -> np.ndarray:
        """Resolved projected-separation bin centers."""
        return makebins(self.nsepp, self.seppmin, self.dsepp, self.logsepp)[1][1]

    @property
    def rp_widths(self) -> np.ndarray:
        """Resolved projected-separation bin widths or logarithmic steps."""
        return makebins(self.nsepp, self.seppmin, self.dsepp, self.logsepp)[1][2]

    @property
    def seppmax(self) -> float:
        """Upper edge of the last projected-separation bin."""
        return float(self.rp_edges[-1])

    @property
    def pi_edges(self) -> np.ndarray:
        """Resolved line-of-sight bin edges."""
        return makebins(self.nsepv, 0.0, self.dsepv, False)[0]

    @property
    def pi_centers(self) -> np.ndarray:
        """Resolved line-of-sight bin centers."""
        return makebins(self.nsepv, 0.0, self.dsepv, False)[1][1]

    @property
    def pi_widths(self) -> np.ndarray:
        """Resolved line-of-sight bin widths."""
        return makebins(self.nsepv, 0.0, self.dsepv, False)[1][2]

    @property
    def sepvmax(self) -> float:
        """Upper edge of the last line-of-sight bin."""
        return float(self.pi_edges[-1])

    def table(self, axis: str = "rp") -> ConfigDescription:
        """Return a plain-text table with the resolved projected bins.

        Parameters
        ----------
        axis : {'rp', 'pi'}, default='rp'
            Which bin family to render. ``'rp'`` shows projected-separation bins
            and ``'pi'`` shows line-of-sight bins.

        Returns
        -------
        ConfigDescription
            Plain-text table with the left edge, right edge, center, and width
            of each requested bin family.
        """
        axis_norm = str(axis).lower()
        if axis_norm == "rp":
            return ConfigDescription(_format_bin_table(self.rp_edges, self.rp_centers, self.rp_widths))
        if axis_norm == "pi":
            return ConfigDescription(_format_bin_table(self.pi_edges, self.pi_centers, self.pi_widths))
        raise ValueError("axis must be either 'rp' or 'pi'.")

    def __str__(self) -> str:
        spacing = "log" if self.logsepp else "linear"
        step_label = "dex" if self.logsepp else "unit"
        return (
            f"ProjectedBinning(rp: {spacing}, {self.nsepp} bins, seppmin={_fmt_bin_value(self.seppmin)}, "
            f"seppmax={_fmt_bin_value(self.seppmax)}, dsepp={_fmt_bin_value(self.dsepp)} {step_label}; "
            f"pi: {self.nsepv} bins, dsepv={_fmt_bin_value(self.dsepv)}, sepvmax={_fmt_bin_value(self.sepvmax)})"
        )

    __repr__ = __str__

    @classmethod
    def describe(cls, recursive: bool = False, _indent: int = 0) -> ConfigDescription:
        """Build a human-readable description of the projected binning schema."""
        prefix = " " * _indent
        base = str(ConfigDocMixin.describe.__func__(cls, recursive=recursive, _indent=_indent))
        extra = [
            f"{prefix}",
            f"{prefix}Constructors",
            f"{prefix}------------",
            f"{prefix}from_binsize(nsepp=20, seppmin=0.1, dsepp=0.1, logsepp=True, nsepv=20, dsepv=2.0)",
            f"{prefix}from_limits(nsepp=20, seppmin=0.1, seppmax=10.0, logsepp=True, nsepv=20, dsepv=2.0)",
            f"{prefix}",
            f"{prefix}Resolved instance attributes",
            f"{prefix}----------------------------",
            f"{prefix}rp_edges, rp_centers, rp_widths, seppmax, pi_edges, pi_centers, pi_widths, sepvmax",
            f"{prefix}",
            f"{prefix}Instance helpers",
            f"{prefix}----------------",
            f"{prefix}table(axis='rp' or 'pi') -> plain-text table of the resolved bins",
        ]
        return ConfigDescription(base + "\n" + "\n".join(extra))


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
    binning: ProjectedBinning = field(default_factory=ProjectedBinning.from_binsize, metadata={"doc": "Projected separation binning specification."})
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
    binning: ProjectedBinning = field(default_factory=ProjectedBinning.from_binsize, metadata={"doc": "Projected separation binning specification."})
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
    binning: ProjectedBinning = field(default_factory=ProjectedBinning.from_binsize, metadata={"doc": "Projected separation binning specification."})
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
    binning: ProjectedBinning = field(default_factory=ProjectedBinning.from_binsize, metadata={"doc": "Projected separation binning specification."})
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
