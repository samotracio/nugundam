"""Angular configuration, prepared-sample, and result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import types
from typing import Literal, Union, get_args, get_origin, get_type_hints

import numpy as np

from ..core.common import makebins


@dataclass(frozen=True, slots=True)
class ConfigDescription:
    """Plain-text wrapper returned by :meth:`ConfigDocMixin.describe`.

    Instances of this class render nicely in terminals, notebooks, and rich
    frontends, while still behaving like ordinary strings when needed.
    """

    text: str

    def __str__(self) -> str:
        """
        Return the plain-text representation of this object.
        
        Returns
        -------
        str
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        return self.text

    def __repr__(self) -> str:
        """
        Return the developer-facing representation of this object.
        
        Returns
        -------
        str
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        return self.text

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        """
        Repr pretty.
        
        Parameters
        ----------
        printer : object
            Value for ``printer``.
        cycle : object
            Value for ``cycle``.
        
        Returns
        -------
        object
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        if cycle:
            printer.text("...")
            return
        printer.text(self.text)

    def _repr_markdown_(self) -> str:
        """
        Repr markdown.
        
        Returns
        -------
        str
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        return f"```text\n{self.text}\n```"


class ConfigDocMixin:
    """Mixin providing lightweight schema introspection for config dataclasses.

    Classes inheriting from this mixin gain the :meth:`describe` class method,
    which formats the dataclass fields and their short metadata descriptions in
    a readable text block.
    """

    @classmethod
    def describe(cls, recursive: bool = False, _indent: int = 0) -> ConfigDescription:
        """Build a human-readable schema summary for a config dataclass.

        Parameters
        ----------
        recursive : bool, default=False
            If True, recurse into nested config dataclasses.
        _indent : int, default=0
            Internal indentation level used while recursing.

        Returns
        -------
        ConfigDescription
            Formatted description object suitable for display.
        """
        lines: list[str] = []
        prefix = " " * _indent
        lines.append(f"{prefix}{cls.__name__}")
        lines.append(f"{prefix}{'-' * len(cls.__name__)}")
        hints = get_type_hints(cls)
        for f in fields(cls):
            ann = hints.get(f.name, f.type)
            type_name = _format_type(ann)
            doc = f.metadata.get("doc", "")
            lines.append(f"{prefix}{f.name} : {type_name} = {doc}".rstrip())
            if recursive and _is_config_dataclass_type(ann):
                nested_cls = _unwrap_config_dataclass_type(ann)
                if nested_cls is not None:
                    lines.append("")
                    lines.append(str(nested_cls.describe(recursive=True, _indent=_indent + 2)))
        return ConfigDescription("\n".join(lines))


def _format_type(tp) -> str:
    """
    Format type.
    
    Parameters
    ----------
    tp : object
        Value for ``tp``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    origin = get_origin(tp)
    if origin is Literal:
        vals = ", ".join(repr(v) for v in get_args(tp))
        return f"Literal[{vals}]"
    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return f"tuple[{_format_type(args[0])}, ...]"
        return "tuple[" + ", ".join(_format_type(a) for a in args) + "]"
    if origin is list:
        args = get_args(tp)
        return f"list[{_format_type(args[0])}]" if args else "list"
    if origin is dict:
        args = get_args(tp)
        return f"dict[{_format_type(args[0])}, {_format_type(args[1])}]" if len(args) == 2 else "dict"
    if origin is None:
        return getattr(tp, "__name__", str(tp).replace("typing.", ""))
    if origin in (types.UnionType, Union):
        args = [a for a in get_args(tp)]
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            return f"{_format_type(other)} | None"
        return " | ".join(_format_type(a) for a in args)
    return getattr(origin, "__name__", str(tp).replace("typing.", ""))


def _is_config_dataclass_type(tp) -> bool:
    """
    Return whether config dataclass type holds for the supplied object.
    
    Parameters
    ----------
    tp : object
        Value for ``tp``.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    nested = _unwrap_config_dataclass_type(tp)
    return nested is not None and issubclass(nested, ConfigDocMixin)


def _unwrap_config_dataclass_type(tp):
    """
    Unwrap config dataclass type.
    
    Parameters
    ----------
    tp : object
        Value for ``tp``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    origin = get_origin(tp)
    if origin is None:
        return tp if isinstance(tp, type) else None
    args = [a for a in get_args(tp) if a is not type(None)]
    if len(args) == 1 and isinstance(args[0], type):
        return args[0]
    return None

def _fmt_bin_value(value: float) -> str:
    """Format bin values compactly for summaries and text tables."""
    return f"{float(value):.6g}"


def _format_bin_table(edges: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> str:
    """Render a simple plain-text bin table."""
    lines = [" i        left        right       center       width"]
    for i, (left, right, center, width) in enumerate(zip(edges[:-1], edges[1:], centers, widths)):
        lines.append(
            f"{i:2d}  {_fmt_bin_value(left):>10}  {_fmt_bin_value(right):>10}  "
            f"{_fmt_bin_value(center):>10}  {_fmt_bin_value(width):>10}"
        )
    return "\n".join(lines)



class ResultIOMixin:
    """Mixin adding I/O and plotting convenience methods to result objects.

    The refactored angular and projected result dataclasses inherit from this
    mixin so that save/load, ASCII export, and plotting all share the same
    object-oriented convenience layer.
    """
    def save(self, path: str | "Path") -> None:
        """
        Save.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        
        Returns
        -------
        object
            Object returned by this helper.
        """
        from ..io import save_result
        save_result(self, path)

    def save_result(self, path: str | "Path") -> None:
        """
        Serialize a nuGUNDAM result to the native compressed ``.npz`` format.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        self.save(path)

    def write(self, path: str | "Path") -> None:
        """
        Write.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        
        Returns
        -------
        object
            Object returned by this helper.
        """
        self.save(path)

    def write_result(self, path: str | "Path") -> None:
        """
        Alias for :func:`save_result` using the legacy verb ``write``.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        self.save(path)

    def to_ascii(self, path: str | "Path", cols: list[str] | tuple[str, ...] | None = None) -> None:
        """
        To ascii.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        cols : object, optional
            Value for ``cols``.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        from ..ascii_io import write_ascii
        write_ascii(self, path, cols=cols)

    def plot(self, *args, **kwargs):
        """Plot this result or count object with nuGUNDAM's result-aware helper.

        This is a convenience method equivalent to calling
        ``nugundam.plot_result(self, *args, **kwargs)``. It understands the
        native nuGUNDAM angular result classes and count containers, infers the
        appropriate x and y fields, and forwards all plotting options to the
        shared plotting backend.

        Parameters
        ----------
        *args, **kwargs
            Positional and keyword arguments forwarded directly to
            :func:`nugundam.plot_result`. Common options include ``ax`` to choose
            the target axes, ``label`` for the legend entry, ``errors`` to
            select how uncertainties are drawn, ``which`` to choose a specific
            count field for count objects, and scale-related options such as
            ``loglog``, ``xscale`` and ``yscale``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that received the plot.
        """
        from ..plotting import plot_result
        return plot_result(self, *args, **kwargs)

    def plot2d(self, *args, **kwargs):
        r"""Plot a projected 2D field associated with this result or count object.

        This is a convenience wrapper around :func:`nugundam.plot_result2d`.
        Projected correlation results can use ``which="xi"`` to display the
        underlying :math:`\xi(r_p, \pi)` field, while projected count objects
        can also display raw 2D count grids such as ``dd`` or ``d1d2``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that received the plot.
        """
        from ..plotting import plot_result2d
        return plot_result2d(self, *args, **kwargs)

    def plot_cov_matrix(self, *args, **kwargs):
        """Plot the covariance matrix associated with this result.

        This is a convenience method equivalent to calling
        ``nugundam.plot_cov_matrix(self, *args, **kwargs)``. It is primarily
        intended for correlation-function results that carry a covariance matrix
        in their ``cov`` attribute.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that received the plot.
        """
        from ..plotting import plot_cov_matrix
        return plot_cov_matrix(self, *args, **kwargs)

    def plot_corr_matrix(self, *args, **kwargs):
        """Plot the correlation matrix associated with this result.

        This is a convenience method equivalent to calling
        ``nugundam.plot_corr_matrix(self, *args, **kwargs)``. The helper converts
        the stored covariance matrix to a correlation matrix before rendering.

        Returns
        -------
        matplotlib.axes.Axes
            The axes that received the plot.
        """
        from ..plotting import plot_corr_matrix
        return plot_corr_matrix(self, *args, **kwargs)

    @classmethod
    def read_result(cls, path: str | "Path"):
        """
        Read a nuGUNDAM result from disk and reconstruct the saved object.
        
        Parameters
        ----------
        path : object
            Value for ``path``.
        
        Returns
        -------
        object
            Object returned by this helper.
        """
        from ..io import read_result
        obj = read_result(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded result is {type(obj)!r}, not {cls.__name__}")
        return obj


Estimator = Literal["NAT", "DP", "LS"]
WeightMode = Literal["auto", "weighted", "unweighted"]
BootstrapMode = Literal["none", "primary", "both"]
PrimaryRole = Literal["data1", "data2"]
GridAutoMode = Literal["legacy", "adaptive"]


@dataclass(slots=True)
class CatalogColumns(ConfigDocMixin):
    """Column-name mapping for angular catalog inputs.

    Attributes
    ----------
    ra, dec : str
        Names of the right ascension and declination columns.
    weight : str
        Name of the per-object weight column.
    """
    ra: str = field(default="ra", metadata={"doc": "Right ascension column name."})
    dec: str = field(default="dec", metadata={"doc": "Declination column name."})
    weight: str = field(default="wei", metadata={"doc": "Weight column name."})
    region: str | None = field(default=None, metadata={"doc": "Optional jackknife-region column name."})


@dataclass(init=False, slots=True)
class AngularBinning(ConfigDocMixin):
    """Angular separation binning specification.

    Instances are created with the explicit named constructors
    :meth:`from_binsize` or :meth:`from_limits`. Both constructors resolve to
    the same internal ``(nsep, sepmin, dsep, logsep)`` representation used by
    the existing pair-counting preparation code.

    Attributes
    ----------
    nsep : int
        Number of angular bins.
    sepmin : float
        Lower edge of the first angular bin.
    dsep : float
        Linear bin width or logarithmic bin step, depending on ``logsep``.
    logsep : bool
        If True, build logarithmically spaced bins.

    Notes
    -----
    Use :meth:`from_binsize` when you want to specify the bin size directly,
    or :meth:`from_limits` when you prefer to specify the lower and upper
    angular limits and let nuGUNDAM derive the implied step.
    """
    nsep: int = field(default=36, metadata={"doc": "Number of angular separation bins."})
    sepmin: float = field(default=0.01, metadata={"doc": "Lower edge of the first angular bin in the native angular unit."})
    dsep: float = field(default=0.1, metadata={"doc": "Resolved bin width; in dex when logsep=True, otherwise linear."})
    logsep: bool = field(default=True, metadata={"doc": "Use logarithmic angular bins if True, linear bins otherwise."})

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "AngularBinning instances must be created with AngularBinning.from_binsize() "
            "or AngularBinning.from_limits()."
        )

    @classmethod
    def _build(cls, *, nsep: int, sepmin: float, dsep: float, logsep: bool) -> "AngularBinning":
        obj = cls.__new__(cls)
        obj.nsep = int(nsep)
        obj.sepmin = float(sepmin)
        obj.dsep = float(dsep)
        obj.logsep = bool(logsep)
        return obj

    @classmethod
    def from_binsize(
        cls,
        nsep: int = 36,
        sepmin: float = 0.01,
        dsep: float = 0.1,
        logsep: bool = True,
    ) -> "AngularBinning":
        """Create angular bins from a lower edge, bin count, and bin size.

        Parameters
        ----------
        nsep : int, default=36
            Number of angular bins.
        sepmin : float, default=0.01
            Lower edge of the first angular bin.
        dsep : float, default=0.1
            Bin width; in dex when ``logsep=True``, otherwise linear.
        logsep : bool, default=True
            If True, use logarithmic spacing.

        Returns
        -------
        AngularBinning
            Binning instance with the resolved internal representation used by
            the pair-counting preparation layer.
        """
        nsep = int(nsep)
        sepmin = float(sepmin)
        dsep = float(dsep)
        logsep = bool(logsep)
        if nsep <= 0:
            raise ValueError("nsep must be positive.")
        if dsep <= 0.0:
            raise ValueError("dsep must be positive.")
        if logsep and sepmin <= 0.0:
            raise ValueError("sepmin must be positive when logsep=True.")
        return cls._build(nsep=nsep, sepmin=sepmin, dsep=dsep, logsep=logsep)

    @classmethod
    def from_limits(
        cls,
        nsep: int = 36,
        sepmin: float = 0.01,
        sepmax: float = 10.0,
        logsep: bool = True,
    ) -> "AngularBinning":
        """Create angular bins from lower and upper separation limits.

        Parameters
        ----------
        nsep : int, default=36
            Number of angular bins.
        sepmin : float, default=0.01
            Lower edge of the first angular bin.
        sepmax : float, default=10.0
            Upper edge of the last angular bin.
        logsep : bool, default=True
            If True, use logarithmic spacing and derive the logarithmic step.

        Returns
        -------
        AngularBinning
            Binning instance whose internal ``dsep`` matches the requested
            angular range.
        """
        nsep = int(nsep)
        sepmin = float(sepmin)
        sepmax = float(sepmax)
        logsep = bool(logsep)
        if nsep <= 0:
            raise ValueError("nsep must be positive.")
        if logsep:
            if sepmin <= 0.0 or sepmax <= 0.0:
                raise ValueError("sepmin and sepmax must be positive when logsep=True.")
            if sepmax <= sepmin:
                raise ValueError("sepmax must be larger than sepmin.")
            dsep = np.log10(sepmax / sepmin) / float(nsep)
        else:
            if sepmax <= sepmin:
                raise ValueError("sepmax must be larger than sepmin.")
            dsep = (sepmax - sepmin) / float(nsep)
        return cls._build(nsep=nsep, sepmin=sepmin, dsep=dsep, logsep=logsep)

    @property
    def edges(self) -> np.ndarray:
        """Resolved angular bin edges."""
        return makebins(self.nsep, self.sepmin, self.dsep, self.logsep)[0]

    @property
    def centers(self) -> np.ndarray:
        """Resolved angular bin centers."""
        return makebins(self.nsep, self.sepmin, self.dsep, self.logsep)[1][1]

    @property
    def widths(self) -> np.ndarray:
        """Resolved angular bin widths or logarithmic steps."""
        return makebins(self.nsep, self.sepmin, self.dsep, self.logsep)[1][2]

    @property
    def sepmax(self) -> float:
        """Upper edge of the last angular bin."""
        return float(self.edges[-1])

    def table(self) -> ConfigDescription:
        """Return a plain-text table with the resolved angular bins.

        Returns
        -------
        ConfigDescription
            Plain-text table with the left edge, right edge, center, and width
            of each angular bin.
        """
        return ConfigDescription(_format_bin_table(self.edges, self.centers, self.widths))

    def __str__(self) -> str:
        spacing = "log" if self.logsep else "linear"
        step_label = "dex" if self.logsep else "unit"
        return (
            f"AngularBinning({spacing}, {self.nsep} bins, sepmin={_fmt_bin_value(self.sepmin)}, "
            f"sepmax={_fmt_bin_value(self.sepmax)}, dsep={_fmt_bin_value(self.dsep)} {step_label})"
        )

    __repr__ = __str__

    @classmethod
    def describe(cls, recursive: bool = False, _indent: int = 0) -> ConfigDescription:
        """Build a human-readable description of the angular binning schema."""
        prefix = " " * _indent
        base = str(ConfigDocMixin.describe.__func__(cls, recursive=recursive, _indent=_indent))
        extra = [
            f"{prefix}",
            f"{prefix}Constructors",
            f"{prefix}------------",
            f"{prefix}from_binsize(nsep=36, sepmin=0.01, dsep=0.1, logsep=True)",
            f"{prefix}from_limits(nsep=36, sepmin=0.01, sepmax=10.0, logsep=True)",
            f"{prefix}",
            f"{prefix}Resolved instance attributes",
            f"{prefix}----------------------------",
            f"{prefix}edges, centers, widths, sepmax",
            f"{prefix}",
            f"{prefix}Instance helpers",
            f"{prefix}----------------",
            f"{prefix}table() -> plain-text table of the resolved bins",
        ]
        return ConfigDescription(base + "\n" + "\n".join(extra))


@dataclass(slots=True)
class AngularGridSpec(ConfigDocMixin):
    """Grid and memory-ordering options for angular pair counting.

    Attributes
    ----------
    autogrid : bool or {"legacy", "adaptive"}
        Automatic grid-selection mode. ``True`` maps to the legacy nuGUNDAM
        heuristic.
    mxh1, mxh2 : int
        Explicit grid dimensions used when ``autogrid`` is disabled.
    pxorder : {"none", "natural", "cell-dec"}
        Preparatory input ordering used before linked-list construction.
    coarse_bins : int
        Coarse footprint probe size used by the adaptive autogrid mode.
    """
    autogrid: bool | GridAutoMode = field(default=True, metadata={"doc": "Grid selection mode: False uses user-supplied mxh1/mxh2, True or 'legacy' uses the original nuGUNDAM heuristic, and 'adaptive' uses the newer runtime-aware count-box probe."})
    mxh1: int = field(default=20, metadata={"doc": "Requested grid size along the first angular dimension when autogrid=False."})
    mxh2: int = field(default=20, metadata={"doc": "Requested grid size along the second angular dimension when autogrid=False."})
    dens: float | None = field(default=None, metadata={"doc": "Optional target particle density used by automatic grid builders."})
    pxorder: str = field(default="natural", metadata={"doc": "Preparatory memory ordering: 'none' keeps the input order, 'natural' follows the exact counter grid order, and 'cell-dec' additionally sorts within each cell by declination."})
    coarse_bins: int = field(default=32, metadata={"doc": "Nominal coarse count-box resolution used by the adaptive autogrid. Values around 16-64 keep the geometry probe cheap while still capturing patchy footprints."})


@dataclass(slots=True)
class WeightSpec(ConfigDocMixin):
    """
    WeightSpec helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    weight_mode: WeightMode = field(default="auto", metadata={"doc": "Weight handling mode: 'auto', 'weighted', or 'unweighted'."})
    data_col: str = field(default="wei", metadata={"doc": "Weight column name for the primary data catalog in auto-correlation runs."})
    random_col: str | None = field(default=None, metadata={"doc": "Weight column name for the random catalog; None reuses the data convention when appropriate."})
    data1_col: str = field(default="wei", metadata={"doc": "Weight column name for the first data catalog in cross-correlation runs."})
    data2_col: str = field(default="wei", metadata={"doc": "Weight column name for the second data catalog in cross-correlation runs."})


@dataclass(slots=True)
class BootstrapSpec(ConfigDocMixin):
    """
    BootstrapSpec helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    enabled: bool = field(default=False, metadata={"doc": "Enable bootstrap resampling for uncertainty estimates."})
    nbts: int = field(default=50, metadata={"doc": "Number of bootstrap resamples."})
    bseed: int = field(default=12345, metadata={"doc": "Random seed used for bootstrap resampling."})
    mode: BootstrapMode = field(default="none", metadata={"doc": "Bootstrap strategy: 'none', 'primary', or 'both'."})
    primary: PrimaryRole = field(default="data1", metadata={"doc": "Primary sample used by cross-correlation bootstrap schemes."})


@dataclass(slots=True)
class JackknifeSpec(ConfigDocMixin):
    """Configuration for region-based delete-one jackknife uncertainties.

    The jackknife machinery divides the survey footprint into sky regions and
    recomputes the statistic while omitting one region at a time. Users may
    either provide explicit region labels through the catalog-column mappings or
    let nuGUNDAM generate shared regions automatically from the survey geometry.

    Attributes
    ----------
    enabled : bool
        When True, compute jackknife error bars and covariance estimates.
    nregions : int or None
        Requested number of jackknife regions. When None, nuGUNDAM chooses a
        practical default from the number of output bins.
    generator : {"kmeans"}
        Automatic sky-region generator used when no region column is supplied.
    geometry_from : {"auto", "randoms", "data"}
        Catalog family used to define the automatic patch geometry.
    seed : int
        Random seed used by the automatic region generator.
    cross_patch_weight : {"simple"}
        Patch-weighting strategy used when constructing delete-one realizations.
        Only the classical ``"simple"`` scheme is implemented at present.
    return_cov : bool
        When True, store the full jackknife covariance matrix in the result.
    return_realizations : bool
        When True, also store the leave-one-region-out realizations.
    """
    enabled: bool = field(default=False, metadata={"doc": "Enable jackknife uncertainty estimation."})
    nregions: int | None = field(default=None, metadata={"doc": "Number of jackknife regions; None chooses a practical default from the binning."})
    generator: Literal["kmeans"] = field(default="kmeans", metadata={"doc": "Automatic region generator used when no region columns are supplied."})
    geometry_from: Literal["auto", "randoms", "data"] = field(default="auto", metadata={"doc": "Catalog family used to derive automatic region geometry."})
    seed: int = field(default=12345, metadata={"doc": "Random seed used when generating automatic jackknife regions."})
    cross_patch_weight: Literal["simple"] = field(default="simple", metadata={"doc": "Patch weighting strategy. Only 'simple' is implemented in v39."})
    return_cov: bool = field(default=True, metadata={"doc": "Return the full jackknife covariance matrix."})
    return_realizations: bool = field(default=False, metadata={"doc": "Return leave-one-region-out realizations in the final result."})


@dataclass(slots=True)
class ProgressSpec(ConfigDocMixin):
    """
    ProgressSpec helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    enabled: bool = field(default=True, metadata={"doc": "Enable runtime progress reporting."})
    progress_file: str | None = field(default=None, metadata={"doc": "Optional path for progress messages; None uses the default backend behavior."})
    poll_interval: float = field(default=0.15, metadata={"doc": "Polling interval in seconds for notebook progress watchers."})


@dataclass(slots=True)
class SplitRandomSpec(ConfigDocMixin):
    """Configuration controlling the optional split-random RR accelerator.

    When enabled, nuGUNDAM keeps the full data-random count but evaluates the
    random-random term as the sum of within-chunk RR counts over a shuffled set
    of smaller random subcatalogs. Each RR chunk is re-autogridded and
    re-sorted before counting so the split runs still benefit from the normal
    nuGUNDAM cache-friendly memory layout. The estimator then uses the exact
    number of included RR pairs stored in the count metadata.

    Notes
    -----
    This option is currently supported only for auto-correlations with the
    Landy-Szalay estimator (``estimator="LS"``). It is not yet available for
    cross-correlations or for jackknife runs.
    """
    enabled: bool = field(default=False, metadata={"doc": "Enable split-random RR counting for auto LS runs."})
    mode: Literal["match_data", "nchunks", "chunk_size"] = field(default="match_data", metadata={"doc": "How to choose the random subcatalog sizes: match the data size, request a fixed number of chunks, or request a target chunk size."})
    nchunks: int | None = field(default=None, metadata={"doc": "Requested number of random chunks when mode='nchunks'."})
    chunk_size: int | None = field(default=None, metadata={"doc": "Requested chunk size when mode='chunk_size'."})
    seed: int = field(default=12345, metadata={"doc": "Random seed used to shuffle the prepared random catalog before assigning chunks."})


@dataclass(slots=True)
class AngularAutoConfig(ConfigDocMixin):
    """Configuration for angular auto-correlation measurements.

    This dataclass groups all options required by :func:`nugundam.acf`, including
    column mappings, angular binning, grid construction, weighting, bootstrap,
    jackknife, progress reporting, and the optional split-random RR
    acceleration controlled through :class:`SplitRandomSpec`.
    """
    estimator: Estimator = field(default="NAT", metadata={"doc": "Angular auto-correlation estimator: 'NAT', 'DP', or 'LS'."})
    columns_data: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the data catalog."})
    columns_random: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the random catalog."})
    binning: AngularBinning = field(default_factory=AngularBinning.from_binsize, metadata={"doc": "Angular separation binning specification."})
    grid: AngularGridSpec = field(default_factory=AngularGridSpec, metadata={"doc": "Grid / linked-list preparation options for pair counting."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec, metadata={"doc": "Bootstrap uncertainty options."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    split_random: SplitRandomSpec = field(default_factory=SplitRandomSpec, metadata={"doc": "Optional split-random RR acceleration for LS auto-correlations."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class AngularCrossConfig(ConfigDocMixin):
    """Configuration for angular cross-correlation measurements.

    Separate column mappings are provided for the two data samples and for the
    random catalogs associated with each of them. Estimator-specific random
    requirements are enforced at API runtime.
    """
    estimator: Estimator = field(default="DP", metadata={"doc": "Angular cross-correlation estimator: 'NAT', 'DP', or 'LS'."})
    columns_data1: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the first data catalog."})
    columns_random1: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the first random catalog when that catalog is required by the chosen estimator."})
    columns_data2: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the second data catalog."})
    columns_random2: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the second random catalog when that catalog is required by the chosen estimator."})
    binning: AngularBinning = field(default_factory=AngularBinning.from_binsize, metadata={"doc": "Angular separation binning specification."})
    grid: AngularGridSpec = field(default_factory=AngularGridSpec, metadata={"doc": "Grid / linked-list preparation options for pair counting."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=lambda: BootstrapSpec(mode="primary"), metadata={"doc": "Bootstrap uncertainty options."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class AngularAutoCountsConfig(ConfigDocMixin):
    """Configuration for count-only angular auto-pair runs.

    This helper mirrors the counting-related subset of
    :class:`AngularAutoConfig` and is used by :func:`nugundam.ang_auto_counts`.
    The ``split_random`` field is kept for schema symmetry and documentation,
    but the count-only helper itself does not estimate ``w(theta)``.
    """
    columns: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the input catalog used by count-only angular auto-pair runs."})
    binning: AngularBinning = field(default_factory=AngularBinning.from_binsize, metadata={"doc": "Angular separation binning specification."})
    grid: AngularGridSpec = field(default_factory=AngularGridSpec, metadata={"doc": "Grid / linked-list preparation options for pair counting."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=BootstrapSpec, metadata={"doc": "Bootstrap resampling options for DD counts."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    split_random: SplitRandomSpec = field(default_factory=SplitRandomSpec, metadata={"doc": "Optional split-random RR acceleration for LS auto-correlations."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class AngularCrossCountsConfig(ConfigDocMixin):
    """
    AngularCrossCountsConfig helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    columns1: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the first input catalog used by count-only angular cross-pair runs."})
    columns2: CatalogColumns = field(default_factory=CatalogColumns, metadata={"doc": "Column names for the second input catalog used by count-only angular cross-pair runs."})
    binning: AngularBinning = field(default_factory=AngularBinning.from_binsize, metadata={"doc": "Angular separation binning specification."})
    grid: AngularGridSpec = field(default_factory=AngularGridSpec, metadata={"doc": "Grid / linked-list preparation options for pair counting."})
    weights: WeightSpec = field(default_factory=WeightSpec, metadata={"doc": "Weight handling options."})
    bootstrap: BootstrapSpec = field(default_factory=lambda: BootstrapSpec(mode="primary"), metadata={"doc": "Bootstrap resampling options for D1D2 counts."})
    jackknife: JackknifeSpec = field(default_factory=JackknifeSpec, metadata={"doc": "Jackknife uncertainty options."})
    progress: ProgressSpec = field(default_factory=ProgressSpec, metadata={"doc": "Progress-reporting options."})
    split_random: SplitRandomSpec = field(default_factory=SplitRandomSpec, metadata={"doc": "Optional split-random RR acceleration for LS auto-correlations."})
    nthreads: int = field(default=-1, metadata={"doc": "Number of OpenMP threads; -1 lets the runtime choose."})
    description: str = field(default="", metadata={"doc": "Optional free-form description stored with the run metadata."})


@dataclass(slots=True)
class PreparedAngularSample:
    """
    PreparedAngularSample helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    table: object
    ra: np.ndarray
    dec: np.ndarray
    weights: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    sk: np.ndarray
    ll: np.ndarray
    wunit: bool
    sbound: tuple[float, float, float, float]
    mxh1: int
    mxh2: int
    region_id: np.ndarray | None = None
    grid_meta: dict = field(default_factory=dict)

    nrows: int | None = None

    def __post_init__(self) -> None:
        if self.nrows is None:
            self.nrows = int(len(self.ra))

    def __len__(self) -> int:
        return int(self.nrows)


@dataclass(slots=True)
class AngularAutoCounts(ResultIOMixin):
    """
    AngularAutoCounts helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    dd: np.ndarray
    rr: np.ndarray | None = None
    dr: np.ndarray | None = None
    dd_boot: np.ndarray | None = None
    norm_dd_boot: np.ndarray | None = None
    sum_w_data_boot: np.ndarray | None = None
    dd_jk_touch: np.ndarray | None = None
    rr_jk_touch: np.ndarray | None = None
    dr_jk_touch: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AngularCrossCounts(ResultIOMixin):
    """
    AngularCrossCounts helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    theta_edges: np.ndarray
    theta_centers: np.ndarray
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
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AngularAutoCountsResult(ResultIOMixin):
    """
    AngularAutoCountsResult helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    dd: np.ndarray
    dd_boot: np.ndarray | None = None
    norm_dd_boot: np.ndarray | None = None
    sum_w_data_boot: np.ndarray | None = None
    dd_jk_touch: np.ndarray | None = None
    rr_jk_touch: np.ndarray | None = None
    dr_jk_touch: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AngularCrossCountsResult(ResultIOMixin):
    """
    AngularCrossCountsResult helper or container class.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    d1d2: np.ndarray
    d1d2_boot: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AngularCorrelationResult(ResultIOMixin):
    """Final angular correlation-function result returned by :func:`nugundam.acf` or :func:`nugundam.accf`.

    The result stores the measured angular statistic ``wtheta`` together with
    its diagonal uncertainties and the underlying count terms. When jackknife
    resampling is enabled, the result may also carry the full covariance matrix
    in ``cov`` and the leave-one-region-out realizations in ``realizations``.
    """
    theta_edges: np.ndarray
    theta_centers: np.ndarray
    wtheta: np.ndarray
    wtheta_err: np.ndarray
    estimator: Estimator
    counts: AngularAutoCounts | AngularCrossCounts
    cov: np.ndarray | None = None
    realizations: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
