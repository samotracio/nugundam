"""Projected sample preparation, distance handling, and 3D gridding utilities."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import nugundam.cflibfor as cff

from ..core.catalogs import catalog_get_column, catalog_has_column, catalog_nrows
from ..core.common import makebins, radec2xyz
from ..core.jackknife import build_shared_sky_regions, choose_default_nregions, normalize_region_labels
from .models import (
    PreparedProjectedSample,
    ProjectedAutoConfig,
    ProjectedCrossConfig,
    ProjectedCatalogColumns,
    ProjectedGridSpec,
)


def _as_1d_float64(values) -> np.ndarray:
    """
    Convert the supplied value to 1d float64.
    
    Parameters
    ----------
    values : object
        Value for ``values``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    return np.ravel(np.asarray(values, dtype=np.float64))


def _col(table, name: str, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(catalog_get_column(table, name, dtype=dtype), dtype=dtype)


def _shared_user_region_ids(pairs):
    arrays = []
    labels = []
    for table, region_col in pairs:
        arr = np.asarray(_col(table, region_col))
        arrays.append(arr)
        labels.append(arr)
    full = normalize_region_labels(np.concatenate(labels))
    out = []
    start = 0
    for arr in arrays:
        stop = start + len(arr)
        out.append(full[start:stop])
        start = stop
    return out


def _auto_region_ids(catalogs, *, nregions: int, seed: int, geometry_from: str = "auto"):
    random_catalogs = [item for item in catalogs if item[3] == "random"]
    data_catalogs = [item for item in catalogs if item[3] == "data"]
    if geometry_from == "randoms":
        geometry = random_catalogs or data_catalogs
    elif geometry_from == "data":
        geometry = data_catalogs or random_catalogs
    else:
        geometry = random_catalogs or data_catalogs
    geometry_pairs = [(_col(table, ra_col), _col(table, dec_col)) for table, ra_col, dec_col, _kind in geometry]
    assign_pairs = [(_col(table, ra_col), _col(table, dec_col)) for table, ra_col, dec_col, _kind in catalogs]
    assignments, centers = build_shared_sky_regions(geometry_pairs, assign_pairs, nregions=nregions, seed=seed)
    return assignments, centers


def subset_prepared_projected_sample(sample: PreparedProjectedSample, keep, *, pi_edges, regrid: bool = False) -> PreparedProjectedSample:
    idx = np.asarray(keep)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    else:
        idx = np.asarray(idx, dtype=np.int64)
    ra = np.asarray(sample.ra[idx], dtype=np.float64)
    dec = np.asarray(sample.dec[idx], dtype=np.float64)
    dist = np.asarray(sample.dist[idx], dtype=np.float64)
    weights = np.asarray(sample.weights[idx], dtype=np.float32)
    x = np.asarray(sample.x[idx], dtype=np.float64)
    y = np.asarray(sample.y[idx], dtype=np.float64)
    z = np.asarray(sample.z[idx], dtype=np.float64)
    region_id = None if sample.region_id is None else np.asarray(sample.region_id[idx], dtype=np.int32)

    mxh1 = int(sample.mxh1)
    mxh2 = int(sample.mxh2)
    mxh3 = int(sample.mxh3)
    grid_meta = dict(sample.grid_meta)
    if regrid and len(idx) > 0 and bool(grid_meta.get("autogrid", True)):
        mxh1, mxh2, mxh3, _ = best_skgrid_3d_legacy(
            len(ra),
            ra,
            sbound=sample.sbound,
            nsepv=int(grid_meta.get("nsepv", max(1, len(pi_edges) - 1))),
            dsepv=float(grid_meta.get("dsepv", np.asarray(pi_edges, dtype=np.float64)[1] - np.asarray(pi_edges, dtype=np.float64)[0] if len(pi_edges) > 1 else 1.0)),
            dens=grid_meta.get("dens", None),
        )
        sidx = _sort_index_3d(ra, dec, dist, sbound=sample.sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=grid_meta.get("pxorder", "natural"))
        ra, dec, dist = ra[sidx], dec[sidx], dist[sidx]
        weights = np.asarray(weights[sidx], dtype=np.float32)
        x, y, z = x[sidx], y[sidx], z[sidx]
        if region_id is not None:
            region_id = np.asarray(region_id[sidx], dtype=np.int32)
    sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist, np.asarray(sample.sbound, dtype=np.float64), np.asarray(pi_edges, dtype=np.float64))
    return PreparedProjectedSample(
        table=None,
        ra=ra,
        dec=dec,
        dist=dist,
        weights=weights,
        x=x,
        y=y,
        z=z,
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.allclose(weights, 1.0)),
        sbound=tuple(float(v) for v in sample.sbound),
        mxh1=int(mxh1),
        mxh2=int(mxh2),
        mxh3=int(mxh3),
        region_id=region_id,
        grid_meta=grid_meta,
        nrows=int(len(idx)),
    )


def bound3d(dec_arrays, dist_arrays):
    """
    Bound3d.
    
    Parameters
    ----------
    dec_arrays : object
        Value for ``dec_arrays``.
    dist_arrays : object
        Value for ``dist_arrays``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    decs = [_as_1d_float64(a) for a in dec_arrays]
    dists = [_as_1d_float64(a) for a in dist_arrays]
    delta = 1.0e-3
    decmin = max(min(float(np.min(a)) for a in decs) - delta, -90.0)
    decmax = min(max(float(np.max(a)) for a in decs) + delta, 90.0)
    dcmin = max(min(float(np.min(a)) for a in dists) - delta, 0.0)
    dcmax = max(float(np.max(a)) for a in dists) + delta
    return (0.0, 360.0, decmin, decmax, dcmin, dcmax)


def _distance_array(table, columns: ProjectedCatalogColumns, config: ProjectedAutoConfig | ProjectedCrossConfig) -> np.ndarray:
    """
    Distance array.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if config.distance.calcdist:
        try:
            from astropy.cosmology import LambdaCDM
        except Exception as exc:  # pragma: no cover - exercised only when optional dependency is missing
            raise ImportError(
                "Projected correlations with distance.calcdist=True require astropy to be installed. "
                "Either install astropy or provide a precomputed comoving-distance column and set calcdist=False."
            ) from exc
        z = _col(table, columns.redshift)
        cosmo = LambdaCDM(H0=config.distance.h0, Om0=config.distance.omegam, Ode0=config.distance.omegal)
        return _as_1d_float64(cosmo.comoving_distance(z).value)
    return _col(table, columns.distance)


def best_skgrid_3d_legacy(npts, ras, *, sbound, nsepv: int, dsepv: float, dens=None):
    """
    Choose skgrid 3d legacy.
    
    Parameters
    ----------
    npts : object
        Value for ``npts``.
    ras : object
        Value for ``ras``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    nsepv : object
        Value for ``nsepv``. This argument is keyword-only.
    dsepv : object
        Value for ``dsepv``. This argument is keyword-only.
    dens : object, optional
        Value for ``dens``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if isinstance(npts, (list, tuple)):
        npts_eff = int(sum(int(v) for v in npts))
        ras_eff = np.concatenate([_as_1d_float64(r) for r in ras])
    else:
        npts_eff = int(npts)
        ras_eff = _as_1d_float64(ras)
    if dens is None:
        dens = 18.0 if npts_eff > 100000 else 8.0
    if ras_eff.size == 0:
        samplewidth = 360.0
    else:
        samplewidth = float(np.max(ras_eff) - np.min(ras_eff))
        if samplewidth <= 0.0:
            samplewidth = 360.0
    dcmin, dcmax = float(sbound[4]), float(sbound[5])
    rvmax = float(nsepv) * float(dsepv)
    h3 = max(int((dcmax - dcmin) / max(rvmax, 1.0e-9)), 1)
    h1h2 = npts_eff / (float(dens) * h3)
    h1 = max(int(np.rint(2.92 + 0.05 * np.sqrt(max(npts_eff, 1)))), 1)
    h2 = max(int(np.rint(h1h2 / h1) * (360.0 / samplewidth)), 1)
    return h1, h2, h3, float(dens)


def _cell_indices_3d(ra, dec, dist, *, sbound, mxh1: int, mxh2: int, mxh3: int):
    """
    Cell indices 3d.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ramin, ramax, decmin, decmax, dmin, dmax = [float(v) for v in sbound]
    ra_span = max(ramax - ramin, 360.0)
    dec_span = max(decmax - decmin, 1.0e-12)
    dist_span = max(dmax - dmin, 1.0e-12)
    qra = np.floor((np.mod(ra - ramin, ra_span)) / ra_span * mxh2).astype(np.int64)
    qdec = np.floor((dec - decmin) / dec_span * mxh1).astype(np.int64)
    qdist = np.floor((dist - dmin) / dist_span * mxh3).astype(np.int64)
    np.clip(qra, 0, mxh2 - 1, out=qra)
    np.clip(qdec, 0, mxh1 - 1, out=qdec)
    np.clip(qdist, 0, mxh3 - 1, out=qdist)
    return qra, qdec, qdist


def _sort_index_3d(ra, dec, dist, *, sbound, mxh1: int, mxh2: int, mxh3: int, pxorder: str) -> np.ndarray:
    """
    Sort index 3d.
    
    Parameters
    ----------
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
    pxorder : object
        Value for ``pxorder``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    mode = str(pxorder or "none").strip().lower()
    if mode in {"none", "null"}:
        return np.arange(len(ra), dtype=np.int64)
    if mode != "natural":
        raise NameError(f"Projected pxorder {pxorder!r} not implemented. Use 'none' or 'natural'.")
    qra, qdec, qdist = _cell_indices_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3)
    return np.lexsort((qra, qdec, qdist))


def _build_skll3d_python(mxh1: int, mxh2: int, mxh3: int, ra, dec, dist, sbound, pi_edges):
    """
    Build skll3d python.
    
    Parameters
    ----------
    mxh1 : object
        Value for ``mxh1``.
    mxh2 : object
        Value for ``mxh2``.
    mxh3 : object
        Value for ``mxh3``.
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``.
    pi_edges : object
        Value for ``pi_edges``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    qra, qdec, qdist = _cell_indices_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3)
    sk = np.zeros((mxh3, mxh2, mxh1), dtype=np.int32)
    ll = np.zeros(len(ra), dtype=np.int32)
    for i in range(len(ra)):
        c3, c2, c1 = int(qdist[i]), int(qra[i]), int(qdec[i])
        prev = sk[c3, c2, c1]
        ll[i] = prev
        sk[c3, c2, c1] = i + 1
    return sk, ll


def _build_skll3d(mxh1: int, mxh2: int, mxh3: int, ra, dec, dist, sbound, pi_edges):
    """
    Build skll3d.
    
    Parameters
    ----------
    mxh1 : object
        Value for ``mxh1``.
    mxh2 : object
        Value for ``mxh2``.
    mxh3 : object
        Value for ``mxh3``.
    ra : object
        Value for ``ra``.
    dec : object
        Value for ``dec``.
    dist : object
        Value for ``dist``.
    sbound : object
        Value for ``sbound``.
    pi_edges : object
        Value for ``pi_edges``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    try:
        return cff.mod.skll3d(mxh1, mxh2, mxh3, len(ra), ra, dec, dist, sbound, pi_edges, len(pi_edges) - 1)
    except Exception:
        return _build_skll3d_python(mxh1, mxh2, mxh3, ra, dec, dist, sbound, pi_edges)


def _prepare_sample(table, columns: ProjectedCatalogColumns, config, *, sbound, mxh1: int, mxh2: int, mxh3: int, pi_edges, use_weights: bool, region_id=None, grid_meta=None):
    """
    Prepare sample.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
    config : object
        Value for ``config``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    mxh1 : object
        Value for ``mxh1``. This argument is keyword-only.
    mxh2 : object
        Value for ``mxh2``. This argument is keyword-only.
    mxh3 : object
        Value for ``mxh3``. This argument is keyword-only.
    pi_edges : object
        Value for ``pi_edges``. This argument is keyword-only.
    use_weights : object
        Value for ``use_weights``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ra = _col(table, columns.ra)
    dec = _col(table, columns.dec)
    dist = _distance_array(table, columns, config)
    if use_weights:
        if catalog_has_column(table, columns.weight):
            weights = _col(table, columns.weight, dtype=np.float64)
        else:
            if config.weights.weight_mode == "weighted":
                raise KeyError(columns.weight)
            weights = np.ones(len(ra), dtype=np.float64)
    else:
        weights = np.ones(len(ra), dtype=np.float64)

    sidx = _sort_index_3d(ra, dec, dist, sbound=sbound, mxh1=mxh1, mxh2=mxh2, mxh3=mxh3, pxorder=config.grid.pxorder)
    ra, dec, dist, weights = ra[sidx], dec[sidx], dist[sidx], weights[sidx]
    if region_id is not None:
        region_id = np.asarray(region_id, dtype=np.int32)[sidx]
    sk, ll = _build_skll3d(mxh1, mxh2, mxh3, ra, dec, dist, np.asarray(sbound, dtype=np.float64), np.asarray(pi_edges, dtype=np.float64))
    x, y, z = radec2xyz(np.deg2rad(ra), np.deg2rad(dec))
    return PreparedProjectedSample(
        table=None,
        ra=ra,
        dec=dec,
        dist=dist,
        weights=np.asarray(weights, dtype=np.float32),
        x=np.asarray(x, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        z=np.asarray(z, dtype=np.float64),
        sk=np.asarray(sk),
        ll=np.asarray(ll),
        wunit=bool(np.allclose(weights, 1.0)),
        sbound=tuple(float(v) for v in sbound),
        mxh1=int(mxh1),
        mxh2=int(mxh2),
        mxh3=int(mxh3),
        region_id=None if region_id is None else np.asarray(region_id, dtype=np.int32),
        grid_meta={} if grid_meta is None else dict(grid_meta),
        nrows=int(len(ra)),
    )


def _grid_for_sample(table, columns, config, *, sbound):
    """
    Grid for sample.
    
    Parameters
    ----------
    table : object
        Value for ``table``.
    columns : object
        Value for ``columns``.
    config : object
        Value for ``config``.
    sbound : object
        Value for ``sbound``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ra = _col(table, columns.ra)
    if config.grid.autogrid:
        return best_skgrid_3d_legacy(len(ra), ra, sbound=sbound, nsepv=config.binning.nsepv, dsepv=config.binning.dsepv, dens=config.grid.dens)[:3]
    return int(config.grid.mxh1), int(config.grid.mxh2), int(config.grid.mxh3)


def prepare_projected_auto(data, random, config: ProjectedAutoConfig):
    """
    Prepare projected auto.
    
    Parameters
    ----------
    data : object
        Value for ``data``.
    random : object
        Value for ``random``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    rp_edges, (rp_edges2, rp_centers, rp_delta) = makebins(config.binning.nsepp, config.binning.seppmin, config.binning.dsepp, config.binning.logsepp)
    pi_edges, (pi_edges2, pi_centers, pi_delta) = makebins(config.binning.nsepv, 0.0, config.binning.dsepv, False)
    data_dist = _distance_array(data, config.columns_data, config)
    rand_dist = _distance_array(random, config.columns_random, config)
    sbound = bound3d([_col(data, config.columns_data.dec), _col(random, config.columns_random.dec)], [data_dist, rand_dist])
    d_mxh1, d_mxh2, d_mxh3 = _grid_for_sample(data, config.columns_data, config, sbound=sbound)
    r_mxh1, r_mxh2, r_mxh3 = _grid_for_sample(random, config.columns_random, config, sbound=sbound)
    data_region = random_region = None
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if config.jackknife.enabled:
        dreg = config.columns_data.region
        rreg = config.columns_random.region
        if dreg is not None and rreg is not None:
            data_region, random_region = _shared_user_region_ids([(data, dreg), (random, rreg)])
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1)})
        elif dreg is not None or rreg is not None:
            raise ValueError("For projected jackknife auto-correlations, either both data/random region columns must be supplied or neither.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsepp)
            assignments, _centers = _auto_region_ids([
                (data, config.columns_data.ra, config.columns_data.dec, "data"),
                (random, config.columns_random.ra, config.columns_random.dec, "random"),
            ], nregions=min(int(nregions), max(1, catalog_nrows(data), catalog_nrows(random))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
            data_region, random_region = assignments
            jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(np.concatenate([data_region, random_region])) + 1), "jk_geometry_from": config.jackknife.geometry_from})
    data_p = _prepare_sample(data, config.columns_data, config, sbound=sbound, mxh1=d_mxh1, mxh2=d_mxh2, mxh3=d_mxh3, pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=data_region, grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder, "nsepv": config.binning.nsepv, "dsepv": config.binning.dsepv})
    rand_p = _prepare_sample(random, config.columns_random, config, sbound=sbound, mxh1=r_mxh1, mxh2=r_mxh2, mxh3=r_mxh3, pi_edges=pi_edges, use_weights=False, region_id=random_region, grid_meta={"autogrid": bool(config.grid.autogrid), "dens": config.grid.dens, "pxorder": config.grid.pxorder, "nsepv": config.binning.nsepv, "dsepv": config.binning.dsepv})
    meta = {
        "rp_edges": np.asarray(rp_edges, dtype=np.float64),
        "rp_centers": np.asarray(rp_centers, dtype=np.float64),
        "rp_delta": np.asarray(rp_delta, dtype=np.float64),
        "pi_edges": np.asarray(pi_edges, dtype=np.float64),
        "pi_centers": np.asarray(pi_centers, dtype=np.float64),
        "pi_delta": np.asarray(pi_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
    }
    meta.update(jk_meta)
    return data_p, rand_p, meta


def prepare_projected_cross(data1, random1, data2, random2, config: ProjectedCrossConfig):
    """
    Prepare projected cross.
    
    Parameters
    ----------
    data1 : object
        Value for ``data1``.
    random1 : object
        Value for ``random1``.
    data2 : object
        Value for ``data2``.
    random2 : object
        Value for ``random2``.
    config : object
        Value for ``config``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    rp_edges, (rp_edges2, rp_centers, rp_delta) = makebins(config.binning.nsepp, config.binning.seppmin, config.binning.dsepp, config.binning.logsepp)
    pi_edges, (pi_edges2, pi_centers, pi_delta) = makebins(config.binning.nsepv, 0.0, config.binning.dsepv, False)
    d1_dist = _distance_array(data1, config.columns_data1, config)
    r1_dist = _distance_array(random1, config.columns_random1, config) if random1 is not None else np.empty(0, dtype=np.float64)
    d2_dist = _distance_array(data2, config.columns_data2, config)
    r2_dist = _distance_array(random2, config.columns_random2, config) if random2 is not None else np.empty(0, dtype=np.float64)
    dec_arrays = [_col(data1, config.columns_data1.dec), _col(data2, config.columns_data2.dec)]
    dist_arrays = [d1_dist, d2_dist]
    if random1 is not None:
        dec_arrays.append(_col(random1, config.columns_random1.dec)); dist_arrays.append(r1_dist)
    if random2 is not None:
        dec_arrays.append(_col(random2, config.columns_random2.dec)); dist_arrays.append(r2_dist)
    sbound = bound3d(dec_arrays, dist_arrays)
    d1_grid = _grid_for_sample(data1, config.columns_data1, config, sbound=sbound)
    r1_grid = _grid_for_sample(random1, config.columns_random1, config, sbound=sbound) if random1 is not None else None
    d2_grid = _grid_for_sample(data2, config.columns_data2, config, sbound=sbound)
    r2_grid = _grid_for_sample(random2, config.columns_random2, config, sbound=sbound) if random2 is not None else None
    reg1 = reg_r1 = reg2 = reg_r2 = None
    jk_meta = {"jk_enabled": bool(config.jackknife.enabled), "jk_region_source": None, "jk_nregions": None, "jk_geometry_from": None}
    if config.jackknife.enabled:
        pairs = [(data1, config.columns_data1.region), (data2, config.columns_data2.region)]
        if random1 is not None:
            pairs.append((random1, config.columns_random1.region))
        if random2 is not None:
            pairs.append((random2, config.columns_random2.region))
        provided = [col is not None for _tab, col in pairs]
        if all(provided):
            regions = _shared_user_region_ids(pairs)
            reg1, reg2 = regions[0], regions[1]
            idx = 2
            if random1 is not None:
                reg_r1 = regions[idx]; idx += 1
            if random2 is not None:
                reg_r2 = regions[idx]
            all_regs = np.concatenate(regions) if regions else np.empty(0, dtype=np.int32)
            jk_meta.update({"jk_region_source": "user", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0})
        elif any(provided):
            raise ValueError("For projected jackknife cross-correlations, region columns must be supplied for all participating catalogs or for none of them.")
        else:
            nregions = config.jackknife.nregions
            if nregions is None:
                nregions = choose_default_nregions(config.binning.nsepp)
            catalogs = [
                (data1, config.columns_data1.ra, config.columns_data1.dec, "data"),
                (data2, config.columns_data2.ra, config.columns_data2.dec, "data"),
            ]
            if random1 is not None:
                catalogs.append((random1, config.columns_random1.ra, config.columns_random1.dec, "random"))
            if random2 is not None:
                catalogs.append((random2, config.columns_random2.ra, config.columns_random2.dec, "random"))
            assignments, _centers = _auto_region_ids(catalogs, nregions=min(int(nregions), max(1, sum(catalog_nrows(cat[0]) for cat in catalogs))), seed=config.jackknife.seed, geometry_from=config.jackknife.geometry_from)
            reg1, reg2 = assignments[0], assignments[1]
            idx = 2
            if random1 is not None:
                reg_r1 = assignments[idx]; idx += 1
            if random2 is not None:
                reg_r2 = assignments[idx]
            all_regs = np.concatenate(assignments) if assignments else np.empty(0, dtype=np.int32)
            jk_meta.update({"jk_region_source": "auto", "jk_nregions": int(np.max(all_regs) + 1) if all_regs.size else 0, "jk_geometry_from": config.jackknife.geometry_from})
    prep1 = _prepare_sample(data1, config.columns_data1, config, sbound=sbound, mxh1=d1_grid[0], mxh2=d1_grid[1], mxh3=d1_grid[2], pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=reg1)
    prep_r1 = None if random1 is None else _prepare_sample(random1, config.columns_random1, config, sbound=sbound, mxh1=r1_grid[0], mxh2=r1_grid[1], mxh3=r1_grid[2], pi_edges=pi_edges, use_weights=False, region_id=reg_r1)
    prep2 = _prepare_sample(data2, config.columns_data2, config, sbound=sbound, mxh1=d2_grid[0], mxh2=d2_grid[1], mxh3=d2_grid[2], pi_edges=pi_edges, use_weights=config.weights.weight_mode != "unweighted", region_id=reg2)
    prep_r2 = None if random2 is None else _prepare_sample(random2, config.columns_random2, config, sbound=sbound, mxh1=r2_grid[0], mxh2=r2_grid[1], mxh3=r2_grid[2], pi_edges=pi_edges, use_weights=False, region_id=reg_r2)
    meta = {
        "rp_edges": np.asarray(rp_edges, dtype=np.float64),
        "rp_centers": np.asarray(rp_centers, dtype=np.float64),
        "rp_delta": np.asarray(rp_delta, dtype=np.float64),
        "pi_edges": np.asarray(pi_edges, dtype=np.float64),
        "pi_centers": np.asarray(pi_centers, dtype=np.float64),
        "pi_delta": np.asarray(pi_delta, dtype=np.float64),
        "sbound": tuple(float(v) for v in sbound),
    }
    meta.update(jk_meta)
    return prep1, prep_r1, prep2, prep_r2, meta
