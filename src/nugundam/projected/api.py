"""High-level projected counting and correlation entry points."""
from __future__ import annotations

import numpy as np

from ..core.jackknife import jackknife_cov, validate_resampling_choice
from ..core.progress import run_with_progress
from ..result_meta import attach_roundtrip_context, provenance_dict
from .estimators import estimate_auto, estimate_cross, compute_auto_xi2d, compute_cross_xi2d
from .fortran_bridge import build_auto_count_result, build_auto_counts, build_cross_count_result, build_cross_counts
from .models import (
    ProjectedAutoConfig,
    ProjectedAutoCountsConfig,
    ProjectedCrossConfig,
    ProjectedCrossCountsConfig,
    ProjectedAutoCounts,
    ProjectedCrossCounts,
)
from .prepare import prepare_projected_auto, prepare_projected_cross, subset_prepared_projected_sample


def _stage(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _normalize_auto_counts_config(config: ProjectedAutoCountsConfig) -> ProjectedAutoCountsConfig:
    if not isinstance(config, ProjectedAutoCountsConfig):
        raise TypeError("proj_auto_counts expects ProjectedAutoCountsConfig")
    validate_resampling_choice(config.bootstrap, config.jackknife)
    if config.jackknife.enabled:
        raise NotImplementedError("Jackknife is currently available for pcf/pccf, not for count-only proj_auto_counts.")
    return config


def _normalize_cross_counts_config(config: ProjectedCrossCountsConfig) -> ProjectedCrossCountsConfig:
    if not isinstance(config, ProjectedCrossCountsConfig):
        raise TypeError("proj_cross_counts expects ProjectedCrossCountsConfig")
    validate_resampling_choice(config.bootstrap, config.jackknife)
    if config.jackknife.enabled:
        raise NotImplementedError("Jackknife is currently available for pcf/pccf, not for count-only proj_cross_counts.")
    return config




def _validate_split_random_auto(run_name: str, config: ProjectedAutoConfig) -> None:
    spec = getattr(config, "split_random", None)
    if spec is None or not bool(spec.enabled):
        return
    est = str(config.estimator).strip().upper()
    if est != "LS":
        raise NotImplementedError(f"{run_name} currently supports split_random only for auto estimator='LS'.")
    if config.jackknife.enabled:
        raise NotImplementedError(f"{run_name} does not yet support split_random together with jackknife.")
    mode = str(spec.mode).strip().lower()
    if mode not in {"match_data", "nchunks", "chunk_size"}:
        raise ValueError(f"Unsupported split_random.mode={spec.mode!r}.")
    if mode == "nchunks" and (spec.nchunks is None or int(spec.nchunks) <= 0):
        raise ValueError("split_random.nchunks must be a positive integer when mode='nchunks'.")
    if mode == "chunk_size" and (spec.chunk_size is None or int(spec.chunk_size) <= 0):
        raise ValueError("split_random.chunk_size must be a positive integer when mode='chunk_size'.")

def _required_cross_randoms(estimator: str, primary: str) -> tuple[bool, bool]:
    est = str(estimator).strip().upper()
    prim = str(primary).strip().lower()
    if est in {"NAT", "LS"}:
        return True, True
    if est == "DP":
        if prim == "data1":
            return False, True
        if prim == "data2":
            return True, False
        raise ValueError(f"Unsupported cross-correlation primary={primary!r}; use 'data1' or 'data2'.")
    raise ValueError(f"Unsupported projected cross estimator={estimator!r}.")


def _validate_cross_randoms(run_name: str, estimator: str, primary: str, random1, random2) -> None:
    need_r1, need_r2 = _required_cross_randoms(estimator, primary)
    missing = []
    if need_r1 and random1 is None:
        missing.append("random1")
    if need_r2 and random2 is None:
        missing.append("random2")
    if missing:
        raise ValueError(f"{run_name} with estimator={estimator!r} and primary={primary!r} requires {', '.join(missing)}.")


def _auto_counts_to_prepare_config(config: ProjectedAutoCountsConfig) -> ProjectedAutoConfig:
    return ProjectedAutoConfig(
        estimator="NAT",
        columns_data=config.columns,
        columns_random=config.columns,
        binning=config.binning,
        grid=config.grid,
        distance=config.distance,
        weights=config.weights,
        bootstrap=config.bootstrap,
        jackknife=config.jackknife,
        progress=config.progress,
        nthreads=config.nthreads,
        description=config.description,
    )


def _cross_counts_to_prepare_config(config: ProjectedCrossCountsConfig) -> ProjectedCrossConfig:
    return ProjectedCrossConfig(
        estimator="DP",
        columns_data1=config.columns1,
        columns_random1=config.columns1,
        columns_data2=config.columns2,
        columns_random2=config.columns2,
        binning=config.binning,
        grid=config.grid,
        distance=config.distance,
        weights=config.weights,
        bootstrap=config.bootstrap,
        jackknife=config.jackknife,
        progress=config.progress,
        nthreads=config.nthreads,
        description=config.description,
    )


def proj_auto_counts(data, config: ProjectedAutoCountsConfig):
    count_config = _normalize_auto_counts_config(config)
    prep_config = _auto_counts_to_prepare_config(count_config)
    _stage(count_config.progress.enabled, "[proj_auto_counts] preparing data")
    data_p, _, meta = prepare_projected_auto(data, data, prep_config)
    _stage(count_config.progress.enabled, "[proj_auto_counts] counting DD(rp,pi)")
    counts = run_with_progress(
        count_config.progress.enabled,
        count_config.progress.progress_file,
        count_config.progress.poll_interval,
        lambda progress_path: build_auto_count_result(
            data_p,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=count_config.nthreads,
            weight_mode=count_config.weights.weight_mode,
            doboot=count_config.bootstrap.enabled,
            dojk=False,
            nreg=0,
            nbts=count_config.bootstrap.nbts,
            bseed=count_config.bootstrap.bseed,
            progress_file=progress_path,
        ),
    )
    return attach_roundtrip_context(counts, config=count_config, provenance=provenance_dict("proj_auto_counts"), extra_metadata=meta)


def proj_cross_counts(data1, data2, config: ProjectedCrossCountsConfig):
    count_config = _normalize_cross_counts_config(config)
    prep_config = _cross_counts_to_prepare_config(count_config)
    _stage(count_config.progress.enabled, "[proj_cross_counts] preparing data")
    prep1, _, prep2, _, meta = prepare_projected_cross(data1, data1, data2, data2, prep_config)
    _stage(count_config.progress.enabled, "[proj_cross_counts] counting D1D2(rp,pi)")
    counts = run_with_progress(
        count_config.progress.enabled,
        count_config.progress.progress_file,
        count_config.progress.poll_interval,
        lambda progress_path: build_cross_count_result(
            prep1,
            prep2,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=count_config.nthreads,
            weight_mode=count_config.weights.weight_mode,
            doboot=count_config.bootstrap.enabled,
            dojk=False,
            nreg=0,
            nbts=count_config.bootstrap.nbts,
            bseed=count_config.bootstrap.bseed,
            primary=count_config.bootstrap.primary,
            progress_file=progress_path,
        ),
    )
    return attach_roundtrip_context(counts, config=count_config, provenance=provenance_dict("proj_cross_counts"), extra_metadata=meta)


def _jackknife_realizations_auto_rerun(data_p, rand_p, config, meta):
    nregions = int(meta.get("jk_nregions") or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_p.wunit)
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        data_sub = subset_prepared_projected_sample(data_p, data_p.region_id != k, pi_edges=meta["pi_edges"])
        rand_sub = subset_prepared_projected_sample(rand_p, rand_p.region_id != k, pi_edges=meta["pi_edges"])
        counts_k = build_auto_counts(data_sub, rand_sub, rp_edges=meta["rp_edges"], rp_centers=meta["rp_centers"], pi_edges=meta["pi_edges"], pi_centers=meta["pi_centers"], pi_delta=meta["pi_delta"], nthreads=config.nthreads, estimator=config.estimator, weight_mode=config.weights.weight_mode, doboot=False, dojk=False, nreg=0, nbts=0, bseed=config.bootstrap.bseed, progress_file=None)
        result_k = estimate_auto(counts_k, estimator=config.estimator, data_weights=(data_sub.weights if weighted else None))
        out[k] = result_k.wp
    return out


def _auto_touch_ready(counts: ProjectedAutoCounts, estimator: str) -> bool:
    est = str(estimator).upper()
    if counts.dd_jk_touch is None:
        return False
    if est == "NAT":
        return counts.rr_jk_touch is not None
    if est == "DP":
        return counts.dr_jk_touch is not None
    if est == "LS":
        return counts.rr_jk_touch is not None and counts.dr_jk_touch is not None
    return False


def _jackknife_realizations_auto_touch(counts: ProjectedAutoCounts, data_p, rand_p, config, meta):
    nregions = int(meta.get("jk_nregions") or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    data_reg = np.asarray(data_p.region_id, dtype=np.int64)
    rand_reg = np.asarray(rand_p.region_id, dtype=np.int64)
    ndata_reg = np.bincount(data_reg, minlength=nregions)
    nrand_reg = np.bincount(rand_reg, minlength=nregions)
    ndata = int(len(data_reg))
    nrand = int(len(rand_reg))
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_p.wunit)
    if weighted:
        w = np.asarray(data_p.weights, dtype=np.float64)
        sumw = float(np.sum(w)); sumw2 = float(np.sum(np.square(w)))
        sumw_reg = np.bincount(data_reg, weights=w, minlength=nregions).astype(np.float64)
        sumw2_reg = np.bincount(data_reg, weights=np.square(w), minlength=nregions).astype(np.float64)
    else:
        sumw = sumw2 = None
        sumw_reg = sumw2_reg = None
    pi_delta = np.asarray(meta["pi_delta"], dtype=np.float64)
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        counts_k = ProjectedAutoCounts(
            rp_edges=counts.rp_edges,
            rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges,
            pi_centers=counts.pi_centers,
            dd=counts.dd - counts.dd_jk_touch[:, :, k],
            rr=None if counts.rr is None else counts.rr - counts.rr_jk_touch[:, :, k],
            dr=None if counts.dr is None else counts.dr - counts.dr_jk_touch[:, :, k],
            metadata={"n_data": ndata - int(ndata_reg[k]), "n_random": nrand - int(nrand_reg[k]), "data_weighted": weighted},
        )
        xi2d = compute_auto_xi2d(counts_k, estimator=config.estimator, sum_w_data=None if not weighted else (sumw - float(sumw_reg[k])), sum_w2_data=None if not weighted else (sumw2 - float(sumw2_reg[k])))
        out[k] = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    return out


def _jackknife_realizations_auto(data_p, rand_p, counts, config, meta):
    if _auto_touch_ready(counts, config.estimator):
        return _jackknife_realizations_auto_touch(counts, data_p, rand_p, config, meta)
    return _jackknife_realizations_auto_rerun(data_p, rand_p, config, meta)


def _jackknife_realizations_cross_rerun(prep1, prep_r1, prep2, prep_r2, config, meta):
    nregions = int(meta.get("jk_nregions") or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not prep1.wunit or not prep2.wunit))
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        d1 = subset_prepared_projected_sample(prep1, prep1.region_id != k, pi_edges=meta["pi_edges"])
        d2 = subset_prepared_projected_sample(prep2, prep2.region_id != k, pi_edges=meta["pi_edges"])
        r1 = None if prep_r1 is None else subset_prepared_projected_sample(prep_r1, prep_r1.region_id != k, pi_edges=meta["pi_edges"])
        r2 = None if prep_r2 is None else subset_prepared_projected_sample(prep_r2, prep_r2.region_id != k, pi_edges=meta["pi_edges"])
        counts_k = build_cross_counts(d1, r1, d2, r2, rp_edges=meta["rp_edges"], rp_centers=meta["rp_centers"], pi_edges=meta["pi_edges"], pi_centers=meta["pi_centers"], pi_delta=meta["pi_delta"], nthreads=config.nthreads, estimator=config.estimator, weight_mode=config.weights.weight_mode, doboot=False, dojk=False, nreg=0, nbts=0, bseed=config.bootstrap.bseed, primary=config.bootstrap.primary, progress_file=None)
        result_k = estimate_cross(counts_k, estimator=config.estimator, sum_w1=(float(d1.weights.sum()) if weighted else None), sum_w2=(float(d2.weights.sum()) if weighted else None))
        out[k] = result_k.wp
    return out


def _cross_touch_ready(counts: ProjectedCrossCounts, estimator: str) -> bool:
    est = str(estimator).upper()
    if counts.d1d2_jk_touch is None:
        return False
    if est == "NAT":
        return counts.r1r2_jk_touch is not None
    if est == "DP":
        return counts.d1r2_jk_touch is not None
    if est == "LS":
        return counts.d1r2_jk_touch is not None and counts.r1d2_jk_touch is not None and counts.r1r2_jk_touch is not None
    return False


def _jackknife_realizations_cross_touch(counts: ProjectedCrossCounts, prep1, prep_r1, prep2, prep_r2, config, meta):
    nregions = int(meta.get("jk_nregions") or 0)
    if nregions <= 1:
        return np.zeros((0, len(meta["rp_centers"])), dtype=np.float64)
    d1reg = np.asarray(prep1.region_id, dtype=np.int64)
    d2reg = np.asarray(prep2.region_id, dtype=np.int64)
    d1n = np.bincount(d1reg, minlength=nregions)
    d2n = np.bincount(d2reg, minlength=nregions)
    r1n = np.zeros(nregions, dtype=np.int64) if prep_r1 is None else np.bincount(np.asarray(prep_r1.region_id, dtype=np.int64), minlength=nregions)
    r2n = np.zeros(nregions, dtype=np.int64) if prep_r2 is None else np.bincount(np.asarray(prep_r2.region_id, dtype=np.int64), minlength=nregions)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not prep1.wunit or not prep2.wunit))
    if weighted:
        w1 = np.asarray(prep1.weights, dtype=np.float64)
        w2 = np.asarray(prep2.weights, dtype=np.float64)
        sw1 = float(np.sum(w1)); sw2 = float(np.sum(w2))
        sw1_reg = np.bincount(d1reg, weights=w1, minlength=nregions).astype(np.float64)
        sw2_reg = np.bincount(d2reg, weights=w2, minlength=nregions).astype(np.float64)
    else:
        sw1 = sw2 = None
        sw1_reg = sw2_reg = None
    pi_delta = np.asarray(meta["pi_delta"], dtype=np.float64)
    out = np.zeros((nregions, len(meta["rp_centers"])), dtype=np.float64)
    for k in range(nregions):
        counts_k = ProjectedCrossCounts(
            rp_edges=counts.rp_edges,
            rp_centers=counts.rp_centers,
            pi_edges=counts.pi_edges,
            pi_centers=counts.pi_centers,
            d1d2=counts.d1d2 - counts.d1d2_jk_touch[:, :, k],
            d1r2=None if counts.d1r2 is None else counts.d1r2 - counts.d1r2_jk_touch[:, :, k],
            r1d2=None if counts.r1d2 is None else counts.r1d2 - counts.r1d2_jk_touch[:, :, k],
            r1r2=None if counts.r1r2 is None else counts.r1r2 - counts.r1r2_jk_touch[:, :, k],
            metadata={"n_data1": int(prep1.nrows) - int(d1n[k]), "n_random1": (0 if prep_r1 is None else int(prep_r1.nrows) - int(r1n[k])), "n_data2": int(prep2.nrows) - int(d2n[k]), "n_random2": (0 if prep_r2 is None else int(prep_r2.nrows) - int(r2n[k])), "primary": counts.metadata.get("primary", config.bootstrap.primary)},
        )
        xi2d = compute_cross_xi2d(counts_k, estimator=config.estimator, sum_w1=None if not weighted else (sw1 - float(sw1_reg[k])), sum_w2=None if not weighted else (sw2 - float(sw2_reg[k])))
        out[k] = 2.0 * np.sum(xi2d * pi_delta[None, :], axis=1)
    return out


def _jackknife_realizations_cross(prep1, prep_r1, prep2, prep_r2, counts, config, meta):
    if _cross_touch_ready(counts, config.estimator):
        return _jackknife_realizations_cross_touch(counts, prep1, prep_r1, prep2, prep_r2, config, meta)
    return _jackknife_realizations_cross_rerun(prep1, prep_r1, prep2, prep_r2, config, meta)


def pcf(data, random, config: ProjectedAutoConfig):
    """Measure the projected auto-correlation function ``w_p(r_p)``.

    Parameters
    ----------
    data : table-like
        Input data catalog. The required columns are described by
        ``config.columns_data``.
    random : table-like
        Random catalog sampling the same survey selection. The required columns
        are described by ``config.columns_random``.
    config : ProjectedAutoConfig
        Full run configuration. When ``config.split_random.enabled`` is True,
        nuGUNDAM keeps the full ``DR`` count, evaluates ``RR`` as the sum of
        within-chunk counts over shuffled random subcatalogs, and uses the
        exact included RR-pair normalization stored in the count metadata.
        Split-random mode currently supports only ``estimator="LS"`` and is
        not available together with jackknife resampling.

    Returns
    -------
    ProjectedCorrelationResult
        Final projected correlation result containing ``wp``, uncertainties,
        the underlying count terms, and run metadata. In split-random mode the
        attached counts metadata also records the split chunk sizes and the RR
        pair normalization actually used by the estimator.
    """
    validate_resampling_choice(config.bootstrap, config.jackknife)
    _validate_split_random_auto("pcf", config)
    _stage(config.progress.enabled, "[pcf] preparing data and randoms")
    data_p, rand_p, meta = prepare_projected_auto(data, random, config)
    _stage(config.progress.enabled, f"[pcf] counting DD / RR / DR with estimator={config.estimator}")
    counts = run_with_progress(
        config.progress.enabled,
        config.progress.progress_file,
        config.progress.poll_interval,
        lambda progress_path: build_auto_counts(
            data_p,
            rand_p,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=config.nthreads,
            estimator=config.estimator,
            weight_mode=config.weights.weight_mode,
            doboot=(config.bootstrap.enabled and not config.jackknife.enabled),
            dojk=config.jackknife.enabled,
            nreg=int(meta.get("jk_nregions") or 0),
            nbts=config.bootstrap.nbts,
            bseed=config.bootstrap.bseed,
            progress_file=progress_path,
            split_random=config.split_random,
        ),
    )
    counts = attach_roundtrip_context(counts, config=config, provenance=provenance_dict("pcf"), extra_metadata=meta)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and not data_p.wunit)
    _stage(config.progress.enabled, "[pcf] estimating correlation")
    result = estimate_auto(counts, estimator=config.estimator, data_weights=(data_p.weights if weighted else None))
    if config.jackknife.enabled:
        _stage(config.progress.enabled, "[pcf] assembling jackknife covariance")
        realizations = _jackknife_realizations_auto(data_p, rand_p, counts, config, meta)
        cov = jackknife_cov(realizations)
        result.wp_err = np.sqrt(np.diag(cov))
        result.cov = cov if config.jackknife.return_cov else None
        result.realizations = realizations if config.jackknife.return_realizations else None
        result.metadata.update({"jackknife": True, "jk_nregions": int(realizations.shape[0]), "jk_region_source": meta.get("jk_region_source"), "jk_touch_fast": bool(counts.metadata.get("jk_touch_available", False))})
    _stage(config.progress.enabled, "[pcf] done")
    return attach_roundtrip_context(result, config=config, provenance=provenance_dict("pcf"), extra_metadata=meta)


def pccf(data1, data2, config: ProjectedCrossConfig, *, random1=None, random2=None):
    validate_resampling_choice(config.bootstrap, config.jackknife)
    _validate_cross_randoms("pccf", config.estimator, config.bootstrap.primary, random1, random2)
    _stage(config.progress.enabled, "[pccf] preparing data and randoms")
    prep1, prep_r1, prep2, prep_r2, meta = prepare_projected_cross(data1, random1, data2, random2, config)
    _stage(config.progress.enabled, f"[pccf] counting cross terms with estimator={config.estimator}")
    counts = run_with_progress(
        config.progress.enabled,
        config.progress.progress_file,
        config.progress.poll_interval,
        lambda progress_path: build_cross_counts(
            prep1,
            prep_r1,
            prep2,
            prep_r2,
            rp_edges=meta["rp_edges"],
            rp_centers=meta["rp_centers"],
            pi_edges=meta["pi_edges"],
            pi_centers=meta["pi_centers"],
            pi_delta=meta["pi_delta"],
            nthreads=config.nthreads,
            estimator=config.estimator,
            weight_mode=config.weights.weight_mode,
            doboot=(config.bootstrap.enabled and not config.jackknife.enabled),
            dojk=config.jackknife.enabled,
            nreg=int(meta.get("jk_nregions") or 0),
            nbts=config.bootstrap.nbts,
            bseed=config.bootstrap.bseed,
            primary=config.bootstrap.primary,
            progress_file=progress_path,
        ),
    )
    counts = attach_roundtrip_context(counts, config=config, provenance=provenance_dict("pccf"), extra_metadata=meta)
    weighted = config.weights.weight_mode == "weighted" or (config.weights.weight_mode == "auto" and (not prep1.wunit or not prep2.wunit))
    _stage(config.progress.enabled, "[pccf] estimating correlation")
    result = estimate_cross(counts, estimator=config.estimator, sum_w1=(float(prep1.weights.sum()) if weighted else None), sum_w2=(float(prep2.weights.sum()) if weighted else None))
    if config.jackknife.enabled:
        _stage(config.progress.enabled, "[pccf] assembling jackknife covariance")
        realizations = _jackknife_realizations_cross(prep1, prep_r1, prep2, prep_r2, counts, config, meta)
        cov = jackknife_cov(realizations)
        result.wp_err = np.sqrt(np.diag(cov))
        result.cov = cov if config.jackknife.return_cov else None
        result.realizations = realizations if config.jackknife.return_realizations else None
        result.metadata.update({"jackknife": True, "jk_nregions": int(realizations.shape[0]), "jk_region_source": meta.get("jk_region_source"), "jk_touch_fast": bool(counts.metadata.get("jk_touch_available", False))})
    _stage(config.progress.enabled, "[pccf] done")
    return attach_roundtrip_context(result, config=config, provenance=provenance_dict("pccf"), extra_metadata=meta)
