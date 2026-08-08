"""Small helpers for inspecting projected pair-count diagnostics."""
from __future__ import annotations

from collections.abc import Mapping


def _fmt(x, ndigits=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{ndigits}g}"
    except Exception:
        return str(x)


def _metadata_from(result_or_counts):
    # Prefer count metadata when a full correlation result is supplied, because
    # pair timings and detailed counters live on result.counts.metadata.
    if hasattr(result_or_counts, "counts"):
        counts_meta = getattr(getattr(result_or_counts, "counts"), "metadata", None)
        if isinstance(counts_meta, Mapping):
            return counts_meta
    meta = getattr(result_or_counts, "metadata", None)
    return meta if isinstance(meta, Mapping) else None


def summarize_pair_diagnostics(result_or_counts, *, include_split: bool = True) -> str:
    """Return a compact text summary of pair timings and 16Quant diagnostics.

    Detailed pruning counters are collected only when ``cfg.pdf.diagnostics`` is
    enabled. Production runs still store lightweight wall times and qchi
    preparation metadata.
    """
    meta = _metadata_from(result_or_counts)
    if not isinstance(meta, Mapping):
        return "No metadata diagnostics found."

    lines = []
    if meta.get('pdf_bootstrap_backend') is not None:
        lines.append(f"pdf_bootstrap_backend: {meta.get('pdf_bootstrap_backend')}")
    if meta.get('jk_touch_fast') is not None:
        lines.append(f"jk_touch_fast: {meta.get('jk_touch_fast')}")
    if meta.get('jk_touch_available') is not None:
        lines.append(f"jk_touch_available: {meta.get('jk_touch_available')}")
    if meta.get('qchi_diagnostics_enabled') is not None:
        lines.append(f"qchi_diagnostics_enabled: {meta.get('qchi_diagnostics_enabled')}")

    pdf_repr = meta.get("pdf_repr") or meta.get("pdf_repr1") or "unknown"
    if meta.get("pdf_nquant") is not None:
        lines.append(f"PDF mode: {pdf_repr}, Nq={meta.get('pdf_nquant')}")
    else:
        lines.append(f"PDF mode: {pdf_repr}")

    times = meta.get("pair_times_s") or {}
    if times:
        total = sum(float(v) for v in times.values() if v is not None)
        pieces = ", ".join(f"{k}={_fmt(v,4)}s" for k, v in times.items())
        lines.append(f"Pair times: {pieces}; total={_fmt(total,4)}s")

    diags = meta.get("pair_diagnostics") or {}
    if diags:
        for label, d in diags.items():
            if not isinstance(d, Mapping):
                continue
            lines.append(
                f"{label}: support reject={_fmt(100*d.get('support_pair_reject_fraction_after_rv',0.0),3)}%, "
                f"rp support={_fmt(100*d.get('rp_support_reject_fraction_after_rv',0.0),3)}%, "
                f"pi support={_fmt(100*d.get('pi_support_reject_fraction_after_rv',0.0),3)}%, "
                f"q accept={_fmt(100*d.get('quantile_accept_fraction',0.0),3)}%, "
                f"entered={d.get('quantile_products_entered','-')}, accepted={d.get('quantile_products_accepted','-')}"
            )
    elif meta.get("pdf_repr") == "quantile_chi" or meta.get("qchi_diagnostics_enabled") is False:
        lines.append("Detailed qchi pair counters were not collected. Set cfg.pdf.diagnostics = True to enable them.")

    if include_split and meta.get("split_random_enabled"):
        lines.append(
            f"Split random: nchunks={meta.get('split_random_nchunks')}, "
            f"RR total={_fmt(meta.get('split_random_rr_total_time_s'),4)}s, "
            f"chunk sizes={meta.get('split_random_chunk_sizes')}"
        )
        chunks = meta.get("split_random_rr_diagnostics") or []
        if chunks:
            entered = sum(int(c.get("quantile_products_entered", 0)) for c in chunks if isinstance(c, Mapping))
            accepted = sum(int(c.get("quantile_products_accepted", 0)) for c in chunks if isinstance(c, Mapping))
            if entered:
                lines.append(f"Split RR aggregate: q accept={_fmt(100*accepted/entered,3)}%, entered={entered}, accepted={accepted}")
        elif meta.get("qchi_diagnostics_enabled") is False:
            lines.append("Split RR detailed counters were not collected.")

    prep_keys = [k for k in meta.keys() if str(k).startswith("qchi_prepare")]
    for k in sorted(prep_keys):
        v = meta.get(k)
        if isinstance(v, Mapping):
            lines.append(
                f"{k}: Nq={v.get('qchi_nquant','-')}, nlib={v.get('qchi_nlib','-')}, "
                f"span median={_fmt(v.get('qchi_span_median'),4)}, "
                f"bytes={v.get('qchi_library_nbytes','-')}, "
                f"compress={_fmt(v.get('qchi_prepare_compress_time_s'),4)}s"
            )
    return "\n".join(lines)


def print_pair_diagnostics(result_or_counts, *, include_split: bool = True) -> None:
    """Print :func:`summarize_pair_diagnostics`."""
    print(summarize_pair_diagnostics(result_or_counts, include_split=include_split))
