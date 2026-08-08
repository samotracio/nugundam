import numpy as np
from types import SimpleNamespace

from nugundam.projected.models import ProjectedAutoConfig, ProjectedPdfSourceSpec, PDFSourceSpec
from nugundam.result_meta import attach_roundtrip_context, provenance_dict


def _obj():
    return SimpleNamespace(metadata={})


def test_store_config_compact_summarizes_large_pdf_matrix():
    pdfs = np.ones((10, 12), dtype=np.float64)
    cfg = ProjectedAutoConfig(
        store_config="compact",
        pdf_source=ProjectedPdfSourceSpec(
            enabled=True,
            z_grid=np.linspace(0.0, 1.0, 13),
            pdf_data=PDFSourceSpec(kind="external_matrix", matrix=pdfs),
        ),
    )
    out = attach_roundtrip_context(_obj(), config=cfg, provenance=provenance_dict("pcf"))
    snap = out.metadata["config"]
    matrix_meta = snap["pdf_source"]["pdf_data"]["matrix"]
    assert out.metadata["config_store"] == "compact"
    assert matrix_meta["__kind__"] == "ndarray_summary"
    assert matrix_meta["shape"] == [10, 12]
    assert matrix_meta["size"] == 120
    assert snap["estimator"] == "NAT"


def test_store_config_full_keeps_full_matrix_payload():
    pdfs = np.arange(24, dtype=np.float64).reshape(4, 6)
    cfg = ProjectedAutoConfig(
        store_config="full",
        pdf_source=ProjectedPdfSourceSpec(
            enabled=True,
            pdf_data=PDFSourceSpec(kind="external_matrix", matrix=pdfs),
        ),
    )
    out = attach_roundtrip_context(_obj(), config=cfg, provenance=provenance_dict("pcf"))
    stored = out.metadata["config"]["pdf_source"]["pdf_data"]["matrix"]
    assert out.metadata["config_store"] == "full"
    assert isinstance(stored, np.ndarray)
    np.testing.assert_array_equal(stored, pdfs)


def test_store_config_none_omits_config_key():
    cfg = ProjectedAutoConfig(store_config="none")
    out = attach_roundtrip_context(_obj(), config=cfg, provenance=provenance_dict("pcf"))
    assert out.metadata["config_store"] == "none"
    assert "config" not in out.metadata
    assert out.metadata["provenance"]["run_kind"] == "pcf"
