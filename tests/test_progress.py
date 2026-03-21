from pathlib import Path

import numpy as np

from nugundam.angular import api as angular_api
from nugundam.angular.models import AngularAutoConfig, AngularAutoCounts, PreparedAngularSample
from nugundam.projected import api as projected_api
from nugundam.projected.models import ProjectedAutoConfig, ProjectedAutoCounts, PreparedProjectedSample
from nugundam.core import progress as progress_mod
from nugundam.core.progress import progress_context, run_with_progress


def _prepared():
    return PreparedAngularSample(
        table=[0, 1],
        ra=np.array([0.0, 1.0]),
        dec=np.array([0.0, 1.0]),
        weights=np.ones(2),
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([1.0, 1.0]),
        sk=np.array([0, 0]),
        ll=np.array([0, 1]),
        wunit=True,
        sbound=(0.0, 1.0, 0.0, 1.0),
        mxh1=2,
        mxh2=2,
    )


def test_progress_context_tempfile_cleanup():
    with progress_context(True, None, 0.01) as progress_path:
        path = Path(progress_path)
        assert path.exists()
        path.write_text('[DD] stripe 1/3\n', encoding='utf-8')
    assert not path.exists()


def test_run_with_progress_cli_uses_stdout_path(monkeypatch):
    monkeypatch.setattr(progress_mod, 'in_notebook', lambda: False)
    seen = []

    def _target(progress_path):
        seen.append(progress_path)
        return 12

    out = run_with_progress(True, None, 0.01, _target)
    assert out == 12
    assert seen == [None]


def test_run_with_progress_notebook_uses_temp_file(monkeypatch):
    monkeypatch.setattr(progress_mod, 'in_notebook', lambda: True)
    monkeypatch.setattr(progress_mod, '_prefer_process_backend', lambda: False)

    class DummyHandle:
        def __init__(self):
            self.updates = []

        def update(self, obj):
            self.updates.append(obj)

    handle = DummyHandle()

    def _display(*args, **kwargs):
        return handle

    monkeypatch.setattr('IPython.display.display', _display)
    monkeypatch.setattr('IPython.display.HTML', lambda s: s)
    seen = []

    def _target(progress_path):
        seen.append(progress_path)
        assert progress_path is not None
        Path(progress_path).write_text('====  Counting DD pairs in 2 DEC strips  ====\n[DD] stripe 1/2\n', encoding='utf-8')
        return 21

    out = run_with_progress(True, None, 0.01, _target)
    assert out == 21
    assert len(seen) == 1
    assert seen[0] is not None


def test_acf_passes_progress_file(monkeypatch, theta_grid, tmp_path):
    edges, centers = theta_grid
    progress_path = tmp_path / 'run.progress'
    captured = {}

    monkeypatch.setattr(
        angular_api,
        'prepare_angular_auto',
        lambda d, r, c: (_prepared(), _prepared(), {'theta_edges': edges, 'theta_centers': centers}),
    )
    monkeypatch.setattr(progress_mod, 'in_notebook', lambda: False)

    def _build(*args, **kwargs):
        captured['progress_file'] = kwargs.get('progress_file')
        return AngularAutoCounts(
            theta_edges=edges,
            theta_centers=centers,
            dd=np.array([1.0, 1.0]),
            rr=np.array([1.0, 1.0]),
            dr=np.array([1.0, 1.0]),
            metadata={'n_data': 3, 'n_random': 3},
        )

    monkeypatch.setattr(angular_api, 'build_auto_counts', _build)
    cfg = AngularAutoConfig()
    cfg.progress.progress_file = str(progress_path)
    angular_api.acf(object(), object(), cfg)
    assert captured['progress_file'] == str(progress_path)



def test_notebook_status_emitter_updates_single_status(monkeypatch):
    class DummyHandle:
        def __init__(self):
            self.updates = []

        def update(self, obj):
            self.updates.append(obj)

    handle = DummyHandle()

    def _display(*args, **kwargs):
        return handle

    monkeypatch.setattr("IPython.display.display", _display)
    monkeypatch.setattr("IPython.display.HTML", lambda s: s)

    emitter = progress_mod._NotebookStatusEmitter(min_update_interval=0.0)
    emitter.emit("====  Counting DD                               pairs in 34 DEC strips  ====\n")
    emitter.emit("[DD                              ] stripe 7/34\n")
    rendered_dd = handle.updates[-1]
    assert "DD 7/34" in rendered_dd
    assert "DD                   7/34" not in rendered_dd
    emitter.emit("====  Counting RR split 1/6 pairs in 52 DEC strips  ====\n")
    emitter.emit("[RR split 1/6] stripe 52/52  particles 98125-100000\n")
    emitter.close()

    assert handle.updates
    rendered = handle.updates[-1]
    assert "RR[split 1/6] 52/52" in rendered
    assert "particles 98125-100000" in rendered
    assert "Counting DD pairs" not in rendered



def _prepared_projected():
    return PreparedProjectedSample(
        table=[0, 1],
        ra=np.array([0.0, 1.0]),
        dec=np.array([0.0, 1.0]),
        dist=np.array([100.0, 101.0]),
        weights=np.ones(2, dtype=np.float32),
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([1.0, 1.0]),
        sk=np.zeros((1, 1, 1), dtype=int),
        ll=np.array([0, 1]),
        wunit=True,
        sbound=(0.0, 360.0, -1.0, 1.0, 99.0, 102.0),
        mxh1=1,
        mxh2=1,
        mxh3=1,
    )


def test_pcf_passes_progress_file(monkeypatch, rp_pi_grid, tmp_path):
    rp_edges, rp_centers, pi_edges, pi_centers = rp_pi_grid
    progress_path = tmp_path / 'proj.progress'
    captured = {}

    monkeypatch.setattr(
        projected_api,
        'prepare_projected_auto',
        lambda d, r, c: (_prepared_projected(), _prepared_projected(), {'rp_edges': rp_edges, 'rp_centers': rp_centers, 'pi_edges': pi_edges, 'pi_centers': pi_centers, 'pi_delta': np.array([1.0, 1.0])}),
    )
    monkeypatch.setattr(progress_mod, 'in_notebook', lambda: False)

    def _build(*args, **kwargs):
        captured['progress_file'] = kwargs.get('progress_file')
        return ProjectedAutoCounts(
            rp_edges=rp_edges,
            rp_centers=rp_centers,
            pi_edges=pi_edges,
            pi_centers=pi_centers,
            dd=np.array([[1.0, 1.0], [1.0, 1.0]]),
            rr=np.array([[1.0, 1.0], [1.0, 1.0]]),
            dr=np.array([[1.0, 1.0], [1.0, 1.0]]),
            metadata={'n_data': 3, 'n_random': 3},
        )

    monkeypatch.setattr(projected_api, 'build_auto_counts', _build)
    cfg = ProjectedAutoConfig()
    cfg.progress.progress_file = str(progress_path)
    projected_api.pcf(object(), object(), cfg)
    assert captured['progress_file'] == str(progress_path)
