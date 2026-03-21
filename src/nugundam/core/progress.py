"""Progress-reporting backends for terminal and notebook executions."""
from __future__ import annotations

import html
import multiprocessing as mp
import os
import queue
import re
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")

_HEADER_RE = re.compile(r"=+\s+Counting\s+(.+?)\s+pairs\s+in\s+(\d+)\s+DEC\s+strips\s+=+")
_STRIPE_RE = re.compile(r"\[(.+?)\]\s+stripe\s+(\d+)/(\d+)(?:\s+(.*))?$")


def _display_phase_label(phase: str) -> str:
    """Return a compact, human-friendly label for notebook progress badges.

    Parameters
    ----------
    phase : str
        Raw counter label emitted by the Fortran kernels. Labels arrive through
        a fixed-width character buffer, so short values such as ``"DD"`` may
        include trailing blanks. Split-random labels use the form
        ``"RR split i/n"``.

    Returns
    -------
    str
        Trimmed label for display. Split-random labels are rendered as
        ``"RR[split i/n]"`` while ordinary labels such as ``"DD"`` and
        ``"DR"`` are returned without trailing blanks.
    """
    phase = phase.strip()
    m = re.match(r"^([A-Za-z0-9]+)\s+split\s+(\d+/\d+)$", phase)
    if m is None:
        return phase
    return f"{m.group(1)}[split {m.group(2)}]"


def in_notebook() -> bool:
    """
    In notebook.
    
    Returns
    -------
    bool
        Object returned by this helper.
    """
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return shell.__class__.__name__ == "ZMQInteractiveShell"


def _prefer_process_backend() -> bool:
    """
    Prefer process backend.
    
    Returns
    -------
    bool
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    if not in_notebook():
        return False
    if os.name != "posix":
        return False
    try:
        return "fork" in mp.get_all_start_methods()
    except Exception:
        return False


class _StreamEmitter:
    """
    Internal helper class ``_StreamEmitter``.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    def emit(self, line: str) -> None:
        """
        Emit.
        
        Parameters
        ----------
        line : object
            Value for ``line``.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        text = line.rstrip()
        if text:
            print(text, flush=True)

    def close(self) -> None:
        """
        Close.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        return None


class _NotebookStatusEmitter:
    """
    Internal helper class ``_NotebookStatusEmitter``.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    def __init__(self, *, min_update_interval: float = 0.20):
        """
        Initialize the object.
        
        Parameters
        ----------
        min_update_interval : object, optional
            Value for ``min_update_interval``. This argument is keyword-only.
        
        Returns
        -------
        None
            Initializes the instance in place.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        self._handle = None
        self._phase: str | None = None
        self._current = 0
        self._total = 0
        self._status = "waiting"
        self._dirty = False
        self._last_render = 0.0
        self._min_update_interval = float(min_update_interval)
        try:
            from IPython.display import HTML, display  # type: ignore
        except Exception:
            self._fallback = _StreamEmitter()
            return
        self._fallback = None
        self._HTML = HTML
        self._handle = display(self._HTML(self._render()), display_id=True)

    def _render(self) -> str:
        """
        Render.
        
        Returns
        -------
        object
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        text = html.escape(self._status)
        return (
            '<div style="font-family:monospace; display:inline-block; '
            'min-width:14ch; padding:0.28em 0.6em; border:1px solid #8884; '
            'border-radius:6px; white-space:pre">'
            f"{text}"
            "</div>"
        )

    def _flush(self, *, force: bool = False) -> None:
        """
        Flush.
        
        Parameters
        ----------
        force : object, optional
            Value for ``force``. This argument is keyword-only.
        
        Returns
        -------
        None
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        if self._fallback is not None:
            return
        if self._handle is None or not self._dirty:
            return
        now = time.monotonic()
        if (not force) and (now - self._last_render < self._min_update_interval):
            return
        self._handle.update(self._HTML(self._render()))
        self._last_render = now
        self._dirty = False

    def _set_status(self, status: str) -> None:
        """
        Set status.
        
        Parameters
        ----------
        status : object
            Value for ``status``.
        
        Returns
        -------
        None
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        if status == self._status:
            return
        self._status = status
        self._dirty = True

    def emit(self, line: str) -> None:
        """
        Emit.
        
        Parameters
        ----------
        line : object
            Value for ``line``.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        text = line.rstrip()
        if not text:
            return
        if self._fallback is not None:
            self._fallback.emit(text)
            return
        mh = _HEADER_RE.match(text)
        if mh is not None:
            self._phase = _display_phase_label(mh.group(1))
            self._current = 0
            self._total = int(mh.group(2))
            self._set_status(f"{self._phase} 0/{self._total}")
            self._flush(force=True)
            return
        ms = _STRIPE_RE.match(text)
        if ms is not None:
            self._phase = _display_phase_label(ms.group(1))
            self._current = int(ms.group(2))
            self._total = int(ms.group(3))
            extra = (ms.group(4) or "").strip()
            status = f"{self._phase} {self._current}/{self._total}"
            if extra:
                status = f"{status}  {extra}"
            self._set_status(status)
            self._flush(force=self._current >= self._total)
            return

    def close(self) -> None:
        """
        Close.
        
        Returns
        -------
        None
            Object returned by this helper.
        """
        if self._fallback is not None:
            self._fallback.close()
            return
        self._flush(force=True)


class _WorkerResult:
    """
    Internal helper class ``_WorkerResult``.
    
    Notes
    -----
    Internal helper documented for completeness.
    """
    def __init__(self):
        """
        Initialize the object.
        
        Returns
        -------
        None
            Initializes the instance in place.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        self.value = None
        self.error: BaseException | None = None


def _emit_pending(fh, emitter) -> None:
    """
    Emit pending.
    
    Parameters
    ----------
    fh : object
        Value for ``fh``.
    emitter : object
        Value for ``emitter``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    while True:
        line = fh.readline()
        if not line:
            break
        emitter.emit(line)


def _process_entry(target: Callable[[str | None], T], progress_path: str, out_queue) -> None:
    """
    Process entry.
    
    Parameters
    ----------
    target : object
        Value for ``target``.
    progress_path : object
        Value for ``progress_path``.
    out_queue : object
        Value for ``out_queue``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    try:
        value = target(progress_path)
    except BaseException:
        out_queue.put(("error", traceback.format_exc()))
        return
    out_queue.put(("ok", value))


def _run_with_thread(progress_path: str, poll_interval: float, target: Callable[[str | None], T], *, notebook: bool) -> T:
    """
    Run with thread.
    
    Parameters
    ----------
    progress_path : object
        Value for ``progress_path``.
    poll_interval : object
        Value for ``poll_interval``.
    target : object
        Value for ``target``.
    notebook : object
        Value for ``notebook``. This argument is keyword-only.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    result = _WorkerResult()

    def _worker() -> None:
        """
        Worker.
        
        Returns
        -------
        object
            Object returned by this helper.
        
        Notes
        -----
        Internal helper used by the refactored nuGUNDAM package.
        """
        try:
            result.value = target(progress_path)
        except BaseException as exc:  # pragma: no cover - re-raised in main thread
            result.error = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    emitter = _NotebookStatusEmitter() if notebook else _StreamEmitter()
    try:
        with Path(progress_path).open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_SET)
            while worker.is_alive():
                _emit_pending(fh, emitter)
                time.sleep(poll_interval)
            _emit_pending(fh, emitter)
    finally:
        worker.join()
        emitter.close()

    if result.error is not None:
        raise result.error
    return result.value  # type: ignore[return-value]


def _run_with_process(progress_path: str, poll_interval: float, target: Callable[[str | None], T]) -> T:
    """
    Run with process.
    
    Parameters
    ----------
    progress_path : object
        Value for ``progress_path``.
    poll_interval : object
        Value for ``poll_interval``.
    target : object
        Value for ``target``.
    
    Returns
    -------
    object
        Object returned by this helper.
    
    Notes
    -----
    Internal helper used by the refactored nuGUNDAM package.
    """
    ctx = mp.get_context("fork")
    out_queue = ctx.Queue()
    proc = ctx.Process(target=_process_entry, args=(target, progress_path, out_queue), daemon=True)
    proc.start()
    emitter = _NotebookStatusEmitter()
    payload = None
    try:
        with Path(progress_path).open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(0, os.SEEK_SET)
            while proc.is_alive():
                _emit_pending(fh, emitter)
                try:
                    payload = out_queue.get_nowait()
                    break
                except queue.Empty:
                    time.sleep(poll_interval)
            _emit_pending(fh, emitter)
        if payload is None:
            try:
                payload = out_queue.get(timeout=max(1.0, 10.0 * poll_interval))
            except queue.Empty:
                payload = None
    finally:
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.join()
        emitter.close()

    if payload is None:
        raise RuntimeError("Notebook progress worker finished without returning a result.")
    status, value = payload
    if status != "ok":
        raise RuntimeError(f"Notebook progress worker failed:\n{value}")
    return value  # type: ignore[return-value]


@contextmanager
def progress_context(enabled: bool, progress_file: str | None = None, poll_interval: float = 0.15) -> Iterator[str | None]:
    """
    Progress context.
    
    Parameters
    ----------
    enabled : object
        Value for ``enabled``.
    progress_file : object, optional
        Value for ``progress_file``.
    poll_interval : object, optional
        Value for ``poll_interval``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    del poll_interval
    if not enabled:
        yield None
        return

    temp_path: Path | None = None
    if progress_file is None:
        fd, tmp = tempfile.mkstemp(prefix="gundam_progress_", suffix=".log")
        os.close(fd)
        temp_path = Path(tmp)
        path = temp_path
    else:
        path = Path(progress_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    path.write_text("", encoding="utf-8")
    try:
        yield str(path)
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def run_with_progress(enabled: bool, progress_file: str | None, poll_interval: float, target: Callable[[str | None], T]) -> T:
    """
    Run with progress.
    
    Parameters
    ----------
    enabled : object
        Value for ``enabled``.
    progress_file : object
        Value for ``progress_file``.
    poll_interval : object
        Value for ``poll_interval``.
    target : object
        Value for ``target``.
    
    Returns
    -------
    object
        Object returned by this helper.
    """
    if not enabled:
        return target(None)

    notebook = in_notebook()

    if not notebook and progress_file is None:
        return target(None)

    if notebook or progress_file is not None:
        with progress_context(True, progress_file, poll_interval) as progress_path:
            assert progress_path is not None
            if notebook and _prefer_process_backend():
                return _run_with_process(progress_path, poll_interval, target)
            return _run_with_thread(progress_path, poll_interval, target, notebook=notebook)

    return target(None)
