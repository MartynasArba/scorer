"""
Background tasks for the GUI. Preprocessing, scoring and file conversion used to run in the Qt
event loop, so the window froze ("Not responding") for minutes and a failure inside a slot could
abort the whole application. Now:

    self._task = run_task(self, fn, *args, on_done=..., on_error=..., on_progress=...)

`fn` runs in a QThread and may accept `progress` (callable(str)) and `cancelled` (callable() -> bool)
keyword arguments; results and exceptions come back to the GUI thread through signals.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import traceback

from PyQt5.QtCore import QObject, QThread, pyqtSignal


class _Worker(QObject):
    """QObject that runs one function in a worker thread and reports through signals.

    Signals:
        done(object): The function's return value.
        failed(str): Error type, message and traceback.
        progress(str): Progress lines from the function.
    """
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, fn, args, kwargs):
        """Store the function and its arguments."""
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self._cancel = False

    def cancel(self) -> None:
        """Ask the function to stop; it sees this through its ``cancelled()`` callback."""
        self._cancel = True

    def run(self) -> None:
        """Call the function (adding ``progress``/``cancelled`` if it accepts them) and emit the outcome."""
        try:
            params = inspect.signature(self.fn).parameters
            kw = dict(self.kwargs)
            if "progress" in params:
                kw["progress"] = self.progress.emit
            if "cancelled" in params:
                kw["cancelled"] = lambda: self._cancel
            self.done.emit(self.fn(*self.args, **kw))
        except Exception as e:  # reported to the GUI thread, never raised into Qt
            self.failed.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")


class Task:
    """Handle for a running background function, returned by ``run_task``."""

    def __init__(self, thread: QThread, worker: _Worker):
        """Keep the thread and worker so the task can be cancelled and polled."""
        self.thread, self.worker = thread, worker

    def cancel(self) -> None:
        """Ask the function to stop at its next ``cancelled()`` check."""
        self.worker.cancel()

    @property
    def running(self) -> bool:
        """True while the thread runs; False once it has finished (even after Qt deleted it)."""
        try:
            return self.thread.isRunning()
        except RuntimeError:  # the QThread was already deleted after finishing
            return False


def run_task(parent, fn, *args, on_done=None, on_error=None, on_progress=None, on_finished=None, **kwargs) -> Task:
    """Run ``fn(*args, **kwargs)`` in a new QThread and return a ``Task``.

    If ``fn`` accepts ``progress`` or ``cancelled`` keyword arguments, it gets a callable
    that sends a line to ``on_progress`` and a callable that returns True after
    ``Task.cancel()``. All callbacks run in the GUI thread.

    Args:
        parent: QObject that owns the thread.
        fn: Function to run; it must not touch widgets.
        *args, **kwargs: Passed to ``fn``.
        on_done: Called with the return value.
        on_error: Called with the error text (an exception never reaches Qt).
        on_progress: Called with each progress line.
        on_finished: Called when the thread has stopped, after success or failure.
    """
    thread = QThread(parent)
    worker = _Worker(fn, args, kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_done:
        worker.done.connect(on_done)
    if on_error:
        worker.failed.connect(on_error)
    if on_progress:
        worker.progress.connect(on_progress)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    if on_finished:
        thread.finished.connect(on_finished)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    task = Task(thread, worker)
    thread.start()
    return task


class _Lines(io.TextIOBase):
    """File-like object that sends each complete printed line to a callback (blank lines skipped)."""

    def __init__(self, emit):
        """Start with an empty buffer."""
        self.emit, self.buf = emit, ""

    def write(self, text):
        """Buffer ``text`` and emit every complete line; returns the number of characters taken."""
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.emit(line)
        return len(text)


def with_prints_to(fn):
    """Wrap ``fn`` so that its ``print()`` output goes to the task's progress log.

    The data modules report progress by printing. ``redirect_stdout`` is process-wide,
    so prints from other threads during the call are shown as well.

    Returns:
        ``job(progress=None, cancelled=None)``, ready for ``run_task``.
    """
    params = inspect.signature(fn).parameters

    def job(progress=None, cancelled=None):
        """Call ``fn`` with stdout redirected to ``progress``."""
        kw = {}
        if "progress" in params:
            kw["progress"] = progress
        if "cancelled" in params:
            kw["cancelled"] = cancelled
        with contextlib.redirect_stdout(_Lines(progress or (lambda s: None))):
            return fn(**kw)
    return job
