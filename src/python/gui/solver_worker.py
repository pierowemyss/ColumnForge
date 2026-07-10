"""Background solver execution on a QThread (roadmap Month 3, closes B10).

SolverWorker wraps one solve job — a callable (report, cancel) -> profile —
and turns it into Qt signals the GUI thread consumes:

    progress(iteration, residual)   from the solver's report hook
    finished(profile)               profile dict, includes aborted runs
    failed(message, traceback, user_error)   user_error: ValueError (bad
                                    config) vs an unexpected solver bug
"""
import traceback

from PySide6.QtCore import QObject, Signal


class SolverWorker(QObject):
    progress = Signal(int, float)
    finished = Signal(dict)
    failed = Signal(str, str, bool)

    def __init__(self, job):
        super().__init__()
        self._job = job
        self._cancelled = False

    def cancel(self):
        """Thread-safe: a bare bool flip read by the solver's cancel hook."""
        self._cancelled = True

    def run(self):
        try:
            profile = self._job(report=self.progress.emit,
                                cancel=lambda: self._cancelled)
        except ValueError as exc:
            self.failed.emit(str(exc), traceback.format_exc(), True)
        except Exception as exc:  # solver bugs must not kill the thread silently
            self.failed.emit(f"{type(exc).__name__}: {exc}",
                             traceback.format_exc(), False)
        else:
            self.finished.emit(profile)
