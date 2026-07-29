"""Background solver execution on a QThread (roadmap Month 3, closes B10).

SolverWorker wraps one solve job — a callable (report, cancel) -> profile —
and turns it into Qt signals the GUI thread consumes:

    progress(done, total, residual) from the job's report hook — `done`/`total`
                                    are the job's own units of work, so a run
                                    whose operating specs need a root-find can
                                    still show one monotonic sweep.
    finished(profile)               profile dict, includes aborted runs; passed
                                    as `object` so key order survives (see below)
    failed(message, traceback, user_error)   user_error: ValueError (bad
                                    config) vs an unexpected solver bug
"""
import traceback

from PySide6.QtCore import QObject, Signal


class SolverWorker(QObject):
    progress = Signal(int, int, float)
    # Signal(object), NOT Signal(dict): a dict argument is marshalled through
    # QVariantMap, which is a sorted map, so every dict in the payload arrives
    # with its keys in ALPHABETICAL order, recursively. That silently reordered
    # a BVM design's `profiles` to extractive/rectifying/stripping and the ternary
    # plot -- which reads them top->bottom -- then labelled the rectifying stages
    # "extractive" and vice versa. `object` hands the payload over untouched.
    finished = Signal(object)
    failed = Signal(str, str, bool)

    def __init__(self, job):
        super().__init__()
        self._job = job
        self._cancelled = False

    def cancel(self):
        """Thread-safe: a bare bool flip read by the solver's cancel hook."""
        self._cancelled = True

    def _emit(self, sig, *args):
        """Emit unless the window went away mid-solve: closing the app deletes
        the C++ half of this worker while run() is still on the stack, and a
        raw emit then raises RuntimeError — out of the except clause too, which
        aborts the process. Nobody is listening at that point; drop it."""
        try:
            sig.emit(*args)
        except RuntimeError:
            pass

    def run(self):
        try:
            profile = self._job(report=lambda *a: self._emit(self.progress, *a),
                                cancel=lambda: self._cancelled)
        except ValueError as exc:
            self._emit(self.failed, str(exc), traceback.format_exc(), True)
        except Exception as exc:  # solver bugs must not kill the thread silently
            self._emit(self.failed, f"{type(exc).__name__}: {exc}",
                       traceback.format_exc(), False)
        else:
            self._emit(self.finished, profile)
