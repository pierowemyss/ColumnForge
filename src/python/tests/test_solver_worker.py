"""Month-3 regression: solves run on a QThread worker with live progress and a
real Abort (B10). Offscreen Qt; the event loop is spun manually."""
import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


def _spin_until(pred, timeout=15.0):
    t0 = time.monotonic()
    while not pred():
        app.processEvents()
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("worker did not finish in time")
        time.sleep(0.005)


def _configured_window():
    from gui.main_window import MainWindow
    from gui.state.window_state import Stream, StreamType
    from core.dof import SpecKind
    from core import component_db

    w = MainWindow()
    ws = w.window_state
    ws.pressure = 1.01325
    ws.num_stages = 20
    ws.light_key_index = 0
    ws.heavy_key_index = 1
    for name in ("benzene", "toluene", "p-xylene"):
        component_db.load_into(ws, name)
    ws.streams.clear()
    ws.add_stream(Stream(id="Feed", stream_type=StreamType.FEED, stage=10,
                         flow=100.0,
                         composition={"benzene": 0.4, "toluene": 0.35,
                                      "p-xylene": 0.25}))
    ws.specs = []
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 3.0)
    ws.upsert_operating_spec(SpecKind.DISTILLATE_RATE, 40.0)
    return w, ws


def test_threaded_run_updates_results_and_progress():
    w, ws = _configured_window()
    ws.results = None
    w.run_simulation()
    assert w._solver_thread.isRunning() or ws.results is not None
    _spin_until(lambda: ws.results is not None)
    _spin_until(lambda: not w._solver_thread.isRunning())

    assert ws.results["found"], ws.results.get("message")
    assert w.results_tab.data_table.rowCount() == 20
    assert w.sim_tab.progress_bar.value() == 100
    assert int(w.sim_tab.iter_label.text()) >= 1
    assert w.sim_tab.run_btn.isEnabled()          # set_running(False) happened
    # a second run while idle is allowed and completes too
    ws.results = None
    w.run_simulation()
    _spin_until(lambda: ws.results is not None)


def test_abort_cancels_running_job():
    w, _ = _configured_window()
    stalled = {"n": 0}

    def slow_job(report, cancel):
        while not cancel():                        # runs until Abort
            stalled["n"] += 1
            report(stalled["n"], 100, 1.0)         # (done, total, residual)
            time.sleep(0.002)
        return {"found": False, "message": "Aborted.", "n_stages": 20,
                "feed_stage": 10, "T": [], "x": [[]]}

    w._start_solver(slow_job)
    _spin_until(lambda: stalled["n"] > 5)          # job is genuinely running
    assert w._solver_thread.isRunning()
    assert not w.sim_tab.run_btn.isEnabled()       # UI is in running state
    w.abort_simulation()
    _spin_until(lambda: not w._solver_thread.isRunning())
    app.processEvents()
    assert w.sim_tab.run_btn.isEnabled()


def test_recovery_spec_run_converges_and_moves_the_bar():
    """An LK-recovery spec makes the run resolve (R, D) by re-solving the column
    inside a root-find. That used to hang the GUI; then it ran but reported
    nothing, so the bar looped 0-100% per trial solve and the iteration count sat
    at zero. The bar must climb, once, and the run must actually converge."""
    from core.dof import SpecKind

    w, ws = _configured_window()
    ws.specs = []
    ws.upsert_operating_spec(SpecKind.REFLUX_RATIO, 3.0)
    ws.upsert_operating_spec(SpecKind.LK_RECOVERY, 0.98)
    seen = []
    w._solver_worker = None
    ws.results = None
    w.run_simulation()
    w._solver_worker.progress.connect(lambda d, t, r: seen.append((d, t)))
    _spin_until(lambda: ws.results is not None, timeout=60.0)
    _spin_until(lambda: not w._solver_thread.isRunning())

    assert ws.results["found"], ws.results.get("message")
    # the recovery target was actually hit: rec = D xD_lk / (F z_lk)
    rec = ws.results["D"] * ws.results["xD"][0] / (100.0 * 0.4)
    assert abs(rec - 0.98) < 2e-3, rec
    assert seen, "no progress reported during the resolve"
    pcts = [min(99, int(100 * d / max(1, t))) for d, t in seen]
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), "progress bar went backwards"
    assert max(pcts) > 0 and w.sim_tab.progress_bar.value() == 100


def test_closing_mid_solve_does_not_crash():
    """closeEvent used to leave the worker running into teardown: the thread's
    next progress emit hit a deleted C++ object and took the process down."""
    w, ws = _configured_window()
    ws.is_modified = False                         # no save prompt
    w.run_simulation()
    assert w._solver_thread.isRunning() or ws.results is not None
    w.close()                                      # joins the solver thread
    assert not w._solver_thread.isRunning()
    app.processEvents()


def test_payload_keeps_its_key_order_across_the_signal():
    """`finished` used to be Signal(dict), which marshals through QVariantMap --
    a SORTED map, recursively. A BVM design's `profiles` came out of the thread
    as extractive/rectifying/stripping, and the ternary plot reads those keys
    top->bottom, so the rectifying stages were drawn labelled "extractive"."""
    from gui.solver_worker import SolverWorker

    sent = {"profiles": {"rectifying": 1, "extractive": 2, "stripping": 3},
            "zz": 0, "aa": 1}
    got = []
    worker = SolverWorker(lambda report, cancel: sent)
    worker.finished.connect(got.append)
    worker.run()
    assert got and got[0] is sent, "payload was copied, not passed through"
    assert list(got[0]) == list(sent)
    assert list(got[0]["profiles"]) == ["rectifying", "extractive", "stripping"]


def test_failed_job_reports_without_crashing(monkeypatch):
    import gui.main_window as mw
    w, _ = _configured_window()
    seen = {}
    monkeypatch.setattr(mw.QMessageBox, "warning",
                        lambda *a, **k: seen.setdefault("msg", a[2]))

    def bad_job(report, cancel):
        raise ValueError("deliberately infeasible")

    w._start_solver(bad_job)
    _spin_until(lambda: "msg" in seen)
    _spin_until(lambda: not w._solver_thread.isRunning())
    assert "infeasible" in seen["msg"]
    assert w.sim_tab.run_btn.isEnabled()


if __name__ == "__main__":
    test_threaded_run_updates_results_and_progress()
    test_abort_cancels_running_job()
    print("solver worker OK (run under pytest for the failure case)")
