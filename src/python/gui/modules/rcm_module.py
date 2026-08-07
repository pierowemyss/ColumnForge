"""RCM side module -- residue curve maps for a ternary.

A residue curve is the still-pot composition path of a simple batch
distillation: dx/dxi = x - y(x). The map of them is the standard first look at
an azeotropic system -- it shows the distillation regions, where the boundaries
run, and which products a column could ever reach, before any column exists.

Pick three components from the list (shift-click; the order sets the triangle's
corners), then either auto-generate a fan of curves or click anywhere inside the
simplex to trace the one curve through that point.

Thermo comes from the shared window_state through `module_thermo.session_models`
-- the same vapour-pressure correlation, activity model and EOS every other
panel uses. That was the point of the rewrite: the predecessor module carried
its own hard-coded Ideal/NRTL/NRTL-SRK ladder, so changing the app's
thermodynamics had no effect on the map it drew.

The curves themselves come from the compiled solver (`core.rcm` ->
`side_features/rcm/RCM_solv.c`), which forwards the app's model
parameters into `src/native/nifco2.f90`. `gui.plotting.residue_curve` computes
the same thing in NumPy about 50x slower and is what an engine switch would
select; everything downstream of the composition arrays is shared already.
"""

import numpy as np
from core import rcm
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..panels.sci_spin_box import SciDoubleSpinBox
from ..plotting import (
    CompactNavigationToolbar,
    composition_from_click,
    plot_residue_curves,
    singular_points,
    ternary_axes,
)
from ..theme import palette
from .module_thermo import session_models

#: 1 atm in bar. The old module defaulted to a flat 1 bar, which is not a
#: pressure anybody actually runs at.
DEFAULT_P_BAR = 1.01325


def seed_points(n):
    """`n` interior seed compositions for an auto-generated map.

    Seeds decide what the map looks like, because every curve passes through
    its own seed. A subsampled corner-to-corner grid leaves the middle of the
    triangle bare — and the middle is where `plot_residue_curves` puts its
    direction arrows, so the arrows end up crowded against the edges.

    These sit on the three transversals of constant x_i = 1/3, each of which
    runs edge to edge *through the centroid*. That covers the middle by
    construction and spreads in all three directions at once. Points closer
    together than `tol` are dropped: the three lines cross at the centroid, and
    three copies of the same curve is just three times the work.
    """
    seeds = []
    for axis in range(3):
        for t in np.linspace(0.27, 0.49, n):
            x = np.array([t, 1.5 - 3 * t, -0.5 + 2 * t])
            seeds.append(x)
    return seeds[: int(n)]


class OrderedComponentList(QListWidget):
    """Component picker where the *order* of selection is the answer.

    Plain click starts a fresh selection at position 1; shift-click appends the
    next position; shift-clicking a picked component removes it and renumbers
    the rest. Qt's own selection is switched off entirely (NoSelection) because
    the contract is that the picked components are the only highlighted ones --
    an ExtendedSelection list would also paint a current-item focus rectangle
    and would not preserve click order in the first place.
    """

    orderChanged = Signal(object)  # object, not list: see solver_worker.py

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QListWidget.NoSelection)
        self._picked = []  # component names, in click order

    def set_components(self, names):
        """Repopulate, keeping the picked components *by name* -- species get
        renamed and reordered upstream, and an index-based memory would then
        silently point at a different component."""
        names = list(names)
        self._picked = [n for n in self._picked if n in names]
        self.clear()
        for n in names:
            item = QListWidgetItem(n)
            # The true name lives in a data role, never in the text: the text
            # grows an order badge, and parsing it back off the label breaks
            # the moment a component is itself named something like "1. foo".
            item.setData(Qt.UserRole, n)
            self.addItem(item)
        self._repaint_items()

    def picked(self):
        return list(self._picked)

    def set_picked(self, names):
        available = {self.item(i).data(Qt.UserRole) for i in range(self.count())}
        self._picked = [n for n in names if n in available]
        self._repaint_items()
        self.orderChanged.emit(self.picked())

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if event.modifiers() & Qt.ShiftModifier:
            if name in self._picked:
                self._picked.remove(name)
            else:
                self._picked.append(name)
        else:
            self._picked = [name]
        self._repaint_items()
        self.orderChanged.emit(self.picked())

    def _repaint_items(self):
        accent = QColor(palette.ACCENT)
        accent.setAlpha(90)
        clear = QBrush(Qt.NoBrush)
        for i in range(self.count()):
            item = self.item(i)
            name = item.data(Qt.UserRole)
            if name in self._picked:
                item.setText(f"{self._picked.index(name) + 1}. {name}")
                item.setBackground(QBrush(accent))
            else:
                item.setText(name)
                item.setBackground(clear)


class _OptionsDialog(QDialog):
    """Small OK/Cancel form. Values land back on the caller only on accept --
    the predecessor's two options windows were write-only, so every box in them
    was silently ignored."""

    def __init__(self, title, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.widgets = {}
        for key, label, widget in rows:
            form.addRow(label, widget)
            self.widgets[key] = widget
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self):
        return {k: w.value() for k, w in self.widgets.items()}


class RCMModuleWidget(QWidget):
    """Component picker + residue curve map for a ternary system."""

    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._curves = []  # list of (n, 3) composition arrays
        self._comps = []  # the 3 names those curves were drawn for
        self._singular = []
        self._thermo_note = ""
        self._status_prefix = ""  # run outcome; the thermo label is appended
        self._restored = False
        self._thread = self._worker = None
        # Solver / plot options, mirrored into window_state.rcm_params.
        self._opts = {
            "n_it": 250,
            "dxi": 0.02,
            "maxiter": 1000,
            "ftol": 1e-12,
            "xtol": 1e-12,
        }
        self._plot_opts = {"linewidth": 1.2, "lines": 15}
        self._setup_ui()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        layout = QHBoxLayout(self)

        left = QWidget()
        left_col = QVBoxLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(360)
        left_scroll.setFrameShape(QScrollArea.NoFrame)

        hint = QLabel("Components (click, then shift-click for 2nd and 3rd):")
        hint.setWordWrap(True)
        left_col.addWidget(hint)
        self.comp_list = OrderedComponentList()
        self.comp_list.setToolTip(
            "Selection order sets the triangle: 1st on the x-axis, 2nd on the\n"
            "y-axis, 3rd at the origin. Shift-click a picked component to drop\n"
            "it; a plain click starts over."
        )
        self.comp_list.orderChanged.connect(self._on_selection_changed)
        left_col.addWidget(self.comp_list, stretch=1)

        self.p_spin = SciDoubleSpinBox()
        self.p_spin.setDecimals(5)
        self.p_spin.setRange(1e-4, 500.0)
        self.p_spin.setSingleStep(0.1)
        self.p_spin.setValue(DEFAULT_P_BAR)
        self.p_spin.setSuffix(" bar")
        p_row = QHBoxLayout()
        p_row.addWidget(QLabel("Pressure:"))
        p_row.addWidget(self.p_spin)
        left_col.addLayout(p_row)

        self.gen_btn = QPushButton("Auto-generate curves")
        self.gen_btn.clicked.connect(self._on_auto_generate)
        left_col.addWidget(self.gen_btn)

        self.clear_btn = QPushButton("Clear plot")
        self.clear_btn.clicked.connect(self._on_clear)
        left_col.addWidget(self.clear_btn)

        self.solver_btn = QPushButton("Solver Options")
        self.solver_btn.clicked.connect(self._on_solver_options)
        left_col.addWidget(self.solver_btn)

        self.plot_btn = QPushButton("Plot Options")
        self.plot_btn.clicked.connect(self._on_plot_options)
        left_col.addWidget(self.plot_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._on_cancel)
        left_col.addWidget(self.cancel_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left_col.addWidget(self.status)

        # The singular-point table lives here rather than under the plot so the
        # whole right-hand side can go to the triangle.
        self.data_table = QTableWidget(0, 3)
        self.data_table.setHorizontalHeaderLabels(
            ["Singular point", "T (degC)", "Type"]
        )
        self.data_table.horizontalHeader().setStretchLastSection(True)
        left_col.addWidget(self.data_table, stretch=1)
        layout.addWidget(left_scroll)

        right = QWidget()
        right_col = QVBoxLayout(right)
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas, stretch=1)
        layout.addWidget(right, stretch=1)

        # One axes for the lifetime of the widget. The predecessor rebuilt the
        # canvas, toolbar and layout on every redraw, leaking a frame per click.
        self.ax = self.figure.add_subplot(111)
        # The simplex is a right triangle with equal legs, so it only looks
        # right at equal aspect; without this it stretches with the window.
        self.ax.set_aspect("equal", adjustable="box")
        self.figure.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)
        self._refresh_species()
        self._redraw()

    # ------------------------------------------------------------- state
    def _species_order(self):
        return self.window_state.get_species_names() if self.window_state else []

    def _refresh_species(self):
        self.comp_list.set_components(self._species_order())
        self._sync_enabled()

    def reload_from_state(self):
        self._refresh_species()
        if not self._restored and self.window_state is not None:
            params = getattr(self.window_state, "rcm_params", None)
            if params:
                self.set_params(dict(params))
            self._restored = True
        self._sync_enabled()

    def showEvent(self, event):
        super().showEvent(event)
        self.reload_from_state()

    def _on_selection_changed(self, _picked):
        self._sync_enabled()
        self._push_params()

    def _sync_enabled(self):
        """Auto-generate is enabled only when it could actually run, and says
        why when it cannot."""
        picked = self.comp_list.picked()
        if not rcm.available():
            reason = (
                f"RCM solver not built. Run `{rcm.BUILD_HINT}`.\n"
                f"{rcm.load_error() or ''}"
            )
        elif len(picked) != 3:
            reason = (
                f"Pick exactly 3 components ({len(picked)} selected) -- a "
                "residue curve map is a ternary diagram."
            )
        else:
            reason = ""
        self.gen_btn.setEnabled(not reason)
        self.gen_btn.setToolTip(reason)
        if reason:
            self.status.setText(reason)
        else:
            # The thermo half is recomputed every time rather than cached with
            # the run: this is also called from showEvent, which is exactly when
            # the user comes back from changing the app's thermodynamics, and a
            # label still naming the old model would be worse than none.
            self.status.setText(
                " ".join(filter(None, (self._status_prefix, self._thermo_label())))
            )

    def _thermo_label(self):
        try:
            _, _, _, label, note = session_models(
                self.window_state, self.comp_list.picked()
            )
        except Exception as exc:
            return f"Thermo unavailable: {exc}"
        return " ".join(filter(None, (label, note)))

    # ------------------------------------------------------------- thermo
    def _gather(self):
        """(comps, P, antoine, gamma_fn, phi_fn) for the picked components, in
        picked order. Raises with a message the status label can show."""
        if self.window_state is None:
            raise ValueError("no case loaded")
        comps = self.comp_list.picked()
        if len(comps) != 3:
            raise ValueError("pick exactly 3 components")

        antoine, gamma_fn, phi_fn, label, note = session_models(
            self.window_state, comps
        )
        self._thermo_note = " ".join(filter(None, (label, note)))
        # session_models hands back parameters in the app's pressure-unit
        # convention, so the pressure has to make the same trip.
        P = self.window_state.thermodynamics_config.pressure_in_psat_unit(
            self.p_spin.value()
        )
        return comps, P, antoine, gamma_fn, phi_fn

    # --------------------------------------------------------------- runs
    def _run_bg(self, label, job, on_done):
        from PySide6.QtCore import QThread

        from ..solver_worker import SolverWorker

        if self._thread is not None:
            return  # ponytail: one RCM run at a time
        self.status.setText(f"{label} ...")
        buttons = (self.gen_btn, self.clear_btn)
        for b in buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._worker = SolverWorker(job)
        self._worker.progress.connect(
            lambda done, total, _r: self.status.setText(
                f"{label} ... {done}/{total} curves"
            )
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(on_done)
        self._worker.failed.connect(
            lambda msg, tb, _user: self._fail(f"{label} failed: {msg}")
        )
        for sig in (self._worker.finished, self._worker.failed):
            sig.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._run_done(buttons))
        self._thread.start()

    def _fail(self, message):
        """Report a failure so it survives the next _sync_enabled -- a run that
        went wrong must not be quietly replaced by the thermo label."""
        self._status_prefix = message
        self.status.setText(message)

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancelling ...")

    def _run_done(self, buttons):
        self._thread = self._worker = None
        self.cancel_btn.setEnabled(False)
        for b in buttons:
            b.setEnabled(True)
        self._sync_enabled()

    def _curve(self, x0, P, antoine, gamma_fn, phi_fn):
        x, _y, _T = rcm.curves(
            x0, P, antoine, gamma_fn=gamma_fn, phi_fn=phi_fn, **self._opts
        )
        # Euler can walk a step outside the simplex at a vertex; drop those
        # rather than drawing a curve that leaves the triangle.
        keep = np.all(x > -1e-9, axis=1) & (np.abs(x.sum(axis=1) - 1.0) < 1e-6)
        return x[keep]

    def _on_auto_generate(self):
        try:
            comps, P, antoine, gamma_fn, phi_fn = self._gather()
        except Exception as exc:
            self._fail(f"Auto-generate failed: {exc}")
            return
        seeds = seed_points(int(self._plot_opts["lines"]))

        def job(report, cancel):
            curves = []
            for i, x0 in enumerate(seeds):
                if cancel():
                    break
                x = self._curve(x0, P, antoine, gamma_fn, phi_fn)
                if len(x) > 1:
                    curves.append(x)
                report(i + 1, len(seeds), 0.0)
            points = singular_points(P, antoine, comps, gamma_fn=gamma_fn)
            return {"comps": comps, "curves": curves, "singular": points}

        self._run_bg("Generating", job, self._on_curves_done)

    def _on_canvas_click(self, event):
        """Click inside the simplex to trace the one curve through that point."""
        if self._thread is not None or self.toolbar.mode:
            return
        x0 = composition_from_click(event)
        if x0 is None:
            return
        try:
            comps, P, antoine, gamma_fn, phi_fn = self._gather()
        except Exception as exc:
            self._fail(f"Curve failed: {exc}")
            return

        def job(report, cancel):
            x = self._curve(x0, P, antoine, gamma_fn, phi_fn)
            return {"comps": comps, "curves": [x] if len(x) > 1 else [], "append": True}

        self._run_bg("Tracing curve", job, self._on_curves_done)

    def _on_curves_done(self, result):
        comps = result["comps"]
        if result.get("append") and comps == self._comps:
            self._curves.extend(result["curves"])
        else:
            self._curves = list(result["curves"])
            self._comps = comps
        if "singular" in result:
            self._singular = result["singular"]
        self._status_prefix = f"{len(self._curves)} curve(s)."
        self._sync_enabled()
        self._redraw()
        self._push_params()

    def _on_clear(self):
        self._curves = []
        self._singular = []
        self._status_prefix = ""
        self._sync_enabled()
        self._redraw()

    # --------------------------------------------------------------- plot
    def _redraw(self):
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")  # clear() drops it
        comps = self._comps or self.comp_list.picked()
        if len(comps) == 3:
            ternary_axes(self.ax, comps)
            # ternary_axes clamps to exactly [0,1], which crops the corner
            # labels it just drew outside that box. Widen rather than trim the
            # figure margins: with equal aspect the triangle stays square.
            self.ax.set_xlim(-0.06, 1.12)
            self.ax.set_ylim(-0.07, 1.08)
        else:
            self.ax.axis("off")
        plot_residue_curves(
            self.ax, self._curves, linewidth=float(self._plot_opts["linewidth"])
        )
        for sp in self._singular:
            x = np.asarray(sp["x"], float)
            # Filled = node (a product the column can reach), open = saddle.
            self.ax.plot(
                x[0],
                x[1],
                "o",
                markersize=5,
                markerfacecolor=("none" if sp["kind"] == "saddle" else palette.ACCENT),
                markeredgecolor=palette.ACCENT,
            )
        self.canvas.draw_idle()
        self._fill_table()

    def _fill_table(self):
        self.data_table.setRowCount(len(self._singular))
        comps = self._comps or self.comp_list.picked()
        for row, sp in enumerate(self._singular):
            x = np.asarray(sp["x"], float)
            if sp.get("pure") is not None:
                name = comps[int(sp["pure"])] if comps else f"comp {sp['pure']}"
            else:
                name = "azeotrope " + ", ".join(f"{v:.3f}" for v in x)
            for col, text in enumerate((name, f"{sp['T']:.2f}", sp["kind"])):
                self.data_table.setItem(row, col, QTableWidgetItem(text))

    # ------------------------------------------------------------ options
    def _on_solver_options(self):
        rows = [
            (
                "n_it",
                "Points per direction:",
                self._int_spin(10, 20000, self._opts["n_it"]),
            ),
            (
                "dxi",
                "Step size dxi:",
                self._spin(1e-4, 1.0, self._opts["dxi"], decimals=4, step=0.01),
            ),
            (
                "maxiter",
                "Maximum iterations:",
                self._int_spin(10, 100000, self._opts["maxiter"]),
            ),
            (
                "ftol",
                "Obj. func. tolerance:",
                self._spin(1e-16, 1e-2, self._opts["ftol"], decimals=16),
            ),
            (
                "xtol",
                "Mol. frac. tolerance:",
                self._spin(1e-16, 1e-2, self._opts["xtol"], decimals=16),
            ),
        ]
        dlg = _OptionsDialog("Solver Options", rows, self)
        if dlg.exec():
            vals = dlg.values()
            self._opts.update(
                n_it=int(vals["n_it"]),
                dxi=float(vals["dxi"]),
                maxiter=int(vals["maxiter"]),
                ftol=float(vals["ftol"]),
                xtol=float(vals["xtol"]),
            )
            self._push_params()

    def _on_plot_options(self):
        rows = [
            (
                "linewidth",
                "Line width:",
                self._spin(
                    0.1, 10.0, self._plot_opts["linewidth"], decimals=2, step=0.1
                ),
            ),
            (
                "lines",
                "Auto-gen # of curves:",
                self._int_spin(1, 100, int(self._plot_opts["lines"])),
            ),
        ]
        dlg = _OptionsDialog("Plot Options", rows, self)
        if dlg.exec():
            vals = dlg.values()
            self._plot_opts.update(
                linewidth=float(vals["linewidth"]), lines=int(vals["lines"])
            )
            self._redraw()
            self._push_params()

    @staticmethod
    def _spin(lo, hi, val, decimals=3, step=0.1):
        s = SciDoubleSpinBox()
        s.setDecimals(decimals)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        return s

    @staticmethod
    def _int_spin(lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(val))
        return s

    # -------------------------------------------------------- persistence
    def get_params(self):
        return {
            "components": self.comp_list.picked(),
            "pressure": self.p_spin.value(),
            **self._opts,
            **self._plot_opts,
        }

    def set_params(self, params):
        if "components" in params:
            self.comp_list.set_picked(list(params["components"]))
        if "pressure" in params:
            self.p_spin.setValue(float(params["pressure"]))
        for key, cast in (
            ("n_it", int),
            ("dxi", float),
            ("maxiter", int),
            ("ftol", float),
            ("xtol", float),
        ):
            if key in params:
                self._opts[key] = cast(params[key])
        for key, cast in (("linewidth", float), ("lines", int)):
            if key in params:
                self._plot_opts[key] = cast(params[key])
        # set_picked above fired orderChanged, which pushed a half-restored
        # snapshot (new components, still-default pressure and options) back
        # into window_state. Overwrite it with the finished one, or a .colx
        # loaded and re-saved without touching this panel would lose them.
        self._push_params()

    def _push_params(self):
        if self.window_state is not None:
            self.window_state.rcm_params = self.get_params()
