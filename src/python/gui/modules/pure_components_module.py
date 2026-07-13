"""Pure Components browser — the bundled database as a first-class tool.

Search/browse the 78 curated records, read the property sheet, plot Psat(T)
over the Antoine valid range (compare up to three), and "Add to column" to load
a record into the session. DB-driven, so it works with zero species loaded.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QCheckBox, QGroupBox,
)

from core import component_db
from core.thermodynamics import antoine_psat
from gui.plotting import CompactNavigationToolbar


class PureComponentsModuleWidget(QWidget):
    def __init__(self, window_state=None, parent=None):
        super().__init__(parent)
        self.window_state = window_state
        self._compare = []                   # pinned record names for overlay
        self._build_ui()
        self._refresh_table("")

    def _build_ui(self):
        layout = QHBoxLayout(self)

        # ---- left: search + table -------------------------------------
        left = QVBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search name, alias, CAS or formula…")
        self.search.textChanged.connect(self._refresh_table)
        left.addWidget(self.search)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["name", "formula", "CAS", "MW", "Tb (K)"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(
            lambda *_: self._on_select())
        self.table.horizontalHeader().setStretchLastSection(True)
        left.addWidget(self.table, 1)
        layout.addLayout(left, 1)

        # ---- right: properties + plot ---------------------------------
        right = QVBoxLayout()
        props = QGroupBox("Properties")
        props_layout = QVBoxLayout(props)
        self.details = QLabel("Select a component.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        props_layout.addWidget(self.details)
        btns = QHBoxLayout()
        self.add_btn = QPushButton("Add to column")
        self.add_btn.clicked.connect(self._add_to_column)
        self.compare_btn = QPushButton("Compare (pin)")
        self.compare_btn.clicked.connect(self._pin_compare)
        self.clear_btn = QPushButton("Clear compare")
        self.clear_btn.clicked.connect(self._clear_compare)
        self.logp = QCheckBox("log P")
        self.logp.toggled.connect(self._draw)
        for w in (self.add_btn, self.compare_btn, self.clear_btn, self.logp):
            btns.addWidget(w)
        btns.addStretch()
        props_layout.addLayout(btns)
        right.addWidget(props)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas, 1)
        self.status = QLabel("")
        right.addWidget(self.status)
        layout.addLayout(right, 1)

    # ------------------------------------------------------------------
    def _refresh_table(self, text):
        recs = component_db.search(text, limit=200)
        self.table.setRowCount(0)
        for rec in recs:
            row = self.table.rowCount()
            self.table.insertRow(row)
            vals = [rec["name"], rec.get("formula", ""), rec.get("cas", ""),
                    _fmt(rec.get("mw")), _fmt(rec.get("tb"))]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if col == 0:
                    item.setData(Qt.UserRole, rec["name"])
                self.table.setItem(row, col, item)
        if self.table.rowCount():
            self.table.setCurrentCell(0, 0)

    def _current_name(self):
        item = self.table.item(self.table.currentRow(), 0)
        return item.data(Qt.UserRole) if item else None

    def _on_select(self):
        name = self._current_name()
        if not name:
            return
        rec = component_db.get(name)
        tmin, tmax, est = component_db.antoine_trange(rec)
        rng = f"{tmin:.0f}–{tmax:.0f} °C" + (" (est.)" if est else "")
        a, b, c = rec["antoine"]
        self.details.setText(
            f"<b>{rec['name']}</b> ({rec.get('formula', '')}, "
            f"{rec.get('cas', '')})<br>"
            f"MW {_fmt(rec.get('mw'))} g/mol · Tb {_fmt(rec.get('tb'))} K · "
            f"Tc {_fmt(rec.get('tc'))} K · Pc {_fmt(rec.get('pc'))} bar · "
            f"ω {_fmt(rec.get('omega'))}<br>"
            f"ρ_liq {_fmt(rec.get('liquid_density'))} · "
            f"cp_liq {_fmt(rec.get('cp_liq'))} · "
            f"hvap_tb {_fmt(rec.get('hvap_tb'))} kJ/mol<br>"
            f"Antoine (log₁₀ mmHg, °C): A={a:g} B={b:g} C={c:g}; valid {rng}")
        self.add_btn.setEnabled(self.window_state is not None)
        self._draw()

    def _draw(self, *_):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        names = list(dict.fromkeys(self._compare + [self._current_name()]))
        for name in names:
            if not name:
                continue
            rec = component_db.get(name)
            tmin, tmax, _ = component_db.antoine_trange(rec)
            T = np.linspace(tmin, tmax, 120)
            ant = np.array([rec["antoine"]], float)
            P = np.array([antoine_psat(t, ant)[0] for t in T])   # mmHg
            P_bar = P * 0.001333224                               # mmHg -> bar
            ax.plot(T, P_bar, label=name)
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("Psat (bar)")
        if self.logp.isChecked():
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if names and any(names):
            ax.legend(fontsize=8)
        self.figure.tight_layout()
        self.canvas.draw()

    def _pin_compare(self):
        name = self._current_name()
        if name and name not in self._compare:
            self._compare = (self._compare + [name])[-3:]     # keep last 3
            self._draw()
            self.status.setText("Comparing: " + ", ".join(self._compare))

    def _clear_compare(self):
        self._compare = []
        self.status.setText("")
        self._draw()

    def _add_to_column(self):
        name = self._current_name()
        if not name or not self.window_state:
            return
        try:
            component_db.load_into(self.window_state, name)
        except KeyError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(f"Added {name} to the column species.")


def _fmt(v, unit=""):
    return f"{v:g}{unit}" if v is not None else "—"
