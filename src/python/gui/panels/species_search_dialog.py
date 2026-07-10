"""Type-ahead component search over the bundled database (roadmap Month 2).

Searches name/alias/CAS/formula via core.component_db.search; shows key
properties for the highlighted hit. accept() leaves the chosen record name
in .selected_name — the caller does the actual load_into(ws, name).
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QVBoxLayout,
)

from core import component_db


class SpeciesSearchDialog(QDialog):
    def __init__(self, parent=None, existing_names=()):
        super().__init__(parent)
        self.setWindowTitle("Add Species from Database")
        self.setMinimumSize(420, 380)
        self.selected_name = None
        self._existing = {n.lower() for n in existing_names}

        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search name, alias, CAS or formula…")
        self.search_edit.textChanged.connect(self._refresh)
        layout.addWidget(self.search_edit)

        self.results = QListWidget()
        self.results.currentItemChanged.connect(self._show_details)
        self.results.itemDoubleClicked.connect(lambda _: self.accept())
        layout.addWidget(self.results, 1)

        self.details = QLabel("")
        self.details.setWordWrap(True)
        layout.addWidget(self.details)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._ok = buttons.button(QDialogButtonBox.Ok)
        layout.addWidget(buttons)

        self._refresh("")
        self.search_edit.setFocus()

    def _refresh(self, text):
        self.results.clear()
        for rec in component_db.search(text, limit=50):
            label = f"{rec['name']}  ({rec['formula']}, {rec['cas']})"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, rec["name"])
            if rec["name"].lower() in self._existing:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip("Already in the species list")
            self.results.addItem(item)
        # preselect the first selectable hit so Enter adds it
        for i in range(self.results.count()):
            if self.results.item(i).flags() & Qt.ItemIsEnabled:
                self.results.setCurrentRow(i)
                break

    def _show_details(self, item, _prev=None):
        if item is None:
            self.details.setText("")
            self._ok.setEnabled(False)
            return
        rec = component_db.get(item.data(Qt.UserRole))
        tmin, tmax, est = component_db.antoine_trange(rec)
        rng = f"{tmin:.0f}–{tmax:.0f} °C" + (" (est.)" if est else "")

        def fmt(v, unit=""):
            return f"{v:g}{unit}" if v is not None else "—"

        self.details.setText(
            f"MW {fmt(rec['mw'])} g/mol · Tb {fmt(rec['tb'])} K · "
            f"Tc {fmt(rec.get('tc'))} K · Pc {fmt(rec.get('pc'))} bar\n"
            f"Antoine valid {rng}"
        )
        self._ok.setEnabled(bool(item.flags() & Qt.ItemIsEnabled))

    def accept(self):
        item = self.results.currentItem()
        if item is None or not (item.flags() & Qt.ItemIsEnabled):
            return
        self.selected_name = item.data(Qt.UserRole)
        super().accept()
