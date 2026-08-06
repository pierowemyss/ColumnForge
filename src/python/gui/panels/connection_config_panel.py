"""Editor for one inter-column connection.

Sixth page of the Specifications overview editor stack, alongside the condenser,
reboiler, stream and module panels. Same contract as those: a pure widget with
`set_config`/`get_config` and a `configChanged` signal — the owning tab does all
the WindowState writing.

Every field here is consumed by core.flowsheet. The thermal quality is the one
that can be left blank, and blank means "the port's natural quality", shown as
placeholder text rather than silently defaulted — an inter-unit heater is a real
decision and must look like one.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QSpinBox,
    QVBoxLayout, QWidget,
)


class ConnectionConfigPanel(QWidget):
    """Split fraction, destination stage, optional thermal-quality override."""

    configChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._conn_id = None
        self._loading = False
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Connection")
        form = QFormLayout(grp)

        self.route_label = QLabel("—")
        self.route_label.setProperty("mono", True)
        form.addRow("Stream:", self.route_label)

        self.stage_spin = QSpinBox(self)
        self.stage_spin.setRange(0, 999)
        self.stage_spin.setToolTip(
            "Destination stage, 0-based from the top (0 = distillate). Must be "
            "an interior tray: the condenser and reboiler stages cannot take a "
            "feed.")
        form.addRow("Feeds stage:", self.stage_spin)

        self.split_spin = QDoubleSpinBox(self)
        self.split_spin.setRange(0.01, 1.0)
        self.split_spin.setSingleStep(0.05)
        self.split_spin.setDecimals(3)
        self.split_spin.setToolTip(
            "Fraction of this outlet sent down this connection. Anything below 1 "
            "leaves the rest as an external product — that is how a purge on a "
            "recycle is expressed.")
        form.addRow("Split fraction:", self.split_spin)

        self.q_edit = QLineEdit(self)
        self.q_edit.setToolTip(
            "Thermal quality of the stream as it arrives: 1 = saturated liquid, "
            "0 = saturated vapour. Leave blank to use the source port's natural "
            "quality; entering a value states an inter-column heater or cooler.")
        form.addRow("Feed quality q:", self.q_edit)

        self.purge_label = QLabel("")
        self.purge_label.setProperty("hint", True)
        self.purge_label.setWordWrap(True)
        form.addRow("", self.purge_label)

        lay.addWidget(grp)
        lay.addStretch()

        self.stage_spin.valueChanged.connect(self._emit)
        self.split_spin.valueChanged.connect(self._emit)
        self.q_edit.editingFinished.connect(self._emit)

    # --- the pure-widget contract -----------------------------------------

    def set_config(self, conn, natural_q=None, max_stage=None):
        """Load a core.flowsheet.Connection (stages shown 0-based from the top)."""
        self._loading = True
        self._conn_id = conn.id
        self.route_label.setText(f"{conn.src}.{conn.port}  →  {conn.dst}")
        if max_stage is not None:
            self.stage_spin.setRange(1, max(1, int(max_stage) - 2))
        self.stage_spin.setValue(max(0, int(conn.stage) - 1))
        self.split_spin.setValue(float(conn.split_fraction))
        self.q_edit.setText("" if conn.q is None else f"{float(conn.q):g}")
        if natural_q is not None:
            self.q_edit.setPlaceholderText(
                f"{float(natural_q):g}  (natural for {conn.port})")
        self._update_purge_hint()
        self._loading = False

    def get_config(self) -> dict:
        """0-based GUI stage; the caller converts to the solver's 1-based."""
        text = self.q_edit.text().strip()
        try:
            q = float(text) if text else None
        except ValueError:
            q = None
        return {"id": self._conn_id, "stage": int(self.stage_spin.value()),
                "split_fraction": float(self.split_spin.value()), "q": q}

    @property
    def current_connection_id(self):
        return self._conn_id

    def _update_purge_hint(self):
        s = self.split_spin.value()
        if s >= 1.0 - 1e-9:
            self.purge_label.setText("All of this outlet goes down this stream.")
        else:
            self.purge_label.setText(
                f"{1 - s:.0%} of this outlet is not sent here — it leaves the "
                "flowsheet as an external product (a purge).")

    def _emit(self):
        if self._loading:
            return
        self._update_purge_hint()
        self.configChanged.emit()
