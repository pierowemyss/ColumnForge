"""Shortcut (FUG) design-report dialog: the Fenske-Underwood-Gilliland numbers
plus the stage-count-vs-reflux curve. A screening result, not a rating solve —
so it gets its own window rather than the per-stage Results tab (which would
have to fabricate stage profiles FUG never computes)."""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas


class FUGReportDialog(QDialog):
    def __init__(self, report, comps, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shortcut (FUG) design report")
        self.resize(640, 520)
        lay = QVBoxLayout(self)

        R, Rmin = report["R"], report["Rmin"]
        head = QLabel(
            f"<b>Minimum stages (Fenske):</b> {report['Nmin']:.1f} &nbsp; "
            f"<b>Minimum reflux (Underwood):</b> {Rmin:.3f}<br>"
            f"<b>At R = {R:.3f} (×{R / Rmin:.2f} Rmin):</b> "
            f"N ≈ {report['N']:.1f} stages, feed at stage {report['feed_stage']} "
            f"(from top)<br>"
            f"<b>D/F = {report['D']:.3f}</b>, B/F = {report['B']:.3f}")
        head.setWordWrap(True)
        lay.addWidget(head)

        # per-component distillate/bottoms composition table
        tbl = QTableWidget(len(comps), 3, self)
        tbl.setHorizontalHeaderLabels(["Component", "xD", "xB"])
        tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i, nm in enumerate(comps):
            tbl.setItem(i, 0, QTableWidgetItem(nm))
            tbl.setItem(i, 1, QTableWidgetItem(f"{report['xD'][i]:.4f}"))
            tbl.setItem(i, 2, QTableWidgetItem(f"{report['xB'][i]:.4f}"))
        tbl.setMaximumHeight(28 * (len(comps) + 1) + 4)
        lay.addWidget(tbl)

        # N-vs-R design curve
        fig = Figure(figsize=(5, 3), tight_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(report["curve_R"], report["curve_N"], color="#218fa7")
        ax.axvline(Rmin, ls="--", color="#d00000", lw=1, label="Rmin")
        ax.plot([R], [report["N"]], "o", color="#fb8500", label="operating")
        ax.set_xlabel("Reflux ratio R"); ax.set_ylabel("Stages N")
        ax.set_title("Gilliland: stages vs reflux"); ax.legend()
        lay.addWidget(FigureCanvas(fig))

        row = QHBoxLayout(); row.addStretch()
        close = QPushButton("Close"); close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
