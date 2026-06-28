import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QTableWidget,
    QTableWidgetItem, QPushButton, QGroupBox, QSplitter, QStackedWidget
)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvas

from gui.panels.sub_tab_bar import SubTabBar


def profile_to_csv_rows(profile: dict) -> list:
    """Header + data rows for CSV export. Pure (no Qt) so it is unit-testable."""
    comps = profile["comps"]
    x, T = profile["x"], profile["T"]
    rows = [["Stage", "T"] + list(comps)]
    for i in range(profile["n_stages"]):
        rows.append([i + 1, float(T[i])] + [float(v) for v in x[i]])
    return rows


class ResultsTab(QWidget):
    """Results tab with visualization and data display."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_state = None
        self._setup_ui()
        self._setup_styles()
        self.view_combo.currentTextChanged.connect(self._on_view_changed)
        self.data_combo.currentTextChanged.connect(lambda _: self._draw_plot())
        self.set_view_type("Plot")

    def set_window_state(self, window_state):
        self.window_state = window_state

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Left: Control Column (LHS Sub-Tab Column)
        self.sub_tab_bar = SubTabBar(self)
        self.sub_tab_bar.addTab("View")
        self.sub_tab_bar.tabClicked.connect(self._on_sub_tab_changed)
        main_layout.addWidget(self.sub_tab_bar)

        # Right: Results Display Pane
        self.stack = QStackedWidget(self)
        
        view_page = QWidget()
        view_layout = QVBoxLayout(view_page)
        view_layout.setSpacing(10)
        view_layout.setContentsMargins(10, 10, 10, 10)

        # Control Row (Dropdowns)
        control_layout = QHBoxLayout()
        
        # View type dropdown
        control_layout.addWidget(QLabel("Display:"))
        self.view_combo = QComboBox(self)
        self.view_combo.addItems(["Plot", "Table"])
        control_layout.addWidget(self.view_combo)

        # Data type dropdown
        control_layout.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox(self)
        self.data_combo.addItems([
            "Compositions",
            "Temperature",
            "Pressure",
            "Liquid Flow",
            "Vapor Flow",
            "K-Values",
            "Enthalpy"
        ])
        control_layout.addWidget(self.data_combo)
        control_layout.addStretch()
        
        view_layout.addLayout(control_layout)

        # Main Display: Plot or Table
        self.display_splitter = QSplitter(Qt.Vertical)

        # Matplotlib plot
        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Placeholder for data table
        table_group = QGroupBox("Data")
        table_layout = QVBoxLayout(table_group)
        self.data_table = QTableWidget(0, 5)
        self.data_table.setHorizontalHeaderLabels(["Stage", "T (K)", "x₁", "x₂", "x₃"])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.data_table)

        # Export button
        self.export_btn = QPushButton("Export CSV")
        table_layout.addWidget(self.export_btn)

        # Add to splitter (one will be hidden based on view_combo)
        self.display_splitter.addWidget(plot_group)
        self.display_splitter.addWidget(table_group)

        view_layout.addWidget(self.display_splitter, 3)

        # Bottom Row: Simulation Summary
        summary_group = QGroupBox("Simulation Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_label = QLabel(
            "Status: Not Run\n"
            "Stages: --\n"
            "Iterations: --\n"
            "Runtime: --"
        )
        self.summary_label.setStyleSheet("font-family: monospace;")
        summary_layout.addWidget(self.summary_label)

        view_layout.addWidget(summary_group, 1)

        self.stack.addWidget(view_page)
        main_layout.addWidget(self.stack)

    def _on_sub_tab_changed(self, index: int):
        """Handle sub-tab change."""
        self.stack.setCurrentIndex(index)
        self.sub_tab_bar.setCurrentIndex(index)

    def _on_view_changed(self, view_type: str):
        """Handle view type change."""
        plot_widget = self.display_splitter.widget(0)
        table_widget = self.display_splitter.widget(1)

        if view_type == "Plot":
            plot_widget.setVisible(True)
            table_widget.setVisible(False)
        else:
            plot_widget.setVisible(False)
            table_widget.setVisible(True)


    def _setup_styles(self):
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                gridline-color: #cccccc;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #cccccc;
            }
        """)

    def set_view_type(self, view_type: str):
        """Set the current view type (Plot or Table)."""
        index = self.view_combo.findText(view_type)
        if index >= 0:
            self.view_combo.setCurrentIndex(index)
            self._on_view_changed(view_type)

    def set_data_type(self, data_type: str):
        """Set the current data type."""
        index = self.data_combo.findText(data_type)
        if index >= 0:
            self.data_combo.setCurrentIndex(index)

    def update_results(self, results: dict):
        """Update summary from the normalized dict; render plot + table from the
        raw profile in window_state.results."""
        self.summary_label.setText(
            f"Status: {results.get('status', 'Unknown')}\n"
            f"Stages: {results.get('stages', '--')}\n"
            f"Iterations: {results.get('iterations', '--')}\n"
            f"Runtime: {results.get('runtime', '--')}"
        )
        self._fill_table()
        self._draw_plot()

    def _profile(self):
        return getattr(self.window_state, "results", None) if self.window_state else None

    def _fill_table(self):
        prof = self._profile()
        if not prof:
            self.data_table.setRowCount(0)
            return
        comps, x, T = prof["comps"], prof["x"], prof["T"]
        headers = ["Stage", "T (K)"] + list(comps)
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(prof["n_stages"])
        for i in range(prof["n_stages"]):
            vals = [i + 1, round(float(T[i]), 2)] + [round(float(v), 4) for v in x[i]]
            for c, v in enumerate(vals):
                self.data_table.setItem(i, c, QTableWidgetItem(str(v)))

    def _draw_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        prof = self._profile()
        if not prof:
            ax.text(0.5, 0.5, "Run a simulation to see results",
                    ha="center", va="center", color="#888888")
            ax.axis("off")
            self.canvas.draw()
            return

        x, T, comps = prof["x"], prof["T"], prof["comps"]
        N = np.arange(1, prof["n_stages"] + 1)
        dtype = self.data_combo.currentText()
        if dtype == "Temperature":
            ax.plot(N, T, "-o", color="#fb8500")
            ax.set_ylabel("Temperature (K)")
        elif dtype == "Compositions":
            for j, name in enumerate(comps):
                ax.plot(N, x[:, j], "-o", label=name)
            ax.set_ylabel("Liquid mole fraction x")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        else:
            # ponytail: BVM profile only carries x / y / T. Flows, K-values and
            # enthalpy come from the rigorous Inside-Out solver — these light up
            # once it emits them (un-gated by keys present in the profile dict).
            avail = dtype.lower().replace(" ", "_").replace("-", "_") in prof
            if avail:
                series = np.asarray(prof[dtype.lower().replace(" ", "_").replace("-", "_")])
                ax.plot(N, series, "-o", color="#219ebc")
                ax.set_ylabel(dtype)
                ax.axvline(prof["feed_stage"], color="grey", ls="--", lw=1)
                ax.set_xlabel("Stage N (bottom → top)")
                self.figure.tight_layout()
                self.canvas.draw()
                return
            ax.text(0.5, 0.5, f"{dtype} available after an Inside-Out run",
                    ha="center", va="center", color="#888888")
            ax.axis("off")
            self.canvas.draw()
            return

        ax.axvline(prof["feed_stage"], color="grey", ls="--", lw=1)
        ax.set_xlabel("Stage N (bottom → top)")
        self.figure.tight_layout()
        self.canvas.draw()

    def clear_results(self):
        """Clear all results."""
        self.summary_label.setText(
            "Status: Not Run\n"
            "Stages: --\n"
            "Iterations: --\n"
            "Runtime: --"
        )
        self.data_table.setRowCount(0)
        self._draw_plot()
