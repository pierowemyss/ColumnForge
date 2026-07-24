#!/usr/bin/env python3
"""
FreeRCM - Integrated into ColumnForge
Containerized module - handles its own imports internally.
"""

import sys
import os

# Set up path for freeRCM - same as launch.py does
_current_dir = os.path.dirname(os.path.abspath(__file__))
_freeRCM_src_path = os.path.dirname(_current_dir)  # src/python
if _freeRCM_src_path not in sys.path:
    sys.path.insert(0, _freeRCM_src_path)

_core_path = os.path.join(_freeRCM_src_path, 'core')
_gui_path = _current_dir
if _core_path not in sys.path:
    sys.path.insert(0, _core_path)
if _gui_path not in sys.path:
    sys.path.insert(0, _gui_path)

import numpy as np
os.environ['QT_API'] = 'pyside6'

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QListWidget, QLineEdit, QComboBox, QFrame, QGridLayout,
    QSizePolicy, QSpacerItem, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import pickle

# Import freeRCM modules
import data_structures
import solver
import plot_widget as plot_widget_mod

dict2struct = data_structures.dict2struct
RCM = solver.RCM
RCMplot = plot_widget_mod.RCMplot


def init_from_window_state(window_state):
    """Initialize global state from ColumnForge's window_state."""
    global P, comps, selected_comps, antoine_params, PLXANT_params
    global NRTL_aij, NRTL_bij, NRTL_cij, TcCel, Pc, omega, allProps
    
    if not window_state:
        return
    
    species_names = list(window_state.species.keys())
    n = len(species_names)
    
    comps = np.array(species_names)
    selected_comps = np.array([])
    
    antoine_params = np.zeros((n, 3))
    PLXANT_params = np.zeros((n, 7))
    NRTL_aij = np.zeros((n, n))
    NRTL_bij = np.zeros((n, n))
    NRTL_cij = np.zeros((n, n))
    
    TcCel = np.zeros(n)
    Pc = np.zeros(n)
    omega = np.zeros(n)
    
    thermo = window_state.thermodynamics_config
    
    for i, name in enumerate(species_names):
        comp_params = thermo.get_component_params(name)
        # columnForge stores tc in Kelvin; freeRCM's SRK expects Celsius (it adds 273.15).
        TcCel[i] = (comp_params.tc - 273.15) if comp_params.tc else 0
        Pc[i] = comp_params.pc if comp_params.pc else 0
        omega[i] = comp_params.omega if comp_params.omega else 0
        antoine_params[i, 0] = comp_params.antoine_a if comp_params.antoine_a else 0
        antoine_params[i, 1] = comp_params.antoine_b if comp_params.antoine_b else 0
        antoine_params[i, 2] = comp_params.antoine_c if comp_params.antoine_c else 0
        for k in range(7):  # PLXANT C1..C7 — without this the antMethod=2 solver gets all zeros
            PLXANT_params[i, k] = getattr(comp_params, f"plxant_c{k+1}", 0) or 0

    for (i_name, j_name), val in thermo.binary.nrtl_aij.items():
        if i_name in species_names and j_name in species_names:
            i = species_names.index(i_name)
            j = species_names.index(j_name)
            NRTL_aij[i, j] = val
    
    for (i_name, j_name), val in thermo.binary.nrtl_bij.items():
        if i_name in species_names and j_name in species_names:
            i = species_names.index(i_name)
            j = species_names.index(j_name)
            NRTL_bij[i, j] = val
    
    for (i_name, j_name), val in thermo.binary.nrtl_cij.items():
        if i_name in species_names and j_name in species_names:
            i = species_names.index(i_name)
            j = species_names.index(j_name)
            NRTL_cij[i, j] = val
    
    allProps.__dict__.update({
        "antoine": antoine_params,
        "PLXANT": PLXANT_params,
        "NRTL_aij": NRTL_aij,
        "NRTL_bij": NRTL_bij,
        "NRTL_cij": NRTL_cij,
        "TcCel": TcCel,
        "Pc": Pc,
        "omega": omega
    })


# Default global state (fallback if no window_state)
P = 1
comps = np.array([])
selected_comps = np.array([])
antoine_params = np.array([])
PLXANT_params = np.array([])
NRTL_aij = np.array([])
NRTL_bij = np.array([])
NRTL_cij = np.array([])
TcCel = np.array([])
Pc = np.array([])
omega = np.array([])

lmopts = {"maxiter": 1000, "ftol": 1e-12, "xtol": 1e-12}
options = {
    "antMethod": 2,
    "activity": 3,
    "lines": 15,
    "linewidth": 1.2,
    "n_it": 250,
    "dxi": 0.02,
    "lmopts": lmopts,
}
opts = dict2struct(options)

allProps_dict = {
    "antoine": np.array([]),
    "PLXANT": np.array([]),
    "NRTL_aij": np.array([]),
    "NRTL_bij": np.array([]),
    "NRTL_cij": np.array([]),
    "TcCel": np.array([]),
    "Pc": np.array([]),
    "omega": np.array([])
}
allProps = dict2struct(allProps_dict)


class NewSimulationWindow(QMainWindow):
    """Component selection window - loads species from ColumnForge."""
    
    def __init__(self, window_state=None):
        super().__init__()
        self.window_state = window_state
        self.setWindowTitle("RCM - Select Components")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize global state from ColumnForge if window_state provided
        if window_state:
            init_from_window_state(window_state)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.create_widgets()

    def create_widgets(self):
        layout = QGridLayout(self.central_widget)

        # Component selection area
        midsection = QFrame()
        midsection_cont = QHBoxLayout(midsection)
        midsection_cont.setSpacing(10)

        # Left: All Components
        left_column = QVBoxLayout()
        self.components_list_label = QLabel("All Components (from ColumnForge)", self)
        self.components_list_label.setFixedHeight(20)
        left_column.addWidget(self.components_list_label)

        self.components_list = QListWidget(self)
        left_column.addWidget(self.components_list)

        midsection_cont.addLayout(left_column, 2)

        # Middle: Arrow buttons
        middle_column = QVBoxLayout()
        middle_column.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        move_to_selected_button = QPushButton(">>", self)
        move_to_selected_button.clicked.connect(self.move_to_selected)
        middle_column.addWidget(move_to_selected_button)

        move_to_master_button = QPushButton("<<", self)
        move_to_master_button.clicked.connect(self.move_to_master)
        middle_column.addWidget(move_to_master_button)
        
        middle_column.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        midsection_cont.addLayout(middle_column)

        # Right: Selected Components
        right_column = QVBoxLayout()
        self.selected_comps_label = QLabel("Selected Components", self)
        self.selected_comps_label.setFixedHeight(20)
        right_column.addWidget(self.selected_comps_label)

        self.selected_comps_list = QListWidget(self)
        right_column.addWidget(self.selected_comps_list)

        midsection_cont.addLayout(right_column, 2)
        layout.addWidget(midsection, 0, 0, 1, 5)

        # Model settings
        vapor_pressure_label = QLabel("Vapor Pressure:", self)
        layout.addWidget(vapor_pressure_label, 2, 0)

        self.vapor_pressure_var = QComboBox(self)
        self.vapor_pressure_var.addItems(["Antoine", "Extended Antoine"])
        self.vapor_pressure_var.setCurrentIndex(opts.antMethod - 1)
        self.vapor_pressure_var.currentIndexChanged.connect(self.update_vapor_pressure_method)
        layout.addWidget(self.vapor_pressure_var, 2, 1)

        dropdown_label = QLabel("Select Model:", self)
        layout.addWidget(dropdown_label, 3, 0)

        self.model_var = QComboBox(self)
        self.model_var.addItems(["Ideal", "NRTL", "NRTL-SRK"])
        self.model_var.setCurrentIndex(opts.activity - 1)
        self.model_var.currentIndexChanged.connect(self.update_eos_method)
        layout.addWidget(self.model_var, 3, 1)

        pressure_label = QLabel("Pressure:", self)
        layout.addWidget(pressure_label, 4, 0)

        self.pressure_entry = QLineEdit(str(P), self)
        self.pressure_entry.returnPressed.connect(self.save_pressure)
        layout.addWidget(self.pressure_entry, 4, 1)

        pressure_unit_label = QLabel(" bar", self)
        layout.addWidget(pressure_unit_label, 4, 2)

        # Run button
        run_button = QPushButton("Run RCM", self)
        run_button.clicked.connect(self.run_rcm)
        layout.addWidget(run_button, 5, 4)

        self.update_components_list()
        self.update_selected_components_list()

    def load_from_window_state(self):
        """Load species and parameters from ColumnForge's window_state."""
        global comps, antoine_params, PLXANT_params, NRTL_aij, NRTL_bij, NRTL_cij, TcCel, Pc, omega
        
        if not self.window_state:
            return

        species_names = list(self.window_state.species.keys())
        n = len(species_names)
        
        comps = np.array(species_names)
        
        antoine_params = np.zeros((n, 3))
        PLXANT_params = np.zeros((n, 7))
        NRTL_aij = np.zeros((n, n))
        NRTL_bij = np.zeros((n, n))
        NRTL_cij = np.zeros((n, n))
        
        TcCel = np.zeros(n)
        Pc = np.zeros(n)
        omega = np.zeros(n)
        
        thermo = self.window_state.thermodynamics_config
        
        for i, name in enumerate(species_names):
            comp_params = thermo.get_component_params(name)
            # columnForge stores tc in Kelvin; freeRCM's SRK expects Celsius (it adds 273.15).
            TcCel[i] = (comp_params.tc - 273.15) if comp_params.tc else 0
            Pc[i] = comp_params.pc if comp_params.pc else 0
            omega[i] = comp_params.omega if comp_params.omega else 0
            antoine_params[i, 0] = comp_params.antoine_a if comp_params.antoine_a else 0
            antoine_params[i, 1] = comp_params.antoine_b if comp_params.antoine_b else 0
            antoine_params[i, 2] = comp_params.antoine_c if comp_params.antoine_c else 0
            for k in range(7):  # PLXANT C1..C7 — without this the antMethod=2 solver gets all zeros
                PLXANT_params[i, k] = getattr(comp_params, f"plxant_c{k+1}", 0) or 0

        for (i_name, j_name), val in thermo.binary.nrtl_aij.items():
            if i_name in species_names and j_name in species_names:
                i = species_names.index(i_name)
                j = species_names.index(j_name)
                NRTL_aij[i, j] = val
        
        for (i_name, j_name), val in thermo.binary.nrtl_bij.items():
            if i_name in species_names and j_name in species_names:
                i = species_names.index(i_name)
                j = species_names.index(j_name)
                NRTL_bij[i, j] = val
        
        for (i_name, j_name), val in thermo.binary.nrtl_cij.items():
            if i_name in species_names and j_name in species_names:
                i = species_names.index(i_name)
                j = species_names.index(j_name)
                NRTL_cij[i, j] = val
        
        allProps.__dict__.update({
            "antoine": antoine_params,
            "PLXANT": PLXANT_params,
            "NRTL_aij": NRTL_aij,
            "NRTL_bij": NRTL_bij,
            "NRTL_cij": NRTL_cij,
            "TcCel": TcCel,
            "Pc": Pc,
            "omega": omega
        })
        
        self.update_components_list()
        self.update_selected_components_list()

    def update_components_list(self):
        self.components_list.clear()
        self.components_list.addItems(comps.tolist())

    def update_selected_components_list(self):
        global selected_comps
        self.selected_comps_list.clear()
        self.selected_comps_list.addItems(selected_comps.tolist())

    def move_to_selected(self):
        selected_items = self.components_list.selectedItems()
        if selected_items:
            global selected_comps
            selected_item = selected_items[0]
            selected_comps = np.append(selected_comps, selected_item.text())
            self.update_components_list()
            self.update_selected_components_list()

    def move_to_master(self):
        selected_items = self.selected_comps_list.selectedItems()
        if selected_items:
            global selected_comps
            selected_item = selected_items[0]
            selected_comps = np.delete(selected_comps, self.selected_comps_list.row(selected_item))
            self.update_components_list()
            self.update_selected_components_list()

    def save_pressure(self):
        global P
        try:
            P = float(self.pressure_entry.text())
        except ValueError:
            pass

    def update_vapor_pressure_method(self):
        global opts
        vapor_pressure_method = self.vapor_pressure_var.currentText()
        if vapor_pressure_method == "Antoine":
            opts.antMethod = 1
        elif vapor_pressure_method == "Extended Antoine":
            opts.antMethod = 2

    def update_eos_method(self):
        global opts
        eos_method = self.model_var.currentText()
        if eos_method == "Ideal":
            opts.activity = 1
        elif eos_method == "NRTL":
            opts.activity = 2
        elif eos_method == "NRTL-SRK":
            opts.activity = 3

    def run_rcm(self):
        global selected_comps
        if len(selected_comps) < 2:
            QMessageBox.information(self, "RCM", "Please select at least 2 components.")
            return
        self.save_pressure()
        self.make_sim_window = MakeSimWindow()
        self.make_sim_window.show()


class MakeSimWindow(QMainWindow):
    """RCM plotting window."""
    
    def __init__(self):
        super().__init__()
        global P, comps, selected_comps, opts
        self.comps = comps
        self.selected_comps = selected_comps
        self.opts = opts
        self.setWindowTitle("RCM Simulation")
        self.create_widgets()

    def create_widgets(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)

        # Button frame
        self.button_frame = QFrame()
        self.button_layout = QVBoxLayout(self.button_frame)

        self.auto_gen_button = QPushButton("Auto-Generate Curves")
        self.auto_gen_button.clicked.connect(self.auto_generate)
        self.button_layout.addWidget(self.auto_gen_button)

        self.clear_button = QPushButton("Clear Plot")
        self.clear_button.clicked.connect(self.clear_plot)
        self.button_layout.addWidget(self.clear_button)

        self.plot_options_button = QPushButton("Plot Options")
        self.plot_options_button.clicked.connect(self.plot_options)
        self.button_layout.addWidget(self.plot_options_button)

        self.solver_options_button = QPushButton("Solver Options")
        self.solver_options_button.clicked.connect(self.solver_options)
        self.button_layout.addWidget(self.solver_options_button)

        self.save_button = QPushButton("Save Simulation")
        self.save_button.clicked.connect(self.save_variables)
        self.button_layout.addWidget(self.save_button)

        self.main_layout.addWidget(self.button_frame)

        # Plot frame
        self.plot_frame = QFrame()
        self.main_layout.addWidget(self.plot_frame)

        self.canvas = None
        self.clear_figure()

    def plot_figure(self):
        global comps, selected_comps, P, allProps, opts
        x0n = np.array([])
        self.x = RCM(comps, selected_comps, P, allProps, opts, x0n, 1).x
        fig, ax = RCMplot(self.x, self.selected_comps, self.opts)

        if self.canvas:
            self.canvas.deleteLater()
            self.toolbar.deleteLater()

        self.canvas = FigureCanvas(fig)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        self.coord_label = QLabel("", self.toolbar)
        self.toolbar.addWidget(self.coord_label)

        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self.click_plot)

    def clear_figure(self):
        global opts
        self.x = np.zeros([2 * opts.n_it, 3, 1])
        fig, ax = RCMplot(self.x, self.selected_comps, self.opts)

        if self.canvas:
            self.canvas.deleteLater()
            self.toolbar.deleteLater()

        self.canvas = FigureCanvas(fig)
        self.toolbar = CompactNavigationToolbar(self.canvas, self)
        self.coord_label = QLabel("", self.toolbar)
        self.toolbar.addWidget(self.coord_label)

        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self.click_plot)

    def genLine(self, event):
        if event.inaxes:
            global comps, selected_comps, P, allProps, opts
            x_click = event.xdata
            y_click = event.ydata
            x0n = np.array([x_click, y_click, 1 - x_click - y_click])
            self.x = np.append(self.x, RCM(comps, selected_comps, P, allProps, opts, x0n, 2).x, 2)
            fig, ax = RCMplot(self.x, self.selected_comps, self.opts)

            if self.canvas:
                self.canvas.deleteLater()
                self.toolbar.deleteLater()

            self.canvas = FigureCanvas(fig)
            self.toolbar = CompactNavigationToolbar(self.canvas, self)
            self.coord_label = QLabel("", self.toolbar)
            self.toolbar.addWidget(self.coord_label)

            self.plot_layout = QVBoxLayout(self.plot_frame)
            self.plot_layout.addWidget(self.toolbar)
            self.plot_layout.addWidget(self.canvas)

            self.canvas.draw()
            self.canvas.mpl_connect("button_press_event", self.click_plot)

    def auto_generate(self):
        self.plot_frame.hide()
        self.plot_frame = QFrame()
        self.main_layout.addWidget(self.plot_frame)
        self.canvas = None
        self.plot_figure()

    def clear_plot(self):
        self.plot_frame.hide()
        self.plot_frame = QFrame()
        self.main_layout.addWidget(self.plot_frame)
        self.canvas = None
        self.clear_figure()

    def click_plot(self, event):
        if event.inaxes:
            self.plot_frame.hide()
            self.plot_frame = QFrame()
            self.main_layout.addWidget(self.plot_frame)
            self.canvas = None
            self.genLine(event)

    def plot_options(self):
        self.plot_options_window = PlotOptsWindow()
        self.plot_options_window.setWindowModality(Qt.ApplicationModal)
        self.plot_options_window.move(self.pos().x() + 50, self.pos().y() + 50)
        self.plot_options_window.show()

    def solver_options(self):
        self.solver_options_window = SolverOptsWindow()
        self.solver_options_window.setWindowModality(Qt.ApplicationModal)
        self.solver_options_window.move(self.pos().x() + 50, self.pos().y() + 50)
        self.solver_options_window.show()

    def save_variables(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Variables", "", "RCM Files (*.rcm)")
        if file_path:
            global P, comps, selected_comps, allProps, lmopts, opts, NRTL_aij, NRTL_bij, NRTL_cij, TcCel, Pc, omega, antoine_params, PLXANT_params
            data = {
                "P": P, "comps": comps, "selected_comps": selected_comps,
                "allProps": allProps, "lmopts": lmopts, "opts": opts,
                "NRTL_aij": NRTL_aij, "NRTL_bij": NRTL_bij, "NRTL_cij": NRTL_cij,
                "TcCel": TcCel, "Pc": Pc, "omega": omega,
                "antoine_params": antoine_params, "PLXANT_params": PLXANT_params
            }
            with open(file_path, "wb") as file:
                pickle.dump(data, file)


class CompactNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent, coordinates=False)


class PlotOptsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plot Options")
        self.create_widgets()

    def create_widgets(self):
        global opts
        self.central_widget = QWidget()
        layout = QGridLayout(self.central_widget)

        line_width_label = QLabel("Line Width:", self)
        layout.addWidget(line_width_label, 0, 0)
        self.linewidth_entry = QLineEdit(str(opts.linewidth), self)
        layout.addWidget(self.linewidth_entry, 0, 1)

        num_lines_label = QLabel("Auto-Gen # of Lines:", self)
        layout.addWidget(num_lines_label, 1, 0)
        self.num_lines_entry = QLineEdit(str(opts.lines), self)
        layout.addWidget(self.num_lines_entry, 1, 1)

        self.setCentralWidget(self.central_widget)


class SolverOptsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solver Options")
        self.create_widgets()

    def create_widgets(self):
        global opts, lmopts
        self.central_widget = QWidget()
        layout = QGridLayout(self.central_widget)

        num_points_label = QLabel("Num. Points fwd/back:", self)
        layout.addWidget(num_points_label, 0, 0)
        self.num_points_entry = QLineEdit(str(opts.n_it), self)
        layout.addWidget(self.num_points_entry, 0, 1)

        dxi_label = QLabel("dxi:", self)
        layout.addWidget(dxi_label, 1, 0)
        self.dxi_entry = QLineEdit(str(opts.dxi), self)
        layout.addWidget(self.dxi_entry, 1, 1)

        maxiter_label = QLabel("Maximum Iterations:", self)
        layout.addWidget(maxiter_label, 2, 0)
        self.maxiter_entry = QLineEdit(str(lmopts["maxiter"]), self)
        layout.addWidget(self.maxiter_entry, 2, 1)

        ftol_label = QLabel("Obj. Func. Tolerance:", self)
        layout.addWidget(ftol_label, 3, 0)
        self.ftol_entry = QLineEdit(str(lmopts["ftol"]), self)
        layout.addWidget(self.ftol_entry, 3, 1)

        xtol_label = QLabel("Mol. Frac. Tolerance:", self)
        layout.addWidget(xtol_label, 4, 0)
        self.xtol_entry = QLineEdit(str(lmopts["xtol"]), self)
        layout.addWidget(self.xtol_entry, 4, 1)

        self.setCentralWidget(self.central_widget)
