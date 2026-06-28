import sys
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QStackedLayout
)
from PySide6.QtCore import Qt, Signal


class ModulesTab(QWidget):
    """Modules tab for side features and additional functionality."""

    launchModule = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.window_state = None
        self.rcm_window = None
        self.bvm_widget = None
        self.modules_loaded = False
        
        self._setup_ui()
        self._setup_styles()
        self._connect_signals()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Select Module:"))
        
        self.module_combo = QComboBox(self)
        self.module_combo.addItems([
            "RCM",
            "BVM",
            "Pure Components (#)",
            "Phase EQ (#)"
        ])
        header_layout.addWidget(self.module_combo)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        self.content_stack = QStackedLayout()
        main_layout.addLayout(self.content_stack)
        
        self.placeholder_widget = QWidget()
        placeholder_layout = QVBoxLayout(self.placeholder_widget)
        placeholder_layout.addStretch()
        self.content_stack.addWidget(self.placeholder_widget)
        
        self.rcm_container = QWidget()
        rcm_layout = QVBoxLayout(self.rcm_container)
        rcm_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.rcm_container)

        self.bvm_container = QWidget()
        bvm_layout = QVBoxLayout(self.bvm_container)
        bvm_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.bvm_container)

        main_layout.addStretch()

    def _setup_styles(self):
        self.setStyleSheet("""
            QLabel {
                font-weight: bold;
            }
        """)

    def _connect_signals(self):
        self.module_combo.currentTextChanged.connect(self._on_module_changed)

    def set_window_state(self, window_state):
        """Set the window state reference."""
        self.window_state = window_state
        self._on_module_changed(self.module_combo.currentText())

    def _on_module_changed(self, module_name: str):
        """Handle module selection change - auto-launches the selected module."""
        if module_name == "RCM":
            self._launch_rcm()
        elif module_name == "BVM":
            self._launch_bvm()
        else:
            self.content_stack.setCurrentWidget(self.placeholder_widget)

    def _setup_paths(self):
        """Lazy load paths only when needed."""
        if self.modules_loaded:
            return
        
        _current_dir = os.path.dirname(os.path.abspath(__file__))                  # .../src/python/gui/tabs
        _src_python = os.path.dirname(os.path.dirname(_current_dir))               # .../src/python
        _repo_root = os.path.dirname(os.path.dirname(_src_python))                 # repo root
        _gui_path = os.path.join(_repo_root, 'src', 'side_features', 'freeRCM', 'src', 'python', 'gui')

        if _gui_path not in sys.path:
            sys.path.insert(0, _gui_path)
        if _src_python not in sys.path:
            sys.path.insert(0, _src_python)
        
        self.modules_loaded = True

    def refresh(self):
        """Refresh the module when tab is selected."""
        current = self.module_combo.currentText()
        if current == "RCM":
            self._launch_rcm()
        elif current == "BVM":
            self._launch_bvm()

    def ensure_bvm(self):
        """Build the BVM widget once and return it, without switching tabs.
        Used by the main Simulation Run so it can solve via the BVM knobs."""
        from ..modules.bvm_module import BVMModuleWidget
        if self.bvm_widget is None:
            self.bvm_widget = BVMModuleWidget(window_state=self.window_state)
            self.bvm_container.layout().addWidget(self.bvm_widget)
        else:
            self.bvm_widget.window_state = self.window_state
        return self.bvm_widget

    def _launch_bvm(self):
        """Launch the BVM module (built once, then reused)."""
        self.ensure_bvm()
        self.content_stack.setCurrentWidget(self.bvm_container)

    def _launch_rcm(self):
        """Launch the RCM interface."""
        self._setup_paths()
        self.content_stack.setCurrentWidget(self.rcm_container)
        
        # Import RCM module - it handles its own imports internally
        import RCM_module_window
        NewSimulationWindow = RCM_module_window.NewSimulationWindow
        
        if self.rcm_window:
            self.rcm_window.deleteLater()
        
        self.rcm_window = NewSimulationWindow(window_state=self.window_state)
        
        self.rcm_window.setParent(self.rcm_container)
        self.rcm_window.setWindowFlags(Qt.WindowType(0))
        self.rcm_container.layout().addWidget(self.rcm_window)
        self.rcm_window.show()

    def get_selected_module(self) -> str:
        """Get the currently selected module name."""
        return self.module_combo.currentText()
