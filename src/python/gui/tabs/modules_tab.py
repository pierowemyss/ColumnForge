import sys
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QStackedLayout
)
from PySide6.QtCore import Qt


class ModulesTab(QWidget):
    """Modules tab for side features and additional functionality."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.window_state = None
        self.rcm_window = None
        self.rcm_placeholder = None
        self.bvm_widget = None
        self.rbm_widget = None
        self.fug_widget = None
        self.txy_widget = None
        self.pure_widget = None
        self.phase_eq_widget = None
        self.modules_loaded = False
        
        self._setup_ui()
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
            "RBM (Rectification Bodies)",
            "Shortcut (FUG)",
            "Txy/Pxy",
            "Pure Components",
            "Phase EQ",
        ])
        header_layout.addWidget(self.module_combo)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        self.content_stack = QStackedLayout()
        main_layout.addLayout(self.content_stack)

        self.rcm_container = QWidget()
        rcm_layout = QVBoxLayout(self.rcm_container)
        rcm_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.rcm_container)

        self.bvm_container = QWidget()
        bvm_layout = QVBoxLayout(self.bvm_container)
        bvm_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.bvm_container)

        self.rbm_container = QWidget()
        rbm_layout = QVBoxLayout(self.rbm_container)
        rbm_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.rbm_container)

        self.fug_container = QWidget()
        fug_layout = QVBoxLayout(self.fug_container)
        fug_layout.setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.fug_container)

        self.txy_container = QWidget()
        QVBoxLayout(self.txy_container).setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.txy_container)

        self.pure_container = QWidget()
        QVBoxLayout(self.pure_container).setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.pure_container)

        self.phase_eq_container = QWidget()
        QVBoxLayout(self.phase_eq_container).setContentsMargins(0, 0, 0, 0)
        self.content_stack.addWidget(self.phase_eq_container)

    def _connect_signals(self):
        self.module_combo.currentTextChanged.connect(self._on_module_changed)

    def set_window_state(self, window_state):
        """Set the window state reference.

        Called on startup and after every File->Load, so it is also where the
        BVM panel's one-shot restore/entrainer-prefill is re-armed: those flags
        latch on first show, and without this a second .colx keeps the first
        file's BVM knobs (load_from_dict reuses the same WindowState object, so
        the panel can't tell the state changed on its own).
        """
        self.window_state = window_state
        for panel in (self.bvm_widget, self.rbm_widget):
            if panel is not None:
                panel._restored = False
                panel._entrainer_prefilled = False
        self._on_module_changed(self.module_combo.currentText())

    def _on_module_changed(self, module_name: str):
        """Handle module selection change - auto-launches the selected module."""
        self._dispatch(module_name)

    def _dispatch(self, module_name: str):
        launchers = {
            "RCM": self._launch_rcm,
            "BVM": self._launch_bvm,
            "RBM (Rectification Bodies)": self._launch_rbm,
            "Shortcut (FUG)": self._launch_fug,
            "Txy/Pxy": self._launch_txy,
            "Pure Components": self._launch_pure,
            "Phase EQ": self._launch_phase_eq,
        }
        launch = launchers.get(module_name)
        if launch:
            launch()

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
        self._dispatch(self.module_combo.currentText())

    def _launch_fug(self):
        """Launch the FUG shortcut module (built once, then reused)."""
        from ..modules.fug_module import FUGModuleWidget
        if self.fug_widget is None:
            self.fug_widget = FUGModuleWidget(window_state=self.window_state)
            self.fug_container.layout().addWidget(self.fug_widget)
        else:
            self.fug_widget.window_state = self.window_state
            self.fug_widget._rebuild_key_combos()
        self.content_stack.setCurrentWidget(self.fug_container)

    def _launch_txy(self):
        """Launch the Txy/Pxy diagram module (built once, then reused)."""
        from ..modules.txy_module import TxyModuleWidget
        if self.txy_widget is None:
            self.txy_widget = TxyModuleWidget(window_state=self.window_state)
            self.txy_container.layout().addWidget(self.txy_widget)
        else:
            self.txy_widget.window_state = self.window_state
            self.txy_widget.refresh()
        self.content_stack.setCurrentWidget(self.txy_container)

    def _launch_pure(self):
        """Launch the Pure Components browser (built once, then reused)."""
        from ..modules.pure_components_module import PureComponentsModuleWidget
        if self.pure_widget is None:
            self.pure_widget = PureComponentsModuleWidget(
                window_state=self.window_state)
            self.pure_container.layout().addWidget(self.pure_widget)
        else:
            self.pure_widget.window_state = self.window_state
        self.content_stack.setCurrentWidget(self.pure_container)

    def _launch_phase_eq(self):
        """Launch the Phase EQ flash module (built once, then reused)."""
        from ..modules.phase_eq_module import PhaseEQModuleWidget
        if self.phase_eq_widget is None:
            self.phase_eq_widget = PhaseEQModuleWidget(
                window_state=self.window_state)
            self.phase_eq_container.layout().addWidget(self.phase_eq_widget)
        else:
            self.phase_eq_widget.window_state = self.window_state
            self.phase_eq_widget.refresh()
        self.content_stack.setCurrentWidget(self.phase_eq_container)

    def _launch_bvm(self):
        """Launch the BVM module (built once, then reused)."""
        from ..modules.bvm_module import BVMModuleWidget
        if self.bvm_widget is None:
            self.bvm_widget = BVMModuleWidget(window_state=self.window_state)
            self.bvm_container.layout().addWidget(self.bvm_widget)
        else:
            self.bvm_widget.window_state = self.window_state
            self.bvm_widget.reload_from_state()
        self.content_stack.setCurrentWidget(self.bvm_container)

    def _launch_rbm(self):
        """Launch the RBM module (built once, then reused)."""
        from ..modules.rbm_module import RBMModuleWidget
        if self.rbm_widget is None:
            self.rbm_widget = RBMModuleWidget(window_state=self.window_state)
            self.rbm_container.layout().addWidget(self.rbm_widget)
        else:
            self.rbm_widget.window_state = self.window_state
            self.rbm_widget.reload_from_state()
        self.content_stack.setCurrentWidget(self.rbm_container)

    def _launch_rcm(self):
        """Launch the RCM interface, or a build hint if its library is missing."""
        self._setup_paths()
        self.content_stack.setCurrentWidget(self.rcm_container)

        if self.rcm_window:
            self.rcm_window.deleteLater()
            self.rcm_window = None

        # RCM needs its compiled library (RCM_solver.so), which the repo ships
        # prebuilt for x86_64 only — on other architectures the load fails. RCM
        # is the *first* entry in the module combo, so it auto-launches when the
        # tab is opened: an unguarded failure here takes down the whole app.
        try:
            err = self._rcm_library_error()
            if err:
                raise OSError(err)
            import RCM_module_window
            self.rcm_window = RCM_module_window.NewSimulationWindow(
                window_state=self.window_state)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "RCM module unavailable; showing build hint")
            self._show_rcm_unavailable()
            return

        if self.rcm_placeholder is not None:
            self.rcm_placeholder.hide()
        self.rcm_window.setParent(self.rcm_container)
        self.rcm_window.setWindowFlags(Qt.WindowType(0))
        self.rcm_container.layout().addWidget(self.rcm_window)
        self.rcm_window.show()

    def _rcm_library_error(self):
        """Reason RCM's native library won't load, or None if it will.

        freeRCM's solver.py only CDLLs when a map is actually computed, so
        importing the module proves nothing: without this probe a bad library
        surfaces as a traceback the first time the user hits Run. The repo
        ships RCM_solver.so built for x86_64 only.
        """
        import ctypes
        _current_dir = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(_current_dir))))
        lib_dir = os.path.join(_repo_root, 'src', 'side_features',
                               'freeRCM', 'lib')
        try:
            # dependency order matters: minpack first, then the solver
            ctypes.CDLL(os.path.join(lib_dir, 'libminpack.so'))
            ctypes.CDLL(os.path.join(lib_dir, 'RCM_solver.so'))
        except OSError as exc:
            return str(exc)
        return None

    def _show_rcm_unavailable(self):
        """Stand in for the RCM view so the rest of the tab stays usable."""
        if self.rcm_placeholder is None:
            self.rcm_placeholder = QLabel(
                "Compile RCM_solv.c to use this module.\n\n"
                "cd src/side_features/freeRCM/build && make",
                self.rcm_container)
            self.rcm_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.rcm_placeholder.setWordWrap(True)
            self.rcm_container.layout().addWidget(self.rcm_placeholder)
        self.rcm_placeholder.show()

    def get_selected_module(self) -> str:
        """Get the currently selected module name."""
        return self.module_combo.currentText()
