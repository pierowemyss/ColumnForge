from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal


class SubTabBar(QFrame):
    """Vertical side navigation bar with clickable tabs aligned to top."""

    tabClicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(140)
        self.setObjectName("subTabBar")

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addStretch()

        self.tab_buttons = []
        self._setup_styles()

    def _setup_styles(self):
        self.setStyleSheet("""
            QFrame#subTabBar {
                background-color: #1a1a1a;
                border-right: 1px solid #333333;
            }
            QPushButton {
                text-align: left;
                padding: 12px 15px;
                border: none;
                background-color: transparent;
                color: #888888;
                font-size: 13px;
                font-weight: normal;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #2d2d2d;
                color: #cccccc;
            }
            QPushButton[selected="true"] {
                background-color: #2d2d2d;
                font-weight: 600;
                color: #ffffff;
                border-left: 3px solid #0078d4;
            }
        """)

    def addTab(self, name: str) -> int:
        """Add a tab button and return its index."""
        index = len(self.tab_buttons)
        btn = QPushButton(name, self)
        btn.setCheckable(False)
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda checked=False, idx=index: self._on_tab_clicked(idx))
        self.layout.insertWidget(self.layout.count() - 1, btn)  # Insert before stretch
        self.tab_buttons.append(btn)
        if index == 0:
            self._select_tab(0)
        return index

    def _on_tab_clicked(self, index: int):
        self._select_tab(index)
        self.tabClicked.emit(index)

    def _select_tab(self, index: int):
        """Set the active tab."""
        if 0 <= index < len(self.tab_buttons):
            for i, btn in enumerate(self.tab_buttons):
                if i == index:
                    btn.setProperty("selected", True)
                else:
                    btn.setProperty("selected", False)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    def setCurrentIndex(self, index: int):
        """Public method to set current tab, compatible with QTabWidget/QStackedWidget."""
        self._select_tab(index)

    def count(self) -> int:
        """Return the number of tabs."""
        return len(self.tab_buttons)

    def tabText(self, index: int) -> str:
        """Return the text of the tab at index."""
        if 0 <= index < len(self.tab_buttons):
            return self.tab_buttons[index].text()
        return ""

    def tabItem(self, index: int):
        """Return the button widget at index."""
        if 0 <= index < len(self.tab_buttons):
            return self.tab_buttons[index]
        return None
