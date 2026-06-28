from PySide6.QtWidgets import QWidget, QHBoxLayout, QDoubleSpinBox, QComboBox
from PySide6.QtCore import Signal


class UnitComboBox(QWidget):
    """Numeric input field with unit dropdown for temperature, pressure, and flow."""

    valueChanged = Signal(float)

    TEMP_UNITS = ["°C", "K", "°F"]
    PRESSURE_UNITS = ["bar", "atm", "kPa", "MPa"]
    FLOW_UNITS = ["kmol/h", "mol/h", "kg/h"]

    def __init__(self, unit_type: str = "pressure", parent=None):
        super().__init__(parent)

        self.unit_type = unit_type
        self.layout = QHBoxLayout(self)
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.spin_box = QDoubleSpinBox(self)
        self.spin_box.setDecimals(4)
        self.spin_box.setRange(-1e10, 1e10)
        self.spin_box.valueChanged.connect(self._on_value_changed)

        self.unit_combo = QComboBox(self)
        self._setup_units()

        self.layout.addWidget(self.spin_box)
        self.layout.addWidget(self.unit_combo)

        self._setup_styles()

    def _setup_units(self):
        if self.unit_type == "temperature":
            self.unit_combo.addItems(self.TEMP_UNITS)
        elif self.unit_type == "pressure":
            self.unit_combo.addItems(self.PRESSURE_UNITS)
        elif self.unit_type == "flow":
            self.unit_combo.addItems(self.FLOW_UNITS)

    def _setup_styles(self):
        self.setStyleSheet("""
            QDoubleSpinBox {
                min-width: 80px;
            }
            QComboBox {
                min-width: 70px;
            }
        """)

    def _on_value_changed(self, value):
        self.valueChanged.emit(value)

    def value(self) -> float:
        """Return the numeric value."""
        return self.spin_box.value()

    def setValue(self, value: float):
        """Set the numeric value."""
        self.spin_box.setValue(value)

    def unit(self) -> str:
        """Return the current unit."""
        return self.unit_combo.currentText()

    def setUnit(self, unit: str):
        """Set the unit."""
        index = self.unit_combo.findText(unit)
        if index >= 0:
            self.unit_combo.setCurrentIndex(index)

    def setRange(self, min_val: float, max_val: float):
        """Set the numeric range."""
        self.spin_box.setRange(min_val, max_val)

    def setDecimals(self, decimals: int):
        """Set the number of decimal places."""
        self.spin_box.setDecimals(decimals)

    def setEnabled(self, enabled: bool):
        """Enable or disable the widget."""
        self.spin_box.setEnabled(enabled)
        self.unit_combo.setEnabled(enabled)

    def convertToSI(self, value: float, unit: str) -> float:
        """Convert a value to SI units (K for temp, bar for pressure, kmol/h for flow)."""
        if self.unit_type == "temperature":
            if unit == "°C":
                return value + 273.15
            elif unit == "°F":
                return (value - 32) * 5/9 + 273.15
            return value  # K
        elif self.unit_type == "pressure":
            if unit == "atm":
                return value * 1.01325
            elif unit == "kPa":
                return value * 0.01
            elif unit == "MPa":
                return value * 10
            return value  # bar
        elif self.unit_type == "flow":
            if unit == "mol/h":
                return value / 1000
            elif unit == "kg/h":
                return value / 1000  # Approximate, assumes avg MW of 100
            return value  # kmol/h
        return value

    def convertFromSI(self, si_value: float, unit: str) -> float:
        """Convert from SI units to the specified unit."""
        if self.unit_type == "temperature":
            if unit == "°C":
                return si_value - 273.15
            elif unit == "°F":
                return (si_value - 273.15) * 9/5 + 32
            return si_value  # K
        elif self.unit_type == "pressure":
            if unit == "atm":
                return si_value / 1.01325
            elif unit == "kPa":
                return si_value * 100
            elif unit == "MPa":
                return si_value / 10
            return si_value  # bar
        elif self.unit_type == "flow":
            if unit == "mol/h":
                return si_value * 1000
            elif unit == "kg/h":
                return si_value * 1000  # Approximate
            return si_value  # kmol/h
        return si_value

    def valueInSI(self) -> float:
        """Return the value converted to SI units."""
        return self.convertToSI(self.value(), self.unit())

    def setValueInSI(self, si_value: float):
        """Set the value from SI units."""
        self.setValue(self.convertFromSI(si_value, self.unit()))
