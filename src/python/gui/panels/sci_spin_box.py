"""A QDoubleSpinBox that speaks scientific notation.

The stock QDoubleSpinBox rounds to a fixed `decimals` and its parser rejects
"4.35e-5", so tiny process values (tolerances, trace mole fractions, rate
constants) can be neither typed nor read. This subclass:

  * accepts scientific input on the way in (validate/valueFromText), and
  * renders with %g on the way out, so small/large magnitudes show as
    "4.35e-05" instead of collapsing to "0.0000".

# ponytail: standard Qt subclass pattern. Swap to QLineEdit + a
# QDoubleValidator(ScientificNotation) only if the step buttons misbehave.
"""

import re

from PySide6.QtWidgets import QDoubleSpinBox
from PySide6.QtGui import QValidator

# A fully-formed float literal (optionally signed, optional exponent).
_FLOAT_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
# A prefix of one that's still being typed ("-", "1.", "4.35e", "1e-", ...).
_PARTIAL_RE = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)?([eE][-+]?)?$")

# Generous internal precision so parsing never clips; display is via %g, so the
# usual `setDecimals(4)` calls only ever cost display digits, not stored value.
_PRECISION = 12


class SciDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super().setDecimals(_PRECISION)

    # Keep full precision regardless of what callers ask for — %g handles
    # readable display, so a requested `decimals` would only reintroduce the
    # rounding this widget exists to avoid.
    def setDecimals(self, _decimals):
        super().setDecimals(_PRECISION)

    def _strip(self, text):
        """Drop the spin box's prefix/suffix so the numeric core is left."""
        p, s = self.prefix(), self.suffix()
        if p and text.startswith(p):
            text = text[len(p):]
        if s and text.endswith(s):
            text = text[:-len(s)]
        return text.strip()

    def validate(self, text, pos):
        core = self._strip(text)
        if _FLOAT_RE.match(core):
            state = QValidator.Acceptable
        elif core == "" or _PARTIAL_RE.match(core):
            state = QValidator.Intermediate
        else:
            state = QValidator.Invalid
        return state, text, pos

    def valueFromText(self, text):
        try:
            return float(self._strip(text))
        except ValueError:
            return 0.0

    def textFromValue(self, value):
        return f"{value:.{_PRECISION}g}"


def fmt(x):
    """Format a number for display, sci-notation for small/large magnitudes.

    Mirrors SciDoubleSpinBox's display so result tables don't round trace
    values to "0.0000". Non-numbers pass through as-is.
    """
    try:
        return f"{float(x):.{_PRECISION}g}"
    except (TypeError, ValueError):
        return str(x)


def _demo():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    box = SciDoubleSpinBox()
    box.setRange(-1e12, 1e12)

    # Input: scientific notation parses.
    assert box.valueFromText("4.35e-5") == 4.35e-5
    assert box.valueFromText("-1.2E3") == -1200.0
    # Display: small value keeps sig figs, doesn't round to zero.
    assert box.textFromValue(4.35e-5) == "4.35e-05"
    assert box.textFromValue(1.5) == "1.5"
    assert box.textFromValue(0.0) == "0"
    # Prefix/suffix are stripped before parsing.
    box.setSuffix(" kW")
    assert box.valueFromText("-1500 kW") == -1500.0
    # Validator states.
    assert box.validate("4.35e-5", 0)[0] == QValidator.Acceptable
    assert box.validate("1e-", 0)[0] == QValidator.Intermediate
    assert box.validate("abc", 0)[0] == QValidator.Invalid
    # fmt() helper.
    assert fmt(4.35e-5) == "4.35e-05"
    assert fmt("n/a") == "n/a"
    print("SciDoubleSpinBox demo OK")


if __name__ == "__main__":
    _demo()
