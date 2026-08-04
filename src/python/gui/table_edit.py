#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-entry helpers shared by every table in the app.

Three bugs live here, all of the "my numbers vanished" family:

* **Stray editors.** Clicking a combo box does not move keyboard focus out of
  an open cell editor, so a half-typed NRTL coefficient was still sitting in
  the editor when the Table: combo repopulated the grid — the value landed in
  the *next* table, or nowhere at all. ``install()`` commits the editor on any
  mouse press outside its view, before the click's own handlers run.
* **Silent parse failures.** ``float()`` rejects the unicode minus, NBSPs and
  decimal commas you get from pasting out of a paper or a spreadsheet, and the
  old callers turned that into ``None`` — the cell blanked on the next reload.
  ``parse_number`` accepts them; callers reject what it can't read *visibly*.
* **Lossy display.** ``f"{v:g}"`` showed 1234.56789 as 1234.57, and the next
  edit anywhere in the table wrote that back to state. ``fmt_number`` keeps
  12 significant digits — enough for any correlation coefficient, still short
  of float noise.

``install(app)`` also gives every item view Ctrl+C / Ctrl+V over the whole
selection (Qt gives item views no clipboard support at all); paste is skipped
on read-only views.

Self-check: QT_QPA_PLATFORM=offscreen python -m gui.table_edit
"""

from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemDelegate, QAbstractItemView, QApplication, QWidget,
)

# Characters that mean "minus" or "digit grouping" in copied text but are not
# what float() expects.
_CLEAN = str.maketrans({
    "−": "-", "–": "-", "—": "-", "‒": "-",  # unicode dashes
    " ": "", " ": "", " ": "", " ": "", "\t": "",  # spaces
    "'": "", "_": "",                                            # group marks
    "D": "E", "d": "e",                                          # Fortran exponent
})


def parse_number(text, default=None):
    """float() that survives text pasted out of a paper or a spreadsheet.

    Returns `default` when the text is not a number at all, so callers can tell
    "empty" from "garbage" and refuse the garbage out loud.
    """
    if text is None:
        return default
    s = str(text).strip().translate(_CLEAN)
    if not s:
        return default
    if "," in s:
        if "." in s:
            s = s.replace(",", "")               # 1,234.5 -> thousands marks
        else:
            head, _, tail = s.rpartition(",")
            # "1,5" is a decimal comma; "1,234" reads as grouping (ambiguous —
            # the 3-digit group is the tie-break, same as every spreadsheet).
            s = s.replace(",", "") if len(tail) == 3 and head else s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def fmt_number(value, sig=12):
    """Cell text for a float, round-tripping through parse_number unchanged."""
    return "" if value is None else f"{value:.{sig}g}"


def commit_open_editor(clicked=None):
    """Commit the cell editor the user is walking away from.

    The open editor is whatever has keyboard focus — the user is typing in it.
    `clicked` is the widget being pressed; presses inside the editing view are
    left to Qt.
    """
    editor = QApplication.focusWidget()
    if editor is None:
        return
    view = _editing_view(editor)
    if view is None:
        return
    if clicked is not None and (clicked is view or view.isAncestorOf(clicked)):
        return                               # still inside the table: Qt's problem
    view.commitData(editor)
    view.closeEditor(editor, QAbstractItemDelegate.NoHint)


def _editing_view(editor):
    """The item view an open cell editor belongs to, else None."""
    parent = editor.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractItemView):
            return parent if parent.state() == QAbstractItemView.EditingState else None
        parent = parent.parentWidget()
    return None


def _focused_view():
    w = QApplication.focusWidget()
    return w if isinstance(w, QAbstractItemView) else None


def copy_selection(view):
    """Selection -> TSV on the clipboard (rows by line, columns by tab)."""
    idxs = view.selectedIndexes()
    if not idxs and view.currentIndex().isValid():
        idxs = [view.currentIndex()]
    if not idxs:
        return False
    rows = sorted({i.row() for i in idxs})
    cols = sorted({i.column() for i in idxs})
    cells = {(i.row(), i.column()): i.data(Qt.DisplayRole) for i in idxs}
    QApplication.clipboard().setText("\n".join(
        "\t".join(str(cells.get((r, c), "") or "") for c in cols) for r in rows))
    return True


def paste_selection(view):
    """TSV from the clipboard into the cells right of/below the anchor cell.

    Anchored at the top-left of the selection (Excel's rule), clipped to the
    existing grid — pasting never grows a table.
    """
    if view.editTriggers() == QAbstractItemView.NoEditTriggers:
        return False                      # read-only view: copy only
    text = QApplication.clipboard().text()
    if not text:
        return False
    idxs = view.selectedIndexes()
    if idxs:
        r0, c0 = min(i.row() for i in idxs), min(i.column() for i in idxs)
    elif view.currentIndex().isValid():
        r0, c0 = view.currentIndex().row(), view.currentIndex().column()
    else:
        return False
    model = view.model()
    for dr, line in enumerate(text.replace("\r\n", "\n").rstrip("\n").split("\n")):
        for dc, value in enumerate(line.split("\t")):
            idx = model.index(r0 + dr, c0 + dc)
            if idx.isValid() and model.flags(idx) & Qt.ItemIsEditable:
                model.setData(idx, value.strip(), Qt.EditRole)
    return True


class _TableAssist(QObject):
    """One application-wide filter — no per-table wiring to forget."""

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype == QEvent.MouseButtonPress:
            commit_open_editor(obj if isinstance(obj, QWidget) else None)
        elif etype == QEvent.KeyPress:
            view = _focused_view()
            if view is not None:
                if event.matches(QKeySequence.Copy):
                    return copy_selection(view)
                if event.matches(QKeySequence.Paste):
                    return paste_selection(view)
        return False


_assist = None


def install(app):
    """Wire the filter into a QApplication. Returns it (keep a reference)."""
    global _assist
    if _assist is None:
        _assist = _TableAssist(app)
        app.installEventFilter(_assist)
    return _assist


def _demo():
    """Self-check: python -m gui.table_edit (needs QT_QPA_PLATFORM=offscreen)."""
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

    assert parse_number("1.5") == 1.5
    assert parse_number("−1234.5") == -1234.5      # unicode minus
    assert parse_number("1 234.5") == 1234.5       # NBSP grouping
    assert parse_number("1,5") == 1.5                   # decimal comma
    assert parse_number("1,234") == 1234.0              # thousands group
    assert parse_number("1,234.5") == 1234.5
    assert parse_number("-7.6521D+03") == -7652.1       # Fortran exponent
    assert parse_number("") is None and parse_number("abc") is None
    assert parse_number("abc", 0.0) == 0.0
    for v in (1234.56789, -0.000123456789, 5.93e-06, 0.1):
        assert parse_number(fmt_number(v)) == v, v
    assert fmt_number(None) == ""

    app = QApplication.instance() or QApplication([])
    install(app)

    t = QTableWidget(2, 2)
    for r in range(2):
        for c in range(2):
            t.setItem(r, c, QTableWidgetItem(f"{r}{c}"))
    t.selectAll()
    assert copy_selection(t)
    assert QApplication.clipboard().text() == "00\t01\n10\t11"

    QApplication.clipboard().setText("7\t8\n9\t10")
    t.setCurrentCell(0, 0)
    assert paste_selection(t)
    assert [t.item(r, c).text() for r in range(2) for c in range(2)] == \
        ["7", "8", "9", "10"]

    # paste into a cell with no item at all, and past the right edge (clipped)
    t.setItem(1, 1, None)
    QApplication.clipboard().setText("42\t99")
    t.setCurrentCell(1, 1)
    paste_selection(t)
    assert t.item(1, 1).text() == "42"

    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    QApplication.clipboard().setText("666")
    assert not paste_selection(t), "read-only table must refuse a paste"
    assert t.item(1, 1).text() == "42"

    print("gui.table_edit self-check OK")


if __name__ == "__main__":
    _demo()
