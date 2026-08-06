"""PFD glyphs: the column shell, its heat exchangers, chips and arrows.

Lifted verbatim out of the old single-column `column_overview_panel`, which
painted them straight onto a QWidget. They are pure — painter in, QRectF in,
pixels out, no widget state — so the flowsheet scene draws exactly the same
column it always did, just once per node instead of once per window.

Everything here works in whatever coordinate system the caller has set up. The
scene keeps a node's shell at the origin of its own item coordinates, so these
never see scene coordinates and never need to know about zoom.
"""

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen, QPolygonF

from ..theme.palette import canvas as _canvas


def stage_y(rect: QRectF, stage: int, n_stages: int) -> float:
    """Centre-line of a stage inside a shell rect.

    Stages are 0-based from the top: 0 = distillate, n_stages-1 = bottoms —
    the app-wide convention (see CLAUDE.md).
    """
    frac = (stage + 0.5) / max(1, n_stages)
    return rect.top() + rect.height() * frac


def stage_at(rect: QRectF, y: float, n_stages: int) -> int:
    """Inverse of `stage_y`: which stage a point lands on. Used to turn a drop
    on a column into a feed stage."""
    if rect.height() <= 0:
        return 0
    frac = (y - rect.top()) / rect.height()
    return max(0, min(n_stages - 1, int(frac * n_stages)))


def draw_column_shell(painter, rect: QRectF, n_trays: int = 4):
    """Capsule shell with dashed tray lines."""
    painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 2))
    painter.setBrush(QBrush(QColor(_canvas.SHELL_FILL)))

    path = QPainterPath()
    cap_h = rect.width() / 2
    path.moveTo(rect.left(), rect.top() + cap_h)
    path.lineTo(rect.left(), rect.bottom() - cap_h)
    path.arcTo(rect.left(), rect.bottom() - rect.width(),
               rect.width(), rect.width(), 180, 180)
    path.lineTo(rect.right(), rect.top() + cap_h)
    path.arcTo(rect.left(), rect.top(), rect.width(), rect.width(), 0, 180)
    path.closeSubpath()
    painter.drawPath(path)

    if n_trays > 0:
        painter.setPen(QPen(QColor(_canvas.TRAY), 1, Qt.DashLine))
        for i in range(1, n_trays + 1):
            y = rect.top() + rect.height() * i / (n_trays + 1)
            painter.drawLine(QPointF(rect.left() + 5, y),
                             QPointF(rect.right() - 5, y))


def draw_hx_coil(painter, rect: QRectF, bumps: float = 2):
    """The serpentine tube that marks a heat exchanger, drawn inside rect."""
    x0 = rect.left() + rect.width() * 0.15
    x1 = rect.right() - rect.width() * 0.15
    mid = rect.center().y()
    amp = rect.height() * 0.22
    n = 24
    path = QPainterPath()
    path.moveTo(x0, mid)
    for i in range(1, n + 1):
        t = i / n
        path.lineTo(x0 + (x1 - x0) * t,
                    mid - amp * math.sin(2 * math.pi * bumps * t))
    painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 1.5))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(path)


def draw_condenser(painter, rect: QRectF, coil: bool = True):
    painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 2))
    painter.setBrush(QBrush(QColor(_canvas.COND_FILL)))
    painter.drawEllipse(rect)
    if coil:
        draw_hx_coil(painter, rect)


def draw_reboiler(painter, rect: QRectF, coil: bool = True):
    painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 2))
    painter.setBrush(QBrush(QColor(_canvas.REBO_FILL)))
    painter.drawRoundedRect(rect, 10, 10)
    if coil:
        draw_hx_coil(painter, rect)


def draw_equip_label(painter, rect: QRectF, text: str, above: bool = False):
    painter.setPen(QPen(QColor(_canvas.LABEL)))
    painter.setFont(QFont("Arial", 8, QFont.Bold))
    y = rect.top() - 16 if above else rect.bottom() + 2
    painter.drawText(QRectF(rect.left() - 20, y, rect.width() + 40, 14),
                     Qt.AlignCenter, text)


def chip_rect(painter, anchor: QPointF, label: str, align_right=False,
              center=False) -> QRectF:
    """Where a chip of this label sits, anchored at the arrow end."""
    painter.setFont(QFont("Arial", 8, QFont.Bold))
    fm = painter.fontMetrics()
    w = fm.horizontalAdvance(label) + 14
    h = fm.height() + 6
    if center:
        x = anchor.x() - w / 2
    elif align_right:
        x = anchor.x() - w
    else:
        x = anchor.x()
    return QRectF(x, anchor.y() - h / 2, w, h)


def paint_chip(painter, rect: QRectF, label: str, color_hex: str,
               selected=False, hover=False):
    """A rounded stream-label chip. Hover tints it, selection fills it."""
    painter.setFont(QFont("Arial", 8, QFont.Bold))
    color = QColor(color_hex)
    if selected:
        bg = QColor(color)
    elif hover:
        bg = QColor(color)
        bg.setAlpha(50)
    else:
        bg = QColor(_canvas.CHIP_BG)
    painter.setPen(QPen(color, 2 if (selected or hover) else 1))
    painter.setBrush(QBrush(bg))
    painter.drawRoundedRect(rect, 4, 4)
    painter.setPen(QPen(QColor(_canvas.CHIP_TEXT_SELECTED if selected
                               else _canvas.CHIP_TEXT)))
    painter.drawText(rect, Qt.AlignCenter, label)


def arrowhead(painter, start: QPointF, end: QPointF, color_hex: str,
              size: float = 8.0):
    """Filled arrowhead at `end`, pointing along start -> end."""
    angle = math.atan2(end.y() - start.y(), end.x() - start.x())
    p1 = QPointF(end.x() - size * math.cos(angle - math.pi / 6),
                 end.y() - size * math.sin(angle - math.pi / 6))
    p2 = QPointF(end.x() - size * math.cos(angle + math.pi / 6),
                 end.y() - size * math.sin(angle + math.pi / 6))
    painter.setPen(QPen(QColor(color_hex), 2))
    painter.setBrush(QBrush(QColor(color_hex)))
    # QPolygonF, not three loose points: PySide6's drawPolygon takes one
    # polygon, and the varargs form C++ allows is not bound.
    painter.drawPolygon(QPolygonF([end, p1, p2]))


def draw_orthogonal_path(painter, points, color_hex: str, label: str = "",
                         arrow_at_end: bool = False):
    """Right-angled run between points, the PFD idiom for a pipe."""
    painter.setPen(QPen(QColor(color_hex), 2))
    painter.setBrush(Qt.NoBrush)
    for a, b in zip(points, points[1:]):
        painter.drawLine(a, b)
    if arrow_at_end and len(points) >= 2:
        arrowhead(painter, points[-2], points[-1], color_hex)
    if label:
        painter.setFont(QFont("Arial", 7))
        painter.setPen(QPen(QColor(_canvas.LABEL)))
        mid = points[len(points) // 2]
        painter.drawText(QRectF(mid.x() - 30, mid.y() - 16, 60, 12),
                         Qt.AlignCenter, label)
