"""QGraphicsItems for the flowsheet editor: columns, ports, edges, chips.

Items are deliberately *dumb* QGraphicsItems, not QGraphicsObjects: the scene
owns every signal, exactly as the single-column canvas had the widget emit for
the things drawn on it. An item that needs to say something calls back into
`self.scene()`. That keeps one signal surface instead of one per item, and
avoids a QObject per chip on a six-column sheet.

Nothing here mutates WindowState. The scene emits a *request* and the tab that
owns the state decides — so the picture can never drift from the model by
editing it behind the model's back.
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainterPath, QPainterPathStroker, QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QToolTip

from ..theme.palette import canvas as _canvas
from . import column_glyphs as glyph

#: Shell geometry in node-local coordinates. The shell's top-left is the item
#: origin, so every other piece is placed relative to it.
SHELL_W = 90.0
SHELL_MIN_H = 200.0
SHELL_MAX_H = 600.0
HX_W, HX_H = 56.0, 40.0
HX_DX = 70.0          # exchanger centre, right of the shell's right edge
HX_DY = 58.0          # ...and above the top / below the bottom
PORT_R = 5.0
MODULE_W, MODULE_H = 64.0, 40.0
MODULE_DX = 118.0     # module corridor, left of the shell


def shell_height(n_stages: int) -> float:
    """Taller column for more stages, clamped so 5 and 60 stages are both
    usable on one sheet."""
    return max(SHELL_MIN_H, min(SHELL_MAX_H, 24.0 + 8.0 * max(1, n_stages)))


class _SceneNotify:
    """Mixin: route a click to the scene, which owns the signals."""

    def _notify(self, method, *args):
        sc = self.scene()
        fn = getattr(sc, method, None)
        if fn is not None:
            fn(*args)


class StreamChip(QGraphicsItem, _SceneNotify):
    """A clickable stream label. `stream_id` is the WindowState key."""

    def __init__(self, unit_id, stream_id, color, parent=None, align_right=False,
                 center=False, module_id=None):
        super().__init__(parent)
        self.unit_id = unit_id
        self.stream_id = stream_id
        self.module_id = module_id      # side-section product chips open a module
        self.color = color
        self.align_right = align_right
        self.center = center
        self._w, self._h = 10.0, 16.0
        self.selected = False
        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(6)
        self._measure()

    def _measure(self):
        from PySide6.QtGui import QFontMetricsF
        fm = QFontMetricsF(QFont("Arial", 8, QFont.Bold))
        self._w = fm.horizontalAdvance(self.stream_id) + 14
        self._h = fm.height() + 6

    def boundingRect(self):
        if self.center:
            x = -self._w / 2
        elif self.align_right:
            x = -self._w
        else:
            x = 0.0
        return QRectF(x, -self._h / 2, self._w, self._h)

    def paint(self, painter, option, widget=None):
        glyph.paint_chip(painter, self.boundingRect(), self.stream_id,
                         self.color, self.selected, self.hovered)

    def hoverEnterEvent(self, event):
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        self.hovered = False
        self.update()

    def mousePressEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        self._notify("activate", self)
        event.accept()


class ModuleBadge(QGraphicsItem, _SceneNotify):
    """A side module (stripper / rectifier / pumparound / interheater) hung off
    a tray, with the lines that say what it does to the column."""

    def __init__(self, unit_id, module, shell_rect, n_stages, parent=None):
        super().__init__(parent)
        self.unit_id = unit_id
        self.module_id = module["id"]
        self.module = module
        self.shell = shell_rect
        self.n_stages = n_stages
        self.setZValue(4)
        self.setAcceptHoverEvents(True)

    @property
    def is_section(self):
        return self.module["type"] in ("Side Stripper", "Side Rectifier")

    def boundingRect(self):
        return QRectF(-MODULE_W / 2 - 22, -MODULE_H / 2 - 20,
                      MODULE_W + 44, MODULE_H + 40)

    def rect(self):
        return QRectF(-MODULE_W / 2, -MODULE_H / 2, MODULE_W, MODULE_H)

    def paint(self, painter, option, widget=None):
        r = self.rect()
        painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 2))
        painter.setBrush(QBrush(QColor(_canvas.MODULE_FILL)))
        if self.is_section:
            painter.drawRoundedRect(r, 6, 6)
            painter.setPen(QPen(QColor(_canvas.TRAY), 1, Qt.DashLine))
            for f in (0.3, 0.5, 0.7):
                y = r.top() + r.height() * f
                painter.drawLine(QPointF(r.left() + 4, y),
                                 QPointF(r.right() - 4, y))
        else:
            painter.drawRect(r)
            glyph.draw_hx_coil(painter, r, bumps=1.5)
        glyph.draw_equip_label(painter, r, self.module_id,
                               above=self.module["type"] == "Side Stripper")

    def mousePressEvent(self, event):
        event.accept()

    def mouseReleaseEvent(self, event):
        self._notify("activate", self)
        event.accept()


class PortItem(QGraphicsItem, _SceneNotify):
    """An outlet you can drag a connection from. `key` is the STABLE port key
    (see core.flowsheet.Port) — never the display label."""

    def __init__(self, unit_id, key, label, parent=None):
        super().__init__(parent)
        self.unit_id = unit_id
        self.key = key
        self.label = label
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{unit_id}.{key} — drag to another column to connect")
        self.setZValue(7)
        self._hover = False

    def boundingRect(self):
        return QRectF(-PORT_R - 2, -PORT_R - 2, 2 * PORT_R + 4, 2 * PORT_R + 4)

    def paint(self, painter, option, widget=None):
        c = QColor(_canvas.NODE_ACTIVE if self._hover else _canvas.PRODUCT)
        painter.setPen(QPen(c, 1.5))
        painter.setBrush(QBrush(c if self._hover else QColor(_canvas.CHIP_BG)))
        painter.drawEllipse(QPointF(0, 0), PORT_R, PORT_R)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        # Starting a connection, not moving anything: the scene takes over.
        self._notify("begin_connect", self, event.scenePos())
        event.accept()


class ColumnNode(QGraphicsItem, _SceneNotify):
    """One Column: shell, condenser, reboiler, its own streams and modules.

    Draggable and selectable. Its children (chips, ports, module badges) handle
    their own clicks, so clicking the *body* is what makes this the active
    column.
    """

    def __init__(self, unit_id, view_model):
        super().__init__()
        self.unit_id = unit_id
        self.vm = view_model
        self.active = False
        self.edges = []                 # ConnectionEdges touching this node
        self.setFlags(QGraphicsItem.ItemIsMovable
                      | QGraphicsItem.ItemIsSelectable
                      | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(2)
        self._build()

    # --- geometry ---------------------------------------------------------

    @property
    def n_stages(self):
        return max(1, int(self.vm.get("num_stages", 20)))

    def shell_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, SHELL_W, shell_height(self.n_stages))

    def condenser_rect(self) -> QRectF:
        s = self.shell_rect()
        return QRectF(s.right() + HX_DX - HX_W / 2, s.top() - HX_DY - HX_H / 2,
                      HX_W, HX_H)

    def reboiler_rect(self) -> QRectF:
        s = self.shell_rect()
        return QRectF(s.right() + HX_DX - HX_W / 2, s.bottom() + HX_DY - HX_H / 2,
                      HX_W, HX_H)

    def stage_y(self, stage: int) -> float:
        return glyph.stage_y(self.shell_rect(), stage, self.n_stages)

    def stage_at(self, local_y: float) -> int:
        return glyph.stage_at(self.shell_rect(), local_y, self.n_stages)

    def stage_y_scene(self, stage: int) -> float:
        return self.mapToScene(QPointF(0.0, self.stage_y(stage))).y()

    def shell_rect_scene(self) -> QRectF:
        return self.mapToScene(self.shell_rect()).boundingRect()

    def boundingRect(self):
        s = self.shell_rect()
        return QRectF(s.left() - MODULE_DX - 60, s.top() - HX_DY - 70,
                      s.width() + MODULE_DX + HX_DX + 180,
                      s.height() + 2 * HX_DY + 140)

    def has_condenser(self):
        return str(self.vm.get("condenser_type", "Total")).lower() != "none"

    def has_reboiler(self):
        return str(self.vm.get("reboiler_type", "Kettle")).lower() != "none"

    # --- children ---------------------------------------------------------

    def _build(self):
        """Chips, ports and module badges as child items, positioned in node
        coordinates so they move with the node for free."""
        s = self.shell_rect()
        self.chips = {}
        self.ports = {}
        self.badges = {}

        for stage, name in self.vm.get("feeds", []):
            chip = StreamChip(self.unit_id, name, _canvas.FEED, self,
                              align_right=True)
            chip.setPos(s.left() - 54, self.stage_y(stage))
            self.chips[name] = chip

        for stage, name, ptype in self.vm.get("products", []):
            if ptype == "distillate":
                pos = QPointF(self.condenser_rect().right() + 58,
                              self.condenser_rect().center().y())
            elif ptype == "bottoms":
                pos = QPointF(self.reboiler_rect().right() + 50,
                              self.reboiler_rect().center().y())
            else:
                pos = QPointF(s.right() + 54, self.stage_y(stage))
            chip = StreamChip(self.unit_id, name, _canvas.PRODUCT, self)
            chip.setPos(pos)
            self.chips[name] = chip

        # Ports: the stable keys core.flowsheet connects. D and B are fixed;
        # a side draw's key is its stream id, a section's is its module id.
        if self.has_condenser():
            self._add_port("D", "Distillate",
                           QPointF(self.condenser_rect().right() + 6,
                                   self.condenser_rect().center().y()))
        if self.has_reboiler():
            self._add_port("B", "Bottoms",
                           QPointF(self.reboiler_rect().right() + 6,
                                   self.reboiler_rect().center().y()))
        for stage, name, ptype in self.vm.get("products", []):
            if ptype not in ("distillate", "bottoms"):
                self._add_port(name, name, QPointF(s.right() + 8,
                                                   self.stage_y(stage)))

        self._place_modules()

    def _add_port(self, key, label, pos):
        p = PortItem(self.unit_id, key, label, self)
        p.setPos(pos)
        self.ports[key] = p

    def _place_modules(self):
        """Park every module beside the tray it hangs off, in the left corridor,
        sliding a crowded one down so two on nearby trays do not overlap. The
        draw/return lines still point at the real tray, so the picture stays
        truthful."""
        s = self.shell_rect()
        last = None
        for m in sorted(self.vm.get("modules", []), key=lambda m: m["stage"]):
            badge = ModuleBadge(self.unit_id, m, s, self.n_stages, self)
            y = self.stage_y(m["stage"])
            if last is not None:
                y = max(y, last + MODULE_H + 52)
            last = y
            badge.setPos(s.left() - MODULE_DX, y)
            self.badges[m["id"]] = badge
            if badge.is_section:
                from_bottom = m["type"] == "Side Stripper"
                label = f"{m['id']} product"
                chip = StreamChip(self.unit_id, label, _canvas.PRODUCT, self,
                                  center=True, module_id=m["id"])
                chip.setPos(s.left() - MODULE_DX,
                            y + (MODULE_H / 2 + 34 if from_bottom
                                 else -MODULE_H / 2 - 34))
                self.chips[label] = chip

    def module_badge(self, module_id):
        return self.badges.get(module_id)

    # --- painting ---------------------------------------------------------

    def paint(self, painter, option, widget=None):
        lod = option.levelOfDetailFromTransform(painter.worldTransform())
        s = self.shell_rect()

        if self.active or self.isSelected():
            painter.setPen(QPen(QColor(_canvas.NODE_ACTIVE), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(s.adjusted(-8, -8, 8, 8), 8, 8)

        if lod < 0.5:
            # Zoomed out: a silhouette and a name. Drawing trays and coils at
            # this scale turns a six-column sheet into hatching.
            painter.setPen(QPen(QColor(_canvas.SHELL_STROKE), 2))
            painter.setBrush(QBrush(QColor(_canvas.SHELL_FILL)))
            painter.drawRoundedRect(s, 8, 8)
            return

        detailed = lod > 1.5
        glyph.draw_column_shell(painter, s, n_trays=(min(self.n_stages - 1, 24)
                                                     if detailed else 4))
        if detailed:
            painter.setPen(QPen(QColor(_canvas.STAGE_LABEL)))
            painter.setFont(QFont("Arial", 6))
            for stage in range(0, self.n_stages, 5):
                y = self.stage_y(stage)
                painter.drawText(QRectF(s.right() + 2, y - 6, 22, 12),
                                 Qt.AlignLeft | Qt.AlignVCenter, str(stage))

        green = _canvas.INTERNAL
        if self.has_condenser():
            c = self.condenser_rect()
            glyph.draw_condenser(painter, c)
            glyph.draw_equip_label(painter, c, "COND")
            top = QPointF(s.center().x(), s.top())
            glyph.draw_orthogonal_path(
                painter, [top, QPointF(top.x(), c.center().y()),
                          QPointF(c.left(), c.center().y())], green)
            # reflux back to the top tray, down its own corridor
            t_split = QPointF(c.center().x(), c.bottom() + 34)
            glyph.draw_orthogonal_path(
                painter, [QPointF(c.center().x(), c.bottom()), t_split], green)
            glyph.draw_orthogonal_path(
                painter, [t_split, QPointF(s.right() + 20, t_split.y()),
                          QPointF(s.right() + 20, self.stage_y(0)),
                          QPointF(s.right(), self.stage_y(0))],
                green, "Reflux", arrow_at_end=True)
            glyph.draw_orthogonal_path(
                painter, [QPointF(c.right(), c.center().y()),
                          QPointF(c.right() + 46, c.center().y())],
                _canvas.PRODUCT, arrow_at_end=True)

        if self.has_reboiler():
            r = self.reboiler_rect()
            glyph.draw_reboiler(painter, r)
            glyph.draw_equip_label(painter, r, "REBO")
            bot = QPointF(s.center().x(), s.bottom())
            glyph.draw_orthogonal_path(
                painter, [bot, QPointF(bot.x(), r.center().y()),
                          QPointF(r.left(), r.center().y())], green)
            glyph.draw_orthogonal_path(
                painter, [QPointF(r.center().x(), r.top()),
                          QPointF(r.center().x(), s.bottom() - 5),
                          QPointF(s.right(), s.bottom() - 5)],
                green, "Boilup", arrow_at_end=True)
            glyph.draw_orthogonal_path(
                painter, [QPointF(r.right(), r.center().y()),
                          QPointF(r.right() + 38, r.center().y())],
                _canvas.PRODUCT, arrow_at_end=True)

        # feed arrows into the shell, side-draw arrows out of it
        for stage, _name in self.vm.get("feeds", []):
            y = self.stage_y(stage)
            glyph.draw_orthogonal_path(
                painter, [QPointF(s.left() - 50, y), QPointF(s.left(), y)],
                _canvas.FEED, arrow_at_end=True)
        for stage, _name, ptype in self.vm.get("products", []):
            if ptype not in ("distillate", "bottoms"):
                y = self.stage_y(stage)
                glyph.draw_orthogonal_path(
                    painter, [QPointF(s.right(), y), QPointF(s.right() + 50, y)],
                    _canvas.PRODUCT, arrow_at_end=True)

        self._paint_module_links(painter)

        # the column's name, always legible
        painter.setPen(QPen(QColor(_canvas.NODE_LABEL)))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(QRectF(s.left() - 40, s.top() - 34, s.width() + 80, 18),
                         Qt.AlignCenter, self.unit_id)

    def _paint_module_links(self, painter):
        """Draw/return (or +Q/-Q) lines from each badge to its tray."""
        s = self.shell_rect()
        green = _canvas.INTERNAL
        for m in self.vm.get("modules", []):
            badge = self.badges.get(m["id"])
            if badge is None:
                continue
            bp = badge.pos()
            r = badge.rect().translated(bp)
            draw_y = self.stage_y(m["stage"])
            ret_y = (draw_y if m.get("return_stage") is None
                     else self.stage_y(m["return_stage"]))
            right = QPointF(r.right(), r.center().y())
            if m["type"] == "Interreboiler":
                heating = (m.get("duty") or 0.0) >= 0.0
                path = [right, QPointF(s.left(), draw_y)]
                glyph.draw_orthogonal_path(
                    painter, path if heating else list(reversed(path)),
                    green, "+Q" if heating else "−Q", arrow_at_end=True)
                continue
            glyph.draw_orthogonal_path(
                painter, [QPointF(s.left(), draw_y),
                          QPointF(r.right() + 12, draw_y),
                          QPointF(r.right() + 12, r.center().y()), right],
                green, "draw", arrow_at_end=True)
            glyph.draw_orthogonal_path(
                painter, [QPointF(r.left(), r.center().y()),
                          QPointF(r.left() - 14, r.center().y()),
                          QPointF(r.left() - 14, ret_y),
                          QPointF(s.left(), ret_y)],
                green, "return", arrow_at_end=True)
            if badge.is_section:
                from_bottom = m["type"] == "Side Stripper"
                start = QPointF(r.center().x(),
                                r.bottom() if from_bottom else r.top())
                end = QPointF(start.x(), start.y() + (24 if from_bottom else -24))
                glyph.draw_orthogonal_path(painter, [start, end],
                                           _canvas.PRODUCT, arrow_at_end=True)

    # --- interaction ------------------------------------------------------

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # snap to a 10-unit grid so a sheet stays tidy without a layout pass
            return QPointF(round(value.x() / 10.0) * 10.0,
                           round(value.y() / 10.0) * 10.0)
        if change == QGraphicsItem.ItemPositionHasChanged:
            for e in self.edges:
                e.update_path()
            self._notify("node_moved", self)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        # a press that did not move is a click: make this the active column
        if (event.scenePos() - event.buttonDownScenePos(Qt.LeftButton)).manhattanLength() < 4:
            self._notify("activate", self)

    def mouseDoubleClickEvent(self, event):
        self._notify("open_unit", self.unit_id)
        event.accept()


class ConnectionEdge(QGraphicsItem, _SceneNotify):
    """A stream from one column's port to another column's stage."""

    def __init__(self, conn, src_node, dst_node, recycle=False, invalid=None):
        super().__init__()
        self.conn = conn
        self.src_node = src_node
        self.dst_node = dst_node
        self.recycle = recycle
        self.invalid = invalid          # None = fine, else the reason
        self._path = QPainterPath()
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        src_node.edges.append(self)
        dst_node.edges.append(self)
        self.update_path()

    def _endpoints(self):
        port = self.src_node.ports.get(self.conn.port)
        if port is not None:
            a = port.scenePos()
        else:
            a = self.src_node.shell_rect_scene().center()
        dst = self.dst_node.shell_rect_scene()
        b = QPointF(dst.left(), self.dst_node.stage_y_scene(self.conn.stage - 1))
        return a, b

    def update_path(self):
        self.prepareGeometryChange()
        a, b = self._endpoints()
        path = QPainterPath(a)
        dx = max(60.0, abs(b.x() - a.x()) * 0.5)
        if self.recycle:
            # bow a recycle well clear of the nodes it loops around, and go the
            # long way (over the top) so it reads as a return, not a shortcut
            lift = 140.0
            path.cubicTo(QPointF(a.x() + dx, a.y() - lift),
                         QPointF(b.x() - dx, b.y() - lift), b)
        else:
            path.cubicTo(QPointF(a.x() + dx, a.y()),
                         QPointF(b.x() - dx, b.y()), b)
        self._path = path
        self.update()

    def boundingRect(self):
        return self._path.boundingRect().adjusted(-20, -20, 20, 20)

    def shape(self):
        """A 2px line is unclickable; widen the hit area."""
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        return stroker.createStroke(self._path)

    def color(self):
        if self.invalid:
            return _canvas.EDGE_INVALID
        return _canvas.RECYCLE if self.recycle else _canvas.INTERNAL

    def paint(self, painter, option, widget=None):
        c = QColor(self.color())
        pen = QPen(c, 3 if self.isSelected() else 2)
        if self.recycle or self.invalid:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._path)

        if self._path.elementCount() >= 2:
            end = self._path.pointAtPercent(1.0)
            near = self._path.pointAtPercent(0.97)
            glyph.arrowhead(painter, near, end, self.color())

        mid = self._path.pointAtPercent(0.5)
        label = ""
        if self.recycle:
            label = "↺"
        if self.conn.split_fraction < 1.0:
            label += f" {self.conn.split_fraction:.0%}"
        if self.invalid:
            label = "⚠"
        if label:
            painter.setPen(QPen(c))
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            painter.drawText(QRectF(mid.x() - 30, mid.y() - 18, 60, 16),
                             Qt.AlignCenter, label.strip())

    def hoverEnterEvent(self, event):
        if self.invalid:
            QToolTip.showText(event.screenPos(), self.invalid)
        elif self.toolTip():
            QToolTip.showText(event.screenPos(), self.toolTip())

    def mouseReleaseEvent(self, event):
        self._notify("activate", self)
        event.accept()


class DragEdge(QGraphicsItem):
    """The rubber line while a connection is being dragged out of a port."""

    def __init__(self, origin: QPointF):
        super().__init__()
        self.origin = origin
        self.tip = origin
        self.ok = True
        self.setZValue(20)

    def set_tip(self, pos: QPointF, ok: bool):
        self.prepareGeometryChange()
        self.tip = pos
        self.ok = ok
        self.update()

    def boundingRect(self):
        return QRectF(self.origin, self.tip).normalized().adjusted(-12, -12, 12, 12)

    def paint(self, painter, option, widget=None):
        c = QColor(_canvas.INTERNAL if self.ok else _canvas.EDGE_INVALID)
        pen = QPen(c, 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(self.origin, self.tip)
        glyph.arrowhead(painter, self.origin, self.tip, c.name())
