from PySide6.QtWidgets import QWidget, QLineEdit
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QIntValidator, QPainterPath

import math


class DraggableItem:
    """Base class for draggable items on the column diagram."""

    def __init__(self, element_type: str, display_name: str, x: float, y: float):
        self.element_type = element_type
        self.display_name = display_name
        self.x = x  # Percentage of width
        self.y = y  # Percentage of height
        self.offset = QPointF(0, 0) # Pixel offset from calculated center
        self.rect = QRectF()
        self.width = 100
        self.height = 50

    def contains(self, point: QPointF) -> bool:
        return self.rect.contains(point)

    def get_center(self, canvas_w: int, canvas_h: int) -> QPointF:
        base_x = canvas_w * self.x
        base_y = canvas_h * self.y
        return QPointF(base_x + self.offset.x(), base_y + self.offset.y())

    def update_rect(self, canvas_w: int, canvas_h: int):
        center = self.get_center(canvas_w, canvas_h)
        self.rect = QRectF(center.x() - self.width/2, center.y() - self.height/2, self.width, self.height)


class ColumnOverviewCanvas(QWidget):
    """Interactive canvas showing an Aspen-style clickable column diagram.

    Every stream (feed, distillate, bottoms, side draw) is a clickable label
    chip: single-click selects it and emits streamClicked with the stream id;
    the condenser/reboiler emit elementClicked the same way. Equipment drags;
    a press that moves is a drag, a press that doesn't is a click.
    """

    elementClicked = Signal(str)  # "condenser" | "reboiler" on click
    streamClicked = Signal(str)   # stream id on click
    specsChanged = Signal()

    # material streams (feeds in, products out) vs internal recycle lines
    FEED_COLOR = "#1971c2"
    PRODUCT_COLOR = "#e03131"
    INTERNAL_COLOR = "#2f9e44"

    def __init__(self, parent=None):
        super().__init__(parent)

        self.num_stages = 20
        self.feed_stage = 10
        self.condenser_type = "Total"
        self.reboiler_type = "Kettle"
        self.window_state = None

        self.items = {} # element_type -> DraggableItem
        self.feed_data = []
        self.product_data = []
        self.module_data = []

        self.stream_hits = {}        # stream id -> chip QRectF (rebuilt per paint)
        self.selected_stream = None
        self.hover_stream = None

        self.dragging_item = None
        self.last_mouse_pos = QPointF()
        self._press_pos = QPointF()
        self._press_moved = False

        self._init_items()
        self._init_stage_input()

        self.setMinimumSize(400, 600)
        self.setStyleSheet("background-color: #f8f9fa;")
        self.setMouseTracking(True)

    def set_window_state(self, window_state):
        self.window_state = window_state
        if window_state:
            self.set_num_stages(window_state.num_stages)

    def _on_stage_input_changed(self):
        try:
            val = int(self.stage_input.text())
            if val != self.num_stages:
                self.num_stages = val
                if self.window_state:
                    self.window_state.num_stages = val
                    self.window_state.mark_modified()
                self.specsChanged.emit()
                self.update()
        except ValueError:
            pass
        self.stage_input.clearFocus()

    def set_column_config(self, num_stages: int, feed_stage: int,
                          condenser_type: str, reboiler_type: str):
        self.num_stages = max(2, num_stages)
        self.feed_stage = max(0, min(feed_stage, self.num_stages - 1))
        self.condenser_type = condenser_type
        self.reboiler_type = reboiler_type
        self.stage_input.setText(str(self.num_stages))
        self.update()

    def set_streams(self, feeds: list, products: list, modules: list):
        self.feed_data = feeds
        self.product_data = products
        self.module_data = modules
        
        for _, name, mtype in modules:
            m_id = f"module_{name}"
            if m_id not in self.items:
                self.items[m_id] = DraggableItem(m_id, name, 0.2, 0.5)
                self.items[m_id].width = 50
                self.items[m_id].height = 40
        
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        self.stream_hits = {}

        for item in self.items.values():
            item.update_rect(w, h)

        col = self.items["column"]
        self._draw_column_shell(painter, col)
        
        input_pos = col.rect.center()
        self.stage_input.move(input_pos.x() - self.stage_input.width()/2, 
                             input_pos.y() - self.stage_input.height()/2)
        if not self.stage_input.isVisible():
            self.stage_input.show()
            
        cond = self.items["condenser"]
        self._draw_condenser(painter, cond)
        
        rebo = self.items["reboiler"]
        self._draw_reboiler(painter, rebo)
        
        for item_id, item in self.items.items():
            if item_id.startswith("module_"):
                self._draw_module(painter, item)
                
        self._draw_all_streams(painter)

    def _draw_column_shell(self, painter, col):
        rect = col.rect
        painter.setPen(QPen(QColor("#495057"), 2))
        painter.setBrush(QBrush(QColor("#e9ecef")))
        
        path = QPainterPath()
        cap_h = rect.width() / 2
        path.moveTo(rect.left(), rect.top() + cap_h)
        path.lineTo(rect.left(), rect.bottom() - cap_h)
        path.arcTo(rect.left(), rect.bottom() - rect.width(), rect.width(), rect.width(), 180, 180)
        path.lineTo(rect.right(), rect.top() + cap_h)
        path.arcTo(rect.left(), rect.top(), rect.width(), rect.width(), 0, 180)
        path.closeSubpath()
        painter.drawPath(path)
        
        painter.setPen(QPen(QColor("#adb5bd"), 1, Qt.DashLine))
        y_steps = [0.2, 0.4, 0.6, 0.8]
        for step in y_steps:
            y = rect.top() + rect.height() * step
            painter.drawLine(rect.left() + 5, y, rect.right() - 5, y)

    def _draw_condenser(self, painter, item):
        rect = item.rect
        painter.setPen(QPen(QColor("#495057"), 2))
        painter.setBrush(QBrush(QColor("#ffec99")))
        
        painter.drawEllipse(rect)
        painter.drawLine(rect.topLeft() + QPointF(10, 10), rect.bottomRight() - QPointF(10, 10))
        painter.drawLine(rect.topRight() + QPointF(-10, 10), rect.bottomLeft() - QPointF(-10, -10))
        
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, "COND")

    def _draw_reboiler(self, painter, item):
        rect = item.rect
        painter.setPen(QPen(QColor("#495057"), 2))
        painter.setBrush(QBrush(QColor("#ffc9c9")))
        
        painter.drawRoundedRect(rect, 10, 10)
        painter.drawLine(rect.left() + 10, rect.top(), rect.left() + 10, rect.bottom())
        
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, "REBO")

    def _draw_module(self, painter, item):
        rect = item.rect
        painter.setPen(QPen(QColor("#495057"), 2))
        painter.setBrush(QBrush(QColor("#a5d8ff")))
        painter.drawRect(rect)
        
        painter.setFont(QFont("Arial", 8))
        painter.drawText(rect, Qt.AlignCenter, item.display_name)

    def _stage_to_y(self, col_rect, stage):
        # Stages are 0-based from the top: 0 = distillate, num_stages-1 = bottoms.
        frac = (stage + 0.5) / self.num_stages
        return col_rect.top() + col_rect.height() * frac

    def _draw_all_streams(self, painter):
        col_rect = self.items["column"].rect
        cond_rect = self.items["condenser"].rect
        rebo_rect = self.items["reboiler"].rect

        green = self.INTERNAL_COLOR

        # 1. Column to Condenser (Vapor Line)
        col_top_center = QPointF(col_rect.center().x(), col_rect.top())
        cond_center = cond_rect.center()

        vapor_path = [
            col_top_center,
            QPointF(col_top_center.x(), cond_center.y()),
            cond_center
        ]
        self._draw_orthogonal_path(painter, vapor_path, "", green)

        # 2. Condenser Output (T-split)
        cond_bottom = QPointF(cond_center.x(), cond_rect.bottom())
        t_split_y = cond_bottom.y() + 20
        t_split_point = QPointF(cond_bottom.x(), t_split_y)

        painter.setPen(QPen(QColor(green), 2))
        painter.drawLine(cond_bottom, t_split_point)

        # Reflux branch: back to the top tray (stage num_stages - 1) from the RIGHT
        top_y = self._stage_to_y(col_rect, self.num_stages - 1)
        reflux_path = [
            t_split_point,
            QPointF(col_rect.right() + 20, t_split_y),
            QPointF(col_rect.right() + 20, top_y),
            QPointF(col_rect.right(), top_y)
        ]
        self._draw_orthogonal_path(painter, reflux_path, "Reflux", green, arrow_at_end=True)

        # Distillate branch: outbound to the RIGHT (opposite to reflux direction)
        dist_target_x = t_split_point.x() + 60
        distillate_path = [
            t_split_point,
            QPointF(dist_target_x, t_split_y)
        ]
        dist_name = next((name for _, name, ptype in self.product_data
                          if ptype == "distillate"), "Distillate")
        self._draw_orthogonal_path(painter, distillate_path, "",
                                   self.PRODUCT_COLOR, arrow_at_end=True)
        self._draw_stream_chip(painter, QPointF(dist_target_x + 4, t_split_y),
                               dist_name, self.PRODUCT_COLOR)

        # 3. Column to Reboiler (Liquid Line)
        col_bottom_center = QPointF(col_rect.center().x(), col_rect.bottom())
        rebo_center = rebo_rect.center()

        liquid_path = [
            col_bottom_center,
            QPointF(col_bottom_center.x(), rebo_center.y()),
            rebo_center
        ]
        self._draw_orthogonal_path(painter, liquid_path, "", green)

        # 4. Reboiler to Column (Boilup Line)
        rebo_top = QPointF(rebo_center.x(), rebo_rect.top())
        boilup_entry_y = col_rect.bottom() - 5

        # Boilup enters from RIGHT (to keep Reflux/Boilup on same side)
        boilup_path = [
            rebo_top,
            QPointF(col_rect.right() + 30, rebo_top.y()),
            QPointF(col_rect.right() + 30, boilup_entry_y),
            QPointF(col_rect.right(), boilup_entry_y)
        ]
        self._draw_orthogonal_path(painter, boilup_path, "Boilup", green, arrow_at_end=True)

        # 5. Bottoms stream: exits the reboiler to the RIGHT
        rebo_side_right = QPointF(rebo_rect.right(), rebo_center.y())
        bott_target_x = rebo_side_right.x() + 50
        bottoms_name = next((name for _, name, ptype in self.product_data
                             if ptype == "bottoms"), "Bottoms")
        bottoms_path = [
            rebo_side_right,
            QPointF(bott_target_x, rebo_side_right.y())
        ]
        self._draw_orthogonal_path(painter, bottoms_path, "",
                                   self.PRODUCT_COLOR, arrow_at_end=True)
        self._draw_stream_chip(painter,
                               QPointF(bott_target_x + 4, rebo_side_right.y()),
                               bottoms_name, self.PRODUCT_COLOR)

        # 6. Draw Feeds (chip sits left of the arrow, pointing into the column)
        for stage, name in self.feed_data:
            stage_y = self._stage_to_y(col_rect, stage)
            feed_path = [
                QPointF(col_rect.left() - 50, stage_y),
                QPointF(col_rect.left(), stage_y)
            ]
            self._draw_orthogonal_path(painter, feed_path, "", self.FEED_COLOR,
                                       arrow_at_end=True)
            self._draw_stream_chip(painter,
                                   QPointF(col_rect.left() - 54, stage_y),
                                   name, self.FEED_COLOR, align_right=True)

        # 7. Draw other products (Sidestreams)
        for stage, name, ptype in self.product_data:
            if ptype not in ["distillate", "bottoms"]:
                stage_y = self._stage_to_y(col_rect, stage)
                side_path = [
                    QPointF(col_rect.right(), stage_y),
                    QPointF(col_rect.right() + 50, stage_y)
                ]
                self._draw_orthogonal_path(painter, side_path, "",
                                           self.PRODUCT_COLOR, arrow_at_end=True)
                self._draw_stream_chip(painter,
                                       QPointF(col_rect.right() + 54, stage_y),
                                       name, self.PRODUCT_COLOR)

    def _draw_stream_chip(self, painter, anchor, label, color_hex,
                          align_right=False):
        """Clickable stream label: a rounded chip anchored at the arrow end.
        Hover tints it, selection fills it; its rect is the click hit-zone."""
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(label) + 14
        h = fm.height() + 6
        x = anchor.x() - w if align_right else anchor.x()
        rect = QRectF(x, anchor.y() - h / 2, w, h)

        selected = label == self.selected_stream
        hover = label == self.hover_stream
        color = QColor(color_hex)
        if selected:
            bg = QColor(color)
        elif hover:
            bg = QColor(color)
            bg.setAlpha(50)
        else:
            bg = QColor("#ffffff")
        painter.setPen(QPen(color, 2 if (selected or hover) else 1))
        painter.setBrush(QBrush(bg))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QPen(QColor("#ffffff" if selected else "#212529")))
        painter.drawText(rect, Qt.AlignCenter, label)
        # ponytail: chips are keyed by stream id == displayed label; give chips
        # their own id->label map if display names ever diverge from ids
        self.stream_hits[label] = rect

    def _draw_orthogonal_path(self, painter, points, label, color_hex, arrow_at_end=False):
        painter.setPen(QPen(QColor(color_hex), 2))
        for i in range(len(points) - 1):
            painter.drawLine(points[i], points[i+1])

        if arrow_at_end and len(points) >= 2:
            start, end = points[-2], points[-1]
            angle = math.atan2(end.y() - start.y(), end.x() - start.x())
            arrow_size = 8
            p1 = end - QPointF(arrow_size * math.cos(angle - math.pi/6), arrow_size * math.sin(angle - math.pi/6))
            p2 = end - QPointF(arrow_size * math.cos(angle + math.pi/6), arrow_size * math.sin(angle + math.pi/6))
            painter.setBrush(QBrush(QColor(color_hex)))
            painter.drawPolygon([end, p1, p2])

        if label:
            painter.setFont(QFont("Arial", 8))
            painter.setPen(QPen(QColor("#495057")))
            # Place label near start of path
            painter.drawText(points[0] + QPointF(5, -5), label)

    def mousePressEvent(self, event):
        pos = event.position()
        self._press_pos = pos
        self._press_moved = False
        self.dragging_item = None

        for item in reversed(list(self.items.values())):
            if item.contains(pos):
                self.dragging_item = item
                self.last_mouse_pos = pos
                break
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.dragging_item:
            if (pos - self._press_pos).manhattanLength() > 3:
                self._press_moved = True
            delta = pos - self.last_mouse_pos
            new_center = self.dragging_item.get_center(self.width(), self.height()) + delta

            if 0 < new_center.x() < self.width() and 0 < new_center.y() < self.height():
                self.dragging_item.offset += delta
                self.last_mouse_pos = pos
                self.update()
            return

        # hover feedback: pointing hand + tinted chip over anything clickable
        hover = next((sid for sid, r in self.stream_hits.items()
                      if r.contains(pos)), None)
        over_equip = any(it.contains(pos) for it in self.items.values())
        self.setCursor(Qt.PointingHandCursor if (hover or over_equip)
                       else Qt.ArrowCursor)
        if hover != self.hover_stream:
            self.hover_stream = hover
            self.update()

    def mouseReleaseEvent(self, event):
        was_drag = self.dragging_item is not None and self._press_moved
        self.dragging_item = None
        if not was_drag:
            self._handle_click(event.position())
        self.update()

    def mouseDoubleClickEvent(self, event):
        # a double-click is two clicks; the first already opened the editor
        self._handle_click(event.position())

    def _handle_click(self, pos):
        """Single-click: streams first (chips sit on top), then equipment."""
        for sid, rect in self.stream_hits.items():
            if rect.contains(pos):
                self.selected_stream = sid
                self.streamClicked.emit(sid)
                return
        for item in reversed(list(self.items.values())):
            if item.contains(pos):
                if item.element_type in ("condenser", "reboiler"):
                    self.selected_stream = None
                    self.elementClicked.emit(item.element_type)
                return

    def set_num_stages(self, num_stages: int):
        self.num_stages = max(2, num_stages)
        self.stage_input.setText(str(self.num_stages))
        self.update()

    def set_feed_stage(self, feed_stage: int):
        self.feed_stage = max(0, min(feed_stage, self.num_stages - 1))
        self.update()

    def _init_items(self):
        # Initial percentage-based positions
        self.items["column"] = DraggableItem("column", "Column", 0.5, 0.5)
        self.items["column"].width = 80
        self.items["column"].height = 300
        
        self.items["condenser"] = DraggableItem("condenser", "Condenser", 0.5, 0.1)
        self.items["condenser"].width = 60
        self.items["condenser"].height = 60
        
        self.items["reboiler"] = DraggableItem("reboiler", "Reboiler", 0.5, 0.9)
        self.items["reboiler"].width = 70
        self.items["reboiler"].height = 50
        # Distillate/Bottoms are stream chips drawn (and hit-tested) in
        # _draw_all_streams, not equipment items.

    def _init_stage_input(self):
        self.stage_input = QLineEdit(str(self.num_stages), self)
        self.stage_input.setFixedWidth(40)
        self.stage_input.setAlignment(Qt.AlignCenter)
        self.stage_input.setValidator(QIntValidator(2, 999))
        self.stage_input.setStyleSheet("""
            QLineEdit {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid #ced4da;
                border-radius: 3px;
                font-weight: bold;
                color: #212529;
            }
        """)
        self.stage_input.returnPressed.connect(self._on_stage_input_changed)
        self.stage_input.hide()
