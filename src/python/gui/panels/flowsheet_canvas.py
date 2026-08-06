"""The flowsheet editor: a QGraphicsScene of columns and the streams between them.

Replaces the single-column hand-painted canvas. The scene owns every signal and
never mutates WindowState — it emits *requests* (`connectionRequested`,
`deleteRequested`) and the tab that owns the state decides, then calls
`rebuild()`. WindowState stays the single source of truth, so the picture cannot
drift from the model by editing it behind the model's back.

Connection legality is not decided here either: `core.flowsheet.validate_connection`
is the one authority, and `is_recycle` comes from the same `tear_set` the solver
uses — so a stream drawn as a recycle is exactly a stream the solver tears.
"""

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView, QToolTip

from ..theme.palette import canvas as _canvas
from .flowsheet_items import (
    ColumnNode, ConnectionEdge, DragEdge, ModuleBadge, StreamChip,
)


class FlowsheetScene(QGraphicsScene):
    """Owns the items and the signals; owns no state."""

    nodeClicked = Signal(str)                  # unit_id -> make it active
    nodeDoubleClicked = Signal(str)
    elementClicked = Signal(str, str)          # unit_id, "condenser"|"reboiler"|"module_X"
    streamClicked = Signal(str, str)           # unit_id, stream_id
    edgeClicked = Signal(str)                  # connection id
    connectionRequested = Signal(object)       # a candidate dict, see _finish_connect
    deleteRequested = Signal(object)           # {"edges": [...], "nodes": [...]}
    nodeMoved = Signal(str, float, float)
    validationRejected = Signal(str)
    specsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QColor(_canvas.BG))
        self.nodes = {}
        self.edges = {}
        self._drag = None                      # (PortItem, DragEdge)
        self._fs = None                        # last core.flowsheet.Flowsheet
        self.active_unit = None
        self.selected_stream = None

    # --- building ---------------------------------------------------------

    def rebuild(self, view_models, connections, flowsheet=None,
                active_unit=None):
        """Redraw from the model.

        view_models: {unit_id: dict} — num_stages, condenser_type,
            reboiler_type, feeds [(stage, name)], products [(stage, name, type)],
            modules [dict], node_pos.
        connections: the live list of core.flowsheet.Connection.
        flowsheet:   a built core.flowsheet.Flowsheet, when one can be built —
            used only to ask which edges are recycles and which are invalid.
        """
        self.clear()
        self.nodes, self.edges = {}, {}
        self._drag = None
        self._fs = flowsheet
        self.active_unit = active_unit

        for uid, vm in view_models.items():
            node = ColumnNode(uid, vm)
            pos = vm.get("node_pos")
            if pos is not None:
                node.setPos(QPointF(float(pos[0]), float(pos[1])))
            node.active = (uid == active_unit)
            self.addItem(node)
            self.nodes[uid] = node

        if any(vm.get("node_pos") is None for vm in view_models.values()):
            self._auto_place(view_models)

        torn, invalid = set(), {}
        if flowsheet is not None:
            try:
                from core.flowsheet import tear_set, validate_connection
                torn = set(tear_set(flowsheet))
                for c in connections:
                    why = validate_connection(flowsheet, c)
                    if why:
                        invalid[c.id] = why
            except Exception:
                pass          # a half-built flowsheet still draws; it just
                              # cannot say which edges recycle

        for c in connections:
            src, dst = self.nodes.get(c.src), self.nodes.get(c.dst)
            if src is None or dst is None:
                continue
            edge = ConnectionEdge(c, src, dst, recycle=c.id in torn,
                                  invalid=invalid.get(c.id))
            edge.setToolTip(
                f"{c.src}.{c.port} → {c.dst} stage {c.stage - 1}"
                + (f"  (split {c.split_fraction:.0%})" if c.split_fraction < 1 else ""))
            self.addItem(edge)
            self.edges[c.id] = edge

        rect = self.itemsBoundingRect()
        self.setSceneRect(rect.adjusted(-160, -160, 160, 160))

    def _auto_place(self, view_models):
        """Place nodes that have no saved position, left to right by dependency,
        using the same layered layout core.flowsheet computes."""
        try:
            from core.flowsheet import auto_layout
            pos = auto_layout(self._fs) if self._fs is not None else {}
        except Exception:
            pos = {}
        for i, (uid, node) in enumerate(self.nodes.items()):
            if view_models.get(uid, {}).get("node_pos") is not None:
                continue
            p = pos.get(uid)
            node.setPos(QPointF(*p) if p else QPointF(i * 340.0, 0.0))

    def set_stream_results(self, result):
        """After a solve, an edge's tooltip carries its converged stream."""
        if result is None:
            return
        for cid, st in getattr(result, "streams", {}).items():
            edge = self.edges.get(cid)
            if edge is None:
                continue
            comps = []
            for ur in result.units.values():
                comps = list(ur.profile.get("comps", []))
                break
            top = ", ".join(f"{n} {v:.3f}" for n, v in
                            sorted(zip(comps, st.comp), key=lambda t: -t[1])[:3])
            edge.setToolTip(f"{st.src}.{st.port} → {st.dst} stage {st.stage - 1}\n"
                            f"{st.flow:.4g} kmol/h, q={st.q:.2f}\n{top}")

    # --- item callbacks (the items are dumb; the scene emits) --------------

    def activate(self, item):
        """One entry point for 'the user clicked this item'."""
        if isinstance(item, StreamChip):
            self.selected_stream = item.stream_id
            if item.module_id:
                self.elementClicked.emit(item.unit_id, f"module_{item.module_id}")
            else:
                self.streamClicked.emit(item.unit_id, item.stream_id)
        elif isinstance(item, ModuleBadge):
            self.elementClicked.emit(item.unit_id, f"module_{item.module_id}")
        elif isinstance(item, ConnectionEdge):
            self.edgeClicked.emit(item.conn.id)
        elif isinstance(item, ColumnNode):
            self.set_active(item.unit_id)
            self.nodeClicked.emit(item.unit_id)

    def open_unit(self, unit_id):
        self.set_active(unit_id)
        self.nodeDoubleClicked.emit(unit_id)

    def node_moved(self, node):
        self.nodeMoved.emit(node.unit_id, node.pos().x(), node.pos().y())

    def set_active(self, unit_id):
        self.active_unit = unit_id
        for uid, node in self.nodes.items():
            node.active = (uid == unit_id)
            node.update()

    def click_equipment(self, unit_id, which):
        self.elementClicked.emit(unit_id, which)

    # --- dragging a new connection ----------------------------------------

    def begin_connect(self, port, scene_pos):
        self._drag = (port, DragEdge(port.scenePos()))
        self.addItem(self._drag[1])

    def _node_at(self, pos):
        for item in self.items(pos):
            node = item
            while node is not None and not isinstance(node, ColumnNode):
                node = node.parentItem()
            if isinstance(node, ColumnNode):
                return node
        return None

    def _candidate(self, port, node, scene_pos):
        """The Connection a drop here would make, and why it can't be made."""
        from core.flowsheet import Connection, validate_connection
        local_y = node.mapFromScene(scene_pos).y()
        stage = node.stage_at(local_y) + 1        # GUI 0-based -> solver 1-based
        conn = Connection(id=f"{port.unit_id}.{port.key}->{node.unit_id}@{stage}",
                          src=port.unit_id, port=port.key,
                          dst=node.unit_id, stage=stage)
        why = None
        if self._fs is not None:
            try:
                why = validate_connection(self._fs, conn)
            except Exception as exc:
                why = str(exc)
        return conn, why

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            port, edge = self._drag
            node = self._node_at(event.scenePos())
            ok, why = False, None
            if node is not None:
                conn, why = self._candidate(port, node, event.scenePos())
                ok = why is None
                if ok:
                    QToolTip.showText(event.screenPos(),
                                      f"→ {node.unit_id} stage {conn.stage - 1}")
                else:
                    QToolTip.showText(event.screenPos(), why)
            edge.set_tip(event.scenePos(), ok)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            port, edge = self._drag
            self.removeItem(edge)
            self._drag = None
            node = self._node_at(event.scenePos())
            if node is not None:
                conn, why = self._candidate(port, node, event.scenePos())
                if why is None:
                    self.connectionRequested.emit(conn)
                else:
                    QToolTip.showText(event.screenPos(), why)
                    self.validationRejected.emit(why)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            payload = {
                "edges": [i.conn.id for i in self.selectedItems()
                          if isinstance(i, ConnectionEdge)],
                "nodes": [i.unit_id for i in self.selectedItems()
                          if isinstance(i, ColumnNode)],
            }
            if payload["edges"] or payload["nodes"]:
                self.deleteRequested.emit(payload)
                event.accept()
                return
        super().keyPressEvent(event)


class FlowsheetView(QGraphicsView):
    """Zoom, pan, rubber-band select. Nothing model-aware lives here."""

    MIN_ZOOM, MAX_ZOOM = 0.2, 4.0

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setMinimumSize(400, 500)
        self._zoom = 1.0

    def wheelEvent(self, event):
        step = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        target = self._zoom * step
        if not (self.MIN_ZOOM <= target <= self.MAX_ZOOM):
            return
        self._zoom = target
        self.scale(step, step)

    def fit(self):
        rect = self.scene().itemsBoundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect.adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
        self._zoom = min(1.0, self.transform().m11())
        # never zoom PAST 1:1 on open — a single small column would balloon
        if self.transform().m11() > 1.0:
            self.resetTransform()
            self._zoom = 1.0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F or (
                event.key() == Qt.Key_0 and event.modifiers() & Qt.ControlModifier):
            self.fit()
            event.accept()
            return
        if event.key() == Qt.Key_Space:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.setDragMode(QGraphicsView.RubberBandDrag)
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.RubberBandDrag)
