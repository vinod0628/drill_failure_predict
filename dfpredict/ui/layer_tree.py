"""Layers panel (TOC): visibility checkboxes, selection, context menu.

Order: every layer can be dragged up or down (the order is saved with the workspace; it does
not change how the view draws - that goes by depth). The project database's layers (Pits,
Drillholes, Mined shapes) are greyed out while nothing is stored in them."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr
from ..scene.layers import (BlockModelLayer, DrillholeLayer, Layer, MinedShapesLayer, StringLayer,
                            SurfaceLayer)
from ..scene.survey import SurveyLayer
from .icons import icon

PIT_ROLE = Qt.UserRole + 1          # pit items under the survey layer: the pit id
DESIGN_ROLE = Qt.UserRole + 2       # designed pit items (under "Designed Pits"): the design id
GROUP_ROLE = Qt.UserRole + 3        # the "Designed Pits" group item: "designs"


def database_kind(layer) -> int | None:
    """Which project-database layer this is (0 Pits, 1 Drillholes, 2 Mined shapes); None for
    the other layers."""
    if isinstance(layer, SurveyLayer):
        return 0
    if isinstance(layer, DrillholeLayer) and layer.source == "database":
        return 1
    if isinstance(layer, MinedShapesLayer):
        return 2
    return None


def has_content(layer) -> bool:
    """False for a project-database layer with nothing stored yet (no pits / drillholes /
    mined shapes): it is listed, greyed out."""
    if isinstance(layer, SurveyLayer):
        return layer.data is not None and (bool(layer.data.pits) or bool(layer.data.designs))
    if isinstance(layer, DrillholeLayer) and layer.source == "database":
        return layer.data is not None and len(layer.data.collars) > 0
    if isinstance(layer, MinedShapesLayer):
        return bool(layer.shapes)
    return True


EMPTY_TIPS = {0: N_("No pits yet - add them with the Pit Manager (Ctrl+U)"),
              1: N_("No drillholes yet - add them with the Drillhole Manager (Ctrl+D)"),
              2: N_("No mined shapes yet - draw an outline (Ctrl+L, 2D view) or load a mined solid "
                    "in the Reserve window")}


LAYER_MIME = "application/x-geoviewer-layer"


class _TreeWidget(QTreeWidget):
    """A tree whose top-level items can be dragged to another place (never into another item).

    The drag carries only a marker: Qt's own drag data would copy every item's data, and the
    layer object kept on each item cannot be copied that way (PySide pickles it - a layer drawn
    in the view cannot be pickled, which crashed the app). Qt moves the item itself."""

    moved = Signal()

    def mimeTypes(self):
        return [LAYER_MIME]

    def mimeData(self, items):
        md = QMimeData()
        md.setData(LAYER_MIME, QByteArray(b"layer"))
        return md

    def dropEvent(self, e):
        super().dropEvent(e)
        QTimer.singleShot(0, self.moved.emit)          # after Qt has finished the move
KIND_ICONS = {
    StringLayer: "vector-polyline",
    SurfaceLayer: "terrain",
    BlockModelLayer: "cube-outline",
    DrillholeLayer: "source-commit",
    MinedShapesLayer: "pickaxe",
    SurveyLayer: "map-marker-path",
}


def swatch_icon(rgb) -> QIcon:
    """A small square of the pit's colour."""
    pm = QPixmap(14, 14)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setPen(QColor("#555b63"))
    p.setBrush(QColor.fromRgbF(*rgb))
    p.drawRoundedRect(1, 1, 12, 12, 2, 2)
    p.end()
    return QIcon(pm)


def _tip(layer: Layer) -> str:
    if getattr(layer, "offline", None):
        return tr("{kind} - not connected\n{reason}\nCheck Settings > Database, then press F5 to try again.",
                  kind=tr(layer.kind), reason=layer.offline)
    if layer.source == "database":
        where = getattr(getattr(layer, "data", None), "source", "")
        tables = "survey_*" if isinstance(layer, SurveyLayer) else "drillhole_*"
        extra = ("\n" + tr("Main pits and their branches are listed below it.")
                 if isinstance(layer, SurveyLayer) else "")
        return tr("{kind}\nProject database: {where}\n({tables} tables)", kind=tr(layer.kind), where=where,
                  tables=tables) + extra
    return tr(layer.kind) + (f"\n{layer.path}" if layer.path is not None else "")


def panel_header(title: str, parent: QWidget) -> tuple[QFrame, QHBoxLayout]:
    f = QFrame(parent)
    f.setObjectName("panelHeader")
    lay = QHBoxLayout(f)
    lay.setContentsMargins(10, 4, 6, 4)
    lay.setSpacing(2)
    lbl = QLabel(title.upper(), f)
    lbl.setObjectName("panelTitle")
    lay.addWidget(lbl)
    lay.addStretch(1)
    return f, lay


def header_button(icon_name: str, tip: str, parent: QWidget) -> QToolButton:
    b = QToolButton(parent)
    b.setObjectName("headerButton")
    b.setIcon(icon(icon_name))
    b.setIconSize(QSize(16, 16))
    b.setToolTip(tip)
    return b


class LayerTree(QWidget):
    layer_selected = Signal(object)          # Layer | None
    visibility_changed = Signal(object, bool)
    remove_requested = Signal(object)
    zoom_requested = Signal(object)
    rename_requested = Signal(object)
    open_requested = Signal()
    order_changed = Signal()                            # the layers were put in another order
    pit_selected = Signal(object, object)               # survey layer, pit id
    pit_visibility_changed = Signal(object, object, bool)
    pit_zoom_requested = Signal(object, object)
    pit_geology_requested = Signal(object, object)    # (survey layer, pit id): open Pit Geology
    design_selected = Signal(object, object)            # survey layer, designed pit id
    design_visibility_changed = Signal(object, object, bool)
    designs_visibility_changed = Signal(object, bool)   # the "Designed Pits" group switched on / off
    designs_selected = Signal(object)                   # survey layer: its "Designed Pits" group selected
    design_zoom_requested = Signal(object, object)      # survey layer, design id (None: all of them)
    design_rename_requested = Signal(object, object)
    design_delete_requested = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        header, hl = panel_header(tr("Layers"), self)
        b_open = header_button("folder-open-outline", tr("Open files (Ctrl+O)"), header)
        b_zoom = header_button("magnify-scan", tr("Zoom to selected layer"), header)
        b_del = header_button("delete-outline", tr("Remove selected layer (Del)"), header)
        self.b_del = b_del
        for b in (b_open, b_zoom, b_del):
            hl.addWidget(b)
        b_open.clicked.connect(self.open_requested)
        b_zoom.clicked.connect(lambda: self.current_layer() and self.zoom_requested.emit(self.current_layer()))
        b_del.clicked.connect(lambda: self.current_layer() and self.remove_requested.emit(self.current_layer()))
        lay.addWidget(header)

        self.tree = _TreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setDragDropMode(QAbstractItemView.InternalMove)   # drag a layer up or down
        self.tree.setDefaultDropAction(Qt.MoveAction)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.moved.connect(self._after_drag)
        self.tree.setIndentation(10)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemDoubleClicked.connect(self._double_clicked)
        lay.addWidget(self.tree, 1)

        self.empty_hint = QLabel(tr("Open a workspace folder (Ctrl+Shift+O),\nor open or drop .str, .dtm, "
                                    "block model CSV,\ndrillhole CSV / .accdb or .dxf files"), self.tree)
        self.empty_hint.setAlignment(Qt.AlignCenter)
        self.empty_hint.setStyleSheet("color: #6c747e; padding: 18px;")
        self.empty_hint.setWordWrap(True)
        self._update_hint()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.empty_hint.setGeometry(self.tree.viewport().geometry())

    def _update_hint(self):
        self.empty_hint.setVisible(self.tree.topLevelItemCount() == 0)
        self.empty_hint.setGeometry(self.tree.viewport().geometry())

    @staticmethod
    def _layer(item: QTreeWidgetItem | None) -> Layer | None:
        return item.data(0, Qt.UserRole) if item is not None else None

    def current_layer(self) -> Layer | None:
        return self._layer(self.tree.currentItem())

    @staticmethod
    def _set_enabled(it: QTreeWidgetItem, layer: Layer) -> None:
        """Offline layers (database not reachable) and database layers with nothing stored yet
        are greyed out and cannot be switched on. Every layer can be dragged; no item takes a
        drop (a layer is never put inside another one)."""
        flags = (it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled) & ~Qt.ItemIsDropEnabled
        if getattr(layer, "offline", None) or not has_content(layer):
            flags &= ~Qt.ItemIsEnabled
        else:
            flags |= Qt.ItemIsEnabled
        it.setFlags(flags)
        rank = database_kind(layer)
        if not getattr(layer, "offline", None) and not has_content(layer) and rank in EMPTY_TIPS:
            it.setToolTip(0, f"{tr(layer.kind)}\n{tr(EMPTY_TIPS[rank])}")
        else:
            it.setToolTip(0, _tip(layer))

    def refresh_state(self, layer: Layer) -> None:
        """Greyed out or not, after the layer's content changed (e.g. the first mined shape)."""
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is layer:
                self.tree.blockSignals(True)
                self._set_enabled(it, layer)
                self.tree.blockSignals(False)
                return

    def _after_drag(self) -> None:
        self.order_changed.emit()

    def add_layer(self, layer: Layer) -> None:
        it = QTreeWidgetItem([layer.name])
        it.setData(0, Qt.UserRole, layer)
        self._set_enabled(it, layer)
        it.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
        it.setIcon(0, icon(KIND_ICONS.get(type(layer), "layers-outline")))
        self.tree.blockSignals(True)
        self.tree.insertTopLevelItem(0, it)
        self._fill_pits(it, layer)
        self._set_enabled(it, layer)
        self.tree.blockSignals(False)
        if it.flags() & Qt.ItemIsEnabled:
            self.tree.setCurrentItem(it)
        self.order_changed.emit()
        self._update_hint()

    def replace_layer(self, old: Layer, new: Layer) -> None:
        """Put a reloaded layer in place of the old one (same row, check state from the layer)."""
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is old:
                cur = self.tree.currentItem()
                mine = cur is not None and self._layer(cur) is old
                cur_pit = cur.data(0, PIT_ROLE) if mine else None
                cur_design = cur.data(0, DESIGN_ROLE) if mine else None
                cur_group = mine and cur.data(0, GROUP_ROLE) is not None
                self.tree.blockSignals(True)
                it.setData(0, Qt.UserRole, new)
                it.setText(0, new.name)
                self._set_enabled(it, new)
                it.setCheckState(0, Qt.Checked if new.visible else Qt.Unchecked)
                again = self._fill_pits(it, new, select=cur_pit)
                again_d = self._find_child(it, DESIGN_ROLE, cur_design) if cur_design is not None else None
                again_g = self._find_child(it, GROUP_ROLE, "designs") if cur_group else None
                self.tree.blockSignals(False)
                if again is not None:
                    self.tree.setCurrentItem(again)
                    self.pit_selected.emit(new, cur_pit)
                elif again_d is not None:
                    self.tree.setCurrentItem(again_d)
                    self.design_selected.emit(new, cur_design)
                elif again_g is not None:
                    self.tree.setCurrentItem(again_g)
                    self.designs_selected.emit(new)
                elif cur is not None and (cur is it or cur_pit is not None or cur_design is not None or cur_group):
                    self.tree.setCurrentItem(it)
                    self.layer_selected.emit(new)
                return

    # --- survey pits -------------------------------------------------------------------
    def _pit_items(self, layer_item) -> list:
        out, todo = [], [layer_item.child(i) for i in range(layer_item.childCount())]
        while todo:
            it = todo.pop()
            out.append(it)
            todo += [it.child(i) for i in range(it.childCount())]
        return out

    def _find_child(self, layer_item, role, value):
        return next((c for c in self._pit_items(layer_item) if c.data(0, role) == value), None)

    def _fill_pits(self, it, layer, select=None):
        """The survey layer's pits under its item: main pits, their branches nested below
        (checkbox = shown, swatch = colour), then the "Designed Pits" group when there are designed
        pits. Expanded / collapsed pits stay so on a reload. Returns the item of pit ``select`` when
        it is still there."""
        if not isinstance(layer, SurveyLayer):
            return None
        collapsed = {c.data(0, PIT_ROLE) for c in self._pit_items(it)
                     if not c.isExpanded() and c.childCount() and c.data(0, PIT_ROLE) is not None}
        group = self._find_child(it, GROUP_ROLE, "designs")
        group_open = group.isExpanded() if group is not None else True
        was_open = it.isExpanded() or it.childCount() == 0
        it.takeChildren()
        d = layer.data
        if d is None or (not d.pits and not d.designs):
            return None
        items, found = {}, None
        ids = {x.id for x in d.pits}
        top = set()                                          # pits listed at the top (main, or parent lost)
        pending = [q for q in d.pits]
        while pending:                                       # parents before their branches
            rest = []
            for q in pending:
                main = q.parent_id is None or q.parent_id not in ids or q.id in top
                parent = it if main else items.get(q.parent_id)
                if parent is None:
                    rest.append(q)
                    continue
                c = QTreeWidgetItem([f"{q.name}  ({q.code})"])
                c.setData(0, Qt.UserRole, layer)
                c.setData(0, PIT_ROLE, q.id)
                c.setFlags((c.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                           & ~Qt.ItemIsDragEnabled & ~Qt.ItemIsDropEnabled)      # pits stay in place
                c.setCheckState(0, Qt.Checked if layer.pit_style(q.id)["visible"] else Qt.Unchecked)
                c.setIcon(0, swatch_icon(layer.pit_color(q.id)))
                tip = (tr("Main pit {name} ({code}), starts at {start}\n"
                          "Tick to show, untick to hide; double-click to zoom",
                          name=q.name, code=q.code, start=q.start_station) if q.parent_id is None else
                       tr("Branch {name} ({code}), starts at {start}\n"
                          "Tick to show, untick to hide; double-click to zoom",
                          name=q.name, code=q.code, start=q.start_station))
                c.setToolTip(0, tip)
                parent.addChild(c)
                items[q.id] = c
                if q.id == select:
                    found = c
            if len(rest) == len(pending):                    # a loop of parents: list them at the top
                top |= {q.id for q in rest}
            pending = rest
        if d.designs:
            self._fill_designs(it, layer).setExpanded(group_open)
        it.setExpanded(was_open)
        for pid, c in items.items():
            c.setExpanded(pid not in collapsed)
        return found

    def _fill_designs(self, it, layer):
        """The "Designed Pits" group (checkbox: all of them shown) and one item per designed pit."""
        g = QTreeWidgetItem([tr("Designed Pits")])
        g.setData(0, Qt.UserRole, layer)
        g.setData(0, GROUP_ROLE, "designs")
        g.setFlags((g.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsDragEnabled
                   & ~Qt.ItemIsDropEnabled)
        g.setCheckState(0, Qt.Checked if layer.show_designs else Qt.Unchecked)
        g.setIcon(0, icon("pencil-ruler"))
        g.setToolTip(0, tr("Drives designed with the route designer and saved to the database (not surveyed)\n"
                           "Tick to show, untick to hide; double-click to zoom"))
        it.addChild(g)
        for x in layer.data.designs:
            c = QTreeWidgetItem([x.name])
            c.setData(0, Qt.UserRole, layer)
            c.setData(0, DESIGN_ROLE, x.id)
            c.setFlags((c.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled) & ~Qt.ItemIsDragEnabled
                       & ~Qt.ItemIsDropEnabled)
            c.setCheckState(0, Qt.Checked if layer.design_style(x.id)["visible"] else Qt.Unchecked)
            c.setIcon(0, swatch_icon(layer.design_color(x.id)))
            c.setToolTip(0, tr("Designed pit {name}: from {a} to {b}, {m:,.0f} m\n"
                               "Tick to show, untick to hide; double-click to zoom; right-click to rename or delete",
                               name=x.name, a=x.start_name or "-", b=x.target_name or "-", m=x.length))
            g.addChild(c)
        return g

    def update_pits(self, layer: Layer) -> None:
        """Check boxes and colour swatches follow the pits' display settings."""
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is layer:
                self.tree.blockSignals(True)
                for c in self._pit_items(it):
                    pid, did = c.data(0, PIT_ROLE), c.data(0, DESIGN_ROLE)
                    if pid is not None:
                        c.setCheckState(0, Qt.Checked if layer.pit_style(pid)["visible"] else Qt.Unchecked)
                        c.setIcon(0, swatch_icon(layer.pit_color(pid)))
                    elif did is not None:
                        c.setCheckState(0, Qt.Checked if layer.design_style(did)["visible"] else Qt.Unchecked)
                        c.setIcon(0, swatch_icon(layer.design_color(did)))
                    elif c.data(0, GROUP_ROLE) is not None:
                        c.setCheckState(0, Qt.Checked if layer.show_designs else Qt.Unchecked)
                self.tree.blockSignals(False)
                return

    def current_pit(self):
        it = self.tree.currentItem()
        return it.data(0, PIT_ROLE) if it is not None else None

    def current_design(self):
        it = self.tree.currentItem()
        return it.data(0, DESIGN_ROLE) if it is not None else None

    def _double_clicked(self, it, _col):
        pid = it.data(0, PIT_ROLE)
        if pid is not None:
            self.pit_zoom_requested.emit(self._layer(it), pid)
        elif it.data(0, DESIGN_ROLE) is not None:
            self.design_zoom_requested.emit(self._layer(it), it.data(0, DESIGN_ROLE))
        elif it.data(0, GROUP_ROLE) is not None:
            self.design_zoom_requested.emit(self._layer(it), None)
        else:
            self.zoom_requested.emit(self._layer(it))

    def update_name(self, layer: Layer) -> None:
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is layer:
                self.tree.blockSignals(True)
                it.setText(0, layer.name)
                self.tree.blockSignals(False)
                return

    def clear(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        self.tree.blockSignals(False)
        self._update_hint()
        self.layer_selected.emit(None)

    def layers_top_down(self) -> list[Layer]:
        return [self._layer(self.tree.topLevelItem(i)) for i in range(self.tree.topLevelItemCount())]

    def remove_layer(self, layer: Layer) -> None:
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is layer:
                self.tree.takeTopLevelItem(i)
                break
        self._update_hint()
        self.layer_selected.emit(self.current_layer())

    def update_check(self, layer: Layer) -> None:
        """The checkbox follows the layer's visibility (changed from elsewhere)."""
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            if self._layer(it) is layer:
                self.tree.blockSignals(True)
                it.setCheckState(0, Qt.Checked if layer.visible else Qt.Unchecked)
                self.tree.blockSignals(False)
                return

    def _current_changed(self, cur, _prev):
        layer = self._layer(cur)
        pid = cur.data(0, PIT_ROLE) if cur is not None else None
        did = cur.data(0, DESIGN_ROLE) if cur is not None else None
        sub = pid is not None or did is not None or (cur is not None and cur.data(0, GROUP_ROLE) is not None)
        self.b_del.setEnabled(layer is not None and layer.removable and not sub)
        if pid is not None:
            self.pit_selected.emit(layer, pid)
        elif did is not None:
            self.design_selected.emit(layer, did)
        elif cur is not None and cur.data(0, GROUP_ROLE) is not None:
            self.designs_selected.emit(layer)
        else:
            self.layer_selected.emit(layer)

    def _item_changed(self, item, _col):
        layer = self._layer(item)
        pid = item.data(0, PIT_ROLE)
        if layer is not None and pid is not None:
            vis = item.checkState(0) == Qt.Checked
            if vis != layer.pit_style(pid)["visible"]:
                self.pit_visibility_changed.emit(layer, pid, vis)
            return
        did = item.data(0, DESIGN_ROLE)
        if layer is not None and did is not None:
            vis = item.checkState(0) == Qt.Checked
            if vis != layer.design_style(did)["visible"]:
                self.design_visibility_changed.emit(layer, did, vis)
            return
        if layer is not None and item.data(0, GROUP_ROLE) is not None:
            vis = item.checkState(0) == Qt.Checked
            if vis != layer.show_designs:
                self.designs_visibility_changed.emit(layer, vis)
            return
        if layer is not None:
            vis = item.checkState(0) == Qt.Checked
            if vis != layer.visible:
                self.visibility_changed.emit(layer, vis)

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        layer = self._layer(item)
        if layer is None:
            return
        m = QMenu(self)
        pid = item.data(0, PIT_ROLE)
        if pid is not None:
            m.addAction(icon("magnify-scan"), tr("Zoom to pit"), lambda: self.pit_zoom_requested.emit(layer, pid))
            m.addAction(icon("layers-outline"), tr("Pit geology ..."),
                        lambda: self.pit_geology_requested.emit(layer, pid))
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        did = item.data(0, DESIGN_ROLE)
        if did is not None:
            m.addAction(icon("magnify-scan"), tr("Zoom to designed pit"),
                        lambda: self.design_zoom_requested.emit(layer, did))
            m.addAction(icon("rename-box"), tr("Rename..."), lambda: self.design_rename_requested.emit(layer, did))
            m.addSeparator()
            m.addAction(icon("delete-outline"), tr("Delete designed pit..."),
                        lambda: self.design_delete_requested.emit(layer, did))
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        if item.data(0, GROUP_ROLE) is not None:
            m.addAction(icon("magnify-scan"), tr("Zoom to designed pits"),
                        lambda: self.design_zoom_requested.emit(layer, None))
            m.exec(self.tree.viewport().mapToGlobal(pos))
            return
        m.addAction(icon("magnify-scan"), tr("Zoom to layer"), lambda: self.zoom_requested.emit(layer))
        m.addAction(icon("rename-box"), tr("Rename..."), lambda: self.rename_requested.emit(layer))
        if layer.removable:                          # not the database drillholes / mined shapes
            m.addSeparator()
            m.addAction(icon("delete-outline"), tr("Remove layer"), lambda: self.remove_requested.emit(layer))
        m.exec(self.tree.viewport().mapToGlobal(pos))
