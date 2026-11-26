"""Properties panel: display settings and information for the selected layer."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..i18n import N_, tr, trn
from ..scene.layers import (COLORMAPS, DEPLETION_ATTRS, MINED_DISPLAYS, BlockModelLayer, DrillholeLayer, Layer,
                            StringLayer, SurfaceLayer)
from ..io.survey_store import STATION_LEVELS
from ..scene.geo_roles import ROLES
from ..scene.survey import SurveyLayer, design_rows
from .icons import icon
from .layer_tree import panel_header


# a pit's profile in its Information, one sentence per station level (tr() where shown)
_PROFILE_TEXT = {"back": N_("{w:g} x {h:g} m, stations in the backs (roof)"),
                 "floor": N_("{w:g} x {h:g} m, stations in the floor"),
                 "middle": N_("{w:g} x {h:g} m, stations in the middle")}
_PROFILE_OTHER = N_("{w:g} x {h:g} m, stations: {level}")


def _breakable(text: str) -> str:
    """Allow line breaks after path separators (zero-width space), so long paths wrap in a
    narrow panel instead of forcing it wider."""
    return text.replace("\\", "\\\u200b").replace("/", "/\u200b")


def _hex(rgb) -> str:
    return QColor.fromRgbF(*rgb).name()


class ColorButton(QPushButton):
    def __init__(self, rgb, on_pick: Callable, parent=None):
        super().__init__(parent)
        self.setObjectName("colorButton")
        self.rgb = rgb
        self.on_pick = on_pick
        self._paint()
        self.clicked.connect(self._choose)

    def _paint(self):
        self.setStyleSheet(f"QPushButton#colorButton {{ background: {_hex(self.rgb)}; }}")

    def _choose(self):
        c = QColorDialog.getColor(QColor.fromRgbF(*self.rgb), self, tr("Layer colour"))
        if c.isValid():
            self.rgb = (c.redF(), c.greenF(), c.blueF())
            self._paint()
            self.on_pick(self.rgb)


def _swatch(rgb) -> QLabel:
    s = QLabel()
    s.setFixedSize(14, 14)
    s.setStyleSheet(f"background: {_hex(rgb)}; border: 1px solid #555; border-radius: 3px;")
    return s


class PropertiesPanel(QWidget):
    def __init__(self, on_changed: Callable[[], None], parent=None):
        super().__init__(parent)
        self.on_changed = on_changed   # called after a change so the view re-renders
        self.layer: Layer | None = None
        self.pit = None                # a pit of the survey layer shown instead of the layer
        # (layer, pit id, **settings): set by the main window so the layers list follows
        self.on_pit_style: Callable | None = None
        self.design = None             # a designed pit of the survey layer shown instead of the layer
        self.designs_group = False     # the survey layer's "Designed Pits" group shown instead of the layer
        self.on_design_style: Callable | None = None
        self.on_designs_shown: Callable | None = None       # (layer, on): the group switched on / off
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        header, _ = panel_header(tr("Properties"), self)
        lay.addWidget(header)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(self.scroll, 1)
        self.set_layer(None)

    # ------------------------------------------------------------------------
    def set_layer(self, layer: Layer | None) -> None:
        self.layer = layer
        self.pit = None
        self.design = None
        self.designs_group = False
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 8)
        v.setSpacing(6)
        if layer is None:
            hint = QLabel(tr("Select a layer to see its properties."))
            hint.setStyleSheet("color: #6c747e; padding: 12px;")
            hint.setAlignment(Qt.AlignCenter)
            v.addWidget(hint)
        else:
            v.addWidget(self._display_group(layer))
            extra = self._extra_group(layer)
            if extra is not None:
                v.addWidget(extra)
            v.addWidget(self._info_group(layer))
        v.addStretch(1)
        self._set_body(body)

    def _set_body(self, body: QWidget) -> None:
        """Show ``body`` in place of the panel's contents. The old contents are deleted later, not now: the
        panel is rebuilt from its own check boxes' signals (e.g. Visible), and deleting the check box while Qt
        is still handling its mouse click crashed the app."""
        old = self.scroll.takeWidget()
        self.scroll.setWidget(body)
        if old is not None:
            old.hide()
            old.deleteLater()

    def refresh(self) -> None:
        if self.pit is not None and isinstance(self.layer, SurveyLayer):
            self.set_pit(self.layer, self.pit)
        elif self.design is not None and isinstance(self.layer, SurveyLayer):
            self.set_design(self.layer, self.design)
        elif self.designs_group and isinstance(self.layer, SurveyLayer):
            self.set_designs(self.layer)
        else:
            self.set_layer(self.layer)

    def set_pit(self, layer: SurveyLayer, pid) -> None:
        """The display settings of one pit of the survey layer."""
        p = layer.data.pit(pid) if layer is not None and layer.data is not None else None
        if p is None:
            self.set_layer(layer)
            return
        self.layer, self.pit, self.design, self.designs_group = layer, pid, None, False
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 8)
        v.setSpacing(6)
        v.addWidget(self._pit_group(layer, p))
        v.addWidget(self._pit_info_group(layer, p))
        v.addStretch(1)
        self._set_body(body)

    def set_design(self, layer: SurveyLayer, did) -> None:
        """A designed pit: shown or not, its colour, and what it is."""
        x = layer.data.design(did) if layer is not None and layer.data is not None else None
        if x is None:
            self.set_layer(layer)
            return
        self.layer, self.pit, self.design, self.designs_group = layer, None, did, False
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 8)
        v.setSpacing(6)
        g, f = self._form(tr("Designed pit"))
        name = QLabel(x.name)
        name.setStyleSheet("font-weight: 600;")
        name.setWordWrap(True)
        f.addRow(tr("Name"), name)
        st = layer.design_style(did)
        vis = QCheckBox()
        vis.setChecked(st["visible"])
        vis.setToolTip(tr("Show or hide this designed pit"))
        vis.toggled.connect(lambda on: self._design_set(layer, did, visible=on))
        f.addRow(tr("Visible"), vis)
        if st["visible"] and not layer.show_designs:
            note = QLabel(tr("Not drawn: Designed Pits is switched off"))
            note.setStyleSheet("color: #e0a030;")
            note.setWordWrap(True)
            f.addRow(note)
        row = QHBoxLayout()
        colour = ColorButton(layer.design_color(did), lambda rgb: self._design_set(layer, did, color=rgb))
        colour.setToolTip(tr("Colour of the designed drive and its centre line"))
        row.addWidget(colour, 1)
        if st["color"] is not None:
            reset = QToolButton()
            reset.setIcon(icon("restore"))
            reset.setToolTip(tr("Back to the default colour"))
            reset.clicked.connect(lambda: self._design_set(layer, did, color=None))
            row.addWidget(reset)
        f.addRow(tr("Colour"), row)
        cl = QCheckBox()
        cl.setChecked(st["centerline"])
        cl.setToolTip(tr("The centre line of this designed pit"))
        cl.toggled.connect(lambda on: self._design_set(layer, did, centerline=on))
        f.addRow(tr("Centre line"), cl)
        v.addWidget(g)
        g2, f2 = self._form(tr("Information"))
        for k, val in design_rows(layer, x)[1:]:
            lbl = QLabel(str(val))
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #c4cad1;")
            key = QLabel(k)
            key.setStyleSheet("color: #8a939d;")
            f2.addRow(key, lbl)
        v.addWidget(g2)
        v.addStretch(1)
        self._set_body(body)

    def set_designs(self, layer: SurveyLayer) -> None:
        """The "Designed Pits" group: all designed pits shown or not, their centre lines, and totals."""
        if layer is None or layer.data is None or not layer.data.designs:
            self.set_layer(layer)
            return
        self.layer, self.pit, self.design, self.designs_group = layer, None, None, True
        ds = layer.data.designs
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(8, 4, 8, 8)
        v.setSpacing(6)
        g, f = self._form(tr("Designed Pits"))
        vis = QCheckBox()
        vis.setChecked(layer.show_designs)
        vis.setToolTip(tr("Show or hide all designed pits (the surveyed pits are not affected)"))
        vis.toggled.connect(lambda on: self._designs_shown(layer, on))
        f.addRow(tr("Visible"), vis)
        lines = [layer.design_style(x.id)["centerline"] for x in ds]
        cl = QCheckBox()
        cl.setTristate(False)
        cl.setChecked(all(lines))
        cl.setToolTip(tr("Centre lines of all designed pits (each one can also be set on its own)"))
        cl.toggled.connect(lambda on: self._designs_centerline(layer, on))
        f.addRow(tr("Centre lines"), cl)
        if any(lines) and not all(lines):
            note = QLabel(tr("Some designed pits have their centre line switched off."))
            note.setWordWrap(True)
            note.setStyleSheet("color: #8a939d;")
            f.addRow(note)
        hint = QLabel(tr("Select a designed pit below this group for its colour and details."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a939d;")
        f.addRow(hint)
        v.addWidget(g)
        g2, f2 = self._form(tr("Information"))
        rows = [(tr("Designed pits"), f"{len(ds):,}"), (tr("Total length"), f"{sum(x.length for x in ds):,.1f} m"),
                (tr("Total volume"), tr("{v:,.0f} m³", v=sum(layer.design_volume(x.id) for x in ds)))]
        for k, val in rows:
            lbl = QLabel(str(val))
            lbl.setStyleSheet("color: #c4cad1;")
            key = QLabel(k)
            key.setStyleSheet("color: #8a939d;")
            f2.addRow(key, lbl)
        v.addWidget(g2)
        v.addStretch(1)
        self._set_body(body)

    def _designs_shown(self, layer, on: bool) -> None:
        if self.on_designs_shown is not None:
            self.on_designs_shown(layer, on)
        else:
            layer.set_show_designs(on)
            self.on_changed()
        self.refresh()

    def _designs_centerline(self, layer, on: bool) -> None:
        layer.set_designs_centerline(on)
        self.on_changed()
        self.refresh()

    def _design_set(self, layer, did, **kw):
        if self.on_design_style is not None:
            self.on_design_style(layer, did, **kw)
        else:
            layer.set_design_style(did, **kw)
            self.on_changed()
        self.refresh()

    def _pit_set(self, layer, pid, **kw):
        if self.on_pit_style is not None:
            self.on_pit_style(layer, pid, **kw)
        else:
            layer.set_pit_style(pid, **kw)
            self.on_changed()
        self.refresh()

    def _pit_group(self, layer: SurveyLayer, p) -> QGroupBox:
        main = p.parent_id is None
        g, f = self._form(tr("Main pit") if main else tr("Branch pit"))
        name = QLabel(f"{p.name}  ({p.code})")
        name.setStyleSheet("font-weight: 600;")
        name.setWordWrap(True)
        f.addRow(tr("Name"), name)
        if not main:
            parent = layer.data.pit(p.parent_id)
            f.addRow(tr("Branch of"), QLabel(f"{parent.name} ({parent.code})" if parent else "-"))
        st = layer.pit_style(p.id)
        vis = QCheckBox()
        vis.setChecked(st["visible"])
        vis.setToolTip(tr("Show or hide this pit") if not layer.data.children(p.id) else
                       tr("Show or hide this pit (its branches are hidden with it)"))
        vis.toggled.connect(lambda on: self._pit_set(layer, p.id, visible=on))
        f.addRow(tr("Visible"), vis)
        hidden_by = [a for a in layer.data.ancestors(p.id) if not layer.pit_style(a)["visible"]]
        if st["visible"] and hidden_by:
            note = QLabel(tr("Not drawn: {name} is hidden", name=layer.data.pit(hidden_by[-1]).name))
            note.setStyleSheet("color: #e0a030;")
            note.setWordWrap(True)
            f.addRow(note)

        colour = ColorButton(layer.pit_color(p.id), lambda rgb: self._pit_set(layer, p.id, color=rgb))
        colour.setToolTip(tr("Colour of the pit's drive and centre line"))
        if main:
            row = QHBoxLayout()
            row.addWidget(colour, 1)
            if st["color"] is not None:
                reset = QToolButton()
                reset.setIcon(icon("restore"))
                reset.setToolTip(tr("Back to the default colour"))
                reset.clicked.connect(lambda: self._pit_set(layer, p.id, color=None))
                row.addWidget(reset)
            f.addRow(tr("Colour"), row)
            if layer.data.children(p.id):
                note = QLabel(tr("Branches without a colour of their own use this colour."))
                note.setWordWrap(True)
                note.setStyleSheet("color: #8a939d;")
                f.addRow(note)
        else:
            inherit = QCheckBox(tr("Inherit parent pit's color"))
            inherit.setChecked(st["color"] is None)
            inherit.setToolTip(tr("Use the colour of the pit this branch starts from; untick to give it its own"))
            colour.setEnabled(st["color"] is not None)
            inherit.toggled.connect(lambda on: self._pit_set(
                layer, p.id, color=None if on else layer.default_color(p.id)))
            f.addRow(inherit)
            f.addRow(tr("Colour"), colour)
        if main:
            c = QCheckBox()
            c.setChecked(st["centerline"])
            c.setToolTip(tr("Centre lines of this main pit and all its branches"))
            c.toggled.connect(lambda on: self._pit_set(layer, p.id, centerline=on))
            f.addRow(tr("Centre lines"), c)
            s_ = QCheckBox()
            s_.setChecked(st["stations"])
            s_.setToolTip(tr("Survey stations (points and names) of this main pit and all its branches"))
            s_.toggled.connect(lambda on: self._pit_set(layer, p.id, stations=on))
            f.addRow(tr("Stations"), s_)
        return g

    def _pit_info_group(self, layer: SurveyLayer, p) -> QGroupBox:
        g, f = self._form(tr("Information"))
        d = layer.data
        n_st = sum(1 for s_ in d.stations.values() if s_.pit_id == p.id)
        profile = tr(_PROFILE_TEXT.get(p.station_level, _PROFILE_OTHER), w=p.profile_width, h=p.profile_height,
                     level=tr(STATION_LEVELS.get(p.station_level, p.station_level)))
        rows = [(tr("Code"), p.code), (tr("Starts at"), p.start_station or "-"),
                (tr("Length (plan)"), f"{layer.pit_length(p.id):,.1f} m"), (tr("Stations"), f"{n_st:,}"),
                (tr("Profile"), profile),
                (tr("Volume"), tr("{v:,.0f} m³", v=layer.pit_volume(p.id))),
                (tr("Branches"), f"{len(d.children(p.id)):,}")]
        if d.children(p.id):
            n_all = len(layer.pit_family(p.id)) - 1
            rows.append((tr("Volume with branches"),
                         trn("{v:,.0f} m³ ({n} branch)", "{v:,.0f} m³ ({n} branches)", n_all,
                             v=layer.pit_volume(p.id, branches=True))))
        for k, val in rows:
            lbl = QLabel(str(val))
            lbl.setWordWrap(True)
            lbl.setStyleSheet("color: #c4cad1;")
            key = QLabel(k)
            key.setStyleSheet("color: #8a939d;")
            f.addRow(key, lbl)
        return g

    # --- helpers ------------------------------------------------------------------
    def _changed(self, fn: Callable, rebuild_panel: bool = False):
        def handler(*args):
            fn(*args)
            self.on_changed()
            if rebuild_panel:
                self.refresh()
        return handler

    @staticmethod
    def _form(title: str) -> tuple[QGroupBox, QFormLayout]:
        g = QGroupBox(title)
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)   # narrow panel: label above its field
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(6)
        return g, f

    def _check(self, value: bool, fn: Callable[[bool], None]) -> QCheckBox:
        cb = QCheckBox()
        cb.setChecked(value)
        cb.toggled.connect(self._changed(fn))
        return cb

    def _spin(self, value, lo, hi, step, fn, decimals=1) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(value)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(self._changed(fn))
        return s

    def _combo(self, items: list, current, fn, rebuild_panel=False) -> QComboBox:
        """``items``: values shown as they are, or (shown text, value) pairs; ``current`` and
        what ``fn`` gets are values."""
        c = QComboBox()
        # do not ask for the width of the longest item: the popup still shows it in full
        c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(4)
        values = []
        for it in items:
            text, value = it if isinstance(it, tuple) else (it, it)
            c.addItem(str(text))
            values.append(value)
        if current in values:
            c.setCurrentIndex(values.index(current))
        handler = self._changed(fn, rebuild_panel)
        c.currentIndexChanged.connect(lambda i: handler(values[i]) if 0 <= i < len(values) else None)
        return c

    def _cmap_combo(self, current: str, fn) -> "ColormapCombo":
        """The colour maps shown as their gradients (the name is the tooltip)."""
        from .colormap_combo import ColormapCombo

        c = ColormapCombo(COLORMAPS, current)
        handler = self._changed(fn, False)
        c.currentIndexChanged.connect(lambda _i: handler(c.currentData()) if c.currentData() else None)
        return c

    # --- groups ----------------------------------------------------------------
    def _display_group(self, layer: Layer) -> QGroupBox:
        g, f = self._form(tr("Display"))
        name = QLabel(layer.name)
        name.setStyleSheet("font-weight: 600;")
        name.setWordWrap(True)
        f.addRow(tr("Name"), name)
        f.addRow(tr("Visible"), self._check(layer.visible, layer.set_visible))

        op = QSlider(Qt.Horizontal)
        op.setRange(5, 100)
        op.setValue(int(round(layer.opacity * 100)))
        op_lbl = QLabel(f"{int(layer.opacity * 100)}%")
        op_lbl.setFixedWidth(34)
        op.valueChanged.connect(lambda v: op_lbl.setText(f"{v}%"))
        op.valueChanged.connect(self._changed(lambda v: layer.set_opacity(v / 100.0)))
        row = QHBoxLayout()
        row.addWidget(op, 1)
        row.addWidget(op_lbl)
        f.addRow(tr("Opacity"), row)

        if isinstance(layer, (StringLayer, SurfaceLayer)):
            has_styles = layer.styles is not None
            if has_styles:
                def set_mode(styles):
                    layer.use_styles = styles
                    layer.rebuild()
                f.addRow(tr("Colour from"), self._combo(
                    [(tr("Style file (.ssi)"), True), (tr("Single colour"), False)],
                    bool(layer.use_styles), set_mode))
            f.addRow(tr("Colour"), ColorButton(layer.color, self._changed(
                lambda rgb: (setattr(layer, "use_styles", False), layer.set_color(rgb)), True)))
        elif isinstance(layer, BlockModelLayer) and layer.color_by is None:
            f.addRow(tr("Colour"), ColorButton(layer.color, self._changed(layer.set_color)))

        if isinstance(layer, StringLayer):
            f.addRow(tr("Line width"), self._spin(layer.line_width, 0.5, 12, 0.5,
                                              lambda v: (setattr(layer, "line_width", v), layer.rebuild())))
            f.addRow(tr("Show points"), self._check(layer.show_points,
                                                lambda v: (setattr(layer, "show_points", v), layer.rebuild())))
        if isinstance(layer, SurfaceLayer):
            f.addRow(tr("Smooth shading"), self._check(layer.smooth,
                                                   lambda v: (setattr(layer, "smooth", v), layer.rebuild())))
            f.addRow(tr("Show edges"), self._check(layer.show_edges,
                                               lambda v: (setattr(layer, "show_edges", v), layer.rebuild())))
        if isinstance(layer, (StringLayer, SurfaceLayer)):
            role = self._combo([(tr(text), key) for key, text in ROLES.items()],
                               layer.role if layer.role in ROLES else "",
                               lambda k: setattr(layer, "role", k))
            role.setToolTip(tr("What this layer is in the geological model: Horizon / contact (the base of a "
                               "formation), Unconformity, Fault, Fold axial surface, Formation solid, Topography. "
                               "The drilling risk prediction reads the model from these roles."))
            f.addRow(tr("Role"), role)
        return g

    def _extra_group(self, layer: Layer) -> QGroupBox | None:
        if isinstance(layer, BlockModelLayer):
            return self._block_group(layer)
        if isinstance(layer, DrillholeLayer):
            return self._drillhole_group(layer)
        if isinstance(layer, SurveyLayer):
            return self._survey_group(layer)
        return None

    def _survey_group(self, layer: SurveyLayer) -> QGroupBox:
        g, f = self._form(tr("Survey"))
        drives = self._check(layer.show_drives, lambda v: (setattr(layer, "show_drives", v), layer.rebuild()))
        drives.setToolTip(tr("The drives: each pit's centre line swept with its rectangular profile"))
        f.addRow(tr("Drives"), drives)
        labels = self._check(layer.show_labels, lambda v: (setattr(layer, "show_labels", v), layer.rebuild()))
        labels.setToolTip(tr("Names of the survey stations"))
        f.addRow(tr("Station names"), labels)
        geo = self._check(layer.show_geology, lambda v: (setattr(layer, "show_geology", v), layer.rebuild()))
        geo.setToolTip(tr("The rocks mapped in the pits, on the walls and the back (Pit geology, Ctrl+J)"))
        f.addRow(tr("Mapped rocks"), geo)
        st = self._check(layer.show_structures, lambda v: (setattr(layer, "show_structures", v), layer.rebuild()))
        st.setToolTip(tr("The structures mapped in the pits: discs tilted to their dip direction and dip"))
        f.addRow(tr("Structures"), st)
        size = self._spin(float(layer.structure_size), 0.1, 20.0, 0.25,
                          lambda v: (setattr(layer, "structure_size", float(v)), layer.rebuild()), 2)
        size.setSuffix(" m")
        size.setToolTip(tr("Radius of a structure's disc"))
        f.addRow(tr("Disc radius"), size)
        sm = self._check(layer.show_samples, lambda v: (setattr(layer, "show_samples", v), layer.rebuild()))
        sm.setToolTip(tr("The samples taken in the pits: narrow bands at their height on the wall"))
        f.addRow(tr("Pit samples"), sm)
        hint = QLabel(tr("Select a pit under this layer for its colour, visibility, centre lines and stations."))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a939d;")
        f.addRow(hint)
        return g

    def _block_group(self, layer: BlockModelLayer) -> QGroupBox:
        g, f = self._form(tr("Block model"))
        bm = layer.data
        names = [(tr("(single colour)"), None)] + [n for n in bm.attribute_names
                                                   if n not in DEPLETION_ATTRS]      # mined: see "Mined blocks"

        def set_attr(t):
            layer.remove_actors()
            layer.color_by = t
            layer.value_min = layer.value_max = None
            layer.clim = None
            layer.rebuild()
        f.addRow(tr("Colour by"), self._combo(names, layer.color_by or None, set_attr, True))

        if layer.color_by and not bm.is_categorical(layer.color_by):
            f.addRow(tr("Colour map"), self._cmap_combo(layer.cmap,
                                                    lambda t: (setattr(layer, "cmap", t), layer.rebuild())))
            rng = layer.value_range() or (0.0, 1.0)
            span = max(rng[1] - rng[0], 1e-9)
            lo = layer.value_min if layer.value_min is not None else rng[0]
            hi = layer.value_max if layer.value_max is not None else rng[1]

            s_min = QDoubleSpinBox()
            s_max = QDoubleSpinBox()
            for s, val in ((s_min, lo), (s_max, hi)):
                s.setDecimals(3)
                s.setRange(rng[0] - span, rng[1] + span)
                s.setSingleStep(span / 50.0)
                s.setValue(val)
            apply_btn = QToolButton()
            apply_btn.setIcon(icon("filter-check-outline"))
            apply_btn.setToolTip(tr("Apply the filter: show the blocks from .. to"))

            def apply():
                layer.value_min = s_min.value() if s_min.value() > rng[0] else None
                layer.value_max = s_max.value() if s_max.value() < rng[1] else None
                layer.rebuild()
            apply_btn.clicked.connect(self._changed(apply, True))
            reset_btn = QToolButton()
            reset_btn.setIcon(icon("filter-remove-outline"))
            reset_btn.setToolTip(tr("Reset the filter: show all blocks"))

            def reset():
                layer.value_min = layer.value_max = None
                layer.rebuild()
            reset_btn.clicked.connect(self._changed(reset, True))
            f.addRow(tr("Show from"), s_min)
            f.addRow(tr("Show to"), s_max)
            for b in (apply_btn, reset_btn):
                b.setIconSize(QSize(18, 18))
                b.setAutoRaise(False)
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addStretch(1)
            row.addWidget(apply_btn)
            row.addWidget(reset_btn)
            f.addRow("", row)

        # constraint: conditions on DTMs and block values, edited in a dialog
        summary = QLabel(layer.constraint_summary() if layer.constraints else tr("none"))
        summary.setWordWrap(True)
        summary.setStyleSheet("color: #c4cad1;")
        missing = layer.missing_dtms()
        if missing:
            summary.setText(summary.text() + "\n" + tr("(not loaded: {names})", names=", ".join(missing)))
            summary.setStyleSheet("color: #e0a030;")
        edit = QToolButton()
        edit.setIcon(icon("filter-cog-outline"))
        edit.setToolTip(tr("Edit the constraint: DTMs and block values, combined by and / or"))

        def open_editor():
            from .constraint_dialog import ConstraintDialog

            def apply(conds, logic):
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    layer.constraints, layer.constraint_logic = conds, logic
                    layer.rebuild()
                finally:
                    QApplication.restoreOverrideCursor()
                self.on_changed()
                self.refresh()

            ConstraintDialog(layer, apply, self.window()).exec()
        edit.clicked.connect(open_editor)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(edit)
        f.addRow(tr("Constraint"), row)
        summary.setContentsMargins(8, 0, 0, 4)
        f.addRow(summary)                       # the whole width: one condition per line
        mined = self._combo([(tr(text), key) for key, text in MINED_DISPLAYS.items()],
                            layer.mined_display if layer.mined_display in MINED_DISPLAYS else "gray",
                            lambda k: (setattr(layer, "mined_display", k), layer.rebuild()))
        mined.setToolTip(tr("How blocks mined by the recorded cuts are shown: Show (like the other blocks), "
                            "Gray (plain gray), Transparent (their colour, see-through) or Hidden. A block counts "
                            "as mined when at least half of it is mined."))
        f.addRow(tr("Mined blocks"), mined)
        f.addRow(tr("Hide null blocks"), self._check(
            layer.hide_null, lambda v: (setattr(layer, "hide_null", v), layer.rebuild())))
        f.addRow(tr("Show block edges"), self._check(
            layer.show_edges, lambda v: (setattr(layer, "show_edges", v), layer.rebuild())))
        return g

    def _drillhole_group(self, layer: DrillholeLayer) -> QGroupBox:
        g, f = self._form(tr("Drillholes"))
        db = layer.data
        counts = layer.all_data.area_counts() if layer.all_data is not None else {}
        if counts:                                   # drilling areas (e.g. the history of distant areas)
            items = [(tr("Automatic (areas being drilled)"), layer.AREA_AUTO),
                     (tr("All areas ({n:,} holes)", n=len(layer.all_data.collars)), layer.AREA_ALL)]
            items += [(tr("{area} ({n:,} holes)", area=a, n=n), a) for a, n in counts.items()]
            box = self._combo(items, layer.area if layer.area in counts or layer.area == layer.AREA_ALL
                              else layer.AREA_AUTO, layer.set_area, True)
            box.setToolTip(tr("The drilling area whose holes are shown. Automatic: the areas with holes in work or "
                              "planned (every hole when there are none)."))
            f.addRow(tr("Area"), box)
        tables = list(db.intervals)
        if tables:
            def set_table(t):
                layer.table = t
                cols = layer.field_columns(t)
                layer.field_name = cols[0] if cols else None
                layer.category_colors.clear()
                layer.rebuild()
            f.addRow(tr("Interval table"), self._combo(tables, layer.table, set_table, True))
            if layer.table:
                def set_field(t):
                    layer.field_name = t
                    layer.category_colors.clear()
                    layer.rebuild()
                f.addRow(tr("Colour by"), self._combo(layer.field_columns(layer.table), layer.field_name,
                                                      set_field, True))
        dia = self._spin(layer.diameter, 0.0, 20.0, 0.5,
                         lambda v: (setattr(layer, "diameter", v), layer.rebuild()), decimals=1)
        dia.setSuffix(" m")
        dia.setSpecialValueText(tr("0 m (lines)"))
        dia.setToolTip(tr("Diameter of the drillhole cylinders in metres (0 draws thin lines - the fastest "
                          "for very many holes)"))
        f.addRow(tr("Diameter"), dia)
        f.addRow(tr("Hole labels"), self._check(layer.show_labels,
                                                lambda v: (setattr(layer, "show_labels", v), layer.rebuild())))
        legend = layer.legend()
        if legend:
            box = QWidget()
            lv = QVBoxLayout(box)
            lv.setContentsMargins(0, 2, 0, 0)
            lv.setSpacing(3)
            for code, desc, rgb in legend:
                r = QHBoxLayout()
                r.setSpacing(6)
                r.addWidget(_swatch(rgb))
                r.addWidget(QLabel(f"{code}" + (f"  —  {desc}" if desc else "")), 1)
                lv.addLayout(r)
            f.addRow(tr("Legend"), box)
        return g

    def _info_group(self, layer: Layer) -> QGroupBox:
        g, f = self._form(tr("Information"))
        for k, val in layer.info():
            lbl = QLabel(_breakable(str(val)))
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lbl.setStyleSheet("color: #c4cad1;")
            key = QLabel(k)
            key.setStyleSheet("color: #8a939d;")
            f.addRow(key, lbl)
        return g
