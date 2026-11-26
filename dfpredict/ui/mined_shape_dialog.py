"""Enter / edit one mined shape: date, name, From / To, checked against the ore.

The depth profile on the right shows the reserve tonnes under the outline level by level
(the part already mined by earlier cuts shaded). The two handles (From / To) can be
dragged; dragging between them moves both. The cut also shows in the view as a
see-through solid, and the figures underneath say what it mines.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QDate, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDateEdit, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from ..scene.production import (AXIS_WORD, MinedShape, metal_text, point_in_polygon, prism_mesh,
                                today)
from ..i18n import N_, tr
from . import theme
from .icons import icon

AXIS_LETTER = {0: "X", 1: "Y", 2: "Z"}
PLANE_TEXT = {"xy": N_("plan view (XY)"), "xz": N_("section XZ"), "yz": N_("section YZ")}
SNAP = 0.1
DEFAULT_THICKNESS = 3.0
BAND = QColor(255, 219, 26)
FROM_COLOR = QColor(80, 220, 120)
TO_COLOR = QColor(80, 190, 255)
ORE = QColor(214, 170, 60)
MINED = QColor(200, 70, 60)


def nice_step(span: float, target: int = 6) -> float:
    raw = max(span, 1e-9) / target
    k = 10 ** math.floor(math.log10(raw))
    return next(m * k for m in (1, 2, 5, 10) if m * k >= raw)


def snap(v: float) -> float:
    return round(v / SNAP) * SNAP


def outlines_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Two outlines (K, 2) overlap: a corner or the centre of one is inside the other."""
    for p, q in ((a, b), (b, a)):
        test = np.vstack([p, p.mean(axis=0)])
        if point_in_polygon(test[:, 0], test[:, 1], q).any():
            return True
    return False


# --- the depth profile -------------------------------------------------------------------------
class DepthProfile(QWidget):
    """Ore tonnes per level under the outline, with draggable From / To handles."""
    changed = Signal(float, float)        # from, to (after a drag)

    PAD_L, PAD_R, PAD_T, PAD_B = 58, 14, 26, 30

    def __init__(self, axis_letter: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.axis_letter = axis_letter
        self.profile = None                 # (edges, ore, old, (lo, hi)) or None
        self.a, self.b = 0.0, 0.0           # from, to
        self._drag = None                   # "a", "b", "band"
        self._grab = 0.0
        self.setToolTip(tr("Ore under the outline, level by level.\nDrag the From / To lines to set the "
                           "mined part; drag between them to move both."))

    def set_profile(self, profile) -> None:
        self.profile = profile
        self.update()

    def set_range(self, a: float, b: float) -> None:
        self.a, self.b = a, b
        self.update()

    # --- coordinates -------------------------------------------------------------------
    def _limits(self) -> tuple[float, float]:
        vals = [self.a, self.b]
        if self.profile is not None:
            vals += [float(self.profile[0][0]), float(self.profile[0][-1])]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.06, 1.0)
        return lo - pad, hi + pad

    def _plot(self) -> QRectF:
        return QRectF(self.PAD_L, self.PAD_T, max(self.width() - self.PAD_L - self.PAD_R, 10),
                      max(self.height() - self.PAD_T - self.PAD_B, 10))

    def _y(self, v: float) -> float:
        r = self._plot()
        lo, hi = self._limits()
        return r.bottom() - (v - lo) / (hi - lo) * r.height()

    def _v(self, y: float) -> float:
        r = self._plot()
        lo, hi = self._limits()
        return lo + (r.bottom() - y) / r.height() * (hi - lo)

    # --- painting ------------------------------------------------------------------------
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.INPUT))
        r = self._plot()
        lo, hi = self._limits()
        f = QFont("Segoe UI")
        f.setPointSizeF(8)
        p.setFont(f)
        muted = QColor(theme.MUTED)
        # level ticks
        step = nice_step(hi - lo)
        v = math.ceil(lo / step) * step
        while v <= hi:
            y = self._y(v)
            p.setPen(QPen(QColor(theme.BORDER), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(muted)
            p.drawText(QRectF(0, y - 8, self.PAD_L - 6, 16), Qt.AlignRight | Qt.AlignVCenter, f"{v:,.0f}")
            v += step
        p.setPen(muted)
        p.drawText(QRectF(0, 2, self.width(), 18), Qt.AlignHCenter, tr("Ore under the outline (t per level)"))
        p.drawText(QRectF(4, 2, self.PAD_L, 18), Qt.AlignLeft, self.axis_letter)
        # ore bars
        if self.profile is not None:
            edges, ore, old, (olo, ohi) = self.profile
            top = float(ore.max()) if len(ore) and ore.max() > 0 else 1.0
            for i in range(len(ore)):
                if ore[i] <= 0:
                    continue
                y0, y1 = self._y(edges[i + 1]), self._y(edges[i])
                w = ore[i] / top * r.width()
                p.fillRect(QRectF(r.left(), y0, w, max(y1 - y0, 1.0)), ORE)
                if old[i] > 0:
                    p.fillRect(QRectF(r.left(), y0, old[i] / top * r.width(), max(y1 - y0, 1.0)), MINED)
            p.setPen(muted)
            p.drawText(QRectF(r.left(), r.bottom() + 4, r.width(), 14), Qt.AlignRight,
                       tr("widest level {t:,.0f} t", t=top))
            # ore top / bottom marks
            pen = QPen(QColor(ORE).lighter(130), 1, Qt.DashLine)
            p.setPen(pen)
            for v, word, is_top in ((ohi, tr("ore top"), True), (olo, tr("ore bottom"), False)):
                y = self._y(v)
                p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
                p.drawText(QRectF(r.right() - 120, y - (15 if is_top else -1), 118, 14),
                           Qt.AlignRight, f"{word} {v:,.1f}")
        else:
            p.setPen(QColor(theme.DANGER))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, tr("No reserve under this outline"))
        # the cut (band) and the handles
        ya, yb = self._y(self.a), self._y(self.b)
        band = QColor(BAND)
        band.setAlpha(60)
        p.fillRect(QRectF(r.left(), min(ya, yb), r.width(), abs(ya - yb)), band)
        fm = p.fontMetrics()
        for y, col, key, val in ((ya, FROM_COLOR, N_("From"), self.a), (yb, TO_COLOR, N_("To"), self.b)):
            p.setPen(QPen(col, 2.5))
            p.drawLine(QPointF(r.left() - 4, y), QPointF(r.right(), y))
            p.setBrush(col)
            p.drawEllipse(QPointF(r.left() - 4, y), 4.5, 4.5)
            # the label on a dark chip, on the outer side of the band (never on the line)
            text = f"{tr(key)} {val:,.1f}"
            outer_below = (key == "From") == (self.a <= self.b)
            h = fm.height() + 2
            chip = QRectF(r.left() + 8, y + 4 if outer_below else y - 4 - h, fm.horizontalAdvance(text) + 10, h)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 22, 26, 215))
            p.drawRoundedRect(chip, 3, 3)
            p.setPen(col)
            p.drawText(chip, Qt.AlignCenter, text)
        p.setPen(QColor(theme.BORDER))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
        p.end()

    # --- dragging -------------------------------------------------------------------------
    def _hit(self, y: float) -> str | None:
        da, db = abs(y - self._y(self.a)), abs(y - self._y(self.b))
        if min(da, db) <= 7:
            return "a" if da <= db else "b"
        lo, hi = sorted((self._y(self.a), self._y(self.b)))
        return "band" if lo < y < hi else None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        y = e.position().y()
        self._drag = self._hit(y)
        self._grab = self._v(y)
        self._limits_at_press = self._limits()

    def mouseMoveEvent(self, e):
        y = e.position().y()
        if self._drag is None:
            h = self._hit(y)
            self.setCursor(Qt.SizeVerCursor if h in ("a", "b") else Qt.OpenHandCursor if h else Qt.ArrowCursor)
            return
        # keep the scale fixed while dragging (the limits follow the handles otherwise)
        r = self._plot()
        lo, hi = self._limits_at_press
        v = lo + (r.bottom() - y) / r.height() * (hi - lo)
        if self._drag == "a":
            self.a = snap(v)
        elif self._drag == "b":
            self.b = snap(v)
        else:
            d = snap(v - self._grab)
            if d:
                self.a, self.b, self._grab = self.a + d, self.b + d, self._grab + d
        self.update()
        self.changed.emit(self.a, self.b)

    def mouseReleaseEvent(self, _e):
        self._drag = None


# --- the dialog ----------------------------------------------------------------------------------
class MinedShapeDialog(QDialog):
    saved = Signal(object)          # MinedShape
    deleted = Signal(int)           # shape id

    def __init__(self, parent, scene, reserve, shape: MinedShape, editing: bool = False):
        super().__init__(parent)
        self.scene = scene
        self.reserve = reserve
        self.dep = reserve.dep
        self.log = reserve.log
        self.shape = shape
        self.editing = editing
        self.unit = reserve.setup.get("unit", "g/t")
        ia = shape.axes[2]
        self.ax = AXIS_LETTER[ia]
        self.setWindowTitle(tr("Edit mined shape") if editing else tr("New mined shape"))
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        # --- form ------------------------------------------------------------------
        self.name = QLineEdit(shape.name)
        self.name.setPlaceholderText(tr("e.g. Stope 3 lift 2"))
        self.date = QDateEdit(QDate.fromString(shape.date, "yyyy-MM-dd"))
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.note = QLineEdit(shape.note)
        self.note.setPlaceholderText("optional")
        outline = QLabel(tr("Drawn in the {plane}: {n} corners, {area:,.0f} m²", plane=tr(PLANE_TEXT[shape.plane]),
                            n=len(shape.points), area=shape.area()))
        outline.setStyleSheet(f"color: {theme.MUTED};")
        outline.setWordWrap(True)

        def spin(tip):
            s = QDoubleSpinBox()
            s.setRange(-1e7, 1e7)
            s.setDecimals(1)
            s.setSingleStep(0.5)
            s.setKeyboardTracking(False)
            s.setToolTip(tip)
            return s

        word = tr(AXIS_WORD[ia])
        self.s_from = spin(tr("Where the mined part starts ({axis})", axis=word))
        self.s_to = spin(tr("Where the mined part ends ({axis})", axis=word))
        self.s_thick = spin(tr("Thickness of the mined part (To - From); changing it moves To"))
        self.s_thick.setRange(0.0, 1e6)
        self.s_thick.setSuffix(" m")

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow(tr("Date"), self.date)
        form.addRow(tr("Name"), self.name)
        form.addRow(outline)
        form.addRow(tr("From  {axis}", axis=self.ax), self.s_from)
        form.addRow(tr("To  {axis}", axis=self.ax), self.s_to)
        form.addRow(tr("Thickness"), self.s_thick)

        self.b_bottom = self._btn("arrow-collapse-down", tr("Use ore bottom"),
                                  tr("Start at the bottom of the ore under the outline (keeps the thickness)"),
                                  self.use_ore_bottom)
        self.b_top = self._btn("arrow-collapse-up", tr("Use ore top"),
                               tr("End at the top of the ore under the outline (keeps the thickness)"),
                               self.use_ore_top)
        self.b_next = self._btn("skip-next", tr("Continue from last cut"),
                                tr("Start where the last cut under this outline ended (same thickness and "
                                   "direction)"),
                                self.continue_last)
        quick = QHBoxLayout()
        for b in (self.b_bottom, self.b_top, self.b_next):
            quick.addWidget(b)
        form.addRow(quick)
        # AI: the best From / To of this thickness under the outline
        self.c_aim = QComboBox()
        for key, text in (("metal", tr("most metal")), ("grade", tr("highest grade")), ("tonnes", tr("most ore"))):
            self.c_aim.addItem(text, key)
        self.c_aim.setToolTip(tr("What the suggestion looks for: the levels with the most metal, the highest grade, "
                                 "or the most ore (keeping the thickness)"))
        self.b_suggest = self._btn("creation", tr("Suggest best levels"),
                                   tr("Find the From / To of this thickness under the outline that gives the most "
                                      "(ore not mined yet only); you can change them afterwards"), self.suggest)
        aim = QHBoxLayout()
        aim.addWidget(self.b_suggest)
        aim.addWidget(QLabel(tr("for")))
        aim.addWidget(self.c_aim, 1)
        form.addRow(aim)
        self.ai_note = QLabel()
        self.ai_note.setWordWrap(True)
        self.ai_note.setStyleSheet("color: #7fb8ff;")
        self.ai_note.setVisible(False)
        form.addRow(self.ai_note)
        form.addRow(tr("Note"), self.note)

        # --- figures ------------------------------------------------------------
        box = QGroupBox(tr("This cut mines"))
        g = QGridLayout(box)
        self.f_new = QLabel()
        self.f_new.setStyleSheet("font-size: 11pt; font-weight: 600;")
        self.f_metal = QLabel()
        self.f_old = QLabel()
        self.f_blocks = QLabel()
        self.f_fill = QLabel()
        rows = ((tr("Ore"), self.f_new), (tr("Metal"), self.f_metal), (tr("Already mined"), self.f_old),
                (tr("Blocks touched"), self.f_blocks), (tr("Ore in the cut"), self.f_fill))
        for i, (k, w) in enumerate(rows):
            lab = QLabel(k)
            lab.setStyleSheet(f"color: {theme.MUTED};")
            g.addWidget(lab, i, 0)
            g.addWidget(w, i, 1)
        self.f_old.setToolTip(tr("Ore inside this cut that earlier cuts mined already - it is not counted again"))
        self.f_fill.setToolTip(tr("How much of the cut's volume is reserve (the rest is outside the ore solid "
                                  "or has no grade)"))
        g.setColumnStretch(1, 1)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #ffb347;")

        left = QVBoxLayout()
        left.addLayout(form)
        left.addWidget(box)
        left.addWidget(self.warn)
        left.addStretch(1)

        self.profile = DepthProfile(self.ax)
        self.profile.changed.connect(self._from_profile)
        legend = QLabel(f"<span style='color:{ORE.name()}'>■</span> {tr('ore')} &nbsp; "
                        f"<span style='color:{MINED.name()}'>■</span> {tr('mined before')} &nbsp; "
                        f"<span style='color:{BAND.name()}'>■</span> {tr('this cut')}")
        legend.setStyleSheet(f"color: {theme.MUTED};")
        right = QVBoxLayout()
        right.addWidget(self.profile, 1)
        right.addWidget(legend)

        body = QHBoxLayout()
        body.addLayout(left, 3)
        body.addLayout(right, 2)

        buttons = QHBoxLayout()
        if editing:
            b_del = QPushButton(icon("delete-outline"), tr("Delete"))
            b_del.clicked.connect(self._delete)
            buttons.addWidget(b_del)
        buttons.addStretch(1)
        b_cancel = QPushButton(tr("Cancel"))
        b_cancel.clicked.connect(self.reject)
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save"))
        self.b_save.clicked.connect(self._save)
        buttons.addWidget(b_cancel)
        buttons.addWidget(self.b_save)

        root = QVBoxLayout(self)
        root.addLayout(body, 1)
        root.addLayout(buttons)
        self.resize(820, 520)

        # --- data ---------------------------------------------------------------
        self._prof = self.dep.profile(shape) if self.dep is not None else None
        self.profile.set_profile(self._prof)
        self._last = self._last_cut()
        self.b_next.setEnabled(self._last is not None)
        if self._last is None:
            self.b_next.setToolTip(tr("No earlier cut under this outline"))
        self.b_bottom.setEnabled(self._prof is not None)
        self.b_top.setEnabled(self._prof is not None)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._update_preview)
        self._busy = False
        if not editing:
            self._default_range()
        self._set(shape.from_, shape.to)
        for s in (self.s_from, self.s_to):
            s.valueChanged.connect(self._from_spins)
        self.s_thick.valueChanged.connect(self._from_thickness)
        self.date.dateChanged.connect(lambda _d: self._timer.start())
        self.finished.connect(lambda _r: self.scene.clear_preview_solid())

    def keyPressEvent(self, e):
        # Enter confirms the value being typed; it does not press a button (Save, ...).
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):
            e.accept()
            return
        super().keyPressEvent(e)

    def _btn(self, ic, text, tip, slot) -> QPushButton:
        b = QPushButton(icon(ic), text)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        return b

    # --- defaults and quick buttons -------------------------------------------------
    def _last_cut(self) -> MinedShape | None:
        """The latest saved cut in the same view whose outline overlaps this one."""
        mine = self.shape.outline()
        for sh in reversed(self.log.sorted_shapes()):
            if sh.id == self.shape.id or sh.plane != self.shape.plane:
                continue
            if outlines_overlap(mine, sh.outline()):
                return sh
        return None

    def _thickness(self) -> float:
        for sh in reversed(self.log.sorted_shapes()):
            if sh.depth > 0 and not sh.is_solid:
                return sh.depth
        return DEFAULT_THICKNESS

    def _default_range(self) -> None:
        if self._last is not None:
            self.continue_last(apply=False)
        elif self._prof is not None:
            lo = snap(self._prof[3][0])
            self.shape.from_, self.shape.to = lo, snap(lo + self._thickness())
        else:
            self.shape.from_, self.shape.to = 0.0, self._thickness()

    def use_ore_bottom(self) -> None:
        if self._prof is None:
            return
        t = abs(self.s_thick.value()) or self._thickness()
        lo = snap(self._prof[3][0])
        self._set(lo, snap(lo + t))

    def use_ore_top(self) -> None:
        if self._prof is None:
            return
        t = abs(self.s_thick.value()) or self._thickness()
        hi = snap(self._prof[3][1])
        self._set(snap(hi - t), hi)

    def continue_last(self, apply: bool = True) -> None:
        last = self._last
        if last is None:
            return
        step = last.to - last.from_
        if not step:
            step = self._thickness()
        a, b = snap(last.to), snap(last.to + step)
        if apply:
            self._set(a, b)
        else:
            self.shape.from_, self.shape.to = a, b

    def suggest(self) -> None:
        """Set From / To to the best window of the current thickness (AI suggestion, explained)."""
        from ..ai.reserve_advisor import suggest_levels

        if self.dep is None:
            return
        t = abs(self.s_thick.value()) or self._thickness()
        before = self.dep.preview(self.current())["new"]
        sug = suggest_levels(self.dep, self.current(), t, self.c_aim.currentData())
        self.ai_note.setVisible(True)
        if not sug.ok:
            self.ai_note.setText("✦ " + sug.message)
            return
        a, b = sug.from_, sug.to
        self._set(a, b)
        gain = ""
        if before.metal > 0 and sug.metal > before.metal * 1.005:
            gain = " " + tr("That is {p:.0%} more metal than the levels you had.", p=sug.metal / before.metal - 1)
        alts = "; ".join(tr("{a:,.1f} → {b:,.1f}: {t:,.0f} t @ {g:,.2f}", a=x, b=y, t=tt, g=mm / tt if tt else 0)
                         for x, y, tt, mm in sug.alternatives)
        self.ai_note.setText("✦ " + tr("Suggested {a:,.1f} → {b:,.1f} ({ax}): {t:,.0f} t of ore not mined yet at "
                                       "{g:,.2f} {unit}.", a=a, b=b, ax=self.ax, t=sug.tonnes, g=sug.grade,
                                       unit=self.unit) + gain
                             + (("  " + tr("Next best: {alts}.", alts=alts)) if alts else ""))

    # --- linked fields ---------------------------------------------------------------
    def _set(self, a: float, b: float) -> None:
        self._busy = True
        self.s_from.setValue(a)
        self.s_to.setValue(b)
        self.s_thick.setValue(abs(b - a))
        self._busy = False
        self.profile.set_range(a, b)
        self._timer.start()

    def _from_spins(self, _v=None) -> None:
        if self._busy:
            return
        self._set(self.s_from.value(), self.s_to.value())

    def _from_thickness(self, t: float) -> None:
        if self._busy:
            return
        a = self.s_from.value()
        sign = -1.0 if self.s_to.value() < a else 1.0
        self._set(a, a + sign * t)

    def _from_profile(self, a: float, b: float) -> None:
        self._busy = True
        self.s_from.setValue(a)
        self.s_to.setValue(b)
        self.s_thick.setValue(abs(b - a))
        self._busy = False
        self._timer.start()

    # --- preview ----------------------------------------------------------------------
    def current(self) -> MinedShape:
        sh = self.shape
        return MinedShape(sh.id, self.date.date().toString("yyyy-MM-dd"), self.name.text().strip(),
                          sh.plane, sh.points, float(self.s_from.value()), float(self.s_to.value()),
                          self.note.text().strip())

    def _update_preview(self) -> None:
        sh = self.current()
        self.scene.set_preview_solid(prism_mesh(sh, self.scene.origin))
        warns = []
        if sh.depth <= 0:
            warns.append(tr("From and To are the same - set the thickness of the mined part."))
        res = self.dep.preview(sh) if self.dep is not None else None
        if res is None:
            for w in (self.f_new, self.f_metal, self.f_old, self.f_blocks, self.f_fill):
                w.setText("-")
            warns.append(self.reserve.error or tr("The reserve is not set up."))
        else:
            new, old = res["new"], res["already"]
            self.f_new.setText(tr("{t:,.0f} t  @ {g:,.2f} {unit}", t=new.tonnes, g=new.grade, unit=self.unit))
            self.f_metal.setText(metal_text(new.metal, self.unit))
            self.f_old.setText(tr("{t:,.0f} t (not counted again)", t=old.tonnes) if old.tonnes > 0.5
                              else tr("none"))
            self.f_blocks.setText(f"{res['blocks']:,}")
            self.f_fill.setText(f"{res['fill']:.0%}")
            if sh.depth > 0 and new.tonnes <= 0.5 and old.tonnes <= 0.5:
                warns.append(tr("No ore between From and To. Check the levels against the profile "
                                "on the right (or use the ore bottom / top buttons)."))
            elif sh.depth > 0 and new.tonnes <= 0.5:
                warns.append(tr("Everything in this cut was mined already by earlier cuts."))
            else:
                if old.tonnes > new.tonnes:
                    warns.append(tr("Most of this cut was mined already by earlier cuts."))
                if 0 < res["fill"] < 0.25:
                    warns.append(tr("Only {fill:.0%} of the cut is ore - much of it is outside the "
                                    "reserve. Is From / To right?", fill=res["fill"]))
        if sh.date > today():
            warns.append(tr("The date is in the future."))
        self.warn.setText("\n".join("⚠ " + w for w in warns))
        self.b_save.setEnabled(sh.depth > 0)
        self._result = res

    # --- save / delete ----------------------------------------------------------------
    def _save(self) -> None:
        self._timer.stop()
        self._update_preview()
        sh = self.current()
        if not sh.name:
            sh.name = tr("Cut {n}", n=sh.id)
        res = getattr(self, "_result", None)
        if res is not None and res["new"].tonnes <= 0.5:
            ans = QMessageBox.question(self, tr("Save mined shape"),
                                       tr("This cut mines no ore (nothing new between From and To).\n\n"
                                          "Save it anyway?"))
            if ans != QMessageBox.Yes:
                return
        self.written = True                 # the receiver sets False when the database write failed
        self.saved.emit(sh)
        if self.written:
            self.accept()

    def _delete(self) -> None:
        ans = QMessageBox.question(self, tr("Delete mined shape"),
                                   tr("Delete {name} ({date})?\n\nIts tonnes go back to the remaining reserve.",
                                      name=self.shape.name or tr("this shape"), date=self.shape.date))
        if ans == QMessageBox.Yes:
            self.written = True
            self.deleted.emit(self.shape.id)
            if self.written:
                self.accept()


def new_shape(log, plane: str, points) -> MinedShape:
    """A new shape for an outline just drawn (From / To filled in by the dialog)."""
    sid = log.next_id()
    pts = [[float(a), float(b)] for a, b in np.asarray(points, dtype=np.float64)]
    return MinedShape(sid, today(), tr("Cut {n}", n=sid), plane, pts, 0.0, 0.0)



class MinedSolidDialog(QDialog):
    """A mined solid (a closed .dtm, e.g. a surveyed stope): date, name, note and what it
    mines - checked live against the reserve like a drawn cut."""

    saved = Signal(object)          # MinedShape
    deleted = Signal(int)           # shape id

    def __init__(self, parent, scene, reserve, shape: MinedShape, editing: bool = False):
        super().__init__(parent)
        self.scene = scene
        self.reserve = reserve
        self.dep = reserve.dep
        self.shape = shape
        self.editing = editing
        self.unit = reserve.setup.get("unit", "g/t")
        self.setWindowTitle(tr("Edit mined solid") if editing else tr("New mined solid"))
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)

        self.name = QLineEdit(shape.name)
        self.name.setPlaceholderText(tr("e.g. Stope 3 (survey 2026-03)"))
        self.date = QDateEdit(QDate.fromString(shape.date, "yyyy-MM-dd"))
        self.date.setCalendarPopup(True)
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.note = QLineEdit(shape.note)
        self.note.setPlaceholderText("optional")
        info = QLabel(tr("Solid from {source}: {n:,} triangles, {volume:,.0f} m³, elevation {lo:,.1f} .. {hi:,.1f}",
                         source=shape.source or tr("a .dtm file"), n=len(shape.triangles), volume=shape.volume(),
                         lo=shape.from_, hi=shape.to))
        info.setStyleSheet(f"color: {theme.MUTED};")
        info.setWordWrap(True)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow(tr("Date"), self.date)
        form.addRow(tr("Name"), self.name)
        form.addRow(info)
        form.addRow(tr("Note"), self.note)

        box = QGroupBox(tr("This solid mines"))
        g = QGridLayout(box)
        self.f_new = QLabel()
        self.f_new.setStyleSheet("font-size: 11pt; font-weight: 600;")
        self.f_metal, self.f_old, self.f_blocks, self.f_fill = QLabel(), QLabel(), QLabel(), QLabel()
        rows = ((tr("Ore"), self.f_new), (tr("Metal"), self.f_metal), (tr("Already mined"), self.f_old),
                (tr("Blocks touched"), self.f_blocks), (tr("Ore in the solid"), self.f_fill))
        for i, (k, w) in enumerate(rows):
            lab = QLabel(k)
            lab.setStyleSheet(f"color: {theme.MUTED};")
            g.addWidget(lab, i, 0)
            g.addWidget(w, i, 1)
        self.f_old.setToolTip(tr("Ore inside this solid that earlier cuts mined already - it is not counted again"))
        self.f_fill.setToolTip(tr("How much of the solid's volume is reserve (the rest is outside the ore solid)"))
        g.setColumnStretch(1, 1)
        self.warn = QLabel()
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #ffb347;")

        buttons = QHBoxLayout()
        if editing:
            b_del = QPushButton(icon("delete-outline"), tr("Delete"))
            b_del.clicked.connect(self._delete)
            buttons.addWidget(b_del)
        buttons.addStretch(1)
        b_cancel = QPushButton(tr("Cancel"))
        b_cancel.clicked.connect(self.reject)
        self.b_save = QPushButton(icon("content-save-outline"), tr("Save"))
        self.b_save.clicked.connect(self._save)
        buttons.addWidget(b_cancel)
        buttons.addWidget(self.b_save)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(box)
        root.addWidget(self.warn)
        root.addStretch(1)
        root.addLayout(buttons)
        self.resize(520, 380)
        self.date.dateChanged.connect(lambda *_: self._update_preview())
        self.finished.connect(lambda _r: self.scene.clear_preview_solid())
        self._update_preview()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter):      # Enter does not press Save
            e.accept()
            return
        super().keyPressEvent(e)

    def current(self) -> MinedShape:
        sh = self.shape
        return MinedShape.from_solid(sh.id, self.date.date().toString("yyyy-MM-dd"), self.name.text().strip(),
                                     sh.vertices, sh.triangles, sh.source, self.note.text().strip())

    def _update_preview(self) -> None:
        sh = self.current()
        if self.scene.origin is not None:
            self.scene.set_preview_solid(prism_mesh(sh, self.scene.origin))
        warns = []
        res = self.dep.preview(sh) if self.dep is not None else None
        if res is None:
            for w in (self.f_new, self.f_metal, self.f_old, self.f_blocks, self.f_fill):
                w.setText("-")
            warns.append(self.reserve.error or tr("The reserve is not set up."))
        else:
            new, old = res["new"], res["already"]
            self.f_new.setText(tr("{t:,.0f} t  @ {g:,.2f} {unit}", t=new.tonnes, g=new.grade, unit=self.unit))
            self.f_metal.setText(metal_text(new.metal, self.unit))
            self.f_old.setText(tr("{t:,.0f} t (not counted again)", t=old.tonnes) if old.tonnes > 0.5
                              else tr("none"))
            self.f_blocks.setText(f"{res['blocks']:,}")
            self.f_fill.setText(f"{res['fill']:.0%}")
            if new.tonnes <= 0.5 and old.tonnes <= 0.5:
                warns.append(tr("The solid holds no reserve ore - is it the right file / place?"))
            elif new.tonnes <= 0.5:
                warns.append(tr("Everything in this solid was mined already by earlier cuts."))
            elif old.tonnes > new.tonnes:
                warns.append(tr("Most of this solid was mined already by earlier cuts."))
        if sh.date > today():
            warns.append(tr("The date is in the future."))
        self.warn.setText("\n".join("⚠ " + w for w in warns))
        self._result = res

    def _save(self) -> None:
        self._update_preview()
        sh = self.current()
        if not sh.name:
            sh.name = tr("Solid {n}", n=sh.id)
        res = getattr(self, "_result", None)
        if res is not None and res["new"].tonnes <= 0.5:
            if QMessageBox.question(self, tr("Save mined solid"),
                                    tr("This solid mines no ore (nothing new inside it).\n\n"
                                       "Save it anyway?")) != QMessageBox.Yes:
                return
        self.written = True                 # the receiver sets False when the database write failed
        self.saved.emit(sh)
        if self.written:
            self.accept()

    def _delete(self) -> None:
        if QMessageBox.question(self, tr("Delete mined solid"),
                                tr("Delete {name} ({date})?\n\nIts tonnes go back to the remaining reserve.",
                                   name=self.shape.name or tr("this solid"),
                                   date=self.shape.date)) == QMessageBox.Yes:
            self.written = True
            self.deleted.emit(self.shape.id)
            if self.written:
                self.accept()


def load_mined_solid(path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, triangles) of the closed solid(s) in a Surpac .dtm; ValueError with a message
    for the user when the file holds no closed solid."""
    from ..io import read_dtm
    from ..scene.solid import closed_mesh

    mesh = closed_mesh(read_dtm(path))
    if mesh is None:
        raise ValueError(tr("This .dtm holds no closed solid (it is a surface, or the solid has holes). "
                            "A mined solid must be closed, like a stope solid."))
    return mesh
