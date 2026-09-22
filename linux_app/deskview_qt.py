"""NAST Deskview — cross-platform Qt port of the WPF desktop app.

Same layout and behaviour as the C# original: left rail (REC / MAP / 3D),
header with dataset + FRAMES + backend status, a fully native recorder tab
(frame stage with ROI drawing, transport bar, objects/ops/jobs column) and
two embedded Chromium panes (QtWebEngine) for the map and 3D viewers.

Runs anywhere PySide6 does (Linux, Windows, macOS):
    venv/bin/python linux_app/deskview_qt.py
Env:
    NAST_API   service base   (default http://127.0.0.1:8130)
    NAST_ROOT  street_video   (default <repo>/viewer/scenes/street_video)
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# WebGL2 for the map and object viewers: Chromium refuses it on a GPU it does not trust -- a laptop whose desktop runs
# on software GL (llvmpipe: an Intel iGPU newer than the Mesa of the distro) gets no WebGL at all without this flag.
_flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
if "--ignore-gpu-blocklist" not in _flags:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (_flags + " --ignore-gpu-blocklist").strip()

from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QUrl
from PySide6.QtGui import (QAction, QColor, QGuiApplication, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QTransform)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QStackedWidget, QVBoxLayout, QWidget)
from PySide6.QtWebEngineWidgets import QWebEngineView

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API = os.environ.get("NAST_API", "http://127.0.0.1:8130")
ROOT = Path(os.environ.get("NAST_ROOT", str(REPO / "viewer" / "scenes" / "street_video")))

# ---- palette (mirrors the WPF resource dictionary) --------------------------------
BG, PANEL, PANEL2 = "#0b0c0e", "#101216", "#16181d"
LINE, LINE2 = "#1d2026", "#262a31"
FG, FG2, DIM, FAINT = "#e8eaee", "#c7ccd4", "#9aa2ad", "#6f7683"
ACCENT, ON_ACCENT, GO, BAD = "#3d7bfd", "#ffffff", "#55dc78", "#ff5f5f"

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG}; font-size: 12px; }}
QFrame#rail {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
QFrame#topbar, QFrame#panel {{ background: {PANEL}; border-bottom: 1px solid {LINE}; }}
QFrame#ops {{ background: {PANEL}; border-left: 1px solid {LINE}; }}
QFrame#strip {{ background: {BG}; border-bottom: 1px solid {PANEL2}; }}
QFrame#transport {{ background: {PANEL}; border-top: 1px solid {LINE}; }}
QLabel#eyebrow {{ color: {FAINT}; font-size: 9px; letter-spacing: 1px; }}
QLabel#mono {{ font-family: Consolas, monospace; }}
QPushButton {{ background: {PANEL2}; color: {FG2}; border: 1px solid {LINE2};
               border-radius: 7px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton#accent {{ background: {ACCENT}; color: {ON_ACCENT}; border: none; font-weight: 600; }}
QPushButton#railbtn {{ background: transparent; border: none; border-radius: 9px;
                       color: {FAINT}; font-size: 9px; padding: 8px 2px; }}
QPushButton#railbtn[active="true"] {{ background: {PANEL2}; color: {ACCENT}; }}
QPushButton#seg[active="true"] {{ background: {ACCENT}; color: {ON_ACCENT}; border: none; }}
QComboBox {{ background: {PANEL2}; border: 1px solid {LINE2}; border-radius: 7px; padding: 5px 9px; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; color: {FG}; selection-background-color: {ACCENT}; }}
QLineEdit {{ background: {PANEL2}; border: 1px solid {LINE2}; border-radius: 7px; padding: 6px 9px; }}
QSlider::groove:horizontal {{ height: 4px; background: {PANEL2}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px; background: {ACCENT}; }}
QScrollArea {{ border: none; }}
"""


def api_get(path, timeout=6):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def api_post(path, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(API + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def api_delete(path):
    req = urllib.request.Request(API + path, method="DELETE")
    urllib.request.urlopen(req, timeout=30).read()


def eyebrow(text):
    l = QLabel(text.upper())
    l.setObjectName("eyebrow")
    return l


# ================================ recorder stage ==================================
class Stage(QLabel):
    """frame display + ROI drawing overlay (image coordinates = rotated frame)"""

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img = None                      # QPixmap of the current frame (rotated)
        self.punch = None                    # variant pixmap for the ROI-normals mode
        self.draw_pts = []                   # in-progress polygon/rect pts (image coords)
        self.setMouseTracking(True)

    # widget <-> image coordinate mapping (uniform fit)
    def _geom(self):
        if self.img is None:
            return None
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        ox, oy = (self.width() - w) / 2, (self.height() - h) / 2
        return s, ox, oy

    def to_img(self, pos):
        g = self._geom()
        if g is None:
            return None
        s, ox, oy = g
        return QPointF((pos.x() - ox) / s, (pos.y() - oy) / s)

    def mousePressEvent(self, e):
        if self.win.tool is None or self.img is None:
            return
        p = self.to_img(e.position())
        if self.win.tool == "rect":
            self.draw_pts = [p, p]
        else:
            if e.type() == e.Type.MouseButtonDblClick:
                return
            self.draw_pts.append(p)
        self.update()

    def mouseMoveEvent(self, e):
        if self.win.tool == "rect" and len(self.draw_pts) == 2:
            self.draw_pts[1] = self.to_img(e.position())
            self.update()

    def mouseReleaseEvent(self, e):
        if self.win.tool == "rect" and len(self.draw_pts) == 2:
            a, b = self.draw_pts
            if abs(a.x() - b.x()) > 4 and abs(a.y() - b.y()) > 4:
                self.win.commit_roi("rect", [a, b])
            self.draw_pts = []
            self.update()

    def mouseDoubleClickEvent(self, e):
        if self.win.tool == "poly" and len(self.draw_pts) >= 3:
            self.win.commit_roi("poly", list(self.draw_pts))
            self.draw_pts = []
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.img is None:
            return
        g = self._geom()
        if g is None:
            return
        s, ox, oy = g
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        target = QRectF(ox, oy, self.img.width() * s, self.img.height() * s)
        p.drawPixmap(target, self.img, QRectF(self.img.rect()))
        # ROI-normals punch-through: variant image clipped to the ROI shapes
        if self.punch is not None:
            path = QPainterPath()
            for o in self.win.rois_on_frame():
                pts = o["_pts"]
                if o["kind"] == "rect" and len(pts) >= 2:
                    r = QRectF(QPointF(ox + min(pts[0][0], pts[1][0]) * s, oy + min(pts[0][1], pts[1][1]) * s),
                               QPointF(ox + max(pts[0][0], pts[1][0]) * s, oy + max(pts[0][1], pts[1][1]) * s))
                    path.addRect(r)
                elif len(pts) >= 3:
                    path.moveTo(ox + pts[0][0] * s, oy + pts[0][1] * s)
                    for q in pts[1:]:
                        path.lineTo(ox + q[0] * s, oy + q[1] * s)
                    path.closeSubpath()
            if not path.isEmpty():
                p.save(); p.setClipPath(path)
                p.drawPixmap(target, self.punch, QRectF(self.punch.rect()))
                p.restore()
        # existing ROIs on this frame
        pen = QPen(QColor(ACCENT)); pen.setWidthF(1.6)
        p.setPen(pen)
        for o in self.win.rois_on_frame():
            pts = o["_pts"]
            if o["kind"] == "rect" and len(pts) >= 2:
                p.drawRect(QRectF(QPointF(ox + min(pts[0][0], pts[1][0]) * s, oy + min(pts[0][1], pts[1][1]) * s),
                                  QPointF(ox + max(pts[0][0], pts[1][0]) * s, oy + max(pts[0][1], pts[1][1]) * s)))
            elif len(pts) >= 3:
                for i in range(len(pts)):
                    a, b = pts[i], pts[(i + 1) % len(pts)]
                    p.drawLine(QPointF(ox + a[0] * s, oy + a[1] * s), QPointF(ox + b[0] * s, oy + b[1] * s))
        # in-progress drawing
        pen = QPen(QColor(GO)); pen.setWidthF(1.6); pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        d = self.draw_pts
        if self.win.tool == "rect" and len(d) == 2:
            p.drawRect(QRectF(QPointF(ox + d[0].x() * s, oy + d[0].y() * s),
                              QPointF(ox + d[1].x() * s, oy + d[1].y() * s)))
        elif self.win.tool == "poly" and len(d) >= 1:
            for i in range(len(d) - 1):
                p.drawLine(QPointF(ox + d[i].x() * s, oy + d[i].y() * s),
                           QPointF(ox + d[i + 1].x() * s, oy + d[i + 1].y() * s))
        p.end()


# ==================================== window ======================================
class Deskview(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deskview")
        self.resize(1500, 950)
        self.cam = "A"
        self.layer = "rgb"
        self.variant = "nxyz"
        self.tool = None
        self.pos = 0
        self.playing = False
        self.reverse = False
        self.speed = 1.0
        self.sel = -1
        self.objects = []
        self.frames = {"A": [], "B": []}
        self.cache = {}
        self.cache_order = []

        root_w = QWidget(); root_l = QHBoxLayout(root_w)
        root_l.setContentsMargins(0, 0, 0, 0); root_l.setSpacing(0)
        root_l.addWidget(self._rail())
        main = QVBoxLayout(); main.setContentsMargins(0, 0, 0, 0); main.setSpacing(0)
        main.addWidget(self._topbar())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._pane_rec())
        self.stack.addWidget(self._pane_web("map"))
        self.stack.addWidget(self._pane_3d())
        main.addWidget(self.stack, 1)
        mw = QWidget(); mw.setLayout(main)
        root_l.addWidget(mw, 1)
        self.setCentralWidget(root_w)

        self.play_timer = QTimer(self); self.play_timer.timeout.connect(self._tick)
        self.play_timer.start(66)
        self.jobs_timer = QTimer(self); self.jobs_timer.timeout.connect(self.poll_jobs)
        self.jobs_timer.start(2500)

        self._decode_overlay()
        self.load_folder()
        self.connect_api()
        self.sync_dataset()
        self.decode_timer = QTimer(self); self.decode_timer.timeout.connect(self.poll_decode)
        self.decode_timer.start(1000)
        self.poll_decode()

    # ------------------------------- chrome ---------------------------------------
    def _rail(self):
        f = QFrame(); f.setObjectName("rail"); f.setFixedWidth(68)
        v = QVBoxLayout(f); v.setContentsMargins(8, 10, 8, 12); v.setSpacing(6)
        logo = QLabel("DV"); logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(f"background:{ACCENT}; color:{ON_ACCENT}; font-weight:700;"
                           f"border-radius:6px; min-height:26px; max-height:26px; font-size:11px;")
        v.addWidget(logo); v.addSpacing(10)
        self.rail_btns = []
        for i, name in enumerate(("REC", "MAP", "3D")):
            b = QPushButton(name); b.setObjectName("railbtn"); b.setFixedHeight(52)
            b.clicked.connect(lambda _, k=i: self.switch_tab(k))
            v.addWidget(b); self.rail_btns.append(b)
        v.addStretch(1)
        self.api_dot = QLabel("● API"); self.api_dot.setAlignment(Qt.AlignCenter)
        self.api_dot.setStyleSheet(f"color:{BAD}; font-size:9px;")
        v.addWidget(self.api_dot)
        self._mark_rail(0)
        return f

    def _mark_rail(self, i):
        for k, b in enumerate(self.rail_btns):
            b.setProperty("active", "true" if k == i else "false")
            b.style().unpolish(b); b.style().polish(b)

    def _topbar(self):
        f = QFrame(); f.setObjectName("topbar"); f.setFixedHeight(48)
        h = QHBoxLayout(f); h.setContentsMargins(18, 0, 18, 0)
        self.lbl_dataset = QLabel("street_video"); self.lbl_dataset.setStyleSheet("font-weight:600; font-size:13px;")
        self.lbl_path = QLabel(str(ROOT)); self.lbl_path.setObjectName("mono")
        self.lbl_path.setStyleSheet(f"color:{FAINT}; font-size:10px;")
        self.lbl_frames = QLabel("FRAMES  —"); self.lbl_frames.setObjectName("mono")
        self.lbl_frames.setStyleSheet(f"background:{PANEL2}; border:1px solid {LINE2};"
                                      f"border-radius:6px; padding:4px 9px; font-size:11px;")
        h.addWidget(self.lbl_dataset); h.addSpacing(9); h.addWidget(self.lbl_path)
        h.addSpacing(16); h.addWidget(self.lbl_frames); h.addStretch(1)
        self.lbl_api = QLabel("Backend…"); self.lbl_api.setObjectName("mono")
        self.lbl_api.setStyleSheet(f"color:{DIM}; font-size:11px;")
        b = QPushButton("Open data folder"); b.clicked.connect(self.open_folder)
        h.addWidget(self.lbl_api); h.addSpacing(14); h.addWidget(b)
        return f

    # ------------------------------ recorder pane ---------------------------------
    def _seg_button(self, text, cb):
        b = QPushButton(text); b.setObjectName("seg"); b.clicked.connect(cb)
        return b

    def _pane_rec(self):
        pane = QWidget(); h = QHBoxLayout(pane); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        left = QVBoxLayout(); left.setContentsMargins(0, 0, 0, 0); left.setSpacing(0)

        strip = QFrame(); strip.setObjectName("strip"); strip.setFixedHeight(52)
        sh = QHBoxLayout(strip); sh.setContentsMargins(18, 0, 18, 0)
        sh.addWidget(eyebrow("cam")); sh.addSpacing(6)
        self.btn_cam = {}
        for c in ("A", "B"):
            b = self._seg_button(c, lambda _, cc=c: self.set_cam(cc))
            self.btn_cam[c] = b; sh.addWidget(b)
        sh.addSpacing(20); sh.addWidget(eyebrow("layer")); sh.addSpacing(6)
        self.btn_layer = {}
        for key, text in (("rgb", "RGB"), ("nxyz", "Normals"), ("roinx", "ROI normals")):
            b = self._seg_button(text, lambda _, kk=key: self.set_layer(kk))
            self.btn_layer[key] = b; sh.addWidget(b)
        self.cmb_variant = QComboBox(); self.cmb_variant.setFixedWidth(180)
        self.cmb_variant.currentIndexChanged.connect(self._variant_changed)
        sh.addSpacing(8); sh.addWidget(self.cmb_variant)
        sh.addSpacing(20); sh.addWidget(eyebrow("draw")); sh.addSpacing(6)
        self.btn_tool = {}
        for key, text in (("rect", "Rectangle"), ("poly", "Polygon")):
            b = self._seg_button(text, lambda _, kk=key: self.set_tool(kk))
            self.btn_tool[key] = b; sh.addWidget(b)
        self.txt_label = QLineEdit(); self.txt_label.setPlaceholderText("Object name (lamp, sign…)")
        self.txt_label.setFixedWidth(150)
        sh.addSpacing(8); sh.addWidget(self.txt_label)
        hint = QLabel("Drag to box · double-click closes a polygon")
        hint.setStyleSheet(f"color:{FAINT}; font-size:11px;")
        sh.addSpacing(16); sh.addWidget(hint); sh.addStretch(1)
        left.addWidget(strip)

        stage_holder = QFrame(); stage_holder.setStyleSheet(f"background:{BG};")
        sv = QVBoxLayout(stage_holder); sv.setContentsMargins(0, 10, 0, 10)
        self.stage = Stage(self)
        sv.addWidget(self.stage, 1)
        self.hud = QLabel(""); self.hud.setParent(self.stage)
        self.hud.setStyleSheet(f"background:rgba(11,12,13,0.72); border:1px solid {LINE2};"
                               f"border-radius:8px; padding:8px 11px; font-family:Consolas; font-size:11px;")
        self.hud.move(14, 14)
        self.toast_lbl = QLabel("", self.stage); self.toast_lbl.hide()
        self.toast_lbl.setStyleSheet(f"background:rgba(11,12,13,0.92); border:1px solid {GO};"
                                     f"border-radius:9px; padding:8px 14px; font-weight:600;")
        left.addWidget(stage_holder, 1)

        tr = QFrame(); tr.setObjectName("transport"); tr.setFixedHeight(64)
        th = QHBoxLayout(tr); th.setContentsMargins(18, 0, 18, 0)
        for text, cb in (("⏮", lambda: self.step(-1)),):
            b = QPushButton(text); b.setFixedSize(34, 32); b.clicked.connect(cb); th.addWidget(b)
        self.btn_play = QPushButton("▶"); self.btn_play.setObjectName("accent")
        self.btn_play.setFixedSize(40, 32); self.btn_play.clicked.connect(self.toggle_play)
        th.addWidget(self.btn_play)
        b = QPushButton("⏭"); b.setFixedSize(34, 32); b.clicked.connect(lambda: self.step(1)); th.addWidget(b)
        self.btn_rev = QPushButton("◀◀ Reverse"); self.btn_rev.clicked.connect(self.toggle_rev)
        th.addWidget(self.btn_rev)
        for s in (0.5, 1.0, 2.0, 4.0):
            b = QPushButton(f"{s:g}×"); b.setFixedHeight(26)
            b.clicked.connect(lambda _, ss=s: setattr(self, "speed", ss))
            th.addWidget(b)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self._slider_changed)
        th.addWidget(self.slider, 1)
        self.lbl_pos = QLabel("0 / 0"); self.lbl_pos.setObjectName("mono")
        th.addWidget(self.lbl_pos)
        left.addWidget(tr)
        lw = QWidget(); lw.setLayout(left)
        h.addWidget(lw, 1)

        ops = QFrame(); ops.setObjectName("ops"); ops.setFixedWidth(372)
        ov = QVBoxLayout(ops); ov.setContentsMargins(16, 14, 16, 12); ov.setSpacing(10)
        head = QHBoxLayout(); head.addWidget(QLabel("Objects"))
        self.lbl_objcount = QLabel("0"); self.lbl_objcount.setStyleSheet(f"color:{FAINT};")
        head.addStretch(1); head.addLayout(head_r := QHBoxLayout()); head_r.addWidget(self.lbl_objcount)
        ov.addLayout(head)
        self.obj_box = QVBoxLayout(); self.obj_box.setSpacing(4)
        objw = QWidget(); objw.setLayout(self.obj_box)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(objw)
        ov.addWidget(sc, 2)
        self.btn_addobs = QPushButton("＋ Add view of selected"); self.btn_addobs.clicked.connect(self.add_obs)
        self.btn_auto = QPushButton("Auto views"); self.btn_auto.clicked.connect(self.auto_views)
        self.btn_rec = QPushButton("Reconstruct →"); self.btn_rec.setObjectName("accent")
        self.btn_rec.clicked.connect(self.reconstruct)
        ov.addWidget(self.btn_addobs); ov.addWidget(self.btn_auto); ov.addWidget(self.btn_rec)
        ov.addSpacing(6); ov.addWidget(QLabel("Jobs"))
        self.jobs_box = QVBoxLayout(); self.jobs_box.setSpacing(4)
        jw = QWidget(); jw.setLayout(self.jobs_box)
        js = QScrollArea(); js.setWidgetResizable(True); js.setWidget(jw)
        ov.addWidget(js, 1)
        h.addWidget(ops)
        self._hilite()
        return pane

    # -------------------------------- web panes -----------------------------------
    def _pane_web(self, kind):
        pane = QWidget(); v = QVBoxLayout(pane); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(0)
        bar = QFrame(); bar.setObjectName("panel"); bar.setFixedHeight(44)
        bh = QHBoxLayout(bar); bh.setContentsMargins(14, 0, 14, 0)
        self.cmb_scene = QComboBox(); self.cmb_scene.setMinimumWidth(260)
        self.cmb_scene.currentIndexChanged.connect(self._scene_changed)
        b = QPushButton("↻"); b.setFixedWidth(34); b.clicked.connect(self.load_scenes)
        bh.addWidget(eyebrow("scene")); bh.addSpacing(8)
        bh.addWidget(self.cmb_scene); bh.addWidget(b)
        bh.addSpacing(22); bh.addWidget(eyebrow("new recording")); bh.addSpacing(6)
        pb = QPushButton("poses")
        pb.setToolTip("camera poses of the opened recording (local VGGT chain, ~1 min per 300 frames; needs decode + depth done). "
                      "Objects and the map build need them; 'build map' runs this by itself when they are missing.")
        pb.clicked.connect(self.build_poses)
        bh.addWidget(pb)
        bh.addSpacing(14); bh.addWidget(eyebrow("build map")); bh.addSpacing(6)
        for cams in ("A", "B", "AB"):
            mb = QPushButton(cams if cams != "AB" else "A + B")
            mb.setToolTip(f"rebuild the street point cloud from camera {cams} (local VGGT, ~10 min/camera)")
            mb.clicked.connect(lambda _, c=cams: self.build_map(c))
            bh.addWidget(mb)
        self.lbl_map = QLabel(""); self.lbl_map.setObjectName("mono")
        self.lbl_map.setStyleSheet(f"color:{DIM}; font-size:11px;")
        bh.addSpacing(14); bh.addWidget(self.lbl_map, 1)
        v.addWidget(bar)
        self.web_map = QWebEngineView(); v.addWidget(self.web_map, 1)
        self.map_job = None
        self.map_timer = QTimer(self); self.map_timer.timeout.connect(self.poll_map_job)
        return pane

    def build_map(self, cams):
        try:
            r = api_post("/api/build_map", {"cams": cams}, timeout=30)
            self.map_job = r.get("job_id")
            self.lbl_map.setText(f"map {cams}: job #{self.map_job} queued")
            self.map_timer.start(3000)
        except Exception as ex:
            self.lbl_map.setText(f"map build failed: {ex}")

    def build_poses(self):
        try:
            r = api_post("/api/build_poses", {}, timeout=30)
            if r.get("error"):
                self.lbl_map.setText(f"poses: {r['error']}"); return
            self.map_job = r.get("job_id")
            self.lbl_map.setText(f"poses: job #{self.map_job} queued")
            self.map_timer.start(3000)
        except Exception as ex:
            self.lbl_map.setText(f"poses failed: {ex}")

    def poll_map_job(self):
        try:
            jobs = api_get("/api/jobs", timeout=3)
        except Exception:
            return
        j = next((x for x in jobs if x.get("id") == self.map_job), None)
        if j is None:
            return
        st = j.get("status", ""); det = str(j.get("detail", ""))
        self.lbl_map.setText(f"job #{j['id']} · {st.upper()} — {det[:220]}")
        self.lbl_map.setToolTip(det)
        if st in ("done", "error"):
            self.map_timer.stop()
            if st == "done":
                self.load_scenes()                      # fresh base map -> reload the viewer
                self.lbl_map.setText(f"job #{j['id']} · DONE — " + ("map rebuilt, viewer reloaded" if j.get("kind") == "map" else det[:200]))

    def _pane_3d(self):
        pane = QWidget(); h = QHBoxLayout(pane); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        side = QFrame(); side.setObjectName("rail"); side.setFixedWidth(230)
        sv = QVBoxLayout(side); sv.setContentsMargins(12, 12, 12, 12); sv.setSpacing(6)
        sv.addWidget(QLabel("Generated objects"))
        hintl = QLabel("drag to rotate · wheel to zoom"); hintl.setStyleSheet(f"color:{FAINT}; font-size:10px;")
        sv.addWidget(hintl)
        self.render_box = QVBoxLayout(); self.render_box.setSpacing(4)
        rw = QWidget(); rw.setLayout(self.render_box)
        rs = QScrollArea(); rs.setWidgetResizable(True); rs.setWidget(rw)
        sv.addWidget(rs, 1)
        rb = QPushButton("↻ Refresh"); rb.clicked.connect(self.load_renders)
        sv.addWidget(rb)
        self.lbl_render = QLabel(""); self.lbl_render.setStyleSheet(f"color:{FAINT}; font-size:10px;")
        sv.addWidget(self.lbl_render)
        h.addWidget(side)
        self.web_3d = QWebEngineView(); h.addWidget(self.web_3d, 1)
        return pane

    # ------------------------------- data / api -----------------------------------
    def switch_tab(self, i):
        self.stack.setCurrentIndex(i); self._mark_rail(i)
        if i == 1 and self.cmb_scene.count() == 0:
            self.load_scenes()
        if i == 2 and self.render_box.count() == 0:
            self.load_renders()

    def load_folder(self):
        rgb = ROOT / "rgb"
        names = sorted(p.name for p in rgb.glob("*.jpg")) if rgb.exists() else []
        self.frames["A"] = [n for n in names if n.startswith("A_")] or names
        self.frames["B"] = [n for n in names if n.startswith("B_")]
        self.lbl_frames.setText(f"FRAMES  A {len(self.frames['A'])}   B {len(self.frames['B'])}")
        self.lbl_dataset.setText(ROOT.name)
        # variants = the locked catalog, always. Inputs are rgb/ + raw/ only:
        # every product materializes on demand (layers/ is just the decode cache)
        self.cmb_variant.blockSignals(True)
        self.cmb_variant.clear()
        for key, label in (("nxyz", "Nxyz"), ("n_xy", "N xy"), ("n_xz", "N xz"),
                           ("nxyz_phys", "phys"), ("nxyz_diffuse", "diffuse"),
                           ("nxyz_specv2", "specv2"), ("edge", "Edge"),
                           ("rgb_deglare", "RGB deglare")):
            self.cmb_variant.addItem(label, f"layers/{key}")
        self.cmb_variant.blockSignals(False)
        self.variant = self.cmb_variant.currentData()
        self.render()

    def connect_api(self):
        try:
            m = api_get("/api/meta")
            self.lbl_api.setText(f"Backend online · {m['count'] / 1e6:.1f}M pts indexed")
            self.api_dot.setStyleSheet(f"color:{GO}; font-size:9px;")
            self.refresh_objects()
        except Exception:
            # try to start the local service, then retry once
            try:
                py = REPO / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
                subprocess.Popen([str(py if py.exists() else sys.executable), "-u", "server.py", "8130"],
                                 cwd=str(REPO / "inspector"))
                for _ in range(25):
                    time.sleep(0.6)
                    try:
                        api_get("/api/meta"); break
                    except Exception:
                        pass
                self.connect_api(); return
            except Exception as ex:
                self.lbl_api.setText(f"Backend offline: {ex}")
                self.api_dot.setStyleSheet(f"color:{BAD}; font-size:9px;")

    # ------------------------------ frames / render --------------------------------
    def cur_name(self):
        fr = self.frames[self.cam]
        return fr[self.pos] if 0 <= self.pos < len(fr) else None

    def frame_pix(self, name, layer):
        key = layer + "/" + name
        if key in self.cache:
            return self.cache[key]
        path = ROOT / layer / name
        img = None
        if path.exists():
            img = QImage(str(path))
        else:
            # live decode: the service renders the product from raw/ on demand
            variant = layer[7:] if layer.startswith("layers/") else layer
            try:
                with urllib.request.urlopen(f"{API}/frames/live/{variant}/{name}",
                                            timeout=30) as r:
                    img = QImage.fromData(r.read())
            except Exception:
                img = None
        if img is None or img.isNull():
            return None
        pix = QPixmap.fromImage(img).transformed(QTransform().rotate(90))
        self.cache[key] = pix; self.cache_order.append(key)
        if len(self.cache_order) > 220:
            old = self.cache_order.pop(0); self.cache.pop(old, None)
        return pix

    def rois_on_frame(self):
        name = self.cur_name()
        out = []
        for o in self.objects:
            if o.get("frame") != name:
                continue
            try:
                pts = json.loads(o["pts"]) if isinstance(o["pts"], str) else o["pts"]
            except Exception:
                continue
            out.append({"kind": o.get("kind", "rect"), "_pts": pts, "id": o["id"]})
        return out

    def render(self):
        name = self.cur_name()
        if name is None:
            return
        base = self.variant if self.layer == "nxyz" else "rgb"
        self.stage.img = self.frame_pix(name, base)
        self.stage.punch = self.frame_pix(name, self.variant) if self.layer == "roinx" else None
        n_roi = len(self.rois_on_frame())
        self.hud.setText(f"FRAME\n{self.cam}_{self.pos:06d}\n"
                         f"{'· ' + str(n_roi) + ' ROI' if n_roi else '1024×1224 · rgb + nxyz'}")
        self.hud.adjustSize()
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(1, len(self.frames[self.cam]) - 1))
        self.slider.setValue(self.pos)
        self.slider.blockSignals(False)
        self.lbl_pos.setText(f"{self.pos} / {max(0, len(self.frames[self.cam]) - 1)}")
        self.stage.update()

    def _tick(self):
        if not self.playing:
            return
        step = max(1, int(self.speed))
        self.pos += -step if self.reverse else step
        self.pos = max(0, min(len(self.frames[self.cam]) - 1, self.pos))
        self.render()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.setText("⏸" if self.playing else "▶")

    def toggle_rev(self):
        self.reverse = not self.reverse

    def step(self, d):
        self.pos = max(0, min(len(self.frames[self.cam]) - 1, self.pos + d))
        self.render()

    def _slider_changed(self, v):
        self.pos = v; self.render()

    def set_cam(self, c):
        if c == "B" and not self.frames["B"]:
            return
        self.cam = c
        self.pos = min(self.pos, len(self.frames[c]) - 1)
        self._hilite(); self.render()

    def set_layer(self, l):
        self.layer = l; self._hilite(); self.render()

    def set_tool(self, t):
        self.tool = None if self.tool == t else t
        self.stage.draw_pts = []
        self._hilite()

    def _variant_changed(self, _):
        self.variant = self.cmb_variant.currentData() or "nxyz"
        self.cache.clear(); self.cache_order.clear()
        self.render()

    def _hilite(self):
        for c, b in self.btn_cam.items():
            b.setProperty("active", "true" if c == self.cam else "false")
        for k, b in self.btn_layer.items():
            b.setProperty("active", "true" if k == self.layer else "false")
        for k, b in self.btn_tool.items():
            b.setProperty("active", "true" if k == self.tool else "false")
        for b in (*self.btn_cam.values(), *self.btn_layer.values(), *self.btn_tool.values()):
            b.style().unpolish(b); b.style().polish(b)

    # ------------------------------ toast / ops ------------------------------------
    def toast(self, msg, bad=False):
        self.toast_lbl.setStyleSheet(self.toast_lbl.styleSheet().replace(GO, BAD) if bad else
                                     self.toast_lbl.styleSheet().replace(BAD, GO))
        self.toast_lbl.setText(msg); self.toast_lbl.adjustSize()
        self.toast_lbl.move((self.stage.width() - self.toast_lbl.width()) // 2,
                            self.stage.height() - self.toast_lbl.height() - 22)
        self.toast_lbl.show()
        QTimer.singleShot(4200, self.toast_lbl.hide)

    def commit_roi(self, kind, qpts):
        name = self.cur_name()
        pts = [[int(p.x()), int(p.y())] for p in qpts]
        label = self.txt_label.text().strip() or f"R{len(self.objects) + 1}"
        self.toast("Solving 3D position…")

        def work():
            try:
                body = api_post("/api/roi", {"frame": name, "cam": self.cam,
                                             "kind": kind, "pts": pts, "label": label})
                self.toast(f"✓ 3D position solved — {body.get('npts', 0):,} points")
                self.sel = body.get("id", -1)
                try:
                    r = api_post(f"/api/objects/{self.sel}/autoviews", {"n": 8})
                    k = r.get("added", 0)
                    if k:
                        self.toast(f"✓ 3D solved · {k} auto views collected")
                except Exception:
                    pass
            except Exception as ex:
                self.toast(f"ROI failed: {ex}", bad=True)
            self.refresh_objects()

        QTimer.singleShot(10, work)

    def refresh_objects(self):
        try:
            self.objects = api_get("/api/objects")
        except Exception:
            self.objects = []
        while self.obj_box.count():
            it = self.obj_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.lbl_objcount.setText(str(len(self.objects)))
        for o in self.objects:
            b = QPushButton(f"{o.get('label', '?')}   ·  {o.get('frame', '')[:16]}")
            b.setStyleSheet("text-align:left;")
            if o["id"] == self.sel:
                b.setObjectName("accent")
            b.clicked.connect(lambda _, oid=o["id"]: self.select_obj(oid))
            self.obj_box.addWidget(b)
        self.obj_box.addStretch(1)
        self.render()

    def select_obj(self, oid):
        self.sel = oid
        o = next((x for x in self.objects if x["id"] == oid), None)
        if o and o.get("frame") in self.frames[self.cam]:
            self.pos = self.frames[self.cam].index(o["frame"])
        self.refresh_objects()

    def add_obs(self):
        self.toast("Draw the extra view now — it merges into the selected object")

    def auto_views(self):
        if self.sel < 0:
            return
        try:
            r = api_post(f"/api/objects/{self.sel}/autoviews", {"n": 8})
            self.toast(f"✓ {r.get('added', 0)} auto views")
        except Exception as ex:
            self.toast(str(ex), bad=True)
        self.refresh_objects()

    def reconstruct(self):
        if self.sel < 0:
            self.toast("Select an object first", bad=True); return
        try:
            body = api_post("/api/reconstruct", {"object_id": self.sel})
            self.toast(f"Reconstruction queued — job #{body.get('job_id')}")
        except Exception as ex:
            self.toast(str(ex), bad=True)

    def poll_jobs(self):
        try:
            jobs = api_get("/api/jobs", timeout=3)
        except Exception:
            return
        while self.jobs_box.count():
            it = self.jobs_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        by_obj = {o["id"]: o.get("label", "") for o in self.objects}
        for j in jobs[:6]:
            title = "Point cloud" if j.get("kind") == "map" else "Camera poses" if j.get("kind") == "poses" else \
                f"Reconstruction · {by_obj.get(j.get('object_id'), '')}"
            st = j.get("status", "")
            col = {"running": ACCENT, "queued": FAINT, "done": GO, "error": BAD}.get(st, FG2)
            l = QLabel(f"#{j['id']}  {title}\n{st.upper()} — {str(j.get('detail', ''))[:46]}")
            l.setStyleSheet(f"background:{PANEL2}; border:1px solid {LINE2}; border-left:3px solid {col};"
                            f"border-radius:6px; padding:6px 8px; font-size:10px;")
            self.jobs_box.addWidget(l)
        self.jobs_box.addStretch(1)

    # ------------------------------ web tabs ---------------------------------------
    def load_scenes(self):
        try:
            scenes = api_get("/api/scenes")
        except Exception:
            return
        self.cmb_scene.blockSignals(True)
        self.cmb_scene.clear()
        for s in scenes:
            n = s["name"]
            if n.startswith("obj_job_"):
                continue
            self.cmb_scene.addItem(n, s["url"])
        i = self.cmb_scene.findText("street")
        self.cmb_scene.setCurrentIndex(max(0, i))
        self.cmb_scene.blockSignals(False)
        self._scene_changed(0)

    def _scene_changed(self, _):
        url = self.cmb_scene.currentData()
        if url:
            self.web_map.setUrl(QUrl(API + url + "?top=1"))

    def load_renders(self):
        while self.render_box.count():
            it = self.render_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        try:
            scenes = api_get("/api/scenes")
            jobs = api_get("/api/jobs")
        except Exception:
            return
        url_by_job = {}
        for s in scenes:
            if s["name"].startswith("obj_job_"):
                try:
                    url_by_job[int(s["name"][8:])] = s["url"]
                except ValueError:
                    pass
        live = {o["id"] for o in self.objects}
        latest = {}
        for j in jobs:
            if j.get("status") == "done" and j.get("kind") != "map" and j["id"] in url_by_job \
                    and j.get("object_id") in live:
                if j["object_id"] not in latest or j["id"] > latest[j["object_id"]]["id"]:
                    latest[j["object_id"]] = j
        first = True
        by_obj = {o["id"]: o.get("label", "") for o in self.objects}
        for j in sorted(latest.values(), key=lambda x: -x["id"]):
            title = by_obj.get(j["object_id"]) or f"object {j['object_id']}"
            url = url_by_job[j["id"]].replace("index.html", "splat.html")
            b = QPushButton(title); b.setStyleSheet("text-align:left;")
            b.setToolTip(f"job #{j['id']}")
            b.clicked.connect(lambda _, u=url, t=title, jid=j["id"]: self._open_render(u, t, jid))
            self.render_box.addWidget(b)
            if first:
                self._open_render(url, title, j["id"]); first = False
        self.render_box.addStretch(1)
        if first:
            self.lbl_render.setText("No generated objects yet — run a reconstruction from the recorder")

    def _open_render(self, url, title, jid):
        self.web_3d.setUrl(QUrl(f"{API}{url}?v={int(time.time() * 1000)}"))
        self.lbl_render.setText(f"{title} · job #{jid}")

    def open_folder(self):
        """pick a scene folder -- a folder of .raw12 frames is enough: the service
        switches to it, decodes rgb + the normals catalog, the recorder reloads"""
        global ROOT
        d = QFileDialog.getExistingDirectory(self, "Open data folder (raw frames)", str(ROOT.parent))
        if not d:
            return
        try:
            r = api_post("/api/open_dataset", {"path": d}, timeout=30)
        except Exception as ex:
            self.toast(f"open failed: {ex}", bad=True); return
        ROOT = Path(r["path"])
        self.lbl_path.setText(str(ROOT))
        self.cache.clear(); self.cache_order.clear(); self.pos = 0
        self.load_folder()
        self.toast(f"dataset: {r['frames']} frames, raw {r['raw']}, decoded {r['decoded']}")
        self.poll_decode()

    def sync_dataset(self):
        """on start: follow whatever folder the service currently serves"""
        global ROOT
        try:
            r = api_get("/api/dataset", timeout=3)
            p = Path(r["path"])
            if p.exists() and p != ROOT:
                ROOT = p
                self.lbl_path.setText(str(ROOT))
                self.load_folder()
        except Exception:
            pass

    # -------------------------- raw-import decode gate -----------------------------
    def _decode_overlay(self):
        self.gate = QWidget(self)
        self.gate.setStyleSheet(f"background: rgba(8,9,10,0.96);")
        v = QVBoxLayout(self.gate); v.setAlignment(Qt.AlignCenter)
        t = QLabel("Decoding raw frames"); t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet(f"color:{FG}; font-size:19px; font-weight:600;")
        self.gate_sub = QLabel("preparing…"); self.gate_sub.setAlignment(Qt.AlignCenter)
        self.gate_sub.setObjectName("mono")
        self.gate_sub.setStyleSheet(f"color:{DIM}; font-size:12px;")
        self.gate_bar = QFrame(); self.gate_bar.setFixedSize(420, 6)
        self.gate_bar.setStyleSheet(f"background:{PANEL2}; border-radius:3px;")
        self.gate_fill = QFrame(self.gate_bar); self.gate_fill.setGeometry(0, 0, 0, 6)
        self.gate_fill.setStyleSheet(f"background:{ACCENT}; border-radius:3px;")
        note = QLabel("Raw in — RGB and the locked normals catalog are decoded once,\n"
                      "then everything is instant.")
        note.setAlignment(Qt.AlignCenter); note.setStyleSheet(f"color:{FAINT}; font-size:11px;")
        v.addWidget(t); v.addSpacing(6); v.addWidget(self.gate_sub); v.addSpacing(14)
        v.addWidget(self.gate_bar, alignment=Qt.AlignCenter); v.addSpacing(16); v.addWidget(note)
        self.gate.hide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "gate"):
            self.gate.setGeometry(self.rect())

    def poll_decode(self):
        try:
            st = api_get("/api/decode_status", timeout=3)
        except Exception:
            return
        depth_note = ""
        if st.get("depth_total") and not st.get("depth_complete", True):
            depth_note = f"   ·   depth {st.get('depth_done', 0)} / {st['depth_total']}"
        if st["total"] == 0 or st.get("raw_complete", st["complete"]):
            # frames exist (raw decoded, or a ready dataset): no gate -- the depth
            # phase (MoGe, phase 2) runs in the background and only annotates the header
            if self.gate.isVisible():
                self.gate.hide()
                self.cache.clear(); self.cache_order.clear()
                self.load_folder()              # rgb/ was written by the decode: reread the frame list
            base = self.lbl_frames.text().split("   ·   depth")[0]
            self.lbl_frames.setText(base + depth_note)
            return
        # incomplete -> show the gate, and make sure the decode is running
        self.gate.setGeometry(self.rect()); self.gate.show(); self.gate.raise_()
        if not st["running"]:
            try:
                api_post("/api/decode", {}, timeout=5)
            except Exception:
                pass
        pct = st["done"] / max(st["total"], 1)
        self.gate_fill.setGeometry(0, 0, int(420 * pct), 6)
        eta = f" · ~{st['eta'] // 60}m {st['eta'] % 60}s left" if st.get("eta") else ""
        self.gate_sub.setText(f"{st['done']} / {st['total']} frames{eta}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = Deskview()
    win.show()
    if "--selftest" in sys.argv:
        tab = {"rec": 0, "map": 1, "3d": 2}.get(os.environ.get("NAST_SELFTEST_TAB", "rec"), 0)
        win.switch_tab(tab)
        def snap():
            pix = win.grab()
            out = Path(sys.argv[sys.argv.index("--selftest") + 1]
                       if len(sys.argv) > sys.argv.index("--selftest") + 1 else "deskview_qt_selftest.png")
            pix.save(str(out))
            print("SELFTEST_SAVED", out, flush=True)
            app.quit()
        QTimer.singleShot(6000, snap)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
