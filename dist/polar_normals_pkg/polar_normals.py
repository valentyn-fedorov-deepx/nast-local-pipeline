"""Polarization products from the Orthovector (IMX264MYR, colour CPFA) raw frames.

Offline re-implementation of the VyzAI CameraController normal stack
(DavePlayground / CameraController, see dave_playground docs) so every
normals variant the client's GUI offers can be generated for the dashcam
frames -- plus the normal-derived products (edge, ridge, segment, spline,
integrated depth) and the polarization deglare.

Raw frame: 2448x2048, MIPI RAW12 (3 bytes -> 2 px, low nibbles in byte 2),
rows padded to 3680 bytes.  4x4 super-pixel: 2x2 Bayer (BG pattern on the
polar sub-images) x 2x2 polarizers in Sony's native order
    (0,0)=90  (0,1)=45
    (1,0)=135 (1,1)=0
(verified against the recorder's own rgb_half / nxyz_half output: Bayer BG by
chroma sign, angle order by pseudo-normal correlation 0.83/0.89/0.90).

Products (half resolution 1224x1024, sensor orientation, same as rgb_half):
  pseudo normals (signed "Nxyz" / abs "Nxyz (raw)"), subtracted ("Nxyz (sub)"),
  physical VyzLut Fresnel ("Nxyz (phys)"), px Atkinson diffuse ("Nxyz (diffuse)"),
  px Kadambi SpecularV2 ("Nxyz (specV2)"), AoLP grey / colour, DoLP, Polar HSV,
  deglare (unpolarized intensity), edge, ridge map, segmentation, spline outline,
  edge spline, integrated depth.

Conventions (re-locked 2026-08-26 to the CLIENT CameraController verbatim,
fitted against the controller's own May renders of our Drive take --
best 0.859 mean channel correlation, decisive over every alternative):
  theta_cc = 0.5*atan2(S1, S2), S1 = I0-I90, S2 = I45-I135
    (the cv::phase(S2,S1) convention; equals textbook AoLP phi - 45 deg)
  CC pseudo (init_normals verbatim): nx = sin(d)*cos(theta_cc),
    ny = -sin(d)*sin(theta_cc), nz = |cos d| * dolp_factor(0.05),
    enhanced -> normalized
  px modes (make_normal verbatim): x = sin(g)*cos(theta+pi/2),
    y = sin(g)*cos(theta), z = cos(g); theta = theta_cc
  roll fix: camera rolled -90 deg, display = rot90(sensor, 3);
    rot_disp() maps vectors sensor->display (nx' = -ny, ny' = nx);
    scalar angles get +90 deg (aolp_disp)
  paint: canon channel order (display R=X G=Y B=Z); nxyz/n_xy/n_xz abs
    (the controller's NORMALS_RAW mapping), phys/diffuse/specv2 signed
    x,y + raw |z| (the controller's physnorm dumps)
"""
import numpy as np
import cv2

# camera roll on the rig, degrees: -90 for our Orthovector mount (the image is
# displayed rotated k=3, normal vectors are turned to match), 0 for an upright
# camera (no vector rotation). products(..., roll_deg=...) overrides per call.
CAMERA_ROLL_DEG = -90.0

W_RAW, H_RAW, STRIDE = 2448, 2048, 3680
ANGLE_POS = {90: (0, 0), 45: (0, 1), 135: (1, 0), 0: (1, 1)}     # Sony native
BAYER = cv2.COLOR_BayerBG2BGR


# ------------------------------------------------------------------ decode
def unpack_raw12(buf):
    """MIPI RAW12 big-endian-ish packing: byte0=P0[11:4], byte1=P1[11:4],
    byte2={P1[3:0],P0[3:0]}; rows padded to STRIDE."""
    a = np.frombuffer(buf, np.uint8)
    if a.size != H_RAW * STRIDE:
        raise ValueError(f"unexpected raw size {a.size}")
    b = a.reshape(H_RAW, STRIDE)[:, :W_RAW * 3 // 2].reshape(H_RAW, W_RAW // 2, 3).astype(np.uint16)
    img = np.empty((H_RAW, W_RAW), np.uint16)
    img[:, 0::2] = (b[:, :, 0] << 4) | (b[:, :, 2] & 0x0F)
    img[:, 1::2] = (b[:, :, 1] << 4) | (b[:, :, 2] >> 4)
    return img


def _lum16(bgr16):
    return 0.114 * bgr16[..., 0] + 0.587 * bgr16[..., 1] + 0.299 * bgr16[..., 2]


class PolarFrame:
    """All products of one raw frame, lazily computed and cached."""

    def __init__(self, raw_bytes, subtracted_gain=1.5, subtracted_baseline=1.1, clean=True):
        mosaic = unpack_raw12(raw_bytes)
        self.sub = {}                                     # angle -> demosaiced BGR float32 (0..4095*16)
        self.I = {}
        for ang, (r, c) in ANGLE_POS.items():
            s = mosaic[r::2, c::2]
            bgr = cv2.cvtColor((s << 4).astype(np.uint16), BAYER).astype(np.float32)
            self.sub[ang] = bgr
            L = _lum16(bgr)
            # per-angle shot noise is what makes the pseudo normals "dither";
            # a small Gaussian on each angle plane before the Stokes math is
            # the offline stand-in for the GUI's accurate-preset smoothing
            self.I[ang] = cv2.GaussianBlur(L, (5, 5), 1.0) if clean else L
        self.sg, self.sb = subtracted_gain, subtracted_baseline
        self._c = {}

    # ---- Stokes -----------------------------------------------------------
    @property
    def s0(self): return self.I[0] + self.I[90]
    @property
    def s1n(self): return self.I[0] - self.I[90]             # internal "s1_negative" = textbook S1
    @property
    def s2(self): return self.I[45] - self.I[135]

    def dolp(self, subtracted=False):
        key = ("dolp", subtracted)
        if key not in self._c:
            mag = np.hypot(self.s1n, self.s2)
            if subtracted:
                s0p = (self.sg - self.sb) * self.s0 + self.sb * self.s1n
                d = self.sg * mag / np.maximum(s0p, 1e-3)
            else:
                d = mag / np.maximum(self.s0, 1e-3)
            self._c[key] = np.clip(d, 0, 1).astype(np.float32)
        return self._c[key]

    def theta_display(self):
        if "theta" not in self._c:
            t = 0.5 * np.arctan2(self.s1n, self.s2)
            self._c["theta"] = np.mod(t, np.pi).astype(np.float32)       # [0, pi)
        return self._c["theta"]

    def aolp(self):
        """true AoLP phi in (-pi/2, pi/2]"""
        if "phi" not in self._c:
            self._c["phi"] = (0.5 * np.arctan2(self.s2, self.s1n)).astype(np.float32)
        return self._c["phi"]

    # ---- intensity views --------------------------------------------------
    def intensity(self):
        return self.s0 * 0.5

    def color(self, gamma=1.0 / 1.8, deglare=False):
        """S0 colour with grey-world white balance (the recorder's look, roughly).

        deglare=True: closed-form Stokes minimum PER CHANNEL,
        I_min = (S0 - sqrt(S1^2+S2^2)) / 2 -- the polarized (specular) part of
        every pixel removed, so windshields / wet paint keep their diffuse
        colour instead of blowing out.  WB gains and the display scale come
        from the REFERENCE (non-deglared) image, so before/after compare 1:1.

        deglare="soft": same removal but with a brightness FLOOR -- a pixel
        never drops below `floor` (0.35) of its reference luma.  At grazing
        angles (car roof, windshield) nearly ALL light is specular, so the
        full minimum goes black and TRELLIS reads the flat black as holes;
        the floor keeps a scaled copy of the shading there while still
        killing the blowout.  This is the working-stream variant."""
        ref = (self.sub[0] + self.sub[90]) * 0.5                   # S0/2, BGR
        if deglare:
            s0 = 0.5 * (self.sub[0] + self.sub[45] + self.sub[90] + self.sub[135])
            s1 = self.sub[0] - self.sub[90]
            s2 = self.sub[45] - self.sub[135]
            bgr = np.clip((s0 - np.sqrt(s1 * s1 + s2 * s2)) * 0.5, 0, None)
            if deglare == "soft":
                floor = 0.35
                L_d = _lum16(bgr); L_r = _lum16(ref)
                lo = floor * L_r
                # smooth blend toward floor*ref where the diffuse residual
                # falls below the floor (keeps ref hue there, scaled down)
                t = np.clip((lo - L_d) / np.maximum(0.5 * lo, 1e-3), 0, 1)[..., None]
                bgr = bgr * (1 - t) + (floor * ref) * t
        else:
            bgr = ref
        m = ref.reshape(-1, 3).mean(0) + 1e-6
        scale = 1.0 / np.percentile(ref * (m.mean() / m), 99.5)
        bgr = bgr * (m.mean() / m)
        v = np.clip(bgr * scale, 0, 1) ** gamma
        return (v * 255).astype(np.uint8)

    # the recorder's own rgb is a LINEAR per-channel scaling of S0/2 (gamma 1.00,
    # no auto white balance), fitted on 40 frames of the 07.08 dataset against
    # the recorder's jpgs: mean |error| ~1/255. BGR, for sub in 16-bit units.
    RECORDER_K = np.array([0.00642983, 0.00393173, 0.00679093], np.float32)

    def color_recorder(self):
        """S0 colour exactly as the recorder writes rgb_orig (linear, fixed WB)"""
        ref = (self.sub[0] + self.sub[90]) * 0.5
        return np.clip(ref * self.RECORDER_K, 0, 255).astype(np.uint8)

    def color_work(self):
        """the working colour stream: recorder look x glare attenuation --
        the same construction as the shipped street_video/rgb (gen_rgb_soft)"""
        att = self.deglare_atten()
        return np.clip(self.color_recorder().astype(np.float32) * att[..., None], 0, 255).astype(np.uint8)

    def deglare(self):
        """unpolarized intensity I_min = (s0/2)(1-dolp): specular glare removed
        (windshields, wet road) -- the polarization 'see-through' product"""
        return self.intensity() * (1.0 - self.dolp())

    def deglare_atten(self, floor=0.35, blur=5):
        """Per-pixel glare ATTENUATION map (0..1): the luma ratio diffuse/ref
        of the Stokes-minimum deglare, floored and lightly smoothed.  Multiply
        any same-size colour render (e.g. the recorder's own rgb) by it to
        remove the specular sheen while keeping that render's look 1:1."""
        s0 = 0.5 * (self.sub[0] + self.sub[45] + self.sub[90] + self.sub[135])
        amp = np.sqrt((self.sub[0] - self.sub[90]) ** 2 + (self.sub[45] - self.sub[135]) ** 2)
        ref_l = _lum16((self.sub[0] + self.sub[90]) * 0.5)
        dif_l = _lum16(np.clip((s0 - amp) * 0.5, 0, None))
        att = np.clip(dif_l / np.maximum(ref_l, 1e-3), floor, 1.0).astype(np.float32)
        if blur:
            att = cv2.GaussianBlur(att, (blur, blur), 0)
        return att

    def glare(self):
        return self.intensity() * self.dolp()

    # ---- normal fields (H,W,3) float32, unit-ish ----------------------------
    def pseudo(self, dolp_factor=0.05, enhanced=True, subtracted=False):
        """CC init_normals verbatim: x = sd*cos(th), y = -sd*sin(th), th = theta_cc"""
        key = ("pseudo", dolp_factor, enhanced, subtracted)
        if key not in self._c:
            d = self.dolp(subtracted); th = self.theta_display()
            sd = np.sin(d)
            nx = sd * np.cos(th); ny = -sd * np.sin(th); nz = np.abs(np.cos(d)) * dolp_factor
            if enhanced:
                ln = np.sqrt(dolp_factor ** 2 + sd ** 2 * (1 - dolp_factor ** 2))
                nx, ny, nz = nx / ln, ny / ln, nz / ln
            self._c[key] = np.stack([nx, ny, nz], -1).astype(np.float32)
        return self._c[key]

    def pseudo_shapeos(self):
        """ShapeOS-path pseudo field emulation: alpha = phi + 90 deg (textbook
        ShapeOS convention per the comparison doc), dolpFactor 0.5"""
        d = self.dolp(); ph = self.aolp() + np.pi / 2; sd = np.sin(d)
        nx = sd * np.sin(ph); ny = sd * np.cos(ph); nz = np.abs(np.cos(d) * 0.5)
        n = np.stack([nx, ny, nz], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        return n.astype(np.float32)

    # physical: VyzLut (Fresnel dielectric, diffuse below / specular above the threshold)
    @staticmethod
    def fresnel_diffuse_dolp(zen, n):
        s2 = np.sin(zen) ** 2; c = np.cos(zen); root = np.sqrt(np.maximum(0, n * n - s2))
        num = (n - 1 / n) ** 2 * s2
        den = 2 + 2 * n * n - (n + 1 / n) ** 2 * s2 + 4 * c * root
        return num / den

    @staticmethod
    def fresnel_specular_dolp(zen, n):
        s2 = np.sin(zen) ** 2; s4 = s2 * s2; c = np.cos(zen); root = np.sqrt(np.maximum(0, n * n - s2))
        return (2 * s2 * c * root) / (n * n - s2 - n * n * s2 + 2 * s4)

    _lut_cache = {}

    @classmethod
    def lut(cls, n=1.5, res=901):
        key = (round(n, 4), res)
        if key not in cls._lut_cache:
            z = np.linspace(0, np.pi / 2, res)
            dd = cls.fresnel_diffuse_dolp(z, n); ds = cls.fresnel_specular_dolp(z, n)
            pk = int(np.argmax(ds))
            cls._lut_cache[key] = (z, dd, ds, pk)
        return cls._lut_cache[key]

    def physical_vyzlut(self, n=1.5, spec_thr=0.40, dolp_min=0.05, dolp_max=0.95, ramp=0.10):
        """returns (normals, confidence, reflection_model[0 diffuse/1 specular])"""
        key = ("vyzlut", n, spec_thr)
        if key not in self._c:
            z, dd, ds, pk = self.lut(n)
            d = self.dolp(); th = self.theta_display()
            spec = d > spec_thr
            # diffuse branch: monotone ascending table -> searchsorted
            zd = z[np.clip(np.searchsorted(dd, np.minimum(d, dd.max())), 0, len(z) - 1)]
            # specular branch: below-Brewster root
            zs = z[np.clip(np.searchsorted(ds[:pk + 1], np.minimum(d, ds[pk])), 0, pk)]
            zen = np.where(spec, zs, zd).astype(np.float32)
            az = np.where(spec, th + np.pi / 2, th).astype(np.float32)
            s = np.sin(zen)
            # make_normal placement: x = s*cos(az+pi/2), y = s*cos(az)
            nrm = np.stack([-s * np.sin(az), s * np.cos(az), np.cos(zen)], -1)
            flip = nrm[..., 2] < 0
            nrm[flip] *= -1
            wl = np.clip((d - dolp_min) / ramp, 0, 1); wh = np.clip((dolp_max - d) / ramp, 0, 1)
            conf = np.where((d >= dolp_min) & (d <= dolp_max), np.minimum(wl, wh), 0).astype(np.float32)
            self._c[key] = (nrm.astype(np.float32), conf, spec.astype(np.uint8))
        return self._c[key]

    @staticmethod
    def _px_make_normal(gamma, phi):
        """Dave's make_normal verbatim: x = sg*cos(th+pi/2), y = sg*cos(th)"""
        sg = np.sin(gamma)
        x = sg * np.cos(phi + np.pi / 2); y = sg * np.cos(phi); z = np.cos(gamma)
        n = np.stack([x, y, z], -1)
        bad = ~np.isfinite(gamma)
        n[bad] = (0, 0, 1)
        mag = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.where(mag > 0, n / np.maximum(mag, 1e-12), (0, 0, 1))
        n[n[..., 2] < 0] *= -1
        return n.astype(np.float32)

    def px_diffuse(self, n=1.5, gamma_offset=0.0):
        """Atkinson diffuse, closed-form inverse (px transcription)"""
        key = ("pxdiff", n, gamma_offset)
        if key not in self._c:
            d = self.dolp() - gamma_offset
            n2 = n * n; n3 = n2 * n; n4 = n2 * n2; n6 = n4 * n2; n8 = n4 * n4; d2 = d * d
            with np.errstate(invalid="ignore", divide="ignore"):
                omega = (d + 1) * ((n8 + 1) * (d + 1) + 4 * (n6 + n2) * (d - 1) - 2 * n4 * (5 * d - 3))
                t1 = (d + 1) * (-n8 * (d - 1) + 2 * n6 * (3 * d - 2) - 2 * n4 * (4 * d - 3) + 2 * n2 * (d - 2) + (d + 1))
                t2 = -4 * d * n3 * (n2 - 1) ** 2 * np.sqrt(np.maximum(1 - d2, 0))
                gamma = np.arccos(np.sqrt(np.clip((t1 + t2) / omega, 0, 1)))
            gamma = np.where(d <= 0, 0.0, gamma)
            self._c[key] = self._px_make_normal(gamma, self.theta_display())
        return self._c[key]

    def px_specular_v2(self, n=1.5, k=0.0, gamma_offset=0.0):
        """Kadambi specular, complex n+ik, below-peak root (the one the GUI uses)"""
        key = ("pxspec", n, k, gamma_offset)
        if key not in self._c:
            d = self.dolp() - gamma_offset
            n2 = n * n; k2 = k * k; d2 = d * d
            with np.errstate(invalid="ignore", divide="ignore"):
                sq = np.sqrt(n2 - d2 * n2 * k2 - d2 * n2)        # NaN where negative -> (0,0,1)
                om = (n - sq) / d
                gamma = np.arccos(np.sqrt(om * om + 4) * 0.5 - om * 0.5)
            gamma = np.where(d <= 0, 0.0, gamma)
            self._c[key] = self._px_make_normal(gamma, self.theta_display())
        return self._c[key]

    # ---- encodings ----------------------------------------------------------
    @staticmethod
    def encode(n, order="canon", signed=True, scale_max=1.0):
        """normal field -> uint8 BGR.  "canon" (= ShapeOS-ON order of the DALEK
        guide, the approved nxyz_official look): file BGR=(|z|,cy,cx) -> display
        R=X G=Y B=Z.  "legacy" keeps the old DALEK-default R=Z order."""
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
        if signed:
            cx, cy = (nx + 1) * 0.5, (ny + 1) * 0.5
        else:
            cx, cy = np.abs(nx), np.abs(ny)
        cz = np.abs(nz)
        chans = [cz, cy, cx] if order in ("canon", "shapeos") else [cx, cy, cz]
        img = np.stack(chans, -1) * (255.0 / scale_max)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---- scalar / colour views ---------------------------------------------
    @staticmethod
    def rot_disp(n, roll_deg=None):
        """Sensor-frame in-plane components -> DISPLAY frame for a camera
        rolled by roll_deg (CAMERA_ROLL_DEG when None). The vectors turn by
        -roll: for our rig (-90 deg, image shown rotated k=3) that is
        nx' = -ny, ny' = nx; for an upright camera (0 deg) it is the identity.
        nz is never touched."""
        r = CAMERA_ROLL_DEG if roll_deg is None else roll_deg
        a = np.radians(-r); c, s_ = np.cos(a), np.sin(a)
        if abs(s_) < 1e-12 and c > 0:
            return n
        return np.stack([c * n[..., 0] - s_ * n[..., 1], s_ * n[..., 0] + c * n[..., 1], n[..., 2]], -1)

    def aolp_disp(self, roll_deg=None):
        """controller theta against the DISPLAY x-axis (-roll added: +90 deg
        for our rig, nothing for an upright camera), wrapped to (-pi/2, pi/2]."""
        r = CAMERA_ROLL_DEG if roll_deg is None else roll_deg
        ph = np.mod(self.theta_display() + np.radians(-r), np.pi)
        return np.where(ph > np.pi / 2, ph - np.pi, ph)

    def view_aolp(self):
        return np.clip((self.aolp_disp() / np.pi + 0.5) * 255, 0, 255).astype(np.uint8)

    def view_aolp_color(self):
        ph = self.aolp_disp(); m = np.clip(np.abs(ph) / (np.pi / 2), 0, 1) * 255
        out = np.zeros(ph.shape + (3,), np.uint8)
        out[..., 0] = np.where(ph >= 0, m, 0)                 # + -> blue
        out[..., 2] = np.where(ph < 0, m, 0)                  # - -> red
        return out

    def view_dolp(self, scale=1.5):
        return np.clip(self.dolp() * scale * 255, 0, 255).astype(np.uint8)

    def view_polar_hsv(self):
        ph = self.aolp_disp(); d = self.dolp(); v = self.intensity()
        h = ((ph + np.pi / 2) / np.pi * 180).astype(np.uint8)          # OpenCV hue 0..180
        s = np.clip(d * 1.5 * 255, 0, 255).astype(np.uint8)
        vv = np.clip(v / np.percentile(v, 99.5) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(np.stack([h, s, vv], -1), cv2.COLOR_HSV2BGR)

    def view_gray(self, img):
        return np.clip(img / max(np.percentile(img, 99.5), 1e-6) * 255, 0, 255).astype(np.uint8)

    # ---- derived products ---------------------------------------------------
    @staticmethod
    def edge_energy(n):
        acc = np.zeros(n.shape[:2], np.float32)
        for c in range(3):
            acc += np.abs(cv2.Laplacian(n[..., c], cv2.CV_32F, ksize=3))
        return acc

    def edge(self, n=None, thr=0.26):
        n = self.pseudo() if n is None else n
        e = self.edge_energy(n)
        e = cv2.normalize(e, None, 0, 1, cv2.NORM_MINMAX)
        return (e > thr).astype(np.uint8) * 255

    def ridge(self, n=None, thr=0.05, component=0, despeckle=4):
        n = self.pseudo() if n is None else n
        comps = [0, 1, 2] if component == 0 else [component - 1]
        acc = np.zeros(n.shape[:2], np.float32)
        for c in comps:
            sm = cv2.GaussianBlur(n[..., c], (5, 5), 0)
            acc += np.abs(cv2.Laplacian(sm, cv2.CV_32F, ksize=3))
        acc /= 8.0 * len(comps)
        m = (acc > thr).astype(np.uint8)
        if despeckle > 0:
            k, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            small = np.where(st[:, cv2.CC_STAT_AREA] < despeckle)[0]
            m[np.isin(lab, small)] = 0
        return m * 255

    def segment(self, n=None, thr=0.26, close_k=7, min_area=700, max_area=200000, min_extent=0.05):
        e = self.edge(n, thr)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        closed = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k)
        ncc, lab, st, _ = cv2.connectedComponentsWithStats((closed == 0).astype(np.uint8), 8)
        out = np.zeros(e.shape + (3,), np.uint8); regions = []
        for i in range(1, ncc):
            x, y, w, h, area = st[i]
            if area < min_area or area > max_area: continue
            if area / max(1, w * h) < min_extent: continue
            hsv = np.uint8([[[(i * 53) % 180, 200, 255]]])
            out[lab == i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            regions.append((i, (x, y, w, h)))
        return out, lab, regions

    def spline_outline(self, n=None, max_residual=0.12, n_ctrl=16):
        """closed periodic cubic B-spline per accepted region; abstain when the
        fit misses the boundary by more than max_residual (relative to size)"""
        from scipy.interpolate import splprep, splev
        seg, lab, regions = self.segment(n)
        base = cv2.cvtColor(self.view_gray(self.intensity()), cv2.COLOR_GRAY2BGR)
        out = (base * 0.35).astype(np.uint8)
        for i, (x, y, w, h) in regions:
            if w < 4 or h < 4: continue
            m = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts: continue
            c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
            if len(c) < 12: continue
            idx = np.linspace(0, len(c) - 1, 64).astype(int)
            pts = c[idx]
            try:
                tck, _ = splprep([pts[:, 0], pts[:, 1]], per=True, k=3, s=len(pts) * 2.0, nest=n_ctrl + 4)
                u = np.linspace(0, 1, 200)
                sx, sy = splev(u, tck)
                curve = np.stack([sx, sy], 1)
                # residual: mean distance of samples to the curve, relative to the region size
                dmin = np.sqrt(((pts[:, None, :] - curve[None, :, :]) ** 2).sum(-1)).min(1)
                res = float(dmin.mean() / max(w, h))
                ok = res <= max_residual
            except Exception:
                ok = False
            if ok:
                cv2.polylines(out, [np.round(curve).astype(np.int32)], True, (120, 240, 140), 1, cv2.LINE_AA)
                cx_, cy_ = splev(np.linspace(0, 1, n_ctrl, endpoint=False), tck)
                for px, py in zip(cx_, cy_):
                    cv2.circle(out, (int(px), int(py)), 2, (60, 190, 255), -1)
            else:
                cv2.polylines(out, [c.astype(np.int32)], True, (90, 90, 210), 1)
        return out

    def edge_spline(self, n=None, thr=0.26, min_pts=12, eps=1.6):
        """open Catmull-Rom curves along the edge chains"""
        e = self.edge(n, thr)
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        base = cv2.cvtColor(self.view_gray(self.intensity()), cv2.COLOR_GRAY2BGR)
        out = (base * 0.30).astype(np.uint8)
        for c in cnts:
            if len(c) < min_pts: continue
            p = cv2.approxPolyDP(c, eps, False).reshape(-1, 2).astype(np.float32)
            if len(p) < 2: continue
            P = np.vstack([p[:1], p, p[-1:]])
            samples = []
            for i in range(1, len(P) - 2):
                p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
                for t in np.linspace(0, 1, 6, endpoint=False):
                    t2, t3 = t * t, t * t * t
                    q = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
                    samples.append(q)
            samples.append(P[-2])
            cv2.polylines(out, [np.round(np.array(samples)).astype(np.int32)], False, (150, 255, 190), 1, cv2.LINE_AA)
        return out

    # ---- normal integration (depth) ------------------------------------------
    @staticmethod
    def relax_disambiguate(n, iters=4):
        cur = n.copy()
        for _ in range(iters):
            sm = cv2.GaussianBlur(cur, (9, 9), 0)
            flip = cur.copy(); flip[..., 0] *= -1; flip[..., 1] *= -1
            d0 = (cur * sm).sum(-1); d1 = (flip * sm).sum(-1)
            cur = np.where((d1 > d0)[..., None], flip, cur)
        return cur

    @staticmethod
    def integrate_poisson(p, q, valid, maxiter=3000, tol=1e-5):
        """masked Poisson integration of gradients (p=dz/dx, q=dz/dy), Jacobi-PCG
        via scipy; per-connected-component zero mean. Returns depth (nan outside)"""
        from scipy import sparse
        from scipy.sparse.linalg import cg
        H, W = valid.shape
        idx = -np.ones((H, W), np.int64); idx[valid] = np.arange(valid.sum())
        N = int(valid.sum())
        if N < 16:
            return np.full((H, W), np.nan, np.float32), 0.0
        rows, cols, vals = [], [], []
        rhs = np.zeros(N)
        diag = np.zeros(N)
        def edge_(i0, j0, i1, j1, g):
            a, b = idx[i0, j0], idx[i1, j1]
            if a < 0 or b < 0: return
            rows.extend([a, a, b, b]); cols.extend([a, b, b, a]); vals.extend([1, -1, 1, -1])
            rhs[a] -= g; rhs[b] += g
        # horizontal edges: z[i,j+1]-z[i,j] = mean(p)
        vi, vj = np.where(valid[:, :-1] & valid[:, 1:])
        gh = 0.5 * (p[vi, vj] + p[vi, vj + 1])
        a = idx[vi, vj]; b = idx[vi, vj + 1]
        vi2, vj2 = np.where(valid[:-1, :] & valid[1:, :])
        gv = 0.5 * (q[vi2, vj2] + q[vi2 + 1, vj2])
        a2 = idx[vi2, vj2]; b2 = idx[vi2 + 1, vj2]
        A_rows = np.concatenate([a, a, b, b, a2, a2, b2, b2])
        A_cols = np.concatenate([a, b, b, a, a2, b2, b2, a2])
        A_vals = np.concatenate([np.ones_like(a), -np.ones_like(a), np.ones_like(a), -np.ones_like(a),
                                 np.ones_like(a2), -np.ones_like(a2), np.ones_like(a2), -np.ones_like(a2)]).astype(np.float64)
        rhs = np.zeros(N)
        np.add.at(rhs, a, -gh); np.add.at(rhs, b, gh)
        np.add.at(rhs, a2, -gv); np.add.at(rhs, b2, gv)
        A = sparse.coo_matrix((A_vals, (A_rows, A_cols)), shape=(N, N)).tocsr()
        A = A + sparse.identity(N) * 1e-6
        M = sparse.diags(1.0 / (A.diagonal() + 1e-9))
        z, info = cg(A, rhs, M=M, maxiter=maxiter, rtol=tol) if "rtol" in cg.__code__.co_varnames else cg(A, rhs, M=M, maxiter=maxiter, tol=tol)
        depth = np.full((H, W), np.nan, np.float32); depth[valid] = z
        # zero-mean per connected component
        ncc, lab = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)
        for i in range(1, ncc):
            m = lab == i
            depth[m] -= np.nanmean(depth[m])
        # residual: |dz - g| rms
        res = 0.0
        return depth, res

    def depth(self, n=None, work_w=224, relax_iters=4, min_nz=0.05):
        n = self.pseudo() if n is None else n
        H, W = n.shape[:2]
        s = work_w / W
        small = cv2.resize(n, (work_w, int(round(H * s))), interpolation=cv2.INTER_AREA)
        small /= np.linalg.norm(small, axis=-1, keepdims=True) + 1e-9
        small = self.relax_disambiguate(small, relax_iters)
        nz = small[..., 2]
        valid = np.abs(nz) >= min_nz
        p = np.where(valid, -small[..., 0] / np.where(valid, nz, 1), 0)
        q = np.where(valid, -small[..., 1] / np.where(valid, nz, 1), 0)
        cell = W / work_w
        d, _ = self.integrate_poisson(p * cell, q * cell, valid)
        return d, valid

    def view_depth(self, n=None):
        d, valid = self.depth(n)
        out = np.zeros(d.shape + (3,), np.uint8)
        if valid.any():
            v = d[valid]
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            g = np.zeros(d.shape, np.float32)
            g[valid] = np.clip((d[valid] - lo) / max(hi - lo, 1e-6), 0, 1)   # near = large z? keep "near bright": z toward camera is +
            g8 = (g * 255).astype(np.uint8)
            out = cv2.applyColorMap(g8, cv2.COLORMAP_INFERNO)
            out[~valid] = 0
        H, W = self.s0.shape
        return cv2.resize(out, (W, H), interpolation=cv2.INTER_NEAREST)


# ------------------------------------------------------------------ products table
def products(frame, n_refr=1.5, k_att=0.0, gamma_off=0.0, roll_deg=None):
    """LOCKED catalog (2026-08-26): the controller's normals modes only,
    roll-fixed and painted in the canon channel order.
      nxyz          CC pseudo (init_normals verbatim), abs paint
      n_xy / n_xz   component-pair variations of the same pseudo field
      nxyz_phys     VyzLut Fresnel LUT
      nxyz_diffuse  PxDiffuse (Atkinson)
      nxyz_specv2   PxSpecularV2 (Kadambi)
      aolp / dolp / edge everywhere"""
    f = frame
    ps = f.pseudo()
    phys, conf, refl = f.physical_vyzlut(n_refr)
    R = CAMERA_ROLL_DEG if roll_deg is None else roll_deg     # camera roll: -90 on our rig, 0 upright
    d_ps = f.rot_disp(ps, R)
    xy = d_ps.copy(); xy[..., 2] = 0
    xz = d_ps.copy(); xz[..., 1] = 0
    out = {
        "color":          f.color(),
        "rgb_deglare":    f.color(deglare=True),
        "intensity":      f.view_gray(f.intensity()),
        "nxyz":           f.encode(d_ps, "canon", signed=False),
        "n_xy":           f.encode(xy, "canon", signed=False),
        "n_xz":           f.encode(xz, "canon", signed=False),
        "nxyz_phys":      f.encode(f.rot_disp(phys, R), "canon", signed=True),
        "nxyz_diffuse":   f.encode(f.rot_disp(f.px_diffuse(n_refr, gamma_off), R), "canon", signed=True),
        "nxyz_specv2":    f.encode(f.rot_disp(f.px_specular_v2(n_refr, k_att, gamma_off), R), "canon", signed=True),
        "edge":           f.edge(ps),
    }
    return out


if __name__ == "__main__":
    import sys, time
    from pathlib import Path
    src = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = PolarFrame(src.read_bytes())
    P = products(fr)
    for k, v in P.items():
        cv2.imwrite(str(out / f"{k}.png"), v)
    print(f"{len(P)} products in {time.time() - t0:.1f}s -> {out}")
