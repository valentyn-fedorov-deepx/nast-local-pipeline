"""Squeeze the most out of an operator crop before TRELLIS: SAM mask + SR.
  1. SAM (facebook/sam-vit-huge via transformers) with the crop's own box as
     the prompt -> a clean object mask that keeps thin structures and drops
     neighbours (a second lamp, a tree crown).
  2. Real-ESRGAN x4 on the crop (falls back to Lanczos x4 if unavailable) ->
     sharper edges for the image encoder.
  3. Compose RGBA on a square canvas (object ~85% of the side), alpha =
     upscaled SAM mask. TRELLIS then skips rembg and uses our alpha.
Usage: python crop_enhance.py <in_dir> <out_dir>
  in_dir  : PNGs; optional sidecar <name>.box.json {"x0","y0","x1","y1"} in
            crop pixel coords for the SAM prompt (default: whole crop with margin)
  out_dir : enhanced RGBA PNGs, same names
Env: REALESRGAN_WEIGHTS (RealESRGAN_x4plus.pth), TRELLIS_MAX_VIEWS (default 5;
     3 on cards under 14 GB — multi-image fusion memory grows with the views)
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
IN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------- SAM ----------------
from transformers import SamModel, SamProcessor
sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
print("SAM loaded", flush=True)
def object_points(img, box, k=6):
    """Positive prompts ON the object, negatives on the background.
    Objects in these crops are darker than their surroundings (silhouettes
    on sky, cars on grey lot): sample the darkest pixels inside the box
    spread over its height; negatives = brightest pixels near the box edge."""
    g = np.array(img.convert("L")).astype(np.float32)
    x0, y0, x1, y1 = [int(v) for v in box]
    win = g[y0:y1, x0:x1]
    if win.size < 50:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return [[cx, cy]], [1]
    thr = np.percentile(win, 12)
    ys, xs = np.where(win <= thr)
    pos = []
    if len(ys):
        # spread along the vertical extent
        order = np.argsort(ys)
        for q in np.linspace(0.05, 0.95, k):
            i = order[int(q * (len(order) - 1))]
            pos.append([float(xs[i] + x0), float(ys[i] + y0)])
    else:
        pos = [[(x0 + x1) / 2, (y0 + y1) / 2]]
    # negatives: bright pixels in a ring just inside the box edge
    H, W = g.shape
    ring = np.zeros_like(g, bool)
    m = max(4, int(0.06 * max(x1 - x0, y1 - y0)))
    ring[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    ring[y0 + m:y1 - m, x0 + m:x1 - m] = False
    ry, rx = np.where(ring & (g > np.percentile(g[ring], 70)))
    neg = []
    if len(ry):
        for i in np.linspace(0, len(ry) - 1, 4).astype(int):
            neg.append([float(rx[i]), float(ry[i])])
    return pos + neg, [1] * len(pos) + [0] * len(neg)
def sam_mask(img, box, hints=None):
    pts, labs = object_points(img, box)
    if hints and hints.get("pos"):
        # geometry-verified prompts from the inspector (object points projected
        # into this frame + depth-disagreeing background) replace the darkness
        # heuristic positives; the bright-ring negatives stay as extra guards
        ring_neg = [p for p, l in zip(pts, labs) if l == 0]
        pos = [[float(x), float(y)] for x, y in hints["pos"]]
        neg = [[float(x), float(y)] for x, y in (hints.get("neg") or [])]
        pts = pos + neg + ring_neg
        labs = [1] * len(pos) + [0] * (len(neg) + len(ring_neg))
    inputs = proc(img, input_points=[[pts]], input_labels=[[labs]], input_boxes=[[list(box)]],
                  return_tensors="pt").to(dev)
    with torch.inference_mode():
        out = sam(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0]
    scores = out.iou_scores[0, 0].cpu().numpy()
    W, H = img.size
    box_area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    best, best_s = None, -1e9
    geo = bool(hints and hints.get("pos"))
    for m, s in zip(masks.numpy(), scores):
        m = m.astype(bool)
        # must contain the positive prompts, must NOT be a background blob
        hit = np.mean([m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] for p, l in zip(pts, labs) if l == 1])
        cover_box = m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].sum() / box_area
        outside = m.sum() - m[int(box[1]):int(box[3]), int(box[0]):int(box[2])].sum()
        # negatives must be OUTSIDE the mask, hard veto if a negative is inside
        neg_in = np.mean([m[int(min(H - 1, p[1])), int(min(W - 1, p[0]))] for p, l in zip(pts, labs) if l == 0]) if 0 in labs else 0.0
        # background masks are vetoed outright: most negatives inside (sky
        # around a pole) or the box nearly filled. Geometry positives are worth
        # a lot but not a veto -- on a 3-px pole a few of them land beside it
        # (a tight operator box is legitimately filled by the object -- only a
        # mask that also spills far outside the box is background)
        outside_frac = outside / max(1, m.sum())
        n_neg = sum(1 for l in labs if l == 0)
        if (neg_in > 0.5 and n_neg >= 3) or (cover_box > 0.7 and outside_frac > 0.25):
            continue
        sc = float(s) + (2.0 if geo else 0.6) * hit - 1.5 * (outside / max(1, m.sum())) \
             - (1.5 if cover_box > 0.55 else 0) - 2.0 * neg_in
        if sc > best_s:
            best, best_s = m, sc
    if best is None:
        return np.zeros((H, W), np.uint8), -9.0
    mask = best.astype(np.uint8) * 255
    # the ROI box is the operator's definition of the object: nothing beyond
    # it (+12%) belongs -- a pole running to the frame bottom, a bush touching it
    bx0, by0, bx1, by1 = box
    bm = 0.12 * max(bx1 - bx0, by1 - by0)
    clip = np.zeros_like(mask)
    clip[int(max(0, by0 - bm)):int(min(H, by1 + bm)), int(max(0, bx0 - bm)):int(min(W, bx1 + bm))] = 255
    mask = np.minimum(mask, clip)
    # keep only components that contain a positive prompt (drops clouds
    # that share the mask with the pole)
    import cv2
    n_lab, lab = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if n_lab > 2:
        keep = set()
        for p, l in zip(pts, labs):
            if l == 1:
                v = lab[int(min(H - 1, p[1])), int(min(W - 1, p[0]))]
                if v > 0:
                    keep.add(int(v))
        if keep:
            # among prompted components keep the dominant one (largest
            # extent) plus anything touching it -- a neighbour of the same
            # class standing apart in the crop is not the target
            ext = {}
            for i in keep:
                ys, xs = np.where(lab == i)
                ext[i] = max(np.ptp(ys) if len(ys) else 0, np.ptp(xs) if len(xs) else 0)
            main = max(ext, key=ext.get)
            grown = cv2.dilate((lab == main).astype(np.uint8), np.ones((15, 15), np.uint8))
            touch = set(np.unique(lab[grown > 0])) - {0}
            mask = np.isin(lab, list(touch)).astype(np.uint8) * 255
    return mask, float(best_s)
# ---------------- SR ----------------
_upsampler = None
def sr4(img):
    global _upsampler
    try:
        if _upsampler is None:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            wpath = os.environ.get("REALESRGAN_WEIGHTS") or str(Path.home() / "realesrgan/RealESRGAN_x4plus.pth")
            _upsampler = RealESRGANer(scale=4, model_path=wpath,
                                      model=model, tile=256, tile_pad=10, pre_pad=0, half=False, device=dev)
        arr = np.array(img.convert("RGB"))[:, :, ::-1]
        out, _ = _upsampler.enhance(arr, outscale=4)
        return Image.fromarray(out[:, :, ::-1]), "realesrgan"
    except Exception as e:
        w, h = img.size
        return img.convert("RGB").resize((w * 4, h * 4), Image.LANCZOS), f"lanczos ({type(e).__name__})"
import cv2
if os.environ.get("TRELLIS_MAX_VIEWS"):
    MAX_VIEWS = int(os.environ["TRELLIS_MAX_VIEWS"])
elif dev == "cuda" and torch.cuda.get_device_properties(0).total_memory < 14e9:
    MAX_VIEWS = 3                                   # 12 GB cards: fewer views, no OOM
elif dev == "cuda" and torch.cuda.get_device_properties(0).total_memory >= 24e9:
    MAX_VIEWS = 8                                   # 24-32 GB cards: the fusion holds eight views
else:
    MAX_VIEWS = 5
print(f"max views: {MAX_VIEWS}", flush=True)
cands = []
for p in sorted(IN.glob("*.png")):
    img = Image.open(p).convert("RGB")
    W, H = img.size
    side = p.with_suffix(".box.json")
    hints = None
    b = {}
    if side.exists():
        b = json.loads(side.read_text())
        box = [float(b["x0"]), float(b["y0"]), float(b["x1"]), float(b["y1"])]
        if b.get("pos"):
            hints = {"pos": [[x * 4, y * 4] for x, y in b["pos"]],
                     "neg": [[x * 4, y * 4] for x, y in (b.get("neg") or [])]}
    else:
        m = 0.06
        box = [W * m, H * m, W * (1 - m), H * (1 - m)]
    # SR first so SAM sees sharp edges, then segment on the upscaled image
    up, how = sr4(img)
    box_up = [v * 4 for v in box]
    mask, sc = sam_mask(up, box_up, hints)
    if hints:
        print(f"  prompts: {len(hints['pos'])} pos / {len(hints['neg'])} neg from geometry", flush=True)
    cover = (mask > 0).mean()
    print(f"{p.name}: SAM score {sc:.2f}, coverage {cover*100:.1f}%", flush=True)
    if cover < 0.001:
        # SAM found nothing usable: a whole-box "mask" only feeds TRELLIS
        # background -- skip this view (the shell falls back to raw crops if
        # every view is skipped)
        print(f"  {p.name}: no usable mask, view skipped", flush=True)
        continue
    a = mask
    # thin structures: give the alpha a little body so the 518px resize
    # inside TRELLIS keeps them; smooth the edge
    a = cv2.dilate(a, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a, (3, 3), 0)
    ys, xs = np.where(a > 128)
    if len(ys) < 10:
        print(f"  {p.name}: empty mask, skipped", flush=True)
        continue
    y0, y1 = ys.min(), ys.max() + 1; x0, x1 = xs.min(), xs.max() + 1
    # objects that were tiny in the frame come out of SR as smears that only
    # confuse TRELLIS: mask extent back in ORIGINAL frame pixels must be usable
    kk = float(b.get("scale", 1.0)) if side.exists() else 1.0
    orig_px = max(y1 - y0, x1 - x0) / (4.0 * max(kk, 1.0))
    if orig_px < 40:
        print(f"  {p.name}: object only ~{orig_px:.0f} px in the frame, view skipped", flush=True)
        continue
    # ---- quality of this view (judged RELATIVE to the set below) ----
    mb = (a[y0:y1, x0:x1] > 128).astype(np.uint8)
    n_cc, cc = cv2.connectedComponents(mb)
    cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts) or 1.0
    solidity = float(mb.sum() / hull_area)                 # holes / partial masks -> low
    # sharpness on the ORIGINAL (pre-SR) crop pixels of the object
    g0 = np.array(img.convert("L")).astype(np.float32)
    m0 = cv2.resize((a > 128).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    lap = cv2.Laplacian(g0, cv2.CV_32F)
    sharp = float(lap[m0].var()) if m0.sum() > 50 else 0.0
    cands.append(dict(name=p.name, up=up, a=a, box=(x0, y0, x1, y1), how=how, orig_px=orig_px,
                      solidity=solidity, sharp=sharp, ncc=int(n_cc - 1),
                      aspect=float((x1 - x0) / max(1, y1 - y0)), fill=float(mb.mean()),
                      bearing=(float(b["bearing"]) if b.get("bearing") is not None else None),
                      manual=p.name.startswith("roi_0") and int(p.stem.split("_")[1]) < 2))
# ---- keep the views that agree with the set: partial masks (holes, cut
# bodies), blurred frames and crumbs distort a multi-image TRELLIS fusion far
# more than a missing view ever could
if cands:
    med_sol = float(np.median([c["solidity"] for c in cands]))
    med_sharp = float(np.median([c["sharp"] for c in cands])) or 1.0
    med_asp = float(np.median([c["aspect"] for c in cands])) or 1.0
    med_fill = float(np.median([c["fill"] for c in cands])) or 1.0
    for c in cands:
        c["rel_sol"] = c["solidity"] / max(med_sol, 1e-6)
        c["rel_sharp"] = c["sharp"] / max(med_sharp, 1e-6)
        # silhouette shape must agree with the set: a bare pole sliver among
        # head+arm views, a half car among full cars -> out
        ra = c["aspect"] / max(med_asp, 1e-6); rf = c["fill"] / max(med_fill, 1e-6)
        shape_ok = (1 / 1.8 <= ra <= 1.8) and (1 / 1.8 <= rf <= 1.8)
        c["shape_ok"] = shape_ok
        ok = c["rel_sol"] >= 0.75 and c["rel_sharp"] >= 0.35 and c["ncc"] <= 3 and shape_ok
        c["ok"] = ok
        c["score"] = (min(c["rel_sol"], 1.2) * min(c["rel_sharp"], 1.5) ** 0.5
                      * np.sqrt(c["orig_px"]) * (1.15 if c["manual"] else 1.0))
    # best view first, then the best view of a side not covered yet: eight near-duplicates of one side teach the
    # fusion nothing that one of them does not, a view of the rear does
    good = sorted([c for c in cands if c["ok"]], key=lambda c: -c["score"]); keep = []
    while good and len(keep) < MAX_VIEWS:
        def gain(c):
            known = [k["bearing"] for k in keep if k.get("bearing") is not None]
            if c.get("bearing") is None or not known:
                return c["score"]
            d = min(abs((c["bearing"] - k + 180.0) % 360.0 - 180.0) for k in known)
            return c["score"] * (0.4 + 0.6 * min(d / 35.0, 1.0))
        best = max(good, key=gain); keep.append(best); good.remove(best)
    if not keep:                                    # never end up with nothing: best 2 by score
        keep = sorted(cands, key=lambda c: -c["score"])[:2]
    kept_names = {c["name"] for c in keep}
    report = {c["name"]: {k: (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in c.items() if k in ("solidity", "sharp", "ncc", "orig_px", "rel_sol",
                                                          "rel_sharp", "ok", "score", "manual", "aspect", "fill", "shape_ok", "bearing")}
              for c in cands}
    for n, r in report.items():
        r["kept"] = n in kept_names
    (OUT / "quality.json").write_text(json.dumps(report, indent=1))
    for c in cands:
        r = report[c["name"]]
        print(f"  {c['name']}: solidity {c['solidity']:.2f} (x{c['rel_sol']:.2f}) sharp x{c['rel_sharp']:.2f} "
              f"cc {c['ncc']} px {c['orig_px']:.0f} -> {'KEEP' if r['kept'] else 'drop'}", flush=True)
    for c in keep:
        x0, y0, x1, y1 = c["box"]
        rgb = np.array(c["up"])[y0:y1, x0:x1]
        al = c["a"][y0:y1, x0:x1]
        h_, w_ = al.shape
        S = int(max(h_, w_) / 0.85)
        canvas = np.zeros((S, S, 4), np.uint8)
        oy, ox = (S - h_) // 2, (S - w_) // 2
        canvas[oy:oy + h_, ox:ox + w_, :3] = rgb
        canvas[oy:oy + h_, ox:ox + w_, 3] = al
        Image.fromarray(canvas, "RGBA").save(OUT / c["name"])
        print(f"  -> {c['name']} {S}x{S} via {c['how']}", flush=True)
print("CROP_ENHANCE_DONE", flush=True)
