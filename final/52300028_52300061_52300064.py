"""
TRUONG DAI HOC TON DUC THANG - KHOA CONG NGHE THONG TIN
BAI THI CUOI KY - NHAP MON XU LY ANH SO (MA MON: 505060)

NHOM SINH VIEN THUC HIEN:
1. Truong Duong Hung - MSSV: 52300028
2. Luong Minh Tan   - MSSV: 52300061
3. Le Quang Thanh   - MSSV: 52300064

YEU CAU DE BAI (FINAL REPORT):
- Tu dong nhan dien va ve bounding box chu nhat bao quanh MOI loai bien bao giao thong
  (tron, tam giac, chu nhat) o giua duong hoac ben phai duong.
- Tu dong SUY LUAN NOI DUNG bien bao va ghi len phia tren (hoac phia duoi) bien bao.
- Chi dung OpenCV (cv2), math, NumPy. Khong dung mo hinh hoc sau / pre-trained model.
- Mot ma nguon duy nhat cho moi video, khong hardcode theo so thu tu frame.
- Video dau ra: MP4, giu nguyen kich thuoc khung hinh, FPS, he mau. Chen MSSV vao moi frame.

Y TUONG TONG QUAT (thi giac may tinh kinh dien):
  HSV -> phan doan mau (do / xanh duong / vang) -> hinh thai hoc -> contour
  -> phan loai HINH DANG (tron / tam giac / chu nhat) -> phan tich BEN TRONG bien
  (ty le mau, thanh trang ngang, duong cheo, huong mui ten) de suy ra NOI DUNG
  -> NMS gop hop -> THEO DOI (tracking) + bo phieu nhan theo thoi gian de on dinh.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os
import time
import math
import cv2
import numpy as np

# ==============================================================================
# CAU HINH
# ==============================================================================
STUDENT_IDS = "MSSV: 52300028 - 52300061 - 52300064"
FILE_PREFIX = "52300028_52300061_52300064"

INPUT_VIDEOS = [
    ("video1.mp4", f"{FILE_PREFIX}_final_video1.mp4"),
    ("video2.mp4", f"{FILE_PREFIX}_final_video2.mp4"),
]

# Vung quan tam (ty le theo kich thuoc anh, KHONG phu thuoc video cu the):
# giua duong + ben phai (x >= 35% chieu rong), bo bau troi va mat duong sat dau xe.
ROI_X_MIN = 0.35
ROI_Y_MIN = 0.025
ROI_Y_MAX = 0.50

MIN_SIZE = 16          # kich thuoc toi thieu cua bien bao (pixel)
MAX_SIZE = 450         # kich thuoc toi da

GREEN = (0, 255, 0)

# ==============================================================================
# BUOC 1-2: PHAN DOAN MAU TRONG HSV
# ==============================================================================
def build_masks(roi_bgr):
    """Chuyen BGR -> HSV, phan doan do (2 dai Hue), xanh duong, vang; sau do closing."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
        cv2.inRange(hsv, (155, 70, 50), (180, 255, 255)))
    blue = cv2.inRange(hsv, (100, 90, 50), (135, 255, 255))
    blue_strong = cv2.inRange(hsv, (100, 120, 90), (130, 255, 255))   # xanh du sang, loai bong toi
    yellow = cv2.inRange(hsv, (15, 90, 110), (35, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 110), (180, 70, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))
    bright = cv2.inRange(hsv, (0, 0, 120), (180, 255, 255))

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_c = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=2)
    blue_c = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k, iterations=2)
    # Phep MO (opening) cat roi cot/than bien dinh vao dia xanh de contour tron hon
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue_o = cv2.morphologyEx(blue_c, cv2.MORPH_OPEN, k7)
    # Bien tam giac: vien do + nen vang -> gop lai thanh mot khoi dac
    warn = cv2.bitwise_or(red, yellow)
    warn_c = cv2.morphologyEx(warn, cv2.MORPH_CLOSE, k, iterations=2)
    return {
        'red': red, 'blue': blue, 'yellow': yellow, 'white': white, 'dark': dark, 'bright': bright, 'blue_strong': blue_strong,
        'red_c': red_c, 'blue_c': blue_c, 'blue_o': blue_o, 'warn_c': warn_c,
    }


# ==============================================================================
# CAC HAM TIEN ICH HINH HOC
# ==============================================================================
def circle_geometry(c):
    """Tra ve (circularity, fit_circle, (cx, cy), radius) cua mot contour."""
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    per = cv2.arcLength(hull, True)
    (cx, cy), r = cv2.minEnclosingCircle(hull)
    if per <= 0 or r <= 0:
        return 0, 0, (cx, cy), r
    circ = 4.0 * math.pi * hull_area / (per ** 2)
    fit = hull_area / (math.pi * r * r)
    return circ, fit, (cx, cy), r


def disc_mask(shape, cx, cy, r_out, r_in=0):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r_out), 255, -1)
    if r_in > 0:
        cv2.circle(m, (int(cx), int(cy)), int(r_in), 0, -1)
    return m


def frac(mask, region):
    n = cv2.countNonZero(region)
    if n == 0:
        return 0.0
    return cv2.countNonZero(cv2.bitwise_and(mask, region)) / float(n)


# ==============================================================================
# BUOC 3: UNG VIEN HINH TRON (do / xanh duong)
# ==============================================================================
def find_circle_candidates(masks):
    cands = []
    for name in ('red_c', 'blue_o'):
        mask = masks[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
                continue
            ar = w / float(h)
            if not (0.70 <= ar <= 1.40):
                continue
            circ, fit, (cx, cy), r = circle_geometry(c)
            min_fit = 0.68 if r < 20 else 0.75
            min_circ = 0.78 if r < 20 else 0.84
            if fit < min_fit or circ < min_circ:
                continue
            # Xac thuc vanh khuyen: mau chu dao tap trung o vanh ngoai [0.6R, R]
            ring = disc_mask(mask.shape, cx, cy, r, 0.60 * r)
            if frac(mask, ring) < 0.28:
                continue
            cands.append((x, y, w, h))
    return cands


# ==============================================================================
# BUOC 4: UNG VIEN HINH TAM GIAC (bien canh bao: vien do + nen vang)
# ==============================================================================
def find_triangle_candidates(masks):
    cands = []
    mask = masks['warn_c']
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
            continue
        ar = w / float(h)
        if not (0.85 <= ar <= 1.45):
            continue
        area = cv2.contourArea(c)
        if area < 0.30 * w * h:         # tam giac deu chiem ~50% bounding box
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < 0.80:
            continue
        # Xap xi da giac: tam giac -> 3 dinh
        tri_area, tri = cv2.minEnclosingTriangle(hull)
        if tri_area <= 0:
            continue
        tri_fit = hull_area / tri_area
        if tri_fit < 0.80:
            continue
        # Phai co du ca mau vang (nen) va mau do (vien)
        sub_y = masks['yellow'][y:y + h, x:x + w]
        sub_r = masks['red'][y:y + h, x:x + w]
        yr = cv2.countNonZero(sub_y) / float(w * h)
        rr = cv2.countNonZero(sub_r) / float(w * h)
        if yr < 0.12 or rr < 0.08:
            continue
        # Loai vat the tron (do tron cao) de tranh nham voi bien tron
        circ, fit, _, _ = circle_geometry(c)
        if circ > 0.86 and fit > 0.80:
            continue
        cands.append((x, y, w, h))
    return cands


# ==============================================================================
# BUOC 5: UNG VIEN HINH CHU NHAT XANH (bien chi dan)
# ==============================================================================
def find_rect_candidates(masks):
    cands = []
    mask = masks['blue_o']
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20 or w > MAX_SIZE or h > MAX_SIZE:
            continue
        ar = w / float(h)
        if not (0.55 <= ar <= 2.6):
            continue
        area = cv2.contourArea(c)
        if area / float(w * h) < 0.80:   # hinh chu nhat lap day bbox
            continue
        circ, fit, _, _ = circle_geometry(c)
        if circ > 0.84 and fit > 0.75:    # la hinh tron -> khong phai
            continue
        # Co chu / hinh ve mau trang ben trong
        wr = cv2.countNonZero(masks['white'][y:y + h, x:x + w]) / float(w * h)
        if wr < 0.04 or wr > 0.60:
            continue
        cands.append((x, y, w, h))
    return cands


# ==============================================================================
# BUOC 6: NMS GOP HOP TRUNG LAP / LONG NHAU
# ==============================================================================
def merge_boxes(boxes, iou_thr=0.25, ov_thr=0.50):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, w1, h1 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = boxes[j]
                ix1, iy1 = max(x1, x2), max(y1, y2)
                ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = w1 * h1 + w2 * h2 - inter
                    iou = inter / float(union) if union > 0 else 0
                    ov = inter / float(min(w1 * h1, w2 * h2))
                    if iou > iou_thr or ov > ov_thr:
                        used[j] = True
                        nx, ny = min(x1, x2), min(y1, y2)
                        w1 = max(x1 + w1, x2 + w2) - nx
                        h1 = max(y1 + h1, y2 + h2) - ny
                        x1, y1 = nx, ny
                        changed = True
            out.append([x1, y1, w1, h1])
        boxes = out
    return [tuple(b) for b in boxes]


# ==============================================================================
# BUOC 7: SUY LUAN NOI DUNG BIEN BAO
# ==============================================================================
def diag_band_coverage(mask, cx, cy, r, sign):
    """
    Do muc phu cua mau (mask) doc theo mot duong cheo di qua tam.
    sign=+1: cheo '\\' (tren-trai -> duoi-phai), sign=-1: cheo '/'.
    """
    band = np.zeros(mask.shape, np.uint8)
    p1 = (int(cx - 0.8 * r), int(cy - sign * 0.8 * r))
    p2 = (int(cx + 0.8 * r), int(cy + sign * 0.8 * r))
    cv2.line(band, p1, p2, 255, max(2, int(0.22 * r)))
    inner = disc_mask(mask.shape, cx, cy, 0.85 * r)
    band = cv2.bitwise_and(band, inner)
    return frac(mask, band)


def arrow_direction(white_inner):
    """
    Suy ra huong mui ten trang tren nen xanh: dung momen anh.
    Tam khoi luong lech ve phia dau mui ten (dau mui ten nang hon than).
    Tra ve (dx, dy) don vi anh (dy > 0 la huong xuong).
    """
    ys, xs = np.nonzero(white_inner)
    if len(xs) < 10:
        return None
    mx, my = xs.mean(), ys.mean()
    bx, by = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    # truc chinh (PCA) de biet mui ten nam theo phuong nao
    pts = np.stack([xs - mx, ys - my], 1).astype(np.float32)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, 1]                     # vector rieng lon nhat
    if w[1] < 1.6 * max(w[0], 1e-6):   # khong keo dai -> khong phai mui ten ro
        return None
    d = np.array([mx - bx, my - by])
    if np.dot(d, axis) < 0:
        axis = -axis
    return float(axis[0]), float(axis[1])


def classify_circle(roi_bgr, masks, box):
    x, y, w, h = box
    sub = {k: v[y:y + h, x:x + w] for k, v in masks.items()}
    H, W = sub['red'].shape
    cx, cy, r = W / 2.0, H / 2.0, min(W, H) / 2.0
    ring = disc_mask((H, W), cx, cy, r, 0.62 * r)
    inner = disc_mask((H, W), cx, cy, 0.62 * r)

    red_ring = frac(sub['red'], ring)
    blue_ring = frac(sub['blue'], ring)
    red_in = frac(sub['red'], inner)
    blue_in = frac(sub['blue'], inner)
    white_in = frac(sub['white'], inner)
    dark_in = frac(sub['dark'], inner)

    # (a) Vien do + nen xanh duong ben trong: bien CAM DUNG / CAM DO XE
    if red_ring >= 0.25 and blue_in >= 0.30:
        c1 = diag_band_coverage(sub['red'], cx, cy, r, +1)
        c2 = diag_band_coverage(sub['red'], cx, cy, r, -1)
        if min(c1, c2) >= 0.12 and min(c1, c2) >= 0.6 * max(c1, c2):
            return "Cam dung xe va do xe"
        return "Cam do xe"

    # (b) Nen do dac + thanh trang ngang: CAM DI NGUOC CHIEU
    if red_in >= 0.45 and red_ring >= 0.45:
        # Thanh trang NGANG o giua, phia tren va phia duoi thanh phai la nen do
        # (loai den giao thong do: loi sang tron, khong co thanh ngang)
        band = np.zeros((H, W), np.uint8)
        cv2.rectangle(band, (int(cx - 0.6 * r), int(cy - 0.18 * r)), (int(cx + 0.6 * r), int(cy + 0.18 * r)), 255, -1)
        above = np.zeros((H, W), np.uint8)
        cv2.rectangle(above, (int(cx - 0.4 * r), int(cy - 0.62 * r)), (int(cx + 0.4 * r), int(cy - 0.28 * r)), 255, -1)
        below = np.zeros((H, W), np.uint8)
        cv2.rectangle(below, (int(cx - 0.4 * r), int(cy + 0.28 * r)), (int(cx + 0.4 * r), int(cy + 0.62 * r)), 255, -1)
        bar = frac(sub['white'], cv2.bitwise_and(band, inner))
        red_ab = frac(sub['red'], above)
        red_bl = frac(sub['red'], below)
        if bar >= 0.25 and red_ab >= 0.6 and red_bl >= 0.6:
            return "Cam di nguoc chieu"
        return None

    # (c) Vien do + nen trang + hinh den: bien CAM
    if red_ring >= 0.25 and white_in >= 0.25:
        ys, xs = np.nonzero(cv2.bitwise_and(sub['dark'], inner))
        if len(xs) > 0.03 * cv2.countNonZero(inner):
            # Mui ten re: than o duoi, dau/uon cong o tren -> so sanh x(phan tren) voi x(phan duoi)
            top, bot = ys < cy - 0.15 * r, ys > cy + 0.15 * r
            if top.sum() > 3 and bot.sum() > 3:
                off = (xs[top].mean() - xs[bot].mean()) / r
                if off <= -0.10:
                    return "Cam re trai"
                if off >= 0.10:
                    return "Cam re phai"
        return "Bien cam"

    # (d) Nen xanh duong + mui ten trang: bien HIEU LENH
    if (blue_ring >= 0.40 or blue_in >= 0.40) and frac(sub['blue_strong'], ring) >= 0.30:
        arrow = cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_not(sub['blue']), sub['bright']), inner)
        if cv2.countNonZero(arrow) < 0.03 * cv2.countNonZero(inner):
            return None            # nen xanh tron khong co hinh ve -> khong phai bien
        d = arrow_direction(arrow)
        if d is not None:
            dx, dy = d
            ang = math.degrees(math.atan2(dy, dx))    # 0: phai, 90: xuong
            if -150 <= ang <= -30:
                return "Di thang"
            if dx > 0.25:
                return "Huong di vong ben phai"
            if dx < -0.25:
                return "Huong di vong ben trai"
        # Bien o xa, khong thay ro hinh ve: chi nhan neu dia xanh dong nhat, rat dac
        if r >= 25 or (blue_ring >= 0.85 and blue_in >= 0.85):
            return "Bien hieu lenh"
        return None

    return None


def classify_triangle(roi_bgr, masks, box):
    return "Bien canh bao"


def classify_rect(roi_bgr, masks, box):
    return "Bien chi dan"


# ==============================================================================
# PHAT HIEN + PHAN LOAI TREN MOT FRAME
# ==============================================================================
def detect_signs(frame):
    """Tra ve danh sach (x, y, w, h, nhan) toa do tren anh goc."""
    H, W = frame.shape[:2]
    x0, y0, y1 = int(ROI_X_MIN * W), int(ROI_Y_MIN * H), int(ROI_Y_MAX * H)
    roi = frame[y0:y1, x0:W]
    masks = build_masks(roi)

    circles = merge_boxes(find_circle_candidates(masks))
    tris = merge_boxes(find_triangle_candidates(masks))
    rects = merge_boxes(find_rect_candidates(masks))

    items = []
    for b in circles:
        items.append((b, 'circle'))
    for b in tris:
        items.append((b, 'tri'))
    for b in rects:
        items.append((b, 'rect'))

    # Uu tien: tron > tam giac > chu nhat khi chong lan; bo hop bi chua boi hop uu tien cao
    prio = {'circle': 0, 'tri': 1, 'rect': 2}
    items.sort(key=lambda t: prio[t[1]])
    kept = []
    for b, kind in items:
        bx, by, bw, bh = b
        dup = False
        for kb, kk in kept:
            kx, ky, kw, kh = kb
            ix1, iy1 = max(bx, kx), max(by, ky)
            ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / float(min(bw * bh, kw * kh)) > 0.5:
                    dup = True
                    break
        if not dup:
            kept.append((b, kind))

    results = []
    for (bx, by, bw, bh), kind in kept:
        box = (bx, by, bw, bh)
        if kind == 'circle':
            label = classify_circle(roi, masks, box)
        elif kind == 'tri':
            label = classify_triangle(roi, masks, box)
        else:
            label = classify_rect(roi, masks, box)
        if label is None:          # hinh dang giong bien nhung noi dung khong hop le -> bo
            continue
        results.append((bx + x0, by + y0, bw, bh, label))
    return results


# ==============================================================================
# THEO DOI (TRACKING) + BO PHIEU NHAN THEO THOI GIAN -> ON DINH
# ==============================================================================
class SignTracker:
    def __init__(self, max_missed=4, history=25, min_hits=2):
        self.tracks = []
        self.max_missed = max_missed
        self.history = history
        self.min_hits = min_hits

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / float(aw * ah + bw * bh - inter)

    def update(self, dets):
        for t in self.tracks:
            t['matched'] = False
        for (x, y, w, h, label) in dets:
            best, best_iou = None, 0.0
            for t in self.tracks:
                if t['matched']:
                    continue
                v = self._iou(t['box'], (x, y, w, h))
                if v > best_iou:
                    best, best_iou = t, v
            if best is not None and best_iou > 0.15:
                best['box'] = (x, y, w, h)
                best['labels'].append(label)
                best['labels'] = best['labels'][-self.history:]
                best['hits'] += 1
                best['missed'] = 0
                best['matched'] = True
            else:
                self.tracks.append({'box': (x, y, w, h), 'labels': [label], 'hits': 1,
                                    'missed': 0, 'matched': True})
        for t in self.tracks:
            if not t['matched']:
                t['missed'] += 1
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]

        out = []
        for t in self.tracks:
            if t['hits'] < self.min_hits:
                continue
            labels = t['labels']
            label = max(set(labels), key=labels.count)   # bo phieu da so
            x, y, w, h = t['box']
            out.append((x, y, w, h, label))
        return out


# ==============================================================================
# VE KET QUA
# ==============================================================================
def draw_annotations(frame, signs, student_ids_text):
    H, W = frame.shape[:2]
    for (x, y, w, h, label) in signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        scale, th = 0.6, 2
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        tx = int(min(max(0, x + w / 2 - tw / 2), W - tw - 2))
        ty = y - 8                                 # phia tren bien
        if ty - tht - 4 < 85:                      # sat mep tren / de len MSSV -> ghi phia duoi
            ty = y + h + tht + 10
        cv2.rectangle(frame, (tx - 3, ty - tht - 4), (tx + tw + 3, ty + base), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 255), th, cv2.LINE_AA)

    cv2.rectangle(frame, (25, 20), (840, 75), (0, 0, 0), -1)
    cv2.putText(frame, student_ids_text, (35, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def process_video_file(input_path, output_path, student_ids_text):
    if not os.path.exists(input_path):
        print(f"[LOI] Khong tim thay tap tin dau vao: {input_path}")
        return
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[LOI] Khong the mo video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("=" * 70)
    print(f"XU LY: {input_path} | {width}x{height} | {fps:.2f} FPS | {total} frames")
    print(f"Xuat ra: {output_path}")
    print("=" * 70)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    tracker = SignTracker()
    idx, t0 = 0, time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        signs = tracker.update(detect_signs(frame))
        out.write(draw_annotations(frame, signs, student_ids_text))
        idx += 1
        if idx % 300 == 0 or idx == total:
            el = time.time() - t0
            print(f"  Tien do: {idx}/{total} ({100.0 * idx / max(total, 1):.1f}%) - {idx / max(el, 1e-6):.1f} FPS")
    cap.release()
    out.release()
    print(f"[HOAN THANH] {output_path} ({time.time() - t0:.1f}s)\n")


def main():
    print("=" * 70)
    print("NHAN DIEN VA SUY LUAN NOI DUNG BIEN BAO GIAO THONG - BAI THI CUOI KY")
    print(f"Nhom sinh vien: {STUDENT_IDS}")
    print("=" * 70)
    for src, dst in INPUT_VIDEOS:
        process_video_file(src, dst, STUDENT_IDS)
    print("HOAN TAT.")


if __name__ == '__main__':
    main()
