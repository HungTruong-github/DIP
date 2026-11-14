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

TIEU CHI DANH GIA (RUBRIC 7.0 DIEM LAP TRINH):
1. Nhan dien dung cac bien bao (31 bien tren 2 video): 3.0 diem
2. Nhan dien on dinh va duy nhat tu xa den gan: 3.0 diem (tru 0.25 diem neu trung lap tren cung 1 bien)
3. Phat hien thua / nhieu (khong phai bien bao): 1.0 diem (< 5 phat hien thua duoc 1.0 diem)
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
# CAU HINH DUONG DAN VA THONG SO
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STUDENT_IDS = "MSSV: 52300028 - 52300061 - 52300064"
FILE_PREFIX = "52300028_52300061_52300064"

INPUT_VIDEOS = [
    (os.path.join(PROJECT_DIR, "video1.mp4"),
     os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_final_video1.mp4")),
    (os.path.join(PROJECT_DIR, "video2.mp4"),
     os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_final_video2.mp4")),
]

# Vung quan tam (ROI):
# - De bai quy dinh: "chi cac bien bao o giua duong hoac ben phai duong"
# - x >= 0.38W: bat dau tu dai phan cach giua duong den het le phai (loai bo xe doi dien va le trai)
# - y tu 0.03H den 0.46H: bao gom tat ca cac bien tren cao va bien ben le duong khi xe den gan;
#   loai bo hoan toan mat duong, dau xe, banh xe va den xe may o phia duoi.
ROI_X_MIN = 0.38
ROI_Y_MIN = 0.03
ROI_Y_MAX = 0.46

MIN_SIZE = 14          # kich thuoc toi thieu cua bien bao o xa (pixel)
MAX_SIZE = 190         # kich thuoc toi da cua bien bao khi den rat gan (pixel)

GREEN = (0, 255, 0)

# ==============================================================================
# BUOC 1-2: PHAN DOAN MAU TRONG HE HSV
# ==============================================================================
def build_masks(roi_bgr):
    """Chuyen BGR -> HSV, phan doan mau theo chuan bien bao giao thong Viet Nam."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Mau do: 2 dai Hue (0-10 va 155-180)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 65, 45), (10, 255, 255)),
        cv2.inRange(hsv, (155, 65, 45), (180, 255, 255)))

    # Mau xanh duong: Hue 100-132, S >= 75
    blue = cv2.inRange(hsv, (100, 75, 45), (132, 255, 255))
    blue_strong = cv2.inRange(hsv, (102, 115, 80), (128, 255, 255))

    # Mau vang: bien canh bao tam giac (Hue 16-35)
    yellow = cv2.inRange(hsv, (16, 90, 100), (34, 255, 255))

    # Mau trang va mau toi
    white = cv2.inRange(hsv, (0, 0, 115), (180, 70, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 85))
    bright = cv2.inRange(hsv, (0, 0, 125), (180, 255, 255))

    # Bien doi hinh thai hoc (Morphology)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_c = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k3, iterations=2)
    blue_c = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k3, iterations=2)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_o = cv2.morphologyEx(blue_c, cv2.MORPH_OPEN, k5)

    warn = cv2.bitwise_or(red, yellow)
    warn_c = cv2.morphologyEx(warn, cv2.MORPH_CLOSE, k3, iterations=2)

    return {
        'red': red, 'blue': blue, 'yellow': yellow, 'white': white, 'dark': dark,
        'bright': bright, 'blue_strong': blue_strong,
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


def _size_position_check(y, w, h, roi_h):
    """
    Kiem tra tinh hop ly hinh hoc phoi canh (Perspective Geometry):
    - Bien bao o xa (kich thuoc nho < 25px) nam o phan tren/giua khung hinh.
    - O phan duoi khung hinh (sat mat duong), bien bao bat buoc phai co kich thuoc lon do o gan.
    - Neu mot vat the rat nho nam o vi tri thap sat mat duong -> chac chan la nhieu / den xe.
    """
    rel_y = (y + h / 2.0) / float(roi_h)
    max_dim = max(w, h)
    if max_dim < 24 and rel_y > 0.72:
        return False
    if max_dim < 32 and rel_y > 0.85:
        return False
    return True


# ==============================================================================
# BUOC 3: UNG VIEN HINH TRON (Bien cam & Bien hieu lenh)
# ==============================================================================
def find_circle_candidates(masks):
    cands = []
    roi_h = masks['red_c'].shape[0]

    for name in ('red_c', 'blue_o'):
        mask = masks[name]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
                continue
            ar = w / float(h)
            if not (0.70 <= ar <= 1.35):
                continue
            if not _size_position_check(y, w, h, roi_h):
                continue

            circ, fit, (cx, cy), r = circle_geometry(c)
            if r < 16:
                min_fit, min_circ = 0.60, 0.70
            elif r < 25:
                min_fit, min_circ = 0.66, 0.76
            else:
                min_fit, min_circ = 0.74, 0.82

            if fit < min_fit or circ < min_circ:
                continue

            # Kiem tra vanh khuyen (annulus ring)
            inner_r = 0.55 * r
            ring = disc_mask(mask.shape, cx, cy, r, inner_r)
            min_ring = 0.25 if r < 20 else 0.32
            if frac(mask, ring) < min_ring:
                continue

            cands.append((x, y, w, h))
    return cands


# ==============================================================================
# BUOC 4: UNG VIEN HINH TAM GIAC (Bien canh bao: Vien do + Nen vang)
# ==============================================================================
def find_triangle_candidates(masks):
    cands = []
    mask = masks['warn_c']
    roi_h = mask.shape[0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < MIN_SIZE or h < MIN_SIZE or w > 130 or h > 120:
            continue
        ar = w / float(h)
        if not (0.80 <= ar <= 1.40):
            continue
        if not _size_position_check(y, w, h, roi_h):
            continue

        area = cv2.contourArea(c)
        if area < 0.25 * w * h:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0 or area / hull_area < 0.76:
            continue

        tri_area, _ = cv2.minEnclosingTriangle(hull)
        if tri_area <= 0:
            continue
        if hull_area / tri_area < 0.75:
            continue

        sub_y = masks['yellow'][y:y + h, x:x + w]
        sub_r = masks['red'][y:y + h, x:x + w]
        yr = cv2.countNonZero(sub_y) / float(w * h)
        rr = cv2.countNonZero(sub_r) / float(w * h)

        if yr < 0.08 or rr < 0.06:
            continue

        circ, fit, _, _ = circle_geometry(c)
        if circ > 0.85 and fit > 0.80:
            continue

        cands.append((x, y, w, h))
    return cands


# ==============================================================================
# BUOC 5: UNG VIEN HINH CHU NHAT (Bien chi dan: Nen xanh + Chu/Hinh trang)
# ==============================================================================
def find_rect_candidates(masks):
    cands = []
    mask = masks['blue_c']
    roi_h = mask.shape[0]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 22 or h < 18 or w > 190 or h > 140:
            continue
        ar = w / float(h)
        if not (0.50 <= ar <= 2.80):
            continue
        if not _size_position_check(y, w, h, roi_h):
            continue

        hull = cv2.convexHull(c)
        ha = cv2.contourArea(hull)
        if ha / float(w * h) < 0.70:
            continue

        circ, fit, _, _ = circle_geometry(c)
        if circ > 0.85 and fit > 0.78:
            continue

        bs_ratio = cv2.countNonZero(masks['blue_strong'][y:y + h, x:x + w]) / float(w * h)
        if bs_ratio < 0.22:
            continue

        wr = cv2.countNonZero(masks['white'][y:y + h, x:x + w]) / float(w * h)
        if wr < 0.03 or wr > 0.60:
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
    band = np.zeros(mask.shape, np.uint8)
    p1 = (int(cx - 0.75 * r), int(cy - sign * 0.75 * r))
    p2 = (int(cx + 0.75 * r), int(cy + sign * 0.75 * r))
    cv2.line(band, p1, p2, 255, max(2, int(0.22 * r)))
    inner = disc_mask(mask.shape, cx, cy, 0.82 * r)
    band = cv2.bitwise_and(band, inner)
    return frac(mask, band)


def arrow_direction(white_inner):
    """Suy luan huong mui ten trang bang PCA."""
    ys, xs = np.nonzero(white_inner)
    if len(xs) < 12:
        return None
    mx, my = xs.mean(), ys.mean()
    bx, by = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    pts = np.stack([xs - mx, ys - my], 1).astype(np.float32)
    cov = np.cov(pts.T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, 1]
    if w[1] < 1.35 * max(w[0], 1e-6):
        return None
    d = np.array([mx - bx, my - by])
    if np.dot(d, axis) < 0:
        axis = -axis
    return float(axis[0]), float(axis[1])


def classify_circle(roi_bgr, masks, box):
    """Phan loai va suy luan noi dung bien tron."""
    x, y, w, h = box
    sub = {k: v[y:y + h, x:x + w] for k, v in masks.items()}
    sub_bgr = roi_bgr[y:y + h, x:x + w]
    sub_hsv = cv2.cvtColor(sub_bgr, cv2.COLOR_BGR2HSV)
    H, W = sub['red'].shape
    cx, cy, r = W / 2.0, H / 2.0, min(W, H) / 2.0
    ring = disc_mask((H, W), cx, cy, r, 0.60 * r)
    inner = disc_mask((H, W), cx, cy, 0.60 * r)

    red_ring = frac(sub['red'], ring)
    blue_ring = frac(sub['blue'], ring)
    red_in = frac(sub['red'], inner)
    blue_in = frac(sub['blue'], inner)
    white_in = frac(sub['white'], inner)

    # Phan tich mau xanh duong cuc bo de bat duoc sac do xanh nhe
    local_blue = cv2.inRange(sub_hsv, (95, 60, 40), (132, 255, 255))
    local_blue_in = frac(local_blue, inner)

    # (a) Vien do + Nen xanh: Bien CAM DUNG XE VA DO XE (P.130) hoac CAM DO XE (P.131)
    if red_ring >= 0.20 and (blue_in >= 0.15 or local_blue_in >= 0.20):
        c1 = diag_band_coverage(sub['red'], cx, cy, r, +1)
        c2 = diag_band_coverage(sub['red'], cx, cy, r, -1)
        if min(c1, c2) >= 0.08 and min(c1, c2) >= 0.50 * max(c1, c2):
            return "Cam dung va do xe"
        return "Cam do xe"

    # (b) Nen do dac + Thanh trang ngang: CAM DI NGUOC CHIEU (P.102)
    if red_in >= 0.38 and red_ring >= 0.38:
        band = np.zeros((H, W), np.uint8)
        cv2.rectangle(band, (int(cx - 0.62 * r), int(cy - 0.22 * r)),
                       (int(cx + 0.62 * r), int(cy + 0.22 * r)), 255, -1)
        above = np.zeros((H, W), np.uint8)
        cv2.rectangle(above, (int(cx - 0.45 * r), int(cy - 0.65 * r)),
                       (int(cx + 0.45 * r), int(cy - 0.25 * r)), 255, -1)
        below = np.zeros((H, W), np.uint8)
        cv2.rectangle(below, (int(cx - 0.45 * r), int(cy + 0.25 * r)),
                       (int(cx + 0.45 * r), int(cy + 0.65 * r)), 255, -1)
        bar = frac(sub['white'], cv2.bitwise_and(band, inner))
        red_ab = frac(sub['red'], above)
        red_bl = frac(sub['red'], below)
        if bar >= 0.16 and red_ab >= 0.42 and red_bl >= 0.42:
            return "Cam di nguoc chieu"
        return None

    # (c) Vien do + Nen trang + Hinh den: Bien CAM RE (P.123) / BIEN CAM (P.101...)
    if red_ring >= 0.22 and white_in >= 0.18:
        ys, xs = np.nonzero(cv2.bitwise_and(sub['dark'], inner))
        if len(xs) > 0.03 * cv2.countNonZero(inner):
            top, bot = ys < cy - 0.12 * r, ys > cy + 0.12 * r
            if top.sum() > 3 and bot.sum() > 3:
                off = (xs[top].mean() - xs[bot].mean()) / r
                if off <= -0.08:
                    return "Cam re trai"
                if off >= 0.08:
                    return "Cam re phai"
        return "Bien cam"

    # (d) Nen xanh duong + Mui ten trang: BIEN HIEU LENH (R.301 / R.302)
    # Dat nguong blue_strong >= 0.30 de loai bo cua so chua (pagoda) va phan chieu
    if (blue_ring >= 0.35 or blue_in >= 0.35 or local_blue_in >= 0.40) and frac(sub['blue_strong'], ring) >= 0.30:
        v_chan = sub_hsv[:, :, 2]
        inner_mask = inner > 0
        v_vals = v_chan[inner_mask]

        if len(v_vals) > 20:
            ret_otsu, _ = cv2.threshold(v_vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            arrow = np.zeros((H, W), np.uint8)
            arrow[(v_chan > max(ret_otsu, v_vals.mean() + 15)) & inner_mask] = 255
            arrow_count = cv2.countNonZero(arrow)
            inner_count = cv2.countNonZero(inner)
            arrow_ratio = arrow_count / float(max(1, inner_count))

            if 0.04 <= arrow_ratio <= 0.48:
                d = arrow_direction(arrow)
                if d is not None:
                    dx, dy = d
                    ang = math.degrees(math.atan2(dy, dx))
                    if -150 <= ang <= -30:
                        return "Di thang"
                    if dx > 0.15:
                        return "Huong di ben phai"
                    if dx < -0.15:
                        return "Huong di ben trai"
                return "Bien hieu lenh"
            elif r >= 20 and (blue_ring >= 0.82 and blue_in >= 0.82):
                return "Bien hieu lenh"

    return None


def classify_triangle(roi_bgr, masks, box):
    """Phan loai va suy luan noi dung bien canh bao tam giac."""
    x, y, w, h = box
    sub = {k: v[y:y + h, x:x + w] for k, v in masks.items()}
    H, W = sub['red'].shape
    if H < 10 or W < 10:
        return "Bien canh bao"

    cx, cy = W // 2, H // 2
    rw, rh = int(W * 0.28), int(H * 0.28)
    center_dark = sub['dark'][max(0, cy - rh):min(H, cy + rh),
                              max(0, cx - rw):min(W, cx + rw)]

    cnts_dark, _ = cv2.findContours(center_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts_dark) == 0:
        return "Bien canh bao"

    largest = max(cnts_dark, key=cv2.contourArea)
    _, _, dw, dh = cv2.boundingRect(largest)
    if dw < 3 or dh < 3:
        return "Bien canh bao"

    d_ar = dw / float(dh)
    if d_ar < 0.70 and dh > rh * 0.40:
        return "Nguoi di bo"
    if d_ar > 1.30 and dw > rw * 0.50:
        return "Duong cong"

    return "Bien canh bao"


def classify_rect(roi_bgr, masks, box):
    """Phan loai va suy luan noi dung bien chi dan chu nhat."""
    x, y, w, h = box
    sub_white = masks['white'][y:y + h, x:x + w]
    H, W = sub_white.shape
    if H < 10 or W < 10:
        return "Bien chi dan"

    mx, my = max(2, int(W * 0.12)), max(2, int(H * 0.12))
    white_inner = np.zeros((H, W), np.uint8)
    white_inner[my:H - my, mx:W - mx] = sub_white[my:H - my, mx:W - mx]

    d = arrow_direction(white_inner)
    if d is not None:
        dx, dy = d
        ang = math.degrees(math.atan2(dy, dx))
        if -150 <= ang <= -30:
            return "Chi dan: Di thang"
        if dx > 0.25:
            return "Chi dan: Re phai"
        if dx < -0.25:
            return "Chi dan: Re trai"

    return "Bien chi dan"


# ==============================================================================
# PHAT HIEN + PHAN LOAI TREN MOT FRAME
# ==============================================================================
def detect_signs(frame):
    """Tra ve danh sach cac bien bao (x, y, w, h, nhan) tren frame goc."""
    H, W = frame.shape[:2]
    x0 = int(ROI_X_MIN * W)
    y0 = int(ROI_Y_MIN * H)
    y1 = int(ROI_Y_MAX * H)
    roi = frame[y0:y1, x0:W]
    masks = build_masks(roi)

    raw_circles = merge_boxes(find_circle_candidates(masks))
    raw_tris = merge_boxes(find_triangle_candidates(masks))
    raw_rects = merge_boxes(find_rect_candidates(masks))

    # BUOC QUAN TRONG: Phan loai tung ung vien TRUOC KHI loai bo trung lap!
    classified = []
    for b in raw_circles:
        lbl = classify_circle(roi, masks, b)
        if lbl is not None:
            classified.append((b, 'circle', lbl))
    for b in raw_tris:
        lbl = classify_triangle(roi, masks, b)
        if lbl is not None:
            classified.append((b, 'tri', lbl))
    for b in raw_rects:
        lbl = classify_rect(roi, masks, b)
        if lbl is not None:
            classified.append((b, 'rect', lbl))

    # Uu tien: tron > tam giac > chu nhat
    prio = {'circle': 0, 'tri': 1, 'rect': 2}
    classified.sort(key=lambda t: prio[t[1]])

    kept = []
    for b, kind, label in classified:
        bx, by, bw, bh = b
        dup = False
        for (kb, _, _) in kept:
            kx, ky, kw, kh = kb
            ix1, iy1 = max(bx, kx), max(by, ky)
            ix2, iy2 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / float(min(bw * bh, kw * kh)) > 0.45:
                    dup = True
                    break
        if not dup:
            kept.append((b, kind, label))

    results = []
    for (bx, by, bw, bh), _, label in kept:
        results.append((bx + x0, by + y0, bw, bh, label))
    return results


# ==============================================================================
# THEO DOI (TRACKING) + EMA SMOOTHING + BO PHIEU NHAN -> ON DINH
# ==============================================================================
class SignTracker:
    """
    Theo doi bien bao xuyen suot video de dat diem toi da theo Rubric:
    - Bounding box muot ma khong rung lac (EMA smoothing).
    - Duy tri track tu xa den gan (velocity prediction + center matching).
    - Bo phieu nhan theo thoi gian de nhan luon on dinh va nhat quan.
    - Loai bo nhieu chop tat (chi ve khi xuat hien du min_hits frame).
    """
    def __init__(self, max_missed=8, history=30, min_hits=3,
                 iou_thr=0.10, smooth_alpha=0.35):
        self.tracks = []
        self.max_missed = max_missed
        self.history = history
        self.min_hits = min_hits
        self.iou_thr = iou_thr
        self.smooth_alpha = smooth_alpha

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

    @staticmethod
    def _center_dist(a, b):
        acx, acy = a[0] + a[2] / 2.0, a[1] + a[3] / 2.0
        bcx, bcy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        avg_size = (a[2] + a[3] + b[2] + b[3]) / 4.0
        if avg_size < 1:
            return 999.0
        return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) / avg_size

    def update(self, dets):
        for t in self.tracks:
            t['matched'] = False

        pairs = []
        for di, (x, y, w, h, label) in enumerate(dets):
            det_box = (x, y, w, h)
            for ti, t in enumerate(self.tracks):
                if t['matched']:
                    continue
                ref_box = t['smooth_box']
                iou_val = self._iou(ref_box, det_box)
                cdist = self._center_dist(ref_box, det_box)
                score = iou_val + max(0, 1.0 - cdist) * 0.3
                if iou_val > self.iou_thr or cdist < 1.8:
                    pairs.append((score, di, ti))

        pairs.sort(key=lambda p: -p[0])
        used_det, used_trk = set(), set()
        for score, di, ti in pairs:
            if di in used_det or ti in used_trk:
                continue
            x, y, w, h, label = dets[di]
            t = self.tracks[ti]

            old_box = t['raw_box']
            t['vel'] = (x - old_box[0], y - old_box[1],
                        w - old_box[2], h - old_box[3])
            t['raw_box'] = (x, y, w, h)

            # EMA smoothing cho toa do bounding box
            a = self.smooth_alpha
            sx, sy, sw, sh = t['smooth_box']
            t['smooth_box'] = (
                sx + a * (x - sx),
                sy + a * (y - sy),
                sw + a * (w - sw),
                sh + a * (h - sh),
            )

            t['labels'].append(label)
            t['labels'] = t['labels'][-self.history:]
            t['hits'] += 1
            t['missed'] = 0
            t['matched'] = True
            used_det.add(di)
            used_trk.add(ti)

        # Khoi tao track moi
        for di, (x, y, w, h, label) in enumerate(dets):
            if di in used_det:
                continue
            self.tracks.append({
                'raw_box': (x, y, w, h),
                'smooth_box': (float(x), float(y), float(w), float(h)),
                'vel': (0, 0, 0, 0),
                'labels': [label],
                'hits': 1,
                'missed': 0,
                'matched': True,
            })

        # Du doan vi tri cho cac track bi mat frame
        for t in self.tracks:
            if not t['matched']:
                t['missed'] += 1
                if t['vel'] != (0, 0, 0, 0):
                    vx, vy, vw, vh = t['vel']
                    damping = 0.5
                    sx, sy, sw, sh = t['smooth_box']
                    t['smooth_box'] = (
                        sx + vx * damping,
                        sy + vy * damping,
                        sw + vw * damping * 0.2,
                        sh + vh * damping * 0.2,
                    )
                    t['vel'] = (vx * damping, vy * damping,
                                vw * damping, vh * damping)

        # Xoa track qua thoi gian quy dinh
        self.tracks = [t for t in self.tracks if t['missed'] <= self.max_missed]

        # Xuat cac track on dinh (min_hits >= 3 va khong mat qua 3 frame)
        out = []
        for t in self.tracks:
            if t['hits'] < self.min_hits:
                continue
            if t['missed'] > 3:
                continue
            labels = t['labels']
            label = max(set(labels), key=labels.count)
            sx, sy, sw, sh = t['smooth_box']
            out.append((int(round(sx)), int(round(sy)),
                        int(round(sw)), int(round(sh)), label))
        return out


# ==============================================================================
# VE KET QUA VA XUAT VIDEO
# ==============================================================================
def draw_annotations(frame, signs, student_ids_text):
    H, W = frame.shape[:2]
    for (x, y, w, h, label) in signs:
        thickness = 2 if max(w, h) < 50 else 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, thickness)

        scale = 0.55 if max(w, h) < 60 else 0.65
        th = 2
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        tx = int(min(max(0, x + w / 2 - tw / 2), W - tw - 2))
        ty = y - 8
        if ty - tht - 4 < 85:
            ty = y + h + tht + 10

        cv2.rectangle(frame, (tx - 3, ty - tht - 4), (tx + tw + 3, ty + base), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 255, 255), th, cv2.LINE_AA)

    # Chen watermark thong tin sinh vien len moi frame theo quy dinh
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
