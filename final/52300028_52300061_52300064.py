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
OUTPUT_DIR = "output"

INPUT_VIDEOS = [
    ("video1.mp4", os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_final_video1.mp4")),
    ("video2.mp4", os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}_final_video2.mp4")),
]

# Vung quan tam (ty le theo kich thuoc anh, KHONG phu thuoc video cu the):
# giua duong + ben phai (x >= 35% chieu rong), bo bau troi va mat duong sat dau xe.
ROI_X_MIN = 0.35
ROI_Y_MIN = 0.055
ROI_Y_MAX = 0.49

POSITION_RULES = False  # luat vi tri tuyet doi ke thua tu bai giua ky (xem bao cao)
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
        cv2.inRange(hsv, (0, 55, 45), (10, 255, 255)),
        cv2.inRange(hsv, (150, 55, 45), (180, 255, 255)))
    blue_raw = cv2.inRange(hsv, (100, 75, 50), (135, 255, 255))
    b_ch = roi_bgr[..., 0].astype(int)
    r_ch = roi_bgr[..., 2].astype(int)
    blue_dom = ((b_ch - r_ch >= 58) & (b_ch >= 115) & (hsv[..., 1] >= 110)).astype(np.uint8) * 255
    blue = cv2.bitwise_and(blue_raw, blue_dom)
    blue_strong = cv2.inRange(hsv, (100, 115, 80), (130, 255, 255))   # xanh du sang, loai bong toi
    yellow = cv2.inRange(hsv, (15, 90, 110), (35, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 110), (180, 70, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))
    bright = cv2.inRange(hsv, (0, 0, 120), (180, 255, 255))

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_c = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k7, iterations=1)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    blue_o = cv2.morphologyEx(blue, cv2.MORPH_OPEN, k5)
    blue_c = cv2.morphologyEx(blue_o, cv2.MORPH_CLOSE, k5)
    # Bien tam giac: vien do + nen vang -> gop lai thanh mot khoi dac
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    warn = cv2.bitwise_or(red, yellow)
    warn_c = cv2.morphologyEx(warn, cv2.MORPH_CLOSE, k, iterations=2)
    return {
        'red': red, 'blue': blue, 'yellow': yellow, 'white': white, 'dark': dark, 'bright': bright, 'blue_strong': blue_strong, 'sat': hsv[..., 1], 'val': hsv[..., 2],
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
def find_rect_candidates(masks, roi_bgr):
    cands = []
    mask = masks['blue_c']
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 40 or h < 25 or w > MAX_SIZE or h > MAX_SIZE:
            continue
        ar = w / float(h)
        if not (1.20 <= ar <= 2.80):
            continue
        area = cv2.contourArea(c)
        if area / float(w * h) < 0.70:   # hinh chu nhat lap day bbox
            continue
        circ, fit, _, _ = circle_geometry(c)
        if circ > 0.84 and fit > 0.75:    # la hinh tron -> khong phai
            continue
        # Co chu / hinh ve mau trang ben trong (S < 110 de bat duoc hoa van trang bi nhoe vao nen xanh)
        crop_hsv = hsv_roi[y:y + h, x:x + w]
        wr = np.sum((crop_hsv[..., 1] < 110) & (crop_hsv[..., 2] > 110)) / float(w * h)
        if wr < 0.04 or wr > 0.65:
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


def diag_contrast(val, cx, cy, r):
    """
    Do do TOI cua hai duong cheo so voi phan con lai trong dia: (V_ngoai - V_cheo) / V_ngoai.
    Vach cheo do/toi tren nen xanh -> duong; mui ten TRANG tren nen xanh -> am.
    """
    h, w = val.shape
    inner = disc_mask((h, w), cx, cy, 0.8 * r)
    v = val.astype(np.float64)
    res = []
    for sg in (1, -1):
        band = np.zeros((h, w), np.uint8)
        cv2.line(band, (int(cx - 0.8 * r), int(cy - sg * 0.8 * r)),
                 (int(cx + 0.8 * r), int(cy + sg * 0.8 * r)), 255, max(2, int(0.25 * r)))
        b = cv2.bitwise_and(band, inner) > 0
        o = cv2.bitwise_and(inner, cv2.bitwise_not(band)) > 0
        if b.sum() == 0 or o.sum() == 0 or v[o].mean() <= 0:
            res.append(0.0)
        else:
            res.append((v[o].mean() - v[b].mean()) / v[o].mean())
    return res[0], res[1]


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
        white_all = cv2.countNonZero(cv2.bitwise_and(sub['white'], inner))
        white_bar = cv2.countNonZero(cv2.bitwise_and(sub['white'], cv2.bitwise_and(band, inner)))
        concentrated = white_all == 0 or white_bar / float(white_all) >= 0.55   # trang gom thanh 1 thanh ngang
        if bar >= 0.25 and red_ab >= 0.6 and red_bl >= 0.6 and concentrated \
                and frac(sub['white'], inner) <= 0.35:
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
        # Vach cheo TOI tren nen xanh (P.130/P.131 khi vanh do bi nhoe) -> uu tien hon mui ten
        if blue_in >= 0.5:
            c1, c2 = diag_contrast(sub['val'], cx, cy, r)
            if c1 >= 0.08 and c2 >= 0.08:
                return "Cam dung xe va do xe"
            if max(c1, c2) >= 0.08 and min(c1, c2) <= 0.04:
                return "Cam do xe"
        # Hinh mui ten trang = pixel it bao hoa hon han nen xanh (thich nghi theo tung bien,
        # vi bien o xa bi nhoe nen mui ten chi con xanh nhat)
        sat_vals = sub['sat'][inner > 0]
        med_s = float(np.median(sat_vals)) if sat_vals.size else 0.0
        lowsat = np.where(sub['sat'] < 0.65 * med_s, 255, 0).astype(np.uint8)
        arrow = cv2.bitwise_and(lowsat, inner)
        if cv2.countNonZero(arrow) < 0.03 * cv2.countNonZero(inner):
            # Khong co mui ten sang. Bien CAM DO XE o xa: vanh do bi nhoe/toi nen khong bat duoc,
            # nhung van thay VACH CHEO TOI tren nen xanh -> nhan dien qua vach cheo.
            if blue_in >= 0.5:
                c1, c2 = diag_contrast(sub['val'], cx, cy, r)
                if c1 >= 0.08 and c2 >= 0.08:
                    return "Cam dung xe va do xe"
                if max(c1, c2) >= 0.08 and min(c1, c2) <= 0.04:
                    return "Cam do xe"
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
# ------------------------------------------------------------------------------
# BO PHAT HIEN BIEN TRON (KE THUA TU BAI GIUA KY - BAN TINH CHINH)
# ------------------------------------------------------------------------------
def detect_circle_boxes(frame):
    """Bo phat hien bien TRON (tu bai giua ky, da tinh chinh): tra ve danh sach (x, y, w, h)."""
    h_img, w_img = frame.shape[:2]
    sx, sy = w_img / 1920.0, h_img / 1080.0      # he so quy doi theo do phan giai (1920x1080 la chuan)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Buoc 1 & 2: Phan doan mau sac (Color Segmentation)
    lower_red1 = np.array([0, 55, 45])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([150, 55, 45])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    lower_blue = np.array([100, 75, 50])
    upper_blue = np.array([135, 255, 255])
    mask_blue_raw = cv2.inRange(hsv, lower_blue, upper_blue)
    b_ch = frame[..., 0].astype(int)
    r_ch = frame[..., 2].astype(int)
    blue_dom = ((b_ch - r_ch >= 58) & (b_ch >= 115) & (hsv[..., 1] >= 110)).astype(np.uint8) * 255
    mask_blue = cv2.bitwise_and(mask_blue_raw, blue_dom)
    
    # Buoc 3: Phep toan hinh thai hoc Morphological Closing / Opening
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_red_closed = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k7, iterations=1)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_blue_opened = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k5)
    mask_blue_closed = cv2.morphologyEx(mask_blue_opened, cv2.MORPH_CLOSE, k5)
    
    candidate_boxes = []
    
    # Buoc 4 & 5: Duyet contours va phan tich dac trung hinh hoc
    for color_name, mask in [('RED', mask_red_closed), ('BLUE', mask_blue_closed)]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            
            # Loc vi tri: Bien bao nam o dai phan cach giua (x >= 0.38*W) hoac le phai
            if bx < 0.38 * w_img or (bx + bw) > 1910 * sx:
                continue
                
            # Loc do cao: Bien bao gan tren cot hoac gia long mon
            if by < 65 * sy or (by + bh) > 505 * sy:
                continue
                
            # Loc tang lau cao oc phia ben phai (cac bien bao ben phai deu nam o by >= 220)
            if POSITION_RULES and bx > 1100 * sx and by < 220 * sy:
                continue
                
            # Loc mat duong le phai: bien bao that treo tren cot cao (by <= 405), khong nam sat via he (by > 410)
            if POSITION_RULES and bx > 1200 * sx and by > 410 * sy:
                continue
                
            # O sat mep phai x > 1350, bien bao that deu o rat gan xe (kich thuoc >= 30)
            # Loai bo tat ca cac cua so, ban cong, tan la nho o x > 1350
            if POSITION_RULES and bx > 1350 * sx and (bw < 30 * sx or bh < 30 * sy):
                continue
                
            # Loc kich thuoc vat the
            if bw < 18 * sx or bh < 18 * sy or bw > 95 * sx or bh > 95 * sy:
                continue
                
            # Loc ty le khung bao (bien tron co ty le w / h gan bang 1.0)
            aspect_ratio = bw / float(bh)
            if not (0.70 <= aspect_ratio <= 1.35):
                continue
                
            c_area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
                
            # Kiem tra do dac (Solidity S / S_hull >= 0.70) de loai bo tan cay va cua so nan go
            solidity = c_area / float(hull_area)
            if solidity < 0.70:
                continue
                
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            circle_area = math.pi * (radius ** 2)
            if circle_area <= 0:
                continue
                
            fit_circle = hull_area / circle_area
            perimeter = cv2.arcLength(hull, True)
            if perimeter <= 0:
                continue
            hull_circ = 4.0 * math.pi * hull_area / (perimeter ** 2)
            
            # Loai bo cac khoi chu nhat dac
            extent = c_area / float(bw * bh)
            if extent > 0.85:
                continue
                
            # Kiem tra do tron thuc su
            if not (fit_circle >= 0.58 and hull_circ >= 0.82):
                continue
                
            # Kiem tra do chiem cho 4 goc (Corner Occupancy Test):
            # Hinh tron noi tiep hinh chu nhat co 4 goc hoan toan trong rong.
            # Cac bien bao hinh vuong / chu nhat (nhu bien chi dan, bang hieu) va cua so se bi loai bo triet de.
            poly = np.zeros((bh, bw), dtype=np.uint8)
            cv2.drawContours(poly, [c - [bx, by]], -1, 255, -1)
            cw_corner = max(2, int(0.18 * bw))
            ch_corner = max(2, int(0.18 * bh))
            tl = np.mean(poly[:ch_corner, :cw_corner] > 0)
            tr = np.mean(poly[:ch_corner, -cw_corner:] > 0)
            bl = np.mean(poly[-ch_corner:, :cw_corner] > 0)
            br = np.mean(poly[-ch_corner:, -cw_corner:] > 0)
            corner_mean = (tl + tr + bl + br) / 4.0
            corner_max = max(tl, tr, bl, br)
            if corner_mean > 0.22 or corner_max > 0.65:
                continue
                
            rx = int(radius)
            icx = int(cx)
            icy = int(cy)
            x_min = max(0, icx - rx)
            x_max = min(w_img, icx + rx + 1)
            y_min = max(0, icy - rx)
            y_max = min(h_img, icy + rx + 1)
            
            sub_mask = mask[y_min:y_max, x_min:x_max]
            
            # 6. Kiem tra vanh khuyen (Annulus Ring Color Verification)
            ring = np.zeros(sub_mask.shape, dtype=np.uint8)
            cv2.circle(ring, (icx - x_min, icy - y_min), rx, 255, -1)
            cv2.circle(ring, (icx - x_min, icy - y_min), int(0.60 * rx), 0, -1)
            
            ring_pixels = cv2.countNonZero(ring)
            if ring_pixels == 0:
                continue
            ring_color_ratio = cv2.countNonZero(cv2.bitwise_and(sub_mask, ring)) / float(ring_pixels)
            if ring_color_ratio < 0.22:
                continue
                
            if color_name == 'BLUE':
                r_exp = int(1.35 * radius)
                x_min_e, x_max_e = max(0, icx - r_exp), min(w_img, icx + r_exp + 1)
                y_min_e, y_max_e = max(0, icy - r_exp), min(h_img, icy + r_exp + 1)
                sub_red = mask_red[y_min_e:y_max_e, x_min_e:x_max_e]
                ring_outer = np.zeros(sub_red.shape, dtype=np.uint8)
                cv2.circle(ring_outer, (icx - x_min_e, icy - y_min_e), r_exp, 255, -1)
                cv2.circle(ring_outer, (icx - x_min_e, icy - y_min_e), int(0.85 * radius), 0, -1)
                outer_pts = cv2.countNonZero(ring_outer)
                red_ratio_outer = cv2.countNonZero(cv2.bitwise_and(sub_red, ring_outer)) / float(outer_pts) if outer_pts > 0 else 0
                if red_ratio_outer >= 0.15:
                    bx = max(0, icx - r_exp)
                    by = max(0, icy - r_exp)
                    bw = min(w_img - bx, 2 * r_exp)
                    bh = min(h_img - by, 2 * r_exp)
                    color_name = 'RED'
            
            # 7. Phan biet noi dung mau sac ben trong (Internal Content Discrimination)
            crop_bgr = frame[by:by+bh, bx:bx+bw]
            crop_hsv = hsv[by:by+bh, bx:bx+bw]
            
            # Kiem tra do bao hoa mau (Saturation)
            max_s = np.max(crop_hsv[:, :, 1])
            mean_s = np.mean(crop_hsv[:, :, 1])
            
            if color_name == 'BLUE':
                if max_s < 155 or mean_s < 90:
                    continue
                # Do vuot troi cua mau xanh (Blue Dominance B > R) loai bo cua kinh, tuong bong va bien quang cao toi mau
                bmask_crop = mask_blue_closed[by:by+bh, bx:bx+bw]
                b_pix = crop_bgr[bmask_crop > 0]
                if len(b_pix) == 0:
                    continue
                mean_b = np.mean(b_pix[:, 0])
                mean_r = np.mean(b_pix[:, 2])
                if mean_b < 105 or (mean_b - mean_r) < 45:
                    continue
            if color_name == 'RED':
                if max_s < 110:
                    continue
            
            crop_r1 = cv2.inRange(crop_hsv, lower_red1, upper_red1)
            crop_r2 = cv2.inRange(crop_hsv, lower_red2, upper_red2)
            crop_rmask = cv2.bitwise_or(crop_r1, crop_r2)
            crop_bmask = cv2.inRange(crop_hsv, lower_blue, upper_blue)
            crop_white = ((crop_hsv[:, :, 1] < 70) & (crop_hsv[:, :, 2] > 140)).astype(np.uint8) * 255
            crop_arrow = ((crop_hsv[:, :, 1] < 140) & (crop_hsv[:, :, 2] > 165)).astype(np.uint8) * 255
            
            # Kiem tra vung loi ben trong duong tron (Inner Core Disc r <= 0.55 * radius)
            # Giup loai bo hoan toan cac mang do don sac nhu la co, bang ron, den long va den tin hieu
            ch_c, cw_c = crop_hsv.shape[:2]
            Y, X = np.ogrid[:ch_c, :cw_c]
            r_inner = max(2, int(0.55 * radius))
            inner_mask = ((X - (icx - bx))**2 + (Y - (icy - by))**2) <= (r_inner ** 2)
            in_pts = float(np.sum(inner_mask))
            if in_pts <= 0:
                continue
                
            r_red_core = np.sum((crop_rmask > 0) & inner_mask) / in_pts
            r_blue_core = np.sum((crop_bmask > 0) & inner_mask) / in_pts
            r_white_core = np.sum((crop_white > 0) & inner_mask) / in_pts
            r_arrow_core = np.sum((crop_arrow > 0) & inner_mask) / in_pts
            
            if color_name == 'RED':
                # Bien do that luon co ruot trang (P.101, P.102, P.123b, P.127) hoac ruot xanh (P.130, P.131)
                # Loai bo la co, den giao thong, den long (ruot do dac khong co ca trang lan xanh)
                if r_red_core > 0.80:
                    continue
                if r_white_core < 0.05 and r_blue_core < 0.08:
                    continue
            else: # BLUE
                # Tat ca cac bien xanh that (R.302b hoac P.130/P.131) deu phai co mui ten trang hoac gach cheo do
                tot_crop = float(bw * bh)
                r_red = cv2.countNonZero(crop_rmask) / tot_crop
                is_p130_p131 = (r_red >= 0.035 or r_red_core >= 0.035)
                
                tot_w = cv2.countNonZero(crop_arrow)
                if tot_w > 0:
                    q_cx, q_cy = bw // 2, bh // 2
                    q_tl = cv2.countNonZero(crop_arrow[:q_cy, :q_cx])
                    q_tr = cv2.countNonZero(crop_arrow[:q_cy, q_cx:])
                    q_bl = cv2.countNonZero(crop_arrow[q_cy:, :q_cx])
                    q_br = cv2.countNonZero(crop_arrow[q_cy:, q_cx:])
                    is_r302b = (tot_w >= 0.03 * tot_crop) and (q_tl >= 0.12 * tot_w) and (q_br >= 0.18 * tot_w) and ((q_tl + q_br) >= (q_tr + q_bl)) and ((q_tr + q_br) >= 0.30 * tot_w)
                else:
                    is_r302b = False
                    
                is_blue_valid = is_p130_p131 or is_r302b or (tot_w >= 0.025 * tot_crop) or (r_white_core >= 0.04)
                if not is_blue_valid:
                    continue
            
            candidate_boxes.append((bx, by, bw, bh, color_name, radius))
    
    # Buoc 8: Non-Maximum Suppression (NMS) gop cac bounding box trung lap / long nhau
    merged_boxes = []
    used = [False] * len(candidate_boxes)
    for i in range(len(candidate_boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1, c1, r1 = candidate_boxes[i]
        for j in range(i + 1, len(candidate_boxes)):
            if used[j]:
                continue
            x2, y2, w2, h2, c2, r2 = candidate_boxes[j]
            
            # Tinh giao nhau IoU va overlap
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                union_area = (w1 * h1) + (w2 * h2) - inter_area
                iou = inter_area / float(union_area) if union_area > 0 else 0
                min_box_area = min(w1 * h1, w2 * h2)
                overlap_ratio = inter_area / float(min_box_area) if min_box_area > 0 else 0
                
                if iou > 0.20 or overlap_ratio > 0.45:
                    used[j] = True
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(x1 + w1, x2 + w2) - x1
                    h1 = max(y1 + h1, y2 + h2) - y1
                    
        merged_boxes.append((x1, y1, w1, h1))
        
    return merged_boxes


def detect_signs(frame):
    """Tra ve danh sach (x, y, w, h, nhan) toa do tren anh goc."""
    H, W = frame.shape[:2]
    x0, y0, y1 = int(ROI_X_MIN * W), int(ROI_Y_MIN * H), int(ROI_Y_MAX * H)
    roi = frame[y0:y1, x0:W]
    masks = build_masks(roi)

    # Bien TRON: dung bo loc da tinh chinh tu bai giua ky (toa do tren anh goc -> doi sang toa do ROI)
    circles = [(bx - x0, by - y0, bw, bh) for (bx, by, bw, bh) in detect_circle_boxes(frame)]
    tris = [b for b in merge_boxes(find_triangle_candidates(masks)) if b[0] + x0 >= 0.38 * W]
    rects = [b for b in merge_boxes(find_rect_candidates(masks, roi)) if b[0] + x0 >= 0.38 * W and (b[1] + y0 + b[3]) <= 460 * (H / 1080.0)]

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
        if label is None:          # hinh dang giong bien nhung noi dung khong hop le -> bo (loai bo ang ten, tru dien, poster)
            continue
        if kind == 'circle' and label in ('Cam di nguoc chieu', 'Bien cam', 'Cam re trai', 'Cam re phai'):
            # Vanh NGOAI bien: neu xung quanh la mang xanh lon (nam tren bang xanh) hoac
            # mang do lon (mai ngoi, bien quang cao do) thi day khong phai bien tron doc lap
            ccx, ccy, rr = bx + bw / 2.0, by + bh / 2.0, min(bw, bh) / 2.0
            outer = disc_mask(masks['red'].shape, ccx, ccy, 1.6 * rr, 1.2 * rr)
            if frac(masks['blue'], outer) > 0.5 or frac(masks['red'], outer) > 0.4:
                continue
        results.append((bx + x0, by + y0, bw, bh, label))
    return results


# ==============================================================================
# THEO DOI NGOAI TUYEN (OFFLINE TRACKING) - CHONG NHAP NHAY, LOC BAO DONG GIA
#   Luot 1: phat hien tung frame -> luu danh sach hop.
#   Giua   : lien ket hop thanh TRACK (moi bien = 1 track), noi cac track bi dut,
#            loc theo do ben vung, noi suy cho frame bi hut, lam muot toa do,
#            bo phieu nhan DUY NHAT cho ca track.
#   Luot 2: ve cac track da duoc xac nhan len video.
# ==============================================================================
GENERIC_LABELS = ('Bien hieu lenh', 'Bien cam', 'Bien tron')


def label_family(label):
    if label == 'Bien canh bao':
        return 'tri'
    if label == 'Bien chi dan':
        return 'rect'
    if label.startswith('Huong') or label in ('Di thang', 'Bien hieu lenh'):
        return 'blue'
    return 'red'


def min_track_frames(label):
    """So frame toi thieu de mot track duoc xem la bien that (nhan mo ho can bang chung dai hon)."""
    if label == 'Bien cam':
        return 60
    if label in ('Bien chi dan', 'Bien hieu lenh') or label.startswith('Huong') or label == 'Di thang':
        return 40 if label != 'Bien hieu lenh' and label != 'Bien chi dan' else 30
    return 15


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / float(aw * ah + bw * bh - inter)


def box_contain(small, big):
    """Ty le dien tich cua hop 'small' nam trong hop 'big'."""
    sx, sy, sw, sh = small
    bx, by, bw, bh = big
    ix1, iy1 = max(sx, bx), max(sy, by)
    ix2, iy2 = min(sx + sw, bx + bw), min(sy + sh, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / float(sw * sh)


def link_detections(dets, max_gap=15):
    """Lien ket cac phat hien theo thoi gian thanh track (cung ho nhan, vi tri/kich thuoc gan nhau)."""
    tracks, active = [], []
    for fi, frame_dets in enumerate(dets):
        used = set()
        for i in sorted(range(len(frame_dets)), key=lambda k: -frame_dets[k][2] * frame_dets[k][3]):
            x, y, w, h, label = frame_dets[i]
            fam = label_family(label)
            best, best_sc = None, 0.0
            for t in active:
                if id(t) in used or t['fam'] != fam:
                    continue
                bx, by, bw, bh = t['box']
                dist = math.hypot((x + w / 2) - (bx + bw / 2), (y + h / 2) - (by + bh / 2))
                ratio = (w * h) / float(bw * bh)
                iou = box_iou((x, y, w, h), t['box'])
                if (iou > 0.1 or dist < 0.6 * max(w, h, bw, bh)) and 0.5 <= ratio <= 2.0:
                    sc = iou + 1.0 / (1.0 + dist)
                    if sc > best_sc:
                        best, best_sc = t, sc
            if best is None:
                best = {'fam': fam, 'frames': {}, 'box': (x, y, w, h), 'last': fi}
                tracks.append(best)
                active.append(best)
            best['frames'][fi] = ((x, y, w, h), label)
            best['box'], best['last'] = (x, y, w, h), fi
            used.add(id(best))
        active = [t for t in active if fi - t['last'] <= max_gap]
    return tracks


def merge_tracks(tracks, max_gap=45):
    """Noi hai track cua cung mot bien bi dut quang (bi che, nhoe) neu vi tri/kich thuoc khop."""
    tracks = sorted(tracks, key=lambda t: min(t['frames']))
    merged = True
    while merged:
        merged = False
        for a in tracks:
            ea = max(a['frames'])
            ba = a['frames'][ea][0]
            for b in tracks:
                if b is a or b['fam'] != a['fam']:
                    continue
                sb = min(b['frames'])
                if not (0 < sb - ea <= max_gap):
                    continue
                bb = b['frames'][sb][0]
                dist = math.hypot((ba[0] + ba[2] / 2) - (bb[0] + bb[2] / 2), (ba[1] + ba[3] / 2) - (bb[1] + bb[3] / 2))
                ratio = (bb[2] * bb[3]) / float(ba[2] * ba[3])
                if dist < 1.2 * max(ba[2], ba[3], bb[2], bb[3]) and 0.5 <= ratio <= 2.0:
                    a['frames'].update(b['frames'])
                    tracks.remove(b)
                    merged = True
                    break
            if merged:
                break
    return tracks


def vote_label(track):
    """Mot nhan duy nhat cho ca track: bo phieu co trong so theo dien tich (box lon -> tin cay hon)."""
    score = {}
    for (x, y, w, h), label in track['frames'].values():
        score[label] = score.get(label, 0.0) + w * h
    specific = {l: v for l, v in score.items() if l not in GENERIC_LABELS}
    pool = specific if specific else score
    return max(pool, key=pool.get)


def finalize_tracks(tracks, n_frames):
    """Loc track yeu, noi suy + lam muot, tra ve: per_frame[fi] = [(x, y, w, h, label)]."""
    kept = []
    for t in tracks:
        fs = sorted(t['frames'])
        t['label'] = vote_label(t)
        span = fs[-1] - fs[0] + 1
        min_f = min_track_frames(t['label'])
        if fs[0] <= 5 or fs[-1] >= n_frames - 10:
            min_f = min(min_f, 15)
        if len(fs) < min_f or len(fs) / float(span) < 0.4:
            continue
        # Noi suy tuyen tinh cac frame bi hut
        full = np.arange(fs[0], fs[-1] + 1)
        arr = np.array([t['frames'][f][0] for f in fs], np.float64)
        interp = np.stack([np.interp(full, fs, arr[:, k]) for k in range(4)], 1)
        # Lam muot bang trung binh truot (cua so le, bien lap) -> khung khong rung
        win = 9
        if len(full) >= win:
            pad = win // 2
            padded = np.pad(interp, ((pad, pad), (0, 0)), mode='edge')
            ker = np.ones(win) / win
            interp = np.stack([np.convolve(padded[:, k], ker, mode='valid') for k in range(4)], 1)
        t['full'] = full
        t['boxes'] = interp
        t['n'] = len(fs)
        kept.append(t)

    # Bo track nam gon trong bien lon khac (vd: chi tiet do ben trong bien chi dan xanh)
    final = []
    for t in kept:
        inside = 0
        both = 0
        for f, bx in zip(t['full'], t['boxes']):
            for o in kept:
                if o is t or o['n'] <= t['n'] or f < o['full'][0] or f > o['full'][-1]:
                    continue
                ob = o['boxes'][f - o['full'][0]]
                if o['fam'] == 'rect' or (ob[2] * ob[3]) > 2.0 * (bx[2] * bx[3]):
                    both += 1
                    if box_contain(tuple(bx), tuple(ob)) > 0.8:
                        inside += 1
                    break
        if both and inside / float(both) > 0.5:
            continue
        final.append(t)

    per_frame = [[] for _ in range(n_frames)]
    for t in final:
        for f, bx in zip(t['full'], t['boxes']):
            if f < n_frames:
                per_frame[f].append((int(round(bx[0])), int(round(bx[1])), int(round(bx[2])),
                                     int(round(bx[3])), t['label'], t['n']))
    # NMS theo tung frame giua cac track (khong de 2 hop chong len cung 1 bien)
    for f in range(n_frames):
        items = sorted(per_frame[f], key=lambda it: -it[5])
        keep = []
        for it in items:
            b = it[:4]
            if any(box_iou(b, k[:4]) > 0.3 or box_contain(b, k[:4]) > 0.6 or box_contain(k[:4], b) > 0.6 for k in keep):
                continue
            keep.append(it)
        per_frame[f] = [(x, y, w, h, lab) for (x, y, w, h, lab, _) in keep]
    return per_frame


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

        def hits_other(ty_):
            # nhan (hinh chu nhat) co de len bien khac khong?
            lx1, ly1, lx2, ly2 = tx - 3, ty_ - tht - 4, tx + tw + 3, ty_ + base
            for (ox, oy, ow, oh, _) in signs:
                if (ox, oy, ow, oh) == (x, y, w, h):
                    continue
                if lx1 < ox + ow and lx2 > ox and ly1 < oy + oh and ly2 > oy:
                    return True
            return False

        ty_above = y - 8
        ty_below = y + h + tht + 10
        if ty_above - tht - 4 >= 85 and not hits_other(ty_above):
            ty = ty_above                      # uu tien phia tren bien
        elif not hits_other(ty_below):
            ty = ty_below                      # vuong bien khac / sat mep -> phia duoi
        else:
            ty = ty_above if ty_above - tht - 4 >= 85 else ty_below
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

    # ---- LUOT 1: phat hien tren tung frame ----
    dets, t0 = [], time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets.append(detect_signs(frame))
        if len(dets) % 300 == 0:
            print(f"  [Luot 1] Phat hien: {len(dets)}/{total} frames")
    cap.release()
    n_frames = len(dets)

    # ---- THEO DOI + LOC + LAM MUOT ----
    tracks = merge_tracks(link_detections(dets))
    per_frame = finalize_tracks(tracks, n_frames)
    print(f"  Theo doi: {len(tracks)} track tho, {sum(1 for fr in per_frame if fr)} frame co bien sau khi loc")

    # ---- LUOT 2: ve ket qua ----
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        signs = per_frame[idx] if idx < n_frames else []
        out.write(draw_annotations(frame, signs, student_ids_text))
        idx += 1
        if idx % 300 == 0 or idx == n_frames:
            print(f"  [Luot 2] Ve ket qua: {idx}/{n_frames} frames")
    cap.release()
    out.release()
    print(f"[HOAN THANH] {output_path} ({time.time() - t0:.1f}s)\n")


def main():
    print("=" * 70)
    print("NHAN DIEN VA SUY LUAN NOI DUNG BIEN BAO GIAO THONG - BAI THI CUOI KY")
    print(f"Nhom sinh vien: {STUDENT_IDS}")
    print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for src, dst in INPUT_VIDEOS:
        process_video_file(src, dst, STUDENT_IDS)
    print("HOAN TAT.")


if __name__ == '__main__':
    main()