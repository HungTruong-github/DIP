"""
TRUONG DAI HOC TON DUC THANG
KHOA CONG NGHE THONG TIN
BAI THI GIUA KY - NHAP MON XU LY ANH SO (MA MON: 505060)

NHOM SINH VIEN THUC HIEN:
1. Truong Duong Hung - MSSV: 52300028
2. Luong Minh Tan   - MSSV: 52300061
3. Le Quang Thanh   - MSSV: 52300064

YEU CAU DE BAI:
- Tu dong nhan dien va ve bounding box chu nhat bao quanh tat ca cac bien bao giao thong
  hinh tron (circular traffic signs) xuat hien o giua duong (dai phan cach) hoac ben phai duong.
- Chi su dung thu vien: OpenCV (cv2), math, Numpy (numpy).
- Khong su dung cac mo hinh hoc sau / pre-trained models.
- Dung chung mot ma nguon cho tat ca cac video dau vao.
- Video dau ra giu nguyen he mau, kich thuoc khung hinh va toc do khung hinh (FPS).
- Chen MSSV cua cac thanh vien vao tat ca cac khung hinh cua video dau ra.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import cv2
import numpy as np
import math
import os
import time

# ==============================================================================
# CAU HINH THONG TIN SINH VIEN VA DUONG DAN
# ==============================================================================
STUDENT_IDS = "MSSV: 52300028 - 52300061 - 52300064"
FILE_PREFIX = "52300028_52300061_52300064"

INPUT_VIDEOS = [
    ("video1.mp4", f"{FILE_PREFIX}_video1.mp4"),
    ("video2.mp4", f"{FILE_PREFIX}_video2.mp4")
]

# Cac moc frame mau cach nhau toi thieu 3 giay (>90 frames) phuc vu trich xuat anh bao cao
REPORT_SAMPLE_FRAMES = {
    "video1.mp4": [227, 540, 810, 1260, 1980],
    "video2.mp4": [60, 630, 1200, 1320, 1860]
}

# ==============================================================================
# HAM PHAN TICH VA NHAN DIEN BIEN BAO HINH TRON TREN TUNG FRAME
# ==============================================================================
def detect_circular_traffic_signs(frame):
    """
    Nhan dien cac bien bao giao thong hinh tron tren mot khung hinh video.
    Ap dung hoan toan cac ky thuat xu ly anh kinh dien:
    1. Chuyen doi khong gian mau BGR sang HSV.
    2. Phan doan mau sac (Mau Do voi 2 dai Hue va Mau Xanh Duong).
    3. Phep toan hinh thai hoc (Morphological Closing) de ket noi vien va lap day vung khuyet.
    4. Gioi han vung quan tam (ROI): Chi xet khu vuc giua va ben phai duong, do cao hop ly.
    5. Trich xuat duong bao (Contours) va phan tich bao loi (Convex Hull).
    6. Kiem tra do tron (Circularity) va ty le lap day duong tron ngoai tiep (Enclosing Circle Fit).
    7. Kiem tra phan bo mau tren vanh khuyen (Annulus Ring Color Verification).
    8. Non-Maximum Suppression (NMS) de loai bo cac box trung lap / long nhau.
    """
    h_img, w_img = frame.shape[:2]
    
    # Buoc 1: Chuyen doi sang HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Buoc 2: Phan doan mau sac (Color Segmentation)
    # Mau do trong HSV bi chia thanh 2 dai: can 0 va can 180
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([155, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    
    # Mau xanh duong cua bien hieu lenh
    lower_blue = np.array([100, 90, 50])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Buoc 3: Phep toan hinh thai hoc Morphological Closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_red_closed = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_blue_closed = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    candidate_boxes = []
    
    # Buoc 4 & 5: Duyet contours theo tung kenh mau
    for color_name, mask in [('RED', mask_red_closed), ('BLUE', mask_blue_closed)]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            bx, by, bw, bh = cv2.boundingRect(c)
            
            # Loc vi tri: Chi xet bien bao o giua duong hoac ben phai (x >= 0.35 * w_img)
            if bx < 0.35 * w_img:
                continue
                
            # Loc cao do: Bien bao tren cot/gia long mon, khong nam tren mat duong nhua
            max_allowed_y = 520 if (bw >= 50 or bh >= 50) else 440
            if by < 25 or by > max_allowed_y:
                continue
                
            # Loc kich thuoc co ban
            if bw < 18 or bh < 18 or bw > 450 or bh > 450:
                continue
                
            # Loc ty le khung bao (bien hinh tron thi w / h ~ 1.0)
            aspect_ratio = bw / float(bh)
            if not (0.70 <= aspect_ratio <= 1.40):
                continue
                
            # Tinh bao loi va duong tron ngoai tiep nho nhat
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            (cx, cy), radius = cv2.minEnclosingCircle(hull)
            circle_area = math.pi * (radius ** 2)
            if circle_area <= 0:
                continue
                
            fit_circle = hull_area / circle_area
            perimeter = cv2.arcLength(hull, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * math.pi * hull_area / (perimeter ** 2)
            
            # Nguong do tron (phu hop voi ca bien xa va gan)
            min_fit = 0.68 if radius < 20 else 0.75
            min_circ = 0.78 if radius < 20 else 0.84
            
            if fit_circle >= min_fit and circularity >= min_circ:
                # Buoc 6: Kiem tra vanh khuyen (Annulus Ring Color Verification)
                # Bien bao hinh tron co phan vien ngoai dac trung boi mau sac chu dao
                rx = int(radius)
                icx = int(cx)
                icy = int(cy)
                x_min = max(0, icx - rx)
                x_max = min(w_img, icx + rx + 1)
                y_min = max(0, icy - rx)
                y_max = min(h_img, icy + rx + 1)
                
                sub_mask = mask[y_min:y_max, x_min:x_max]
                ring = np.zeros(sub_mask.shape, dtype=np.uint8)
                cv2.circle(ring, (icx - x_min, icy - y_min), rx, 255, -1)
                cv2.circle(ring, (icx - x_min, icy - y_min), int(0.60 * rx), 0, -1)
                
                ring_pixels = cv2.countNonZero(ring)
                if ring_pixels > 0:
                    overlap = cv2.countNonZero(cv2.bitwise_and(sub_mask, ring))
                    color_ratio = overlap / float(ring_pixels)
                else:
                    color_ratio = 0.0
                    
                # Chap nhan ung vien neu ty le mau tren vien du cao
                if color_ratio >= 0.28:
                    candidate_boxes.append((bx, by, bw, bh, color_name, radius))
    
    # Buoc 7: Non-Maximum Suppression (NMS) gop cac bounding box trung lap
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
            
            # Tinh giao nhau IoU
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
                
                # Neu hai hop giao nhau nhieu (IoU > 0.25 hoac mot hop long trong hop kia)
                if iou > 0.25 or overlap_ratio > 0.50:
                    used[j] = True
                    x1 = min(x1, x2)
                    y1 = min(y1, y2)
                    w1 = max(x1 + w1, x2 + w2) - x1
                    h1 = max(y1 + h1, y2 + h2) - y1
                    
        merged_boxes.append((x1, y1, w1, h1))
        
    return merged_boxes

# ==============================================================================
# HAM VE BOUNDING BOX VA CHEN THONG TIN SINH VIEN LEN KHUNG HINH
# ==============================================================================
def draw_annotations(frame, bounding_boxes, student_ids_text):
    """
    Ve bounding box mau xanh la cay xung quanh cac bien bao hinh tron,
    va chen MSSV vao goc tren cung cua khung hinh.
    """
    # 1. Ve bounding box cho tung bien bao hinh tron
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # 2. Chen thong tin MSSV vao goc tren ben trai
    cv2.rectangle(frame, (25, 20), (840, 75), (0, 0, 0), -1)
    cv2.putText(
        frame,
        student_ids_text,
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return frame

# ==============================================================================
# HAM XU LY TOAN BO VIDEO VA XUAT RA TAP TIN VIDEO MOI
# ==============================================================================
def process_video_file(input_path, output_path, student_ids_text, sample_frame_indices=None):
    """
    Doc video dau vao, xu ly tung khung hinh, ve bounding box va MSSV,
    sau do luu thanh video MP4 voi cung kich thuoc, he mau va FPS.
    """
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 70)
    print(f"BAT DAU XU LY VIDEO: {input_path}")
    print(f"- Do phan giai: {width} x {height}")
    print(f"- Toc do khung hinh (FPS): {fps:.2f}")
    print(f"- Tong so khung hinh: {total_frames} (~{total_frames/fps:.1f}s)")
    print(f"- Tap tin xuat ra: {output_path}")
    print("=" * 70)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    os.makedirs("report_frames", exist_ok=True)
    
    frame_idx = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        boxes = detect_circular_traffic_signs(frame)
        annotated_frame = draw_annotations(frame, boxes, student_ids_text)
        
        if sample_frame_indices and frame_idx in sample_frame_indices:
            vid_base = os.path.splitext(os.path.basename(input_path))[0]
            sample_img_name = f"report_frames/{vid_base}_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(sample_img_name, annotated_frame)
            print(f"  -> Luu frame mau bao cao: {sample_img_name} (giay {frame_idx/fps:.1f}s)")
            
        out.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 300 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            percent = (frame_idx / total_frames) * 100.0
            print(f"  Tien do: {frame_idx}/{total_frames} frames ({percent:.1f}%) - Toc do: {proc_fps:.1f} FPS")
            
    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"[HOAN THANH] Da luu video ket qua vao: {output_path} ({total_time:.1f}s)\n")

# ==============================================================================
# HAM CHINH (MAIN ENTRY POINT)
# ==============================================================================
def main():
    print("=" * 70)
    print("CHUONG TRINH NHAN DIEN BIEN BAO GIAO THONG HINH TRON - BAI THI GIUA KY")
    print(f"Nhom sinh vien: {STUDENT_IDS}")
    print("=" * 70)
    
    for in_vid, out_vid in INPUT_VIDEOS:
        samples = REPORT_SAMPLE_FRAMES.get(in_vid, [])
        process_video_file(in_vid, out_vid, STUDENT_IDS, sample_frame_indices=samples)
        
    print("=" * 70)
    print("TAT CA CAC VIDEO DA DUOC XU LY VA XUAT THANH CONG!")
    print(f"1. {INPUT_VIDEOS[0][1]}")
    print(f"2. {INPUT_VIDEOS[1][1]}")
    print("Cac khung hinh mau phuc vu bao cao da duoc luu trong: report_frames/")
    print("=" * 70)

if __name__ == '__main__':
    main()
