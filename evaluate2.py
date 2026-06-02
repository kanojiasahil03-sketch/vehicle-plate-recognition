from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import easyocr
import cv2
import re
import os
import random
import time
import numpy as np

plate_model = YOLO('/home/sahil/anpr_project/models/plate_detector_v2/weights/best.pt')
fast_ocr    = LicensePlateRecognizer('european-plates-mobile-vit-v2-model')
easy_reader = easyocr.Reader(['en'], gpu=False, verbose=False)

timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir  = f"/home/sahil/anpr_project/results/run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"\n📁 Saving results to: {save_dir}")

def clean_plate(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned[:8] if len(cleaned) > 8 else cleaned

def is_registration(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(cleaned) < 4 or len(cleaned) > 8:
        return False, 0
    has_digit  = any(c.isdigit() for c in cleaned)
    has_letter = any(c.isalpha() for c in cleaned)
    if not has_digit or not has_letter:
        return False, 0
    digit_ratio = sum(c.isdigit() for c in cleaned) / len(cleaned)
    return True, digit_ratio + 0.5

def fast_read(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = fast_ocr.run(gray)
        if result:
            plate = result[0].plate.upper()
            cleaned = clean_plate(plate)
            if len(cleaned) >= 4:
                return cleaned, "fast-plate-ocr"
        return None, None
    except Exception as e:
        return None, None

def easy_read(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        best_result = None
        best_len = 0
        for variant in [crop, gray, otsu, cv2.bitwise_not(otsu)]:
            result = easy_reader.readtext(variant)
            for (bbox, text, conf) in result:
                cleaned = clean_plate(text)
                valid, _ = is_registration(cleaned)
                if valid and len(cleaned) > best_len:
                    best_result = cleaned
                    best_len = len(cleaned)
        return best_result, "EasyOCR"
    except:
        return None, None

def perspective_correct(crop):
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < (crop.shape[0] * crop.shape[1] * 0.1):
            return crop
        rect  = cv2.minAreaRect(largest)
        box   = cv2.boxPoints(rect).astype(int)
        w, h  = int(rect[1][0]), int(rect[1][1])
        if w == 0 or h == 0:
            return crop
        if h > w:
            w, h = h, w
        target_w = max(w, 400)
        target_h = int(target_w / 4.5)
        src = np.float32(sorted(box.tolist(), key=lambda x: (x[1], x[0])))
        if src[0][0] > src[1][0]: src[[0,1]] = src[[1,0]]
        if src[2][0] < src[3][0]: src[[2,3]] = src[[3,2]]
        dst = np.float32([[0,0],[target_w,0],[target_w,target_h],[0,target_h]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(crop, M, (target_w, target_h))
    except:
        return crop

def preprocess_crop(crop):
    h, w = crop.shape[:2]
    if w < 600:
        scale = 600 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return crop

def process_and_save(image_path, index):
    img = cv2.imread(image_path)
    if img is None:
        return 0, "LOAD_ERROR", "—"

    results = plate_model(img, verbose=False)
    boxes   = results[0].boxes

    if len(boxes) == 0:
        out = img.copy()
        cv2.putText(out, "NO PLATE DETECTED", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite(f"{save_dir}/{index:02d}_NO_PLATE.jpg", out)
        return 0, "NO_PLATE", "—"

    best = max(boxes, key=lambda b: b.conf)
    conf = float(best.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    pad = 5
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(img.shape[1], x2+pad); y2p = min(img.shape[0], y2+pad)
    crop = img[y1p:y2p, x1p:x2p]
    crop = perspective_correct(crop)
    crop = preprocess_crop(crop)

    # Save crop
    cv2.imwrite(f"{save_dir}/{index:02d}_crop.jpg", crop)

    # OCR
    plate, engine = fast_read(crop)
    if not plate:
        plate, engine = easy_read(crop)
    if not plate:
        plate, engine = "UNREAD", "—"

    # Save annotated result
    # Save annotated result
    color     = img.copy()
    box_color = (0, 255, 0) if plate != "UNREAD" else (0, 0, 255)
    cv2.rectangle(color, (x1, y1), (x2, y2), box_color, 3)

    # Smart font size based on plate width
    plate_w    = max(x2 - x1, 50)
    font_scale = min(0.9, plate_w / 150)
    font_scale = max(0.5, font_scale)
    thickness  = 2

    label = f"{plate} [{engine}]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)
    tx = max(0, x1)
    ty = max(th + 8, y1 - 8)

    # Black background box
    cv2.rectangle(color, (tx - 2, ty - th - 4),
                  (tx + tw + 4, ty + 4), (0, 0, 0), -1)
    cv2.putText(color, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, thickness)

    cv2.imwrite(f"{save_dir}/{index:02d}_result_{plate}.jpg", color)
    return conf, plate, engine

# ── Run ───────────────────────────────────────────────────────────────────────
test_dir   = '/home/sahil/anpr_project/Number-Plates-1/train/images/'
all_images = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
random.shuffle(all_images)
images = all_images[:20]
total  = len(images)

print(f"\n{'#':<4} {'Filename':<25} {'Conf':<7} {'Engine':<16} {'OCR Output'}")
print("─" * 72)

ocr_success = 0
no_plate    = 0
unread      = 0

for i, fname in enumerate(images):
    path = os.path.join(test_dir, fname)
    conf, plate, engine = process_and_save(path, i+1)
    short = fname[:23] + ".."

    if plate == "NO_PLATE":    no_plate += 1
    elif plate == "UNREAD":    unread += 1
    elif plate == "LOAD_ERROR": pass
    else:                      ocr_success += 1

    print(f"{i+1:<4} {short:<25} {conf:<7.2f} {engine:<16} {plate}")

print("─" * 72)
print(f"\n📊 FINAL EVALUATION SUMMARY")
print(f"Total tested          : {total}")
print(f"OCR success           : {ocr_success}/{total} ({round(ocr_success/total*100)}%)")
print(f"No plate detected     : {no_plate}/{total} ({round(no_plate/total*100)}%)")
print(f"Unread                : {unread}/{total} ({round(unread/total*100)}%)")
print(f"\n✅ Success Rate : {round(ocr_success/total*100)}%")
print(f"\n📁 Results saved to : {save_dir}")