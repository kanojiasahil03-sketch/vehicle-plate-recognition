from ultralytics import YOLO
import easyocr
import cv2
import re
import os
import random
import time
import numpy as np

# ── Experiment 1 Models ───────────────────────────────────────────────────────
plate_model = YOLO('/home/sahil/anpr_project/models/plate_detector-3/weights/best.pt')
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

timestamp = time.strftime("%Y%m%d_%H%M%S")
save_dir  = f"/home/sahil/anpr_project/results/exp1_run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"\n📁 Saving Experiment 1 results to: {save_dir}")

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

def _preprocess_variants(crop):
    variants = [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants.append(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    variants.append(denoised)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    variants.append(clahe.apply(gray))
    adapt = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    variants.append(adapt)
    b, g, r = cv2.split(crop)
    red = cv2.subtract(r, g)
    _, red_bin = cv2.threshold(red, 30, 255, cv2.THRESH_BINARY)
    variants.append(cv2.bitwise_not(red_bin))
    return variants

def extract_plate_text(ocr_results):
    candidates = []
    for (bbox, text, conf) in ocr_results:
        cleaned = clean_plate(text)
        valid, score = is_registration(cleaned)
        if valid:
            candidates.append((cleaned, conf * score))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

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
        return 0, "LOAD_ERROR"

    results = plate_model(img, verbose=False)
    boxes   = results[0].boxes

    if len(boxes) == 0:
        out = img.copy()
        cv2.putText(out, "NO PLATE", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imwrite(f"{save_dir}/{index:02d}_NO_PLATE.jpg", out)
        return 0, "NO_PLATE"

    best = max(boxes, key=lambda b: b.conf)
    conf = float(best.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])

    pad = 5
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(img.shape[1], x2+pad); y2p = min(img.shape[0], y2+pad)
    crop = img[y1p:y2p, x1p:x2p]
    crop = perspective_correct(crop)
    crop = preprocess_crop(crop)
    cv2.imwrite(f"{save_dir}/{index:02d}_crop.jpg", crop)

    # EasyOCR with 8 variants
    best_result = None
    best_len = 0
    for variant in _preprocess_variants(crop):
        ocr_results = reader.readtext(variant)
        candidate = extract_plate_text(ocr_results)
        if candidate and len(candidate) > best_len:
            best_result = candidate
            best_len = len(candidate)

    plate = best_result if best_result else "UNREAD"

    # Save annotated image
    color = img.copy()
    box_color = (0,255,0) if plate != "UNREAD" else (0,0,255)
    cv2.rectangle(color, (x1,y1), (x2,y2), box_color, 3)
    plate_w = max(x2-x1, 50)
    font_scale = min(0.9, plate_w/150)
    font_scale = max(0.5, font_scale)
    label = f"{plate} [EasyOCR]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 2)
    tx = max(0, x1)
    ty = max(th+8, y1-8)
    cv2.rectangle(color, (tx-2, ty-th-4), (tx+tw+4, ty+4), (0,0,0), -1)
    cv2.putText(color, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, 2)
    cv2.imwrite(f"{save_dir}/{index:02d}_result_{plate}.jpg", color)

    return conf, plate

# ── Run Experiment 1 ──────────────────────────────────────────────────────────
test_dir = '/home/sahil/anpr_project/Number-Plates-1/train/images/'
all_images = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
random.seed(42)
random.shuffle(all_images)
images = all_images[:20]
total  = len(images)

print(f"\n{'#':<4} {'Filename':<25} {'Conf':<7} {'OCR Output'}")
print("─" * 55)

detected = 0
ocr_success = 0
no_plate = 0
unread = 0

for i, fname in enumerate(images):
    path = os.path.join(test_dir, fname)
    conf, plate = process_and_save(path, i+1)
    short = fname[:23] + ".."

    if plate == "NO_PLATE":    no_plate += 1
    elif plate == "UNREAD":    unread += 1
    elif plate == "LOAD_ERROR": pass
    else:                      detected += 1; ocr_success += 1

    print(f"{i+1:<4} {short:<25} {conf:<7.2f} {plate}")

print("─" * 55)
print(f"\n📊 EXPERIMENT 1 SUMMARY")
print(f"Total tested          : {total}")
print(f"Plates detected       : {detected + unread}/{total} ({round((detected+unread)/total*100)}%)")
print(f"OCR success (≥4 chars): {ocr_success}/{total} ({round(ocr_success/total*100)}%)")
print(f"No plate found        : {no_plate}/{total} ({round(no_plate/total*100)}%)")
print(f"Unread                : {unread}/{total} ({round(unread/total*100)}%)")
print(f"\n✅ Detection Rate : {round((detected+unread+no_plate==0 and 100 or (detected+unread)/total*100))}%")
print(f"✅ OCR Success    : {round(ocr_success/total*100)}%")
print(f"\n📁 Results saved to: {save_dir}")