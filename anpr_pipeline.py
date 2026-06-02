from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import cv2
import re
import sys
import os
import numpy as np

# ── Models ───────────────────────────────────────────────────────────────────
plate_model = YOLO('/home/sahil/anpr_project/models/plate_detector_v2/weights/best.pt')
fast_ocr    = LicensePlateRecognizer('european-plates-mobile-vit-v2-model')

# ── Helpers ───────────────────────────────────────────────────────────────────
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
    """Primary OCR — fast-plate-ocr (European plate specialist)"""
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = fast_ocr.run(gray)
        if result:
            plate = result[0].plate.upper()
            cleaned = clean_plate(plate)
            if len(cleaned) >= 4:
                return cleaned
        return None
    except Exception as e:
        print(f"fast-plate-ocr error: {e}")
        return None

def _preprocess_variants(crop):
    variants = [crop]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants.append(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    variants.append(clahe.apply(gray))
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    variants.append(denoised)
    adapt = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    variants.append(adapt)
    b, g, r = cv2.split(crop)
    red = cv2.subtract(r, g)
    _, red_bin = cv2.threshold(red, 30, 255, cv2.THRESH_BINARY)
    variants.append(cv2.bitwise_not(red_bin))
    return variants

def perspective_correct(crop):
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
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    w, h = int(rect[1][0]), int(rect[1][1])
    if w == 0 or h == 0:
        return crop
    if h > w:
        w, h = h, w
    target_w = max(w, 400)
    target_h = int(target_w / 4.5)
    src = np.float32(sorted(box.tolist(), key=lambda x: (x[1], x[0])))
    if src[0][0] > src[1][0]:
        src[[0, 1]] = src[[1, 0]]
    if src[2][0] < src[3][0]:
        src[[2, 3]] = src[[3, 2]]
    dst = np.float32([[0,0],[target_w,0],[target_w,target_h],[0,target_h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(crop, M, (target_w, target_h))

def preprocess_crop(crop):
    h, w = crop.shape[:2]
    target_w = 600
    if w < target_w:
        scale = target_w / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return crop

def detect_and_read(image_path):
    print(f"\nLoading: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not load image")
        return

    print(f"Image shape: {img.shape}")
    results = plate_model(img, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("No plate detected.")
        return

    best = max(boxes, key=lambda b: b.conf)
    x1, y1, x2, y2 = map(int, best.xyxy[0])
    conf = float(best.conf)
    print(f"Plate detected | Conf: {conf:.2f} | Box: {x1},{y1},{x2},{y2}")

    pad = 5
    x1p = max(0, x1-pad); y1p = max(0, y1-pad)
    x2p = min(img.shape[1], x2+pad); y2p = min(img.shape[0], y2+pad)
    crop = img[y1p:y2p, x1p:x2p]
    crop = perspective_correct(crop)
    crop = preprocess_crop(crop)
    cv2.imwrite("results/plate_crop.jpg", crop)

    # ── OCR: fast-plate-ocr first, EasyOCR fallback ──────────────────────────
    print("Running fast-plate-ocr...")
    plate_text = fast_read(crop)

    if plate_text:
        print(f"fast-plate-ocr: {plate_text}")
    else:
        plate_text = "UNREAD"
        print("OCR failed — plate unread.")

    print(f"\n>>> Final plate: {plate_text} <<<")
    color = img.copy()
    cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Smart text sizing — fits within image width
    plate_w = x2 - x1
    font_scale = min(1.2, plate_w / 120)
    font_scale = max(0.6, font_scale)

    # Draw black background behind text for readability
    (tw, th), _ = cv2.getTextSize(plate_text,
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, 2)
    tx = max(0, x1)
    ty = max(th + 5, y1 - 5)
    cv2.rectangle(color, (tx, ty - th - 5),
                  (tx + tw + 5, ty + 5), (0, 0, 0), -1)
    cv2.putText(color, plate_text, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    # color = img.copy()
    # cv2.rectangle(color, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # cv2.putText(color, plate_text, (x1, y1 - 15),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/annotated.jpg", color)
    print("Saved → results/annotated.jpg")
    return plate_text

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_car.jpg"
    detect_and_read(image_path)