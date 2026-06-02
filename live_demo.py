"""
Live ANPR Demo — Webcam
Press Q to quit, S to save screenshot
"""
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import cv2
import re
import time

# ── Models ───────────────────────────────────────────────────────────────────
plate_model = YOLO('/home/sahil/anpr_project/models/plate_detector_v2/weights/best.pt')
ocr         = LicensePlateRecognizer('european-plates-mobile-vit-v2-model')

def clean_plate(text):
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned[:8] if len(cleaned) > 8 else cleaned

def read_plate(crop):
    try:
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = ocr.run(gray)
        if result:
            return clean_plate(result[0].plate)
    except:
        pass
    return None

# ── Camera ───────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Live ANPR Demo started — press Q to quit, S to save screenshot")

last_plate  = ""
last_time   = 0
screenshot  = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # Run detection every 0.5 seconds to reduce CPU load
    now = time.time()
    if now - last_time > 0.5:
        last_time = now
        results = plate_model(frame, verbose=False)
        boxes   = results[0].boxes

        if len(boxes) > 0:
            best = max(boxes, key=lambda b: b.conf)
            conf = float(best.conf)

            if conf > 0.4:
                x1, y1, x2, y2 = map(int, best.xyxy[0])

                # Crop and read
                pad  = 5
                x1p  = max(0, x1-pad); y1p = max(0, y1-pad)
                x2p  = min(frame.shape[1], x2+pad)
                y2p  = min(frame.shape[0], y2+pad)
                crop = frame[y1p:y2p, x1p:x2p]

                if crop.size > 0:
                    # Upscale
                    h, w = crop.shape[:2]
                    if w < 300:
                        scale = 300 / w
                        crop  = cv2.resize(crop, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_CUBIC)
                    plate = read_plate(crop)
                    if plate:
                        last_plate = plate

                # Draw box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display, f"Conf: {conf:.2f}",
                           (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)

    # Always show last detected plate
    if last_plate:
        # Black background for text
        (tw, th), _ = cv2.getTextSize(last_plate,
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        cv2.rectangle(display, (20, 20), (40+tw, 80), (0, 0, 0), -1)
        cv2.putText(display, last_plate,
                   (30, 65), cv2.FONT_HERSHEY_SIMPLEX,
                   2, (0, 255, 0), 3)

    # Instructions
    cv2.putText(display, "Q: Quit  S: Screenshot",
               (10, display.shape[0]-15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("ANPR Live Demo", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"results/screenshot_{screenshot:03d}.jpg"
        cv2.imwrite(fname, display)
        print(f"Saved: {fname}")
        screenshot += 1

cap.release()
cv2.destroyAllWindows()
print("Demo ended.")