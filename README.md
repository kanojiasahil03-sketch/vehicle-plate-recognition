# 🚗 Vehicle Plate Recognition (ANPR)

Automatic Number Plate Recognition system built with YOLOv8 and fast-plate-ocr.  
Detects and reads vehicle registration plates from images.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OCR](https://img.shields.io/badge/OCR-fast--plate--ocr-orange)

---

## 📊 Results

| Metric | Score |
|---|---|
| Detection Rate | 100% |
| OCR Accuracy | 85% |
| YOLO mAP50 | 96.5% |
| mAP50-95 | 88.2% |

---

## 🔧 How It Works

```
Car Image → YOLOv8 (finds plate) → Perspective Correction → fast-plate-ocr (reads text) → "AB12CDE"
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install ultralytics easyocr fast-plate-ocr opencv-python

# Run on a single image
python anpr_pipeline.py path/to/car.jpg

# Batch evaluate 20 random images
python evaluate2.py
```

---

## 📁 Files

| File | Purpose |
|---|---|
| `anpr_pipeline.py` | Main pipeline — single image |
| `evaluate2.py` | Batch evaluator with saved results |
| `train_model2.py` | YOLOv8 training script |
| `download_data2.py` | Dataset downloader |

---

## 🛠 Tech Stack

- **Detection:** YOLOv8n (custom trained, 1,114 images)
- **OCR:** fast-plate-ocr (European plates) + EasyOCR fallback
- **CV:** OpenCV (perspective correction, preprocessing)
- **Dataset:** Roboflow — Number Plates Ukraine

---

## 📸 Example Output

- Green box = plate detected
- Text overlay = OCR result
- Engine label = which OCR model was used

---

---

## 👤 Author

**Sahil Kanojia**  
Robotics Engineer  
[LinkedIn](https://www.linkedin.com/in/sahil-kanojia-7a38901a9/) | [GitHub](https://github.com/kanojiasahil03-sketch)
