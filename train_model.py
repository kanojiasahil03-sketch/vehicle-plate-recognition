import sys
print("Python:", sys.version)
print("Starting training...")

try:
    from ultralytics import YOLO
    print("YOLO imported OK")
    
    model = YOLO('yolov8n.pt')
    print("Model loaded OK")
    
    results = model.train(
        data='/home/sahil/anpr_project/License-Plate-Recognition-1/data.yaml',
        epochs=20,
        imgsz=640,
        batch=8,
        name='plate_detector',
        project='/home/sahil/anpr_project/models'
    )
    print("Training complete!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()