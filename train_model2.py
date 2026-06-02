import sys
print("Starting training on new dataset...")

try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='/home/sahil/anpr_project/Number-Plates-1/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,
        name='plate_detector_v2',
        project='/home/sahil/anpr_project/models'
    )
    print("Training complete!")

except Exception as e:
    import traceback
    traceback.print_exc()
EOF