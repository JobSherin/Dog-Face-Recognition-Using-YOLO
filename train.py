from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(data = 'data.yaml',epochs =5, imgsz=416,batch=2,fraction = 0.1)