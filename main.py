from ultralytics import YOLO

# Загрузка COCO-предобученной модели YOLOv8n
model = YOLO("yolov8n.pt")

# Отображение информации о модели (опционально)
model.info()

# Обучение модели на примерах COCO8 в течение 10 эпох
results = model.train(data="C:\\Users\\User\\Desktop\\samolet.yaml", workers=0, epochs=10, imgsz=640)