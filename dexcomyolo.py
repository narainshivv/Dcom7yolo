from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2
MODEL = YOLO("Downloads/best1.pt")
results = MODEL.predict(source="0",show= True)
print(results)