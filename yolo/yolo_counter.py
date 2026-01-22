from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

def get_vehicle_counts(frame, rois):
    results = model(frame, conf=0.4, classes=[2,3,5,7])
    counts = []
    for roi in rois:
        x1,y1,x2,y2 = roi
        count = 0
        for box in results[0].boxes.xyxy:
            cx = (box[0]+box[2])/2
            cy = (box[1]+box[3])/2
            if x1 < cx < x2 and y1 < cy < y2:
                count += 1
        counts.append(count)
    return np.array(counts)
