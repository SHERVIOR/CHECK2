



import cv2
import torch
from tracker import *
import numpy as np
import serial

model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

cap = cv2.VideoCapture("../Videos/1st.mp4")

ser = serial.Serial('/dev/rfcomm0', baudrate=9600) #replace this one

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()

# Area coordinates
area_1 = [(220, 461), (411, 120), (593, 122), (756, 452)]  # to sm bacoor

# Parameters for congestion detection
congestion_threshold = 7  # Number of vehicles threshold to consider a rectangle congested
congested_rectangles = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    cv2.polylines(frame, [np.array(area_1, np.int32)], True, (255, 0, 255), 3)

    results = model(frame)

    boxes = []
    centroids = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        b = str(row['name'])
        if b in ['car', 'motorcycle', 'bus', 'truck']:
            boxes.append((x1, y1, x2, y2))
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            centroids.append(centroid)

    boxes_ids = tracker.update(boxes)

    vehicles_in_roi = set()
    for centroid, box_id in zip(centroids, boxes_ids):
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        #cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

        if cv2.pointPolygonTest(np.array(area_1, np.int32), centroid, False) > 0:
            vehicles_in_roi.add(id)

    p = len(vehicles_in_roi)
    cv2.putText(frame, str(p), (20, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    # Congestion detection
    if p >= congestion_threshold:
        congested_rectangles.append(area_1)  # Add the congested rectangle to the list
    else:
        congested_rectangles = []  # Reset the congested rectangles if congestion is not detected

    for rect in congested_rectangles:
        cv2.polylines(frame, [np.array(rect, np.int32)], True, (0, 0, 255), 3)

    # Output warning if congested
    if len(congested_rectangles) > 0:
        warning_message = "WARNING! CONGESTED LANE"
        ser.write(warning_message.encode())  # Send the warning message to Arduino via Bluetooth

    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == 27:
         break

# Retrieve the last number in the count variable every 30 seconds
current_time = time.time()
if current_time - last_update_time >= 30:
    last_count = p
    last_update_time = current_time



last_count_integer = int(last_count)
print(last_count_integer)

ser.close()
cap.release()
cv2.destroyAllWindows()


