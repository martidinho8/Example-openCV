import cv2 as cv
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv.VideoCapture("/home/martijn/Documenten/develop/object_tracking/highway.mp4")

# Object detection from Stable camera
object_detector = cv.createBackgroundSubtractorMOG2(history=100,varThreshold=40)



while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Extract Region of interest
    roi = frame[340: 720,500: 800]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _ , mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)

    # _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(cnt)
        if area > 100:
            # cv.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            detections.append([x, y, w, h])

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    print(boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    
    print(detections)
    cv.imshow("roi", roi)
    cv.imshow("Frame", frame)
    cv.imshow("Mask", mask)

    key = cv.waitKey(30)
    if key == 27:
        break




cap.release()
cv.destroyAllWindows()