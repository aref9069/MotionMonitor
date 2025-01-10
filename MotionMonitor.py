import numpy as np
import cv2 as cv
import cvzone
import math
from ultralytics import YOLO
from sort.sort import Sort

# Initialize video capture and model
cap = cv.VideoCapture("cars.mp4")  
model = YOLO("../Yolo-Weights/yolo11n.pt")

# Define class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "stop sign", "parking meter", "bench", "bird", "cat", "skateboard", 
              ]

# # Load mask and car illustration images
mask = cv.imread("mask.png")
imgCar = cv.imread("Car-Illustration.png", cv.IMREAD_UNCHANGED)
imgCounting= cv.resize(imgCar, (457,161))
                        

# Initialize tracker and line parameters
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
line_points = [400, 297, 673, 297]
totalCount = []
 
while True:
    # Read frame from video
    success, img = cap.read()

    # Apply mask and overlay counting image
    imgRegion = cv.bitwise_and(img, mask)
    img = cvzone.overlayPNG(img, imgCounting, (0, 0))

    # Perform object detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 4))
 
    for r in results:
        boxes = r.boxes
        
        for box in boxes:
            # Extract bounding box coordinates and confidence score
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]


            # Filter for relevant classes based on confidence threshold
            if currentClass in ["car", "truck", "bus", "motorbike", "bicycle"] and conf > 0.3:

                currentArray = np.array([x1, y1, x2, y2])
                detections = np.vstack((detections, currentArray))

    # Update tracker with detected objects
    resultsTracker = tracker.update(detections)
 
    # Draw the tracking line on the image
    cv.line(img, (line_points[0], line_points[1]), (line_points[2], line_points[3]), (0, 0, 255), 5)

    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and ID on the image
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {obj_id}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv.circle(img, (cx, cy), 3, (0, 255, 255), cv.FILLED)
 
        # Check if the object crosses the tracking line
        if line_points[0] < cx < line_points[2] and line_points[1] - 15 < cy < line_points[1] + 15:
            if obj_id not in totalCount:
                totalCount.append(obj_id)
                cv.line(img, (line_points[0], line_points[1]), (line_points[2], line_points[3]), (0, 255, 0), 5)
 
    # Display the total count of detected vehicles
    cv.putText(img,str(len(totalCount)),(255,100),cv.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    
    cv.imshow("Image", img)
    # Break loop on key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()