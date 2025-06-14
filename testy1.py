import cv2
import numpy as np
import os
import csv
import time
from collections import deque

# Parameters
MAX_DISTANCE = 50
MIN_AREA = 500
COUNT_LINE_OFFSET = 200
MEMORY = 30
EXTRACTION_INTERVAL = 5  # seconds

# Directories
os.makedirs("pictures", exist_ok=True)
csv_file = "bboxes.csv"

# CSV header initialization
if not os.path.isfile(csv_file):
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "area", "aspect_ratio", "solidity", "extent", "label"])

# Initialize
fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
next_id = 0
tracks = {}
track_paths = {}
object_types = {}
last_bbox_info = {}
frame_counter = 0
bbox_id = 1
last_save_time = time.time()

def merge_boxes(boxes, iou_threshold=0.4):
    if not boxes:
        return []

    merged = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, w1, h1 = boxes[i]
        x2, y2, w2, h2 = x1, y1, w1, h1

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bx = boxes[j]
            if boxes_overlap((x2, y2, w2, h2), bx, threshold=iou_threshold):
                x2 = min(x2, bx[0])
                y2 = min(y2, bx[1])
                w2 = max(x2 + w2, bx[0] + bx[2]) - x2
                h2 = max(y2 + h2, bx[1] + bx[3]) - y2
                used[j] = True

        merged.append((x2, y2, w2, h2))
        used[i] = True

    if len(merged) < len(boxes):
        return merge_boxes(merged, iou_threshold)
    else:
        return merged

def boxes_overlap(box1, box2, threshold=0.4):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-5)
    return iou > threshold

def remove_inner_boxes(boxes):
    filtered = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        is_inner = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j:
                continue
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                is_inner = True
                break
        if not is_inner:
            filtered.append((x1, y1, w1, h1))
    return filtered

def extract_features(cnt, w, h):
    area = w * h
    aspect_ratio = h / w if w > 0 else 0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    cnt_area = cv2.contourArea(cnt)
    solidity = cnt_area / hull_area if hull_area > 0 else 0
    rect_area = w * h
    extent = cnt_area / rect_area if rect_area > 0 else 0
    return area, aspect_ratio, solidity, extent

# Load video
cap = cv2.VideoCapture("filmy/00000.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * EXTRACTION_INTERVAL)
line_x = frame_width // 2 - COUNT_LINE_OFFSET

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    fgmask = fgbg.apply(blurred)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, thresh = cv2.threshold(fgmask, 170, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (15, 15))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (105, 105))
    dilation = cv2.dilate(closing, (305, 305), iterations=15)
    filtered = dilation

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_boxes = []
    contour_map = {}
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA:
            continue
        filtered_boxes.append((x, y, w, h))
        contour_map[(x, y, w, h)] = cnt

    merged_boxes = merge_boxes(filtered_boxes, iou_threshold=0.4)
    merged_boxes = remove_inner_boxes(merged_boxes)

    # Every N frames = ~5 seconds
    if frame_counter % frame_interval == 0:
        for box in merged_boxes:
            x, y, w, h = box
            crop = frame[y:y + h, x:x + w]
            filename = f"pictures/{bbox_id}.png"
            cv2.imwrite(filename, crop)

            cnt = contour_map.get(box)
            if cnt is not None:
                area, ar, solidity, extent = extract_features(cnt, w, h)
            else:
                area, ar, solidity, extent = w * h, h / w if w > 0 else 0, 0, 0

            with open(csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([bbox_id, area, ar, solidity, extent, ""])
            bbox_id += 1

    # Draw
    for (x, y, w, h) in merged_boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)
    cv2.imshow("Bounding Boxes", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
