import cv2
import numpy as np
from collections import deque

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("bboxes.csv")
df["label"] = df["label"].fillna("n")

features = ["area", "aspect_ratio", "solidity", "extent"]
X = df[features]
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_full, _, y_train, _, = train_test_split(X_scaled, y, test_size=0.1, random_state=42, stratify=(y != "n"))
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_full)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

label_map = {
    "a": "car",
    "c": "truck",
    "t": "tram",
    "p": "person",
    "b": "bike",
    "n": None
}

MAX_DISTANCE = 50
MIN_AREA = 500
COUNT_LINE_OFFSET = 200
MEMORY = 30

fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
next_id = 0
tracks = {}
track_paths = {}
object_types = {}
counts = {
    'car': {'left_to_right': 0, 'right_to_left': 0},
    'truck': {'left_to_right': 0, 'right_to_left': 0},
    'tram': {'left_to_right': 0, 'right_to_left': 0},
    'person': {'left_to_right': 0, 'right_to_left': 0}
}
last_bbox_info = {}

def classify(w, h):
    area = w * h
    aspect_ratio = h / w if w > 0 else 0

    if area > 300000 and aspect_ratio < 0.6:
        return 'tram'
    elif 40000 < area <= 250000 and 0.7 <= aspect_ratio <= 1.2:
        return 'truck'
    elif 2500 < area <= 40000 and 0.4 <= aspect_ratio <= 0.7:
        return 'car'
    elif 500 < area < 3000 and aspect_ratio >= 1.6:
        return 'person'
    else:
        return None

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

def merge_boxes(boxes, iou_threshold=0.1):
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

def get_direction(path, line_x):
    if len(path) < 2:
        return None
    x0 = path[0][0]
    x1 = path[-1][0]
    if x0 < line_x and x1 > line_x:
        return 'left_to_right'
    elif x0 > line_x and x1 < line_x:
        return 'right_to_left'
    return None

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

cap = cv2.VideoCapture("filmy/00000.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_x = frame_width // 2 - COUNT_LINE_OFFSET

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    fgmask = fgbg.apply(blurred)

    fgmask = cv2.medianBlur(fgmask, 5)
    
    _, thresh = cv2.threshold(fgmask, 170, 255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, (15, 15))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (105, 105))
    dilation = cv2.dilate(closing,(305, 305),iterations = 15)

    filtered = dilation
    filtered1 = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA:
            continue
        filtered_boxes.append((x, y, w, h))

    merged_boxes = merge_boxes(filtered_boxes, iou_threshold=0.4)
    merged_boxes = remove_inner_boxes(merged_boxes)

    detections = []
    for (x, y, w, h) in merged_boxes:
        cx = x + w // 2
        cy = y + h // 2
        detections.append(((cx, cy), (x, y, w, h)))

    used = set()
    new_tracks = {}

    for (cx, cy), (x, y, w, h) in detections:
        matched = False
        for oid, (px, py) in tracks.items():
            if oid in used:
                continue
            if np.hypot(cx - px, cy - py) < MAX_DISTANCE:
                new_tracks[oid] = (cx, cy)
                track_paths[oid].append((cx, cy))
                used.add(oid)
                matched = True
                break
        if not matched:
            new_tracks[next_id] = (cx, cy)
            track_paths[next_id] = deque(maxlen=MEMORY)
            track_paths[next_id].append((cx, cy))

            area, ar, solidity, extent = extract_features(cnt, w, h)

            feature_vec = np.array([[area, ar, solidity, extent]])
            feature_scaled = scaler.transform(feature_vec)
            feature_pca = pca.transform(feature_scaled)
            predicted_label = knn.predict(feature_pca)[0]
            obj_type = label_map.get(predicted_label, None)
            object_types[next_id] = obj_type
            aspect_ratio = round(h / w, 2) if w > 0 else 0
            last_bbox_info[next_id] = (x, y, w, h, aspect_ratio)
            next_id += 1
        else:
            obj_type = object_types.get(oid)
            aspect_ratio = round(h / w, 2) if w > 0 else 0
            last_bbox_info[oid] = (x, y, w, h, aspect_ratio)

    ids_to_remove = []
    for oid in tracks:
        if oid not in new_tracks:
            path = track_paths[oid]
            obj_type = object_types.get(oid)
            direction = get_direction(path, line_x)
            if obj_type and direction:
                counts[obj_type][direction] += 1
            ids_to_remove.append(oid)

    for oid in ids_to_remove:
        track_paths.pop(oid, None)
        object_types.pop(oid, None)
        last_bbox_info.pop(oid, None)

    tracks = new_tracks

    cv2.line(frame, (line_x, 0), (line_x, frame_height), (255, 0, 0), 2)

    for oid, (cx, cy) in tracks.items():
        obj_type = object_types.get(oid, '?')

        if oid in last_bbox_info:
            x, y, w, h, aspect_ratio = last_bbox_info[oid]
            label = f"{obj_type}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    y_pos = 30
    for cls in ['car', 'truck', 'tram', 'person']:
        cv2.putText(frame, f"{cls}s L->R: {counts[cls]['left_to_right']}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"{cls}s R->L: {counts[cls]['right_to_left']}", (250, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 30

    cv2.drawContours(filtered1, contours, -1, (0,255,0), 3)
    cv2.imshow("Counter", frame)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    elif key != ord('d'):
        continue

cap.release()
cv2.destroyAllWindows()
