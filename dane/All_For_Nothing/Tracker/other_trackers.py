import cv2
import numpy as np
import uuid

VIDEO_PATH = "filmy/00000.mp4"
LINE_OFFSET = 200 
MIN_CONTOUR_AREA = 1500
MAX_MISSED_FRAMES = 10
ADD_TRACKER_EVERY_N_FRAMES = 10
MAX_TRACKERS = 10

TARGET_W, TARGET_H = 1280, 720

trackers = dict() 
objects = dict()
counts = {
    'car_LR': 0, 'car_RL': 0,
    'truck_LR': 0, 'truck_RL': 0,
    'tram_LR': 0, 'tram_RL': 0,
    'person_LR': 0, 'person_RL': 0,
}

def classify(w, h, area):
    ratio = h / float(w)
    if area < 2000 or ratio > 1.5:
        return 'person'
    elif area > 120000 and ratio < 0.5:
        return 'tram'
    elif 30000 < area < 100000 and ratio < 0.8:
        return 'truck'
    else:
        return 'car'

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print("Failed to open video")
    exit()

orig_h, orig_w = frame.shape[:2]
scale_x = orig_w / TARGET_W
scale_y = orig_h / TARGET_H

count_line_x = (TARGET_W // 2) - LINE_OFFSET

fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    small_frame = cv2.resize(frame, (TARGET_W, TARGET_H))

    fgmask = fgbg.apply(small_frame)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (50, 50))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (105, 105))
    dilation = cv2.dilate(closing, (305, 305), iterations=15)
    filtered = dilation
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    remove_ids = []
    for obj_id, trk in trackers.items():
        success, bbox = trk.update(small_frame)
        if not success:
            objects[obj_id]['missed'] += 1
            if objects[obj_id]['missed'] > MAX_MISSED_FRAMES:
                remove_ids.append(obj_id)
            continue

        x, y, w, h = [int(v) for v in bbox]
        cx = x + w // 2

        prev_x = objects[obj_id]['last_x']
        objects[obj_id]['last_x'] = cx
        objects[obj_id]['bbox'] = bbox
        objects[obj_id]['missed'] = 0

        if not objects[obj_id]['counted']:
            if prev_x < count_line_x <= cx:
                dir = 'LR'
            elif prev_x > count_line_x >= cx:
                dir = 'RL'
            else:
                continue
            obj_type = objects[obj_id]['type']
            key = f"{obj_type}_{dir}"
            counts[key] += 1
            objects[obj_id]['counted'] = True
            print(f"{obj_type}s from {('left to right' if dir == 'LR' else 'right to left')}: {counts[key]}")

    for rid in remove_ids:
        del trackers[rid]
        del objects[rid]

    if frame_idx % ADD_TRACKER_EVERY_N_FRAMES == 0:
        if len(trackers) >= MAX_TRACKERS:
            pass
        else:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w // 2
                cy = y + h // 2

                overlap = False
                for obj_id, data in objects.items():
                    bx, by, bw, bh = data['bbox']
                    center_x = bx + bw // 2
                    center_y = by + bh // 2
                    dist = np.hypot(center_x - cx, center_y - cy)
                    if dist < 50:
                        overlap = True
                        break
                if overlap:
                    continue

                obj_type = classify(w, h, area)
                tracker = cv2.TrackerCSRT_create()
                tracker.init(small_frame, (x, y, w, h))
                obj_id = str(uuid.uuid4())

                trackers[obj_id] = tracker
                objects[obj_id] = {
                    'bbox': (x, y, w, h),
                    'last_x': cx,
                    'counted': False,
                    'missed': 0,
                    'type': obj_type,
                }

    line_x_orig = int(count_line_x * scale_x)
    cv2.line(frame, (line_x_orig, 0), (line_x_orig, orig_h), (0, 255, 0), 2)

    for obj_id, data in objects.items():
        x, y, w, h = data['bbox']
        x1 = int(x * scale_x)
        y1 = int(y * scale_y)
        w1 = int(w * scale_x)
        h1 = int(h * scale_y)
        label = data['type']
        cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # cv2.drawContours(filtered1, contours, -1, (0,255,0), 3)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
