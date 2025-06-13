import argparse
import json
from pathlib import Path
from collections import deque
import numpy as np
import cv2

# ---- PARAMETRY PRZETWARZANIA ----
MAX_DISTANCE = 50
MIN_AREA = 500
COUNT_LINE_OFFSET = 200
MEMORY = 30

def classify(w, h):
    area = w * h
    aspect_ratio = h / w if w > 0 else 0
    if area > 300000 and aspect_ratio < 0.6:
        return 'tram'
    elif 40000 < area <= 250000 and 0.7 <= aspect_ratio <= 1.2:
        return 'truck'
    elif 2500 < area <= 40000 and 0.4 <= aspect_ratio <= 0.7:
        return 'car'
    elif area > 500 and aspect_ratio >= 1.6:
        return 'person'
    else:
        return None

def merge_boxes(boxes, iou_threshold=0.3):
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        x1, y1, w1, h1 = boxes[i]
        x2, y2, w2, h2 = x1, y1, w1, h1
        for j in range(i + 1, len(boxes)):
            if used[j]: continue
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
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-5)
    return iou > threshold

def get_direction(path, line_x):
    if len(path) < 2: return None
    x0, x1 = path[0][0], path[-1][0]
    if x0 < line_x and x1 > line_x:
        return 'left_to_right'
    elif x0 > line_x and x1 < line_x:
        return 'right_to_left'
    return None

def perform_processing(cap, visualize=False):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
    next_id, tracks, track_paths, object_types, last_bbox_info = 0, {}, {}, {}, {}
    counts = {k: {'left_to_right': 0, 'right_to_left': 0} for k in ['car', 'truck', 'tram', 'person']}
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    line_x = frame_width // 2 - COUNT_LINE_OFFSET

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        fgmask = fgbg.apply(blurred)
        fgmask = cv2.medianBlur(fgmask, 5)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_boxes = [(x, y, w, h) for cnt in contours if (w := cv2.boundingRect(cnt)[2]) * (h := cv2.boundingRect(cnt)[3]) >= MIN_AREA]
        merged_boxes = merge_boxes(filtered_boxes, iou_threshold=0.1)
        detections = [((x + w // 2, y + h // 2), (x, y, w, h)) for (x, y, w, h) in merged_boxes]

        used, new_tracks = set(), {}
        for (cx, cy), (x, y, w, h) in detections:
            matched = False
            for oid, (px, py) in tracks.items():
                if oid in used: continue
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
                object_types[next_id] = classify(w, h)
                next_id += 1

        for oid in list(tracks.keys()):
            if oid not in new_tracks:
                direction = get_direction(track_paths[oid], line_x)
                obj_type = object_types.get(oid)
                if obj_type and direction:
                    counts[obj_type][direction] += 1
                track_paths.pop(oid, None)
                object_types.pop(oid, None)

        tracks = new_tracks

        if visualize:
            for oid, path in track_paths.items():
                for i in range(1, len(path)):
                    cv2.line(frame, path[i - 1], path[i], (0, 255, 0), 2)
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 2)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    return counts

def main():
    # ---- OPCJE UŻYTKOWNIKA ----
    prt_json = 0       # 0: nie zapisuje JSON, 1: zapisuje
    wiz = 1            # 0: bez wizualizacji, 1: z wizualizacją
    pth = 1            # 0: z argumentów, 1: ze zmiennej
    video_path = "filmy/00000.mp4"  # ścieżka do pojedynczego pliku .mp4 jeśli pth == 1
    
    if pth == 1:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        results = {Path(video_path).name: perform_processing(cap, visualize=(wiz == 1))}
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('videos_dir', type=str)
        parser.add_argument('results_file', type=str, nargs='?')
        args = parser.parse_args()

        videos_dir = Path(args.videos_dir)
        videos_paths = sorted([p for p in videos_dir.iterdir() if p.suffix == '.mp4'])
        results = {}
        for video_path in videos_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f'Error loading video {video_path}')
                continue
            print(f'Processing {video_path.name}')
            results[video_path.name] = perform_processing(cap, visualize=(wiz == 1))

        if prt_json == 1 and args.results_file:
            results_file = Path(args.results_file)
            with results_file.open('w') as f:
                json.dump(results, f, indent=4)
        elif prt_json == 1:
            with open("results.json", 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
