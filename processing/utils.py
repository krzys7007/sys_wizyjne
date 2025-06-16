import cv2
import numpy as np
from collections import deque

def convert_result(counts):
    carl = counts.get('car', {}).get('left_to_right', 0)
    carr = counts.get('car', {}).get('right_to_left', 0)
    truckl = counts.get('truck', {}).get('left_to_right', 0)
    truckr = counts.get('truck', {}).get('right_to_left', 0)
    trams = counts.get('tram', {}).get('left_to_right', 0) + counts.get('tram', {}).get('right_to_left', 0)
    people = counts.get('person', {}).get('left_to_right', 0) + counts.get('person', {}).get('right_to_left', 0)
    bikes = counts.get('bike', {}).get('left_to_right', 0) + counts.get('bike', {}).get('right_to_left', 0)
    result = {
        "liczba_samochodow_osobowych_z_prawej_na_lewa": carr,
        "liczba_samochodow_osobowych_z_lewej_na_prawa": carl,
        "liczba_samochodow_ciezarowych_autobusow_z_prawej_na_lewa": truckr,
        "liczba_samochodow_ciezarowych_autobusow_z_lewej_na_prawa":truckl,
        "liczba_tramwajow": trams,
        "liczba_pieszych": people,
        "liczba_rowerzystow": bikes,
    }
    return result

def perform_processing(cap: cv2.VideoCapture):
    MAX_DISTANCE = 50
    MIN_AREA = 500
    COUNT_LINE_OFFSET = 350
    MEMORY = 60

    fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
    counted_ids = {}

    next_id = 0
    crossed_line = {}

    tracks = {}
    track_paths = {}
    object_types = {}
    counts = {
        'car': {'left_to_right': 0, 'right_to_left': 0},
        'truck': {'left_to_right': 0, 'right_to_left': 0},
        'tram': {'left_to_right': 0, 'right_to_left': 0},
        'person': {'left_to_right': 0, 'right_to_left': 0},
        'bike': {'left_to_right': 0, 'right_to_left': 0}
    }
    last_bbox_info = {}

    def remove_crooked_line(image, pt1, pt2, width):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)
        norm_dx = dx / length
        norm_dy = dy / length
        perp_dx = -norm_dy * width / 2
        perp_dy = norm_dx * width / 2

        p1 = (int(pt1[0] + perp_dx), int(pt1[1] + perp_dy))
        p2 = (int(pt1[0] - perp_dx), int(pt1[1] - perp_dy))
        p3 = (int(pt2[0] - perp_dx), int(pt2[1] - perp_dy))
        p4 = (int(pt2[0] + perp_dx), int(pt2[1] + perp_dy))

        strip_pts = np.array([[p1, p2, p3, p4]], dtype=np.int32)
        cv2.fillPoly(mask, strip_pts, 255)

        image_without_pole = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return image_without_pole

    def remove_multiple_crooked_lines(image, pole_list, width):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for pt1, pt2 in pole_list:
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            length = np.sqrt(dx**2 + dy**2)
            norm_dx = dx / length
            norm_dy = dy / length
            perp_dx = -norm_dy * width / 2
            perp_dy = norm_dx * width / 2

            p1 = (int(pt1[0] + perp_dx), int(pt1[1] + perp_dy))
            p2 = (int(pt1[0] - perp_dx), int(pt1[1] - perp_dy))
            p3 = (int(pt2[0] - perp_dx), int(pt2[1] - perp_dy))
            p4 = (int(pt2[0] + perp_dx), int(pt2[1] + perp_dy))

            strip_pts = np.array([[p1, p2, p3, p4]], dtype=np.int32)
            cv2.fillPoly(mask, [strip_pts], 255)

        result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return result

    def classify(w, h, y):
        area = w * h
        aspect_ratio = h / w if w > 0 else 0

        roi_name = None
        for name, y_start, y_end in roi_definitions:
            if y_start <= y <= y_end:
                roi_name = name
                break

        if roi_name == "cl_str":
            if area < 160000:
                return 'car'
        elif roi_name == "fh_str":
            if area < 40000:
                return 'car'

        if roi_name == "sidewalk":
            if area < 20000 and 1.3 <= aspect_ratio <= 3.0:
                return 'person'
            else:
                return 'bike'
            
        if roi_name == "cl_str":
            if area > 140000:
                return 'truck'
        elif roi_name == "fh_str":
            if area > 40000:
                return 'truck'

        if roi_name == "tram_line":
            if area > 90000 and aspect_ratio <= 0.9:
                return 'tram'
        
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
        iopercent = inter_area / float(box1_area + box2_area - inter_area + 1e-5)

        return iopercent > threshold

    def merge_boxes(boxes, iopercent_threshold=0.1):
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
                if boxes_overlap((x2, y2, w2, h2), bx, threshold=iopercent_threshold):
                    x2 = min(x2, bx[0])
                    y2 = min(y2, bx[1])
                    w2 = max(x2 + w2, bx[0] + bx[2]) - x2
                    h2 = max(y2 + h2, bx[1] + bx[3]) - y2
                    used[j] = True

            merged.append((x2, y2, w2, h2))
            used[i] = True

        if len(merged) < len(boxes):
            return merge_boxes(merged, iopercent_threshold)
        else:
            return merged

    def get_direction(path, line_x, obj_type):
        if len(path) < 2:
            return None
        x0 = path[0][0]
        x1 = path[-1][0]
        if obj_type == "person":
            return 'left_to_right'
        elif obj_type == "bike":
            return 'left_to_right'
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

    def near_line(cx, threshold=100):
        return abs(cx - line_x) <= threshold

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_x = frame_width // 2 - COUNT_LINE_OFFSET
    roi_definitions = [
        ("sidewalk", 630, frame_height),
        ("cl_str", 440, 620),
        ("tram_line", 330, 460),
        ("fh_str", 200, 360)
    ]

    frame_count = 0

    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break
        
        pole_list = []

        pole_start = (1042, 400)
        pole_end = (1032, 700)
        pole_list.append([pole_start, pole_end])

        pole_start = (320, 0)
        pole_end = (368, 600)
        pole_list.append([pole_start, pole_end])
        

        
        pole_start = (1396, 0)
        pole_end = (1378, 332)
        pole_list.append([pole_start, pole_end])

        # frame = remove_crooked_line(frame, pole_start, pole_end, pole_width)

        pole_start = (1566, 356)
        pole_end = (1534, 700)
        pole_list.append([pole_start, pole_end])
        pole_width = 20

        pole_width = 30
        
        frame = remove_multiple_crooked_lines(frame, pole_list, pole_width)

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

        merged_boxes = merge_boxes(filtered_boxes, iopercent_threshold=0.4)
        merged_boxes = remove_inner_boxes(merged_boxes)

        detections = []
        for (x, y, w, h) in merged_boxes:
            cx = x + w // 2
            cy = y + h // 2
            detections.append(((cx, cy), (x, y, w, h)))

        used = set()
        new_tracks = {}

        for detection in detections:
            center = detection[0]
            bbox = detection[1]

            cx = center[0]
            cy = center[1]

            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            
            matched = False
            matched_oid = None
            for oid, (px, py) in tracks.items():
                if oid in used:
                    continue
                if np.hypot(cx - px, cy - py) < MAX_DISTANCE:
                    new_tracks[oid] = (cx, cy)
                    track_paths[oid].append((cx, cy))
                    used.add(oid)
                    matched = True
                    matched_oid = oid
                    break

            if not matched:
                oid = next_id
                new_tracks[oid] = (cx, cy)
                track_paths[oid] = deque(maxlen=MEMORY)
                track_paths[oid].append((cx, cy))
                if near_line(cx):
                    obj_type = classify(w, h, y + h)
                    object_types[oid] = obj_type
                    if w > 0:
                        aspect_ratio = h / w
                        aspect_ratio = round(aspect_ratio, 2)
                    else:
                        aspect_ratio = 0
                    last_bbox_info[oid] = (x, y, w, h, aspect_ratio)
                next_id += 1
            else:
                oid = matched_oid
                if near_line(cx) and oid not in object_types:
                    obj_type = classify(w, h, y + h)
                    object_types[oid] = obj_type
                    aspect_ratio = round(h / w, 2) if w > 0 else 0
                    last_bbox_info[oid] = (x, y, w, h, aspect_ratio)

        ids_to_remove = []
        for oid in tracks:
            if oid not in new_tracks:
                path = track_paths[oid]
                obj_type = object_types.get(oid)
                direction = get_direction(path, line_x, obj_type)
                if obj_type and direction and oid not in counted_ids:
                    if obj_type in counts:
                        if direction in counts[obj_type]:
                            counts[obj_type][direction] = counts[obj_type][direction] + 1
                    counted_ids[oid] = direction
                    crossed_line[oid] = True
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
                roi_name = None
                for name, y_start, y_end in roi_definitions:
                    if y + h >= y_start and y + h <= y_end:
                        roi_name = name
                        break
                if roi_name:
                    roi_text = "ROI: " + roi_name
                else:
                    roi_text = "ROI: unknown"


                label = f"{obj_type} AR:{aspect_ratio} A:{w*h}"
                direction = get_direction(track_paths[oid], line_x, obj_type)
                dir_text = f"DIR: {direction}" if direction else "DIR: X"

                if oid in counted_ids:
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, f"{label} {dir_text}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                cv2.putText(frame, roi_text, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)

        y_pos = 30
        for cls in ['car', 'truck', 'tram', 'person']:
            cv2.putText(frame, f"{cls}s L->R: {counts[cls]['left_to_right']}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"{cls}s R->L: {counts[cls]['right_to_left']}", (250, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30

        for name, y_start, y_end in roi_definitions:
            color = (0, 255, 0) if name == "cl_str" else (255, 0, 255) if name == "fh_str" else (255, 255, 0)
            cv2.rectangle(frame, (0, y_start), (frame_width, y_end), color, 2)
            cv2.putText(frame, name, (10, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


        cv2.drawContours(filtered1, contours, -1, (0,255,0), 3)

        frame = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8)
        
        cv2.imshow("Counter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key != ord('d'):
            continue

    print(convert_result(counts))
    cap.release()
    cv2.destroyAllWindows()
    return(convert_result(counts))
