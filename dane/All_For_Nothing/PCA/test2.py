import os
import cv2
import pandas as pd
import numpy as np

PICTURE_FOLDER = "pictures/"
CSV_FILE = "bboxes.csv"
LABELS = {'a': 'car', 'c': 'truck', 't': 'tram', 'p': 'person', 'b': 'bike', 'n': 'nothing'}

BUTTON_WIDTH = 100
BUTTON_HEIGHT = 40
PADDING = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
selected_label = None
buttons = []
save_triggered = False

df = pd.read_csv(CSV_FILE)

def draw_buttons_window():
    global buttons
    buttons = []
    canvas_width = len(LABELS) * (BUTTON_WIDTH + PADDING) + BUTTON_WIDTH + PADDING * 2
    canvas_height = BUTTON_HEIGHT + PADDING * 2
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    x_start = PADDING
    y_start = PADDING

    for i, (key, label) in enumerate(LABELS.items()):
        x1 = x_start + i * (BUTTON_WIDTH + PADDING)
        x2 = x1 + BUTTON_WIDTH
        y1 = PADDING
        y2 = y1 + BUTTON_HEIGHT
        color = (0, 200, 0) if selected_label == key else (200, 200, 200)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
        cv2.putText(canvas, label, (x1 + 5, y1 + 25), FONT, 0.6, (0, 0, 0), 1)
        buttons.append({'key': key, 'label': label, 'rect': (x1, y1, x2, y2)})

    x1 = x_start + len(LABELS) * (BUTTON_WIDTH + PADDING)
    x2 = x1 + BUTTON_WIDTH
    y1 = PADDING
    y2 = y1 + BUTTON_HEIGHT
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.putText(canvas, "Save", (x1 + 10, y1 + 25), FONT, 0.6, (255, 255, 255), 1)
    buttons.append({'key': 'save', 'label': 'Save', 'rect': (x1, y1, x2, y2)})

    return canvas

def mouse_callback_buttons(event, x, y, flags, param):
    global selected_label, save_triggered
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in buttons:
            x1, y1, x2, y2 = button['rect']
            if x1 <= x <= x2 and y1 <= y <= y2:
                if button['key'] == 'save':
                    save_triggered = True
                else:
                    selected_label = button['key']

image_files = sorted([f for f in os.listdir(PICTURE_FOLDER) if f.endswith(".png")], key=lambda x: int(x.split('.')[0]))

for filename in image_files:
    img_id = int(filename.split(".")[0])
    row_index = df.index[df['id'] == img_id].tolist()
    if not row_index:
        print(f"No CSV entry for image {filename}")
        continue

    idx = row_index[0]
    if pd.notna(df.loc[idx, 'label']) and df.loc[idx, 'label'].strip() != "":
        continue

    img_path = os.path.join(PICTURE_FOLDER, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image {filename}")
        continue

    selected_label = None
    save_triggered = False

    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Labels", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Labels", mouse_callback_buttons)

    while True:
        button_img = draw_buttons_window()
        cv2.imshow("Image", img)
        cv2.imshow("Labels", button_img)
        key = cv2.waitKey(50)

        if save_triggered:
            if selected_label:
                df.at[idx, 'label'] = selected_label
                print(f"Labeled {filename} as '{LABELS[selected_label]}'")
            else:
                print(f"No label selected for {filename}, skipping...")
            break

    cv2.destroyWindow("Image")
    cv2.destroyWindow("Labels")

df.to_csv(CSV_FILE, index=False)
print("All images labeled and CSV saved.")
