import os
import csv
import cv2
import pandas as pd

# Constants
PICTURE_FOLDER = "pictures/"
CSV_FILE = "bboxes.csv"
LABELS = {'a': 'car', 'c': 'truck', 't': 'tram', 'p': 'person', 'b': 'bike', 'n': 'nothing'}

# Load CSV
df = pd.read_csv(CSV_FILE)

# Iterate over image files in folder
for filename in sorted(os.listdir(PICTURE_FOLDER), key=lambda x: int(x.split('.')[0])):
    if not filename.endswith(".png"):
        continue

    img_id = int(filename.split(".")[0])

    # Find the corresponding row in the CSV
    row_index = df.index[df['id'] == img_id].tolist()
    if not row_index:
        print(f"No CSV entry for image {filename}")
        continue

    idx = row_index[0]

    # Skip if label already filled
    if pd.notna(df.loc[idx, 'label']) and df.loc[idx, 'label'].strip() != "":
        continue

    # Show image
    img_path = os.path.join(PICTURE_FOLDER, filename)
    img = cv2.imread(img_path)
    cv2.imshow(f"Label {filename}", img)
    cv2.waitKey(0)  # Ensure the image window appears

    # Get label from user
    user_input = input(f"Label for {filename} (a,c,t,p,b,n): ").strip().lower()
    cv2.destroyAllWindows()

    # Optionally validate input here
    if all(c in LABELS for c in user_input):
        df.at[idx, 'label'] = user_input
    else:
        print(f"Invalid label '{user_input}' for {filename}, skipping...")

# Save updated CSV
df.to_csv(CSV_FILE, index=False)
print("Labeling complete. CSV updated.")
