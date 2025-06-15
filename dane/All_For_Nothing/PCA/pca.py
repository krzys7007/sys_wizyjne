import pandas as pd
import cv2
import os
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

pca = PCA(n_components=2)
X_train_full, X_test_full, y_train, y_test, df_train, df_test = train_test_split(
    X_scaled, y, df, test_size=0.3, random_state=42, stratify=(y != "n")
)

X_train_pca = pca.fit_transform(X_train_full)
X_test_pca = pca.transform(X_test_full)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)

predictions = knn.predict(X_test_pca)
df_test = df_test.copy()
df_test["predicted_label"] = predictions

image_dir = "pictures"

for _, row in df_test.iterrows():
    img_id = int(row["id"])
    predicted_label = str(row["predicted_label"])
    img_path = os.path.join(image_dir, f"{img_id}.png")

    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found: {img_path}")
        continue

    cv2.putText(image,
                predicted_label,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3)

    cv2.imshow("Predicted Label", image)
    print(f"ID: {img_id} | Predicted label: {predicted_label}")
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
