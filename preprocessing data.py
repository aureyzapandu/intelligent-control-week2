import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset dari file CSV
color_data = pd.read_csv('colors.csv')

X = color_data[['R', 'G', 'B']].values
y = color_data['ColorName'].values

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Training model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi model
y_pred = knn.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ambil pixel tengah gambar
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    pixel_center = frame[center_y, center_x]
    pixel_center_scaled = scaler.transform([pixel_center])

    # Prediksi warna
    color_pred = knn.predict(pixel_center_scaled)[0]
    cv2.putText(frame, f'Color: {color_pred}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Tambahkan bounding box di sekitar pixel tengah
    box_size = 60  # Ukuran bounding box (bisa diubah sesuai kebutuhan)
    top_left = (center_x - box_size, center_y - box_size)
    bottom_right = (center_x + box_size, center_y + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
