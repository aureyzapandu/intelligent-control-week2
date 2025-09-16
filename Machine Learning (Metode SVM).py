import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from collections import Counter

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

# Training model SVM
svm_model = SVC(kernel='rbf', C=1, gamma='scale')
svm_model.fit(X_train, y_train)

# Evaluasi model (hanya di terminal, tidak ditampilkan di frame)
y_pred = svm_model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Uji (SVM):", model_accuracy)

# Integrasi dengan webcam
cap = cv2.VideoCapture(0)

color_counter = Counter()
total_frames = 0

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
    color_pred = svm_model.predict(pixel_center_scaled)[0]
    color_counter[color_pred] += 1
    total_frames += 1

    # Hitung persentase dominasi real-time
    if total_frames > 0:
        dominant_color = color_counter.most_common(1)[0]
        dominant_count = dominant_color[1]
        dominant_percent = (dominant_count / total_frames) * 100
    else:
        dominant_percent = 0

    # Tampilkan di frame
    cv2.putText(frame, f'Color: {color_pred}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f'Akurasi: {dominant_percent:.2f}%', (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Bounding box di pixel tengah
    box_size = 50
    top_left = (center_x - box_size, center_y - box_size)
    bottom_right = (center_x + box_size, center_y + box_size)
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Statistik prediksi di terminal setiap 30 frame
    if total_frames % 30 == 0:
        print("Statistik prediksi warna ({} frame):".format(total_frames))
        for color, count in color_counter.items():
            print(f"{color}: {count} ({count/total_frames*100:.2f}%)")
        print("-" * 30)

cap.release()
cv2.destroyAllWindows()
