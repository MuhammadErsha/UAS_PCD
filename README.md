# UAS_PCD
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Masukkan URL model Roboflow yang telah Anda deploy
ROBOFLOW_API_URL = "https://detect.roboflow.com/{project_id}/1"  # Ganti {project_id} dengan nama project Anda
ROBOFLOW_API_KEY = "your_api_key"  # Ganti dengan API key Anda

def detect_shoes(image_path):
    # Buka file gambar
    image = open(image_path, "rb").read()
    
    # Kirim gambar ke endpoint API Roboflow
    response = requests.post(
        f"{ROBOFLOW_API_URL}?api_key={ROBOFLOW_API_KEY}",
        files={"file": image}
    )
    
    # Parse JSON response
    response_data = response.json()
    
    if "predictions" not in response_data:
        print("Tidak ada objek yang terdeteksi.")
        return

    # Load gambar untuk visualisasi
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Visualisasi bounding box
    for prediction in response_data["predictions"]:
        x, y, width, height = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
        confidence = prediction["confidence"]
        
        # Gambar bounding box
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        color = (255, 0, 0)  # Warna merah
        thickness = 2
        cv2.rectangle(img, start_point, end_point, color, thickness)
        
        # Tambahkan label confidence
        label = f"Shoe: {confidence:.2f}"
        cv2.putText(img, label, start_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Tampilkan gambar
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# Uji deteksi pada gambar
detect_shoes("path/to/your/image.jpg")  # Ganti dengan path gambar Anda
