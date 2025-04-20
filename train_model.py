import cv2
import os
import numpy as np
import pickle

data_dir = 'dataset'
faces, labels = [], []
label_map = {}
current_label = 0
image_extensions = ('.jpg', '.jpeg', '.png')

for user in os.listdir(data_dir):
    path = os.path.join(data_dir, user)
    if os.path.isdir(path):
        label_map[current_label] = user
        for img_file in os.listdir(path):
            if img_file.lower().endswith(image_extensions):
                img_path = os.path.join(path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.equalizeHist(img)
                    faces.append(img)
                    labels.append(current_label)
        current_label += 1

print(f"[INFO] Total user: {len(label_map)} | Total gambar: {len(faces)}")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)
recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("[INFO] Training selesai. Model disimpan.")
