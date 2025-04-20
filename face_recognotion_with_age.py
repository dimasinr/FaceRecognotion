import cv2
import pickle
import numpy as np

# ==== Load model pengenalan wajah ====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

# ==== Load model estimasi umur dan gender ====
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
gender_list = ['Male', 'Female']

# Fungsi untuk prediksi umur
def predict_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    age_net.setInput(blob)
    preds = age_net.forward()
    return age_list[preds[0].argmax()]

# Fungsi untuk prediksi gender
def predict_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    return gender_list[preds[0].argmax()]

# ==== Jalankan webcam ====
cam = cv2.VideoCapture(0)
threshold = 60  # confidence threshold

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))

        label, confidence = recognizer.predict(roi_gray)
        name = "Unknown"
        color = (0, 0, 255)

        if confidence < threshold:
            name = labels.get(label, "Unknown")
            color = (0, 255, 0)

        # Estimasi usia
        face_rgb = frame[y:y+h, x:x+w]
        face_rgb = cv2.resize(face_rgb, (227, 227))
        age = predict_age(face_rgb)
        
        # Estimasi gender
        gender = predict_gender(face_rgb)

        # Menampilkan hasil
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{name} ({round(confidence, 2)}) | Age: {age} | Gender: {gender}"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition + Age & Gender Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
