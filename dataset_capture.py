import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

user_id = input("Masukkan ID/Nama Pengguna: ").strip().lower()
save_dir = f'dataset/{user_id}'
os.makedirs(save_dir, exist_ok=True)

print("[INFO] Mulai ambil gambar. Tekan 'q' untuk keluar.")
count = 0
total = 480  # ambil 60 gambar

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        cv2.imwrite(f"{save_dir}/{str(count)}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{count}/{total}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Capture Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= total:
        break

cam.release()
cv2.destroyAllWindows()
print("[INFO] Pengambilan selesai.")
