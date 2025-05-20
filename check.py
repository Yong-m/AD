import cv2

cap = cv2.VideoCapture(r"C:\Users\User\PycharmProjects\AD\data\input.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(frame.shape)  # (480, 848, 3) 이어야 정상
cap.release()
