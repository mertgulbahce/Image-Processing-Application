import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    frame1 = np.array(255 * (frame / 255) ** 2.2, dtype='uint8')

    cv2.imshow('Original Video', frame)
    cv2.imshow('Gamma(2.2) Transformation', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()