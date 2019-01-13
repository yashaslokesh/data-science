import cv2
import numpy as np

# I used the 'pip install opencv-python' version of opencv, so cv2.data.haarcascades is a shortcut 
# to the data folder
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Uses first camera
cap = cv2.VideoCapture(0)

# Press q to quit
while True:
    # Capture each frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, width, height in faces:

        cv2.rectangle(frame, 
                     (x, y), 
                     (x + width, y + height), 
                     (255, 0, 0), 
                     2)

        # the boxes are regions of interest (ROI)
        box_gray = gray[y:y + height, x:x + width]
        box_frame = frame[y:y + height, x:x + width]

        eyes = eye_cascade.detectMultiScale(box_gray)

        for eye_x, eye_y, eye_width, eye_height in eyes:
            cv2.rectangle(box_frame, 
                         (eye_x, eye_y), 
                         (eye_x + eye_width, eye_y + eye_height), 
                         (0, 255, 0), 
                         2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()