import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bane mask image downloaded from Deviant Art by artist Billelbe
bane_mask = cv2.imread('images/the_dark_knight_rises___bane_s_mask_png_by_billelbe_d6x1ah2-pre.png')

# Uses first camera available
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces from the gray-converted image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Gets the values needed to build a rectangular bounding region around candidate faces
    for x, y, width, height in faces:

        # Manually adjusted to allow the bane mask to fit
        x = int(1.1 * x)
        width = int(0.8 * width)

        # The region of interest
        box_frame = frame[y:y + height, x:x + width]

        # According to OpenCV docs, we should generally use cv2.INTER_AREA as the interpolation 
        # if we want to shrink an image, which is probably what we are going to do here.
        resized_bane_mask = cv2.resize(bane_mask, (width, height), interpolation=cv2.INTER_AREA)

        gray_bane_mask = cv2.cvtColor(resized_bane_mask, cv2.COLOR_BGR2GRAY)

        # 127 is the threshold value, 255 is the max value, we create a mask which will be white
        # in the black areas so that we can AND it with the frame image to get a region for applying
        # the bane image.
        _, ba_mask = cv2.threshold(gray_bane_mask, 50, 255, cv2.THRESH_BINARY_INV)

        # get the part of the face that will fit with the mask
        bane_face_mask = cv2.bitwise_and(resized_bane_mask, resized_bane_mask, ba_mask)

        # Now we get the rest of the region of interest
        _, ba_mask = cv2.threshold(gray_bane_mask, 50, 255, cv2.THRESH_BINARY)

        bane_frame_mask = cv2.bitwise_and(box_frame, box_frame, ba_mask)

        # Combine the mask and the rest of the face region 
        if cv2.waitKey(1) & 0xFF == ord('n'):
            frame[y:y + height, x:x + width] = cv2.add(bane_face_mask, bane_frame_mask)

    cv2.imshow('Bane Mask!', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()