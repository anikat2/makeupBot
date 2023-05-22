import cv2
import dlib
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# pre-trained facial landmark model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
def avgcolor(image, region):
    x, y, w, h = region
    face_image = image[y:y+h, x:x+w]
    average_color = np.mean(face_image, axis=(0, 1))
    return average_color.astype(int)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        average_color = avgcolor(frame, (x, y, w, h))
        print("Average Face Color (BGR):", average_color)
        landmarks = predictor(gray, face_rect)

        for i in range(68):
            x_lm = landmarks.part(i).x
            y_lm = landmarks.part(i).y
            cv2.circle(frame, (x_lm, y_lm), 2, (0, 0, 255), -1)

        lm1 = (landmarks.part(30).x, landmarks.part(30).y)  # Nose tip
        lm2 = (landmarks.part(8).x, landmarks.part(8).y)    # Chin
        lm3 = (landmarks.part(36).x, landmarks.part(36).y)  # Left eye corner
        lm4 = (landmarks.part(45).x, landmarks.part(45).y)  # Right eye corner

        dist1 = math.sqrt((lm2[0] - lm1[0])**2 + (lm2[1] - lm1[1])**2)
        dist2 = math.sqrt((lm4[0] - lm3[0])**2 + (lm4[1] - lm3[1])**2)

        if dist1 < dist2:
            if dist1 < 0.9 * dist2:
                face_shape = "round"
            else:
                face_shape = "oval"
        else:
            if dist1 > 1.1 * dist2:
                face_shape = "square"
            else:
                face_shape = "heart"

        if "round" in face_shape:   
            makeup = "apply highlighter on places near the nose, apply blush from under eye to ear"
        elif "oval" in face_shape:
            makeup = "blush on hairline and cheeks and jawline and ears"
        elif "square" in face_shape: 
            makeup =  "contour at chin and jawline, smokey eye, bold lip"
        else:
            makeup = "bronzer on chin and temples, rosy blush, bright and glossy lips, and light eyeshadow"

        cv2.putText(frame, f"Face Shape: {face_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{makeup}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Face Shape Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

