import cv2
import numpy as np
cap =cv2.VideoCapture("Book.mp4")
# cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    try:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured = cv2.GaussianBlur(gray,(13,13),10)
        dst = cv2.cornerHarris(blured,2,7,0.05)
        # dst = cv2.dilate(dst,None)
        frame[dst>0.01*dst.max()]=[0,255,0]
        # cv2.imshow('blured',blured)
        cv2.imshow('frame',frame)
        # cv2.imshow('gray',gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
cv2.destroyAllWindows()