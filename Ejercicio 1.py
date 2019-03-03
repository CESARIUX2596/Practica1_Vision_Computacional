import cv2
import numpy as np
cap = cv2.VideoCapture("Book.mp4")
reference = cv2.imread('book.png')
reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
# cap = cv2.VideoCapture(0)

def Detector(reference, video):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(video, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(reference,kp1,video,kp2,matches[:30], None, flags = 2)

while (cap.isOpened()):
    try:

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        taco = Detector(reference,video)
        # cv2.imshow('frame',frame)
        cv2.imshow('gray',taco)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
cv2.destroyAllWindows()