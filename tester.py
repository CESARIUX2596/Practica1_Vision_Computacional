import cv2
import numpy as np

detector = cv2.xfeatures2d.SIFT_create()

flannParam = dict(algorithm = 0, tree = 5)
searchParam = dict(checks = 50)
flann = cv2.FlannBasedMatcher(flannParam,searchParam)

sample = cv2.imread("book.png",0)
trainKP,trainDesc = detector.detectAndCompute(sample,None)

cap = cv2.VideoCapture("Book.mp4")
while (cap.isOpened()):
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameKP,frameDesc = detector.detectAndCompute(gray,None)
        matches = flann.knnMatch(frameDesc,trainDesc,k = 2)

        good = []
        for m,n in matches:
            if(m.distance < 0.7 * n.distance):
                good.append(m)
        if (len(good) > 10):
            tp = []
            qp = []
            for m in good:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(frameKP[m.franeIdx].pt)
            tp, qp = np.float32((tp,qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = sample.shape
            trainBorder = np.float32([[[0,0],[0,h-1], [w-1,h-1],[w-1,0]]])
            frameborder = cv2.perspectiveTransform(trainBorder, H)
            cv2.polylines(frame,[np.int32(frameBorder)], True, (0,255,0), 5)
        else:
            print("Matches not found")
        cv2.imshow(result,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
cv2.destroyAllWindows()