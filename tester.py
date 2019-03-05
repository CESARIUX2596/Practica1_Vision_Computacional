import cv2
import numpy as np

detector = cv2.xfeatures2d.SIFT_create()

flannParam = dict(algorithm = 0, tree = 5)
searchParam = dict(checks = 10)
flann = cv2.FlannBasedMatcher(flannParam,searchParam)

bf = cv2.BFMatcher()
sample = cv2.imread("book.png")
sample = cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
trainKP,trainDesc = detector.detectAndCompute(sample,None)

cap = cv2.VideoCapture("Book.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
frameBoard= None
while (cap.isOpened()):
    try:

        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frameKP,frameDesc = detector.detectAndCompute(gray,None)
        matches = bf.knnMatch(frameDesc,trainDesc,k = 2)

        good = []
        for m,n in matches:
            if(m.distance < 0.75 * n.distance):
                good.append(m)
        if (len(good) > 10):
            tp = []
            qp = []
            for m in good:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(frameKP[m.queryIdx].pt)
            tp, qp = np.float32((tp,qp))
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 2.0)
            h, w = sample.shape
            trainBorder = np.float32([[[0,0],[0,h-1], [w-1,h-1],[w-1,0]]])
            frameBorder = cv2.perspectiveTransform(trainBorder, H)
        if frameBorder is not None:
            cv2.polylines(frame,[np.int32(frameBorder)], True, (0,255,0), 3)
##        cv2.imshow("result.mp4",frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
