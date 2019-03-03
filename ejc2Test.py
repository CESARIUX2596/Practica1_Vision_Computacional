import cv2
import numpy as np

def FeatureMatching(img2, img1):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2) 
    good = []
    for m in matches:
        if m[0].distance < 0.5*m[1].distance:         
            good.append(m)
    matches = np.asarray(good)
    if len(matches[:,0]) >= 4:
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    else:
        pass
    dst = cv2.warpPerspective(img1,H,(img2.shape[1] + img1.shape[1], img2.shape[0]))
    dst[0:img1.shape[0], 0:img2.shape[1]-10] = img2[:,:img2.shape[1]-10]
    return CropBlackSpace(dst)

def CropBlackSpace(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = gray>0
    gray = gray[np.ix_(mask.any(1),mask.any(0))]
    return img[:gray.shape[0],:gray.shape[1]]

img1 = cv2.imread('Database/fig1.png')
img2 = cv2.imread('Database/fig2.png')
img3 = cv2.imread('Database/fig3.png')
img4 = cv2.imread('Database/fig4.png')
out1_2 = FeatureMatching(img1,img2)
out3_4 = FeatureMatching(img3,img4)
out = FeatureMatching(out1_2,out3_4)
cv2.imwrite('out.png',out)
##cv2.imwrite('out3_4.png',out3_4)
# cv2.imshow('out2_3',out2_3)
# cv2.imshow('out3_4',out3_4)

cv2.waitKey(0)
cv2.destroyAllWindows()
