import cv2
import numpy as np

def FeatureMatching(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30], None, flags = 2)
    return img3, kp1, des1

img1 = cv2.imread('Database/fig1.png')
img2 = cv2.imread('Database/fig2.png')
img3 = cv2.imread('Database/fig3.png')
img4 = cv2.imread('Database/fig4.png')

out1_2,kp1,des1 = FeatureMatching(img1,img2)
print (kp1)
print(des1)
# out2_3 = FeatureMatching(img2,img3)
# out3_4 = FeatureMatching(img3,img4)
cv2.imshow('out1_2',out1_2)
# cv2.imshow('out2_3',out2_3)
# cv2.imshow('out3_4',out3_4)

cv2.waitKey(0)
cv2.destroyAllWindows()