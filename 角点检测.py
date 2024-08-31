import cv2 as cv
import numpy as np
filename = "opencv_logo.jpg"
image = cv.imread(filename)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
dst = cv.dilate(dst,None)
image[dst>0.01*dst.max()] = [0,0,255]

cv.imshow("dst",image)
cv.waitKey()
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()