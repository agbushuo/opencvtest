import cv2 as cv
import numpy as np
filename = "opencv_logo.jpg"
image = cv.imread(filename)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gary = np.float32(gray)
dst = cv.cornerHarris(gary,2,3,0.06)
dst = cv.dilate(dst,None)
ret,dst = cv.threshold(dst,0.01*dst.max(),255,0)
dst= np.uint8(dst)
ret,labels,stats,centroids =cv.connectedComponentsWithStats(dst)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,1000,0.0001)
corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
res = np.hstack((centroids,corners))
res = np.int32(res)
image[res[:,1],res[:,0]]=[0,0,255]
image[res[:,3],res[:,2]]=[0,255,0]
cv.imwrite("subpixel7.png",image)
cv.imshow("dst",image)
cv.waitKey()
cv.destroyAllWindows()
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
