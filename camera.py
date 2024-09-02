import numpy as np
import cv2 as cv
cap =cv.VideoCapture(2)
fourcc = cv.VideoWriter_fourcc('X','V','I','D')
out = cv.VideoWriter("output.avi",fourcc,30,(640,480))

if not cap.isOpened():
    print("Cannot open canera")
    exit()
while True:
    ret,frame =cap.read()
    if not ret:
        print("Can't receive frame （stream end？）.正在退出...")
    frame=frame[:,::-1]
    # frame=cv.flip(frame,1)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    out.write(frame)
    cv.imshow("frame",gray)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
out.release()
cv.destroyAllWindows()