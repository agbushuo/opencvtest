import cv2 as cv
import numpy as np
cap = cv.VideoCapture(2)

while(1):
    _,frame = cap.read()
    frame = frame[:,::-1]
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_red = np.array([30,30,20])
    upper_red = np.array([230,230,200])

    mask = cv.inRange(hsv,lower_red, upper_red)
    res= cv.bitwise_and(frame,frame,mask=mask)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k ==27:
        break
cv.destroyAllWindows()
