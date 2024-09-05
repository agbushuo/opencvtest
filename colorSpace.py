import cv2 as cv
import numpy as np
cap = cv.VideoCapture(2)

while(1):
    _,frame = cap.read()
    frame = frame[:,::-1]
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 48, 80])
    upper_red = np.array([20, 255, 255])

    mask = cv.inRange(hsv,lower_red, upper_red)
    kernel = np.ones((5,5),np.uint8)
    closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    res= cv.bitwise_and(frame,frame,mask=opening)
    cv.imshow('frame',frame)
    cv.imshow('mask',opening)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k ==27:
        break
cv.destroyAllWindows()
