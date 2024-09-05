import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.asns-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
cap = cv.VideoCapture(2)
plt.ion()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
while(1):
    ret,img=cap.read()
    if not ret:
        print('无法获取摄像头帧')
        break
    img=img[:,::1]
    ax1.clear()
    ax1.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    ax1.set_title('摄像头')
    ax2.clear()
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        ax2.plot(histr,color=col)
    ax2.set_xlim([0,256])
    ax2.set_title('直方图')
    plt.pause(0.01)

    if cv.waitKey(5) & 0xff==27:
        plt.close()
        break
cap.release()
cv.destroyAllWindows()
plt.ioff()
