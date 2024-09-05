import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

cap = cv.VideoCapture(2)
plt.ion()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
exit_flag=False
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
    ax2.set_title('色彩直方图')
    plt.pause(0.01)
    if cv.waitKey(5) & 0xff==27:
        exit_flag = True
        break
    if exit_flag:
        plt.close()  # 关闭 matplotlib 窗口
        cap.release()  # 释放摄像头
        cv.destroyAllWindows()  # 关闭 OpenCV 窗口
        plt.ioff()  # 关闭 matplotlib 的交互模式

