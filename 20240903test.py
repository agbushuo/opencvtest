import numpy as np
import cv2 as cv
# 回调函数（空函数，不执行操作）
def nothing(x):
    pass
# 创建一个黑色的画布
img = np.zeros((512, 512, 3), np.uint8)
cv.namedWindow('image')
# 创建颜色调节的 Trackbars
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)
# 用于存储鼠标点击状态的变量
drawing = False
ix, iy = -1, -1
mode = True
# 定义鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode
    # 按下左键开始绘图
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (b, g, r), -1)
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv.rectangle(img, (ix, iy), (x, y), (b, g, r), -1)
            else:
                cv.circle(img, (x, y), 5, (b, g, r), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv.rectangle(img, (ix, iy), (x, y), (b, g, r), -1)
        else:
            cv.circle(img, (x, y), 5, (b, g, r), -1)
# 将鼠标回调函数绑定到窗口
cv.setMouseCallback('image', draw_circle)
while True:
    # 显示图像
    cv.imshow('image', img)
    # 获取当前 Trackbars 的值
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    # 按 'ESC' 键退出
    k = cv.waitKey(1) & 0xff
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break
# 释放资源并关闭窗口
cv.destroyAllWindows()
