# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import cv2
def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi(cv2.getVersionString())
    image = cv2.imread("opencv_logo.jpg")
    print(image.shape)
    cv2.imshow("image",image)
    cv2.imshow("image0",image[:,:,0])
    cv2.imshow("image1",image[:,:,1])
    cv2.imshow("image2",image[:,:,2])
    cv2.waitKey()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
