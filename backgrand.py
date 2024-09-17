from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='影像 ')
parser.add_argument('--input', help='Path to a video or a sequence of image.', default='output.avi')
parser.add_argument('--algo', help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('不能打开' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgmask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('frame', frame)
    cv.imshow('FG Mask', fgmask)

    keyword = cv.waitKey(30)
    if keyword == ord("q") or keyword == 27:
        break

cv.destroyAllWindows()
