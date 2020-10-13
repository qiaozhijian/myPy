import os, sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
 
cap = cv2.VideoCapture(2)
# ret = cap.set(3, 320)
# ret = cap.set(4, 240)
# 设置摄像头分辨率
width = 1280
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    left_img = frame[:, 0:width//2, :]
    right_img = frame[:, width//2:width, :]
    if ret:
        # 显示两幅图片合成的图片
        #cv2.imshow('img', frame)
        # 显示左摄像头视图
        cv2.imshow('left', frame)
        # cv2.imshow('left', left_img)
        # 显示右摄像头视图
        # cv2.imshow('right', right_img)
        cv2.imwrite('./img/' + str(i) + '.jpg', frame)
    i=i+1
    key = cv2.waitKey(delay=30)
    if key == ord('t'):
        cv2.imwrite('./img/test' + str(i) + '.jpg', frame)#
        i += 1
    if key == ord("q") or key == 27:
        break
 
cap.release()
cv2.destroyAllWindows()

