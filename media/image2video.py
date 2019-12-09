#import cv2
#fps = 25
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  
#videoWriter = cv2.VideoWriter('/media/zhijian/Document/grow/slam/avideo.mp4', fourcc, fps, (1280,720))  
#for i in range(1,2):
#    img = cv2.imread('/media/zhijian/Document/grow/slam/front2/tra'+str(i)+'.png')
#    img.show()
##    cv2.imshow('img', img12)
##    cv2.waitKey(1000/int(fps))
#    videoWriter.write(img)
#videoWriter.release()

import cv2
videoPath = './map2019-08-21-15-47-38_0.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter(videoPath, fourcc, 25, (512, 288))
for i in range(1,6714):
    framePath = '/media/zhijian/Document/grow/slam/pyTest/front/tra'+str(i)+'.png'
    frame = cv2.imread(framePath)
    frame = cv2.resize(frame, (512,288))
    videoWriter.write(frame)
videoWriter.release()