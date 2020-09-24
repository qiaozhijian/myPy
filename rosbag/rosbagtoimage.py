# # coding:utf-8
# #!/usr/bin/python
#
# # Extract images from a bag file.
#
# #PKG = 'beginner_tutorials'
# import roslib;   #roslib.load_manifest(PKG)
# import rosbag
# import rospy
# import cv2
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# from cv_bridge import CvBridgeError
#
# # Reading bag filename from command line or roslaunch parameter.
# import os
# import sys
# #strbag='2019-08-21-13-58-34_6'
# strbag='2019-08-21-13-40-32_0'
# #strbag='2019-08-21-13-55-34_5'
#
# back_path = '/media/zhijian/My Book/hactl_data/'+strbag+'/back/'
# forward_path= '/media/zhijian/My Book/hactl_data/'+strbag+'/front/'
#
# class ImageCreator():
#
#
#     def __init__(self):
#         self.bridge = CvBridge()
#         with rosbag.Bag('/media/zhijian/My Book/hactl_data/'+strbag+'.bag', 'r') as bag:  #要读取的bag文件；
#             for topic,msg,t in bag.read_messages():
#                 if topic == "/spinnaker/backview/image_raw": #图像的topic；
#                         try:
#                             cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
#                         except CvBridgeError as e:
#                             print e
#                         timestr = "%.6f" %  msg.header.stamp.to_sec()
#                         #%.6f表示小数点后带有6位，可根据精确度需要修改；
#                         image_name = timestr+ ".png" #图像命名：时间戳.png
#                         cv2.imwrite(back_path + image_name, cv_image)  #保存；
#                 elif topic == "/spinnaker/frontview/image_raw": #图像的topic；
#                         try:
#                             cv_image = self.bridge.imgmsg_to_cv2(msg,"16UC1")
#                         except CvBridgeError as e:
#                             print e
#                         timestr = "%.6f" %  msg.header.stamp.to_sec()
#                         #%.6f表示小数点后带有6位，可根据精确度需要修改；
#                         image_name = timestr+ ".png" #图像命名：时间戳.png
#                         cv2.imwrite(forward_path + image_name, cv_image)  #保存；
#
# if __name__ == '__main__':
#
#     #rospy.init_node(PKG)
#
#     try:
#         image_creator = ImageCreator()
#     except rospy.ROSInterruptException:
#         pass
