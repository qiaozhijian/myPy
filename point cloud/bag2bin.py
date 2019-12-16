import os
import sys
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import pcl


class Converter:

    def __init__(self, topic, folder):
        self.subscriber_ = rospy.Subscriber(topic, PointCloud2, self.on_topic_received, queue_size=10)
        print('Converter: subscribed ' + topic)
        self.topic_name_ = topic
        self.folder_ = folder
        self.count_ = 0
        print('Converter: initialized')

    def save_lidar(self, lidar_file, lidar):
        if lidar.shape[1] == 3:
            intensity = np.ones(len(lidar)).reshape(-1, 1)
            lidar = np.hstack((lidar, intensity))
            with open(lidar_file, 'wb') as f:
                lidar.astype(np.float32).tofile(f)
                print("Converter: " + lidar_file + " saved")

    def on_topic_received(self, data):
        print(rospy.get_caller_id() + " " + self.topic_name_ + " received")
        points = pcl.pointcloud2_to_xyz_array(data, dtype=np.float32)
        print("Converter: " + str(len(points)) + " points converted")
        lidar_file = "lidar" + str(self.count_) + ".bin"
        lidar_file = os.path.join(self.folder_, lidar_file)
        self.save_lidar(lidar_file, points)
        self.count_ += 1


if __name__ == '__main__':
    topic = sys.argv[1]
    folder = sys.argv[2]
    converter = Converter(topic, folder)
    rospy.init_node('rosbag2bin', anonymous=True)
    print('rosbag2bin:  start')
    rospy.spin()
    print('rosbag2bin: finished spinning')