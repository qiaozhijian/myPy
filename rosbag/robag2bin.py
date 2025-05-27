import argparse
import sys
import rospy
import roslib
import rosbag
import os
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def pc2_to_pcd(pc2_list, index, dir):
    pcl = np.asarray(pc2_list)
    print(pcl.shape)
    pcl.tofile(os.path.join(dir, str(index) + ".bin"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert select ROS PointCloud2 msg to PCD')
    parser.add_argument('-f', '--filepath', help='Input bag files', default="/home/qzj/Desktop/1.bag")
    parser.add_argument('-o', '--output_dir', help='Output directory', default="/home/qzj/Desktop/pcd")

    args = parser.parse_args()

    # Input file source validation
    if args.filepath is not None:
        filepath = args.filepath
    else:
        print('No source bag provided')
        filepath = sys.path[0]

    try:
        bag = rosbag.Bag(filepath)
    except:
        print('Unhandled exception occured when opening file')

    cloud_topics, types, bag_msg_cnt = [], [], []
    msgInfo=bag.get_type_and_topic_info()[0]
    topicInfo=bag.get_type_and_topic_info()[1]
    topics = bag.get_type_and_topic_info()[1].keys()
    topicsList=list(topics)
    for i in range(0, len(bag.get_type_and_topic_info()[1].values())):
        types.append(bag.get_type_and_topic_info()[1][topicsList[i]][0])
        bag_msg_cnt.append(bag.get_type_and_topic_info()[1][topicsList[i]][1])

    topics = zip(topics, types, bag_msg_cnt)
    index = 0

    for topic, type, count in topics:
        if type == 'sensor_msgs/PointCloud2':
            print('Topic(s) Found:')
            print('   ' + topic)
            cloud_topics.append(topic)

    for topic, msg, t in bag.read_messages(topics=cloud_topics):
        p_ = []
        gen = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        for p in gen:
            p_.append(p)

        pc2_to_pcd(p_, index, args.output_dir)
        index = index + 1