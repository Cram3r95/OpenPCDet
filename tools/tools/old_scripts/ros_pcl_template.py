
import rospy
import ros_numpy
import numpy as np
import copy
import os
import sys
import time 
import argparse
import glob
import math
from pathlib import Path

from sensor_msgs.msg import PointCloud2, PointField

def get_xyz_points(cloud_array, remove_nans=True, dtype=np.float):
    '''
    '''
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]

    points = np.zeros(cloud_array.shape + (4,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']
    return points

def xyz_array_to_pointcloud2(points_sum, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points_sum.shape[0]
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
        # PointField('i', 12, PointField.FLOAT32, 1)
        ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = points_sum.shape[0]
    msg.is_dense = int(np.isfinite(points_sum).all())
    msg.data = np.asarray(points_sum, np.float32).tostring()
    return msg


def callback(msg):
    t_t = time.time()

    #ROS PointCloud2 to numpy array
    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    point_cloud = get_xyz_points(msg_cloud, True)

    #Numpy array to ROS PointCloud2
    msg_cloud = xyz_array_to_pointcloud2(point_cloud)
    
    #Publish ROS message
    pub_.publish(msg_cloud)
   
if __name__ == "__main__":

    rospy.init_node('centerpoint_ros_node')
    sub_lidar_topic = [ "/velodyne_points", 
                        "/carla/ego_vehicle/lidar/lidar1/point_cloud",
                        "/kitti_player/hdl64e", 
                        "/lidar_protector/merged_cloud", 
                        "/merged_cloud",
                        "/lidar_top", 
                        "/roi_pclouds",
                        "/livox/lidar",
                        "/SimOneSM_PointCloud_0"]

    #Subscribe to ROS topic with name sub_lidar_topic. Execute callback with every point cloud
    sub_ = rospy.Subscriber(sub_lidar_topic[2], PointCloud2, callback, queue_size=1, buff_size=2**24)
    
    #ROS message publisher. Publishes a topic named /output_pointcloud
    pub_ = rospy.Publisher("/output_pointcloud", PointCloud2, queue_size=1)

    print("[+] ros_node has started!")    
    rospy.spin()
