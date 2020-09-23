
import rospy
import ros_numpy
import numpy as np
import copy
import json
import os
import sys
import time 
import argparse
import glob
import math

from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from kitti_player_tracking.msg import matrices
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField, Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import MarkerArray, Marker
from t4ac_perception_msgs.msg import bev_obstacle, bev_obstacles_list, bev_obstacle_3D, bev_obstacles_3D_list, Object_kitti_list, Object_kitti
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
import cv2

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


def getCalibfromROS(calibmsg):

    
    Extrinsic = np.asarray(calibmsg.T).reshape([3,4])
    Extrinsic = np.vstack((Extrinsic, np.array([0,0,0,1]) ))
    Rectify   = np.asarray(calibmsg.R).reshape([3,3])
    Rectify   = np.hstack((Rectify, np.zeros((3,1)) ))
    Rectify   = np.vstack((Rectify, np.array([0,0,0,1])))
    Intrinsic = np.asarray(calibmsg.P).reshape([3,4])
    #Intrinsic = np.vstack((Intrinsic, np.ones(4)))
    #M = np.matmul(Intrinsic, Extrinsic)
    M = np.matmul(np.matmul(Intrinsic, Rectify), Extrinsic)
    '''
    P = np.asarray(calibmsg.P).reshape([3,4])
    P2 = np.concatenate([P, np.array([[0., 0., 0., 1.]])], axis=0)
    R = np.asarray(calibmsg.R).reshape([3,3])
    R = np.hstack((R, np.zeros((3,1)) ))
    R0_4x4 = np.vstack((R, np.array([0,0,0,1])))
    T = np.asarray(calibmsg.T).reshape([3,4])
    V2C_4x4 = np.concatenate([T, np.array([[0., 0., 0., 1.]])], axis=0)
    calib_dict = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}
    calib = get_calib(calib_dict)
    '''
    return M

def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = np.matmul(proj_mat, points)
    points[:2, :] /= points[2, :]
    return points[:2, :]


def rslidar_callback(msg, calibmsg, imagemsg):

    print("CALLBACK")
    t_t = time.time()

    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(imagemsg, desired_encoding='passthrough')

    
    #calib = getCalibfromFile(calib_file)
    proj_velo2cam2 = getCalibfromROS(calibmsg)

    msg_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    np_p = get_xyz_points(msg_cloud, True)
    #np_p = np.array([[10,0,-2],
    #                 [10,1,-2]])
    pts_velo = np.copy(np_p[:, :3])
    print("msg_cloud", msg_cloud)

    pts_2d = project_to_image(pts_velo.transpose(), proj_velo2cam2)
    
    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_velo = pts_velo[inds, :]
    imgfov_pc_velo = np.hstack((imgfov_pc_velo, np.ones((imgfov_pc_velo.shape[0], 1))))
    imgfov_pc_cam2 = np.matmul(proj_velo2cam2, imgfov_pc_velo.transpose())
    print("imgfov_pc_pixel", imgfov_pc_pixel)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        #print("depth", depth)
        try:
            color = cmap[int(640.0 / depth), :]
            cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                     2, color=tuple(color), thickness=-1)
        except:
            continue

    image_message = bridge.cv2_to_imgmsg(img, encoding="passthrough")
    pubimage.publish(image_message)


    print("total callback time: ", time.time() - t_t)

   
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

    
    image_shape  = np.asarray([1242, 375])
    img_width  = image_shape[0]
    img_height = image_shape[1]

    
    sub_       = Subscriber(sub_lidar_topic[2],                PointCloud2)
    sub_calib  = Subscriber("/kitti_player/matrices/matriz",   matrices)
    sub_img    = Subscriber("/kitti_player/color/left/image_rect",   Image)
    ats = ApproximateTimeSynchronizer([sub_, sub_calib, sub_img], queue_size=5, slop=1)
    #ts = TimeSynchronizer([sub_, sub_calib, sub_img], queue_size=5)
    ats.registerCallback(rslidar_callback)

    pubimage = rospy.Publisher("/lidar_to_image", Image, queue_size=10)

    print("[+] PCDet ros_node has started!")    
    rospy.spin()
