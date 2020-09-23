

#!/usr/bin/env python
# license removed for brevity
import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, Pose, Point, Quaternion
from nav_msgs.msg import Path, Odometry

def talker():
    pub  = rospy.Publisher("/mapping_planning/waypoints", Path, queue_size=10)
    pub2 = rospy.Publisher("/localization/pose1", Odometry, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    print("Publishing empty waypoints")

    while not rospy.is_shutdown():

        #Create waypoint msg
        posestamped = PoseStamped()
        posestamped.header.stamp = rospy.Time.now()
        waypoints = Path()
        waypoints.header.stamp = rospy.Time.now()
        waypoints.poses.append(posestamped)
        waypoints.poses.append(posestamped)

        #Create odometry msg
        odometry_msg = Odometry()
        odometry_msg.header.stamp = rospy.Time.now()
        odometry_msg.header.frame_id = "ego_vehicle"
        odometry_msg.child_frame_id = "map"
        pose1 = PoseWithCovariance()
        pose2 = Pose()

        point = Point()
        point.x = 5
        point.y = -45
        point.z = 0.01

        quat = Quaternion()
        quat.x = 0
        quat.y = 0
        quat.z = 0.7071
        quat.w = 0.7071

        pose2.position    = point
        pose2.orientation = quat
        pose1.pose        = pose2
        odometry_msg.pose = pose1
   
        #Publish msgs
        pub.publish(waypoints)
        pub2.publish(odometry_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
