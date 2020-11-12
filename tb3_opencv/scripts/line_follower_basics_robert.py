#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from move_robot import MoveTurtlebot3

class LineFollower(object):
    
    def __init__(self):
        
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.moveTurtlebot3_object = MoveTurtlebot3()
    
    def camera_callback(self,data):
        
        try:
	        # We select bgr8 because its the OpneCV encoding by default
	        cv_image = self.bridge_object.imgmsg_to_cv2(data,desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
	    
        #cv2.imshow("Image window", cv_image)
        #cv2.waitKey(1)

        height, width, channels = cv_image.shape
        crop_img = cv_image[(height)/2+100:(height)/2+120][1:width]

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([50,255,255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        m = cv2.moments(mask, False)
        
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
            
        except ZeroDivisionError:
            cy, cx = height/2, width/2

        cv2.circle(mask,(int(cx), int(cy)), 10,(0,0,255),-1)

        cv2.imshow("Original", cv_image)
        cv2.imshow("MASK", mask)
        cv2.waitKey(1)

        error_x = cx - width / 2
        twist_object = Twist()
        twist_object.linear.x = 0.2
        twist_object.angular.z = -error_x /100
        rospy.loginfo("ANGULAR VALUE SENT ===>"+str(twist_object.angular.z))

        self.moveTurtlebot3_object.move_robot(twist_object)

        def clean_up(self):
            self.moveTurtlebot3_object.clean_class()
            cv2.destroyAllWindows()

def main():
    rospy.init_node('line_following_node', anonymous=True)
    line_follower_object = LineFollower()

    rate = rospy.Rate(5)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
