#!/usr/bin/env python
import roslib
import sys
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

import math
from move_robot import MoveTurtlebot3

Kp = 0.8 

class LineFollower(object):
    def __init__(self):
        self.bridge_object = CvBridge()
	self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.camera_callback)
        self.moveTurtlebot3_object = MoveTurtlebot3()
        self.twist_object = Twist()
        # Give some initial velocity to make the robot move 
        self.twist_object.linear.x = 0.1
        self.twist_object.angular.z = 0

    def angular_error(self, ref, target):
        error = 0

        # calc distance between ref point and moment point
        distance = math.sqrt((ref[0] - target[0])**2 + (ref[1] - target[1])**2)
        error = math.asin((ref[0] - target[0])/distance)

        return error

    def p_controller(self, error):
        global Kp
        self.twist_object.angular.z = Kp * error

        if abs(error) > 0.1:
            self.twist_object.linear.x = 0
        else:
            self.twist_object.linear.x = 0.1

        return self.twist_object.angular.z

    def camera_callback(self,data):
        try:
	    # We select bgr8 because its the OpneCV encoding by default
	    cv_image = self.bridge_object.imgmsg_to_cv2(data,desired_encoding="bgr8")
	except CvBridgeError as e:
	    print(e)
        
        # manipulate image
        # Crop image
        height, width, channels = cv_image.shape
        crop_img = cv_image[(height)/2+100:(height)/2+120][1:width]

        # Convert to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range of color yellow
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([50,255,255])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask2 = cv2.inRange(hsv2, lower_yellow, upper_yellow)

        # Calculate centroid of the blob (cx, cy) of binary image using ImageMoments
        m = cv2.moments(mask, False)
        m2 = cv2.moments(mask2, False)
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
            cx2, cy2 = m2['m10']/m2['m00'], m2['m01']/m2['m00']
        except ZeroDivisionError:
            cy, cx = height/2, width/2
            cy2, cx2 = height/2, width/2 

        refx = width/2
        refy = height
        #print("cam ref point: " + str(refx) + ", " + str(refy))
        #print(cx, cy)

        # Calculate angular error and feed to p controller
        self.p_controller(self.angular_error([refx, refy], [cx, cy]))
        #print(self.twist_object.angular.z)

        # Draw the centroid in the resultant image
        cv2.circle(mask,(int(cx), int(cy)), 10,(0,0,255),-1)
        #cv2.circle(cv_image,(int(cx2), int(cy2)), 10,(0,0,255),-1)

        # Move the robot
        self.moveTurtlebot3_object.move_robot(self.twist_object)

        # Display image and mask
        cv2.imshow("Original", cv_image)
        cv2.imshow("MASK", mask)
        cv2.waitKey(1)

def main():
 
    rospy.init_node('line_following_node', anonymous=True)
    line_follower_object = LineFollower()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
