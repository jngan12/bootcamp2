#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Int64
from std_msgs.msg import Float32
import math

steering_angle = 0
velocity = 0.3 # (m/s)

# Self defined global variables


class laneKeeping(object):
    def __init__(self):

        self.pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)
        self.bridge_object = CvBridge()
	self.image_sub = rospy.Subscriber("/camera/zed/rgb/image_rect_color", Image, self.camera_callback)
        # Give some initial velocity to make the robot move
        self.linear  = 1
        self.angular = 0
        # Coefficients 
        self.keslope  = 0.9 # Slope error contribution to overall error
        self.kecenter = 0.1 # Distance to center contribution to overall error
        self.kp       = 1.0 # P controller coefficients

    def p_controller(self, error):
        if error != 0:
            self.linear = 0.3
            self.angular = self.kp * error
        else:
            self.linear = 1
            self.angular = 0

        return 

    # Controlls robot movement
    def movement(self, linear_velocity, angular_velocity):

        msg = AckermannDriveStamped()
        msg.drive.speed = linear_velocity
        msg.drive.steering_angle = angular_velocity 
        self.pub.publish(msg)

        return

    def canny(self, image):
        # TO-DO: Extract the canny lines
        # ---
        canny_img = cv2.Canny(image, image.max(), image.min())

        # ---
        return canny_img

    def region_of_interest(self, image):
        triangle = np.array([[(0, 480), (0, 288), (639, 288), (639, 480)]])
        # TO:DO Find the  Region of Interest
        # ---
        masked_image = image[triangle[0][1][1]:triangle[0][0][1], triangle[0][0][0]:triangle[0][2][0]]        
        # ---
        return masked_image


    def average_slope_intercept(self, image, lines):
        # TO-DO: Get and average of the left and right Hough Lines and extract the centerline. 
        # The angle between the extracted centerline and desired centerline will be the error. 
        # Use cv2.line to display the lines.
        # ---
        eslope  = 0
        ecenter = 0

        # Array storing the lane slopes 
        slope = []
        avg_center = [0, 0]

        try:
            for line in lines:
                print line
                for x1,y1,x2,y2 in line:
                    image = cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 4 )
                    # Calculate slope 
                    if x1 == x2:
                        slope.append(float("inf"))
                    else:
                        slope.append( 1.0*(y2-y1)/(x2-x1) )
                    avg_center[0] = (avg_center[0] +  (x1+x2)/2)/2
                    avg_center[1] = (avg_center[1] +  (y1+y2)/2)/2
        except:
            avg_center[0] = 320
            avg_center[1] = 0
            pass

        #cv_cropped = cv2.circle(cv_cropped, (avg_center[0], avg_center[1]), 4, (0,255,0), 4 )
       
        # Calculate slope error
        if len(slope) != 0:
            eslope = sum(slope)/len(slope)
        else:
            eslope = 0

        # Calculate normalized distance-to-center error
        ecenter = (avg_center[0] - 320) * math.pi/(180 * 640)

        # Calculate final error
        error = self.keslope * eslope + self.kecenter * ecenter

        # ---
        return error, image

    def camera_callback(self, data):
    
        global steering_angle
        global velocity

        # TO-DO: Convert the ROS Image to CV type.
        # ---
        cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="passthrough") 
        # ---

        # TO-DO: Extract the canny lines
        canny_image = self.canny(cv_image)

        # TO:DO Find the  Region of Interest
        cropped_image = self.region_of_interest(canny_image)

        # Extract the Hough Lines
        rho = 1
        theta = np.pi/180
        threshold = 10
        minLineLength = 100
        maxLineGap = 20
        lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, minLineLength, maxLineGap)
        cv_cropped = self.region_of_interest(cv_image)

        # TO-DO: Get and average of the left and right Hough Lines and extract the centerline. 
        # The angle between the extracted centerline and desired centerline will be the error. 
        # Use cv2.line to display the lines.
        error, cv_cropped = self.average_slope_intercept(cv_cropped, lines)

        # TO-DO: Implement the final controller
        # ---

        # ---
        self.p_controller(error)

        cv2.imshow('canny', canny_image)
        cv2.imshow('ROI', cv_cropped)
        cv2.waitKey(1)

        # Move the robot
        self.movement(self.linear, self.angular)
        return

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            # TO-DO: Publish the steering angle and velocity
            # ---

            # ---

            vel.header.stamp = rospy.Time.now()
            vel.header.frame_id = "base_link"

            print("Steering angle: %f" % m)
            self.pub.publish(vel)

            rate.sleep()

        self.pub.publish(vel)

        # Display converted images
        cv2.imshow('canny',canny_image)
        cv2.imshow('ROI',cropped_image)
        cv2.waitKey(1)

def main():
    rospy.init_node('lane_keeping', anonymous=True)
    lane_keeping_obj = laneKeeping()

    #bridge_object = CvBridge()

    # TO-DO: Publish and subscribe to the correct topics.
    # ---

    # ---
    #pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)
    #rospy.Subscriber("/scan", LaserScan, camera_callback)
    #vel = AckermannDriveStamped()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()



