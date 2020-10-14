#!/usr/bin/env python

import rospy
import math
import numpy as np
import yaml
import sys
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
import pdb

import time

# Vehicle parameters
ANGLE_RANGE = 270           # Hokuyo 10LX has 270 degree scan.
DISTANCE_THRESHOLD = 3      # Distance threshold before collision (m)
VELOCITY = 0.5              # Maximum Velocity of the vehicle
TIME_THRESHOLD = 1          # Time threshold before collision (s)
STEERING_ANGLE = 0          # Steering angle is uncontrolled

EMERGENCY_BRAKE_ENGAGED = 1 # Time when the emergency brake is engaged

# P-Controller Parameters
kp_dist = 0    # parameter range: 0 - 1 
kp_ttc = 0     # parameter range: 0 - 1

dist_error = 0.0
time_error = float("inf")

control_mechanism = None
current_velocity = VELOCITY

pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)

def p_control(delta, k, V, norm):
    '''
    P controller implementation. 
    Input:
        delta - distance or TTC delta value
        k     - kp_dist or kp_ttc
        V     - maximum velocity
        norm  - normalizing factor
    Output: velocity
    '''

    EMERGENCY_BRAKE_ENGAGED = 1

    if delta > EMERGENCY_BRAKE_ENGAGED:
        velocity = V
    else:
        velocity = max(0, k * delta * norm)

    return velocity

def dist_control(distance):
	global kp_dist
	global VELOCITY
	global DISTANCE_THRESHOLD 
	global STEERING_ANGLE

        EMERGENCY_BRAKE_ENGAGED = 1
	# TO-DO: Calculate Distance to Collition Error
	# ---

	# ---

        # Start with max velocity
        delta = distance - DISTANCE_THRESHOLD
       
        # Apply P controller here 
        velocity = p_control(delta, kp_dist, VELOCITY, VELOCITY/EMERGENCY_BRAKE_ENGAGED)

	print("Distance before collision is = ", distance)
	print("Vehicle velocity= ", velocity)

	msg = AckermannDriveStamped()
	msg.drive.speed = velocity
	msg.drive.steering_angle = STEERING_ANGLE
	pub.publish(msg)

        return velocity

def TTC_control(distance):
	global kp_ttc
	global TIME_THRESHOLD
	global VELOCITY
	global STEERING_ANGLE
        global time_error
        global current_velocity

	# TO-DO: Calculate Time To Collision Error
	# ---

	# ---

        ttc = float("inf")

        if time_error == float("inf"):  # kick off
            velocity = VELOCITY
            ttc      = distance/velocity
            time_error = ttc
        elif current_velocity != 0 :
            ttc      = distance/current_velocity
            delta    = ttc - TIME_THRESHOLD

            # Apply P controller here 
            velocity = p_control(delta, kp_ttc, VELOCITY, VELOCITY/time_error)
        else:
            velocity = 0

        current_velocity = velocity

	print("Time to collision in seconds is = ", ttc)
	print("Vehicle velocity = ", velocity)

	msg = AckermannDriveStamped()
	msg.drive.speed = velocity
	msg.drive.steering_angle = STEERING_ANGLE
	pub.publish(msg)

        return velocity

def get_index(angle, data):
	# TO-DO: For a given angle, return the corresponding index for the data.ranges array
	# ---

	# ---
        
        # Initial value for return array index
        angle_index = 0

        # Extract parameters from lidar data packet:
        # minimum angle, maximum angle, angle increment, length of range array
        min_angle = data.angle_min
        max_angle = data.angle_max
        angle_increment = data.angle_increment
        range_len = len(data.ranges)

        # 
        values_per_degree = int(1/(180*angle_increment/math.pi))
        angle_index = (angle - int(180* min_angle/math.pi) ) * values_per_degree 
        print("Angel index: " + str(angle_index))

        return angle_index 

# Use this function to find the average distance of a range of points directly in front of the vehicle.
def get_distance(data): 
	global ANGLE_RANGE
	
	angle_front = 90   # Range of angle in the front of the vehicle we want to observe
	avg_dist = 0
	
	# Get the corresponding list of indices for given range of angles
	index_front = get_index(angle_front/2, data)
        index_front_mirror = get_index(-(angle_front/2), data)
        
	# TO-DO: Find the avg range distance
	# ---
        # Slice array to contain only front indecies
        front_range_array = data.ranges[index_front_mirror:index_front]
        # Further extract non-inf values from the array
        distance_values = [ x for x in front_range_array if x != float("inf") ]
        avg_dist = sum(distance_values)/len(distance_values) 

        # ---
	
	print("Average Distance = ", avg_dist)

	return avg_dist

def callback(data):

	# TO-DO: Complete the Callback. Get the distance and input it into the controller
	# ---

	# ---
        global control_mechanism

        distance = get_distance(data)

        if control_mechanism == "dist_ctrl": 
            velocity = dist_control(distance)
        else:
            velocity = TTC_control(distance)

        return

def main():
        global control_mechanism
        global kp_dist
        global kp_ttc

        time.sleep(5)
        print("AEB started")


        # Create control node and subscribe to scan message
        rospy.init_node('aeb',anonymous = True)
        rospy.Subscriber("/scan",LaserScan,callback)

        # Initialize parameters
        try:
            control_mechanism = rospy.get_param('~control')
        except KeyError:    # In case running standalone:
            control_mechanism = "dist_ctrl"

        try:
            k = rospy.get_param('~k')
        except KeyError:    # In case running standalone:
            k = 0.8

        if control_mechanism == "dist_ctrl":
            kp_dist = k
        else:
            kp_ttc = k

        # Let the ROS spin!
        rospy.spin()

if __name__ == '__main__':
        main()
