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

# Vehicle parameters
ANGLE_RANGE = 270 				# Hokuyo 10LX has 270 degrees scan
DISTANCE_RIGHT_THRESHOLD = 0.5 	# (m)
VELOCITY = 1.0 					# meters per second

# Controller parameters
kp = 0.8
kd = 0.8  # To be updated 

# Other global variables
current_error = 0.0
prev_error = 0.0
prev_timestamp = 0
current_timestamp = 0

# cycle count
cycle_count = 0

# Following selection
following = "center"

pub = None     # publisher for controlling robot movement

# implement brake
def brake(data):
    BREAKDIST = 1.5
    brake = False
    if data.ranges[get_index(0, data)] <= BREAKDIST:
        brake = True
    return brake

# Controlls robot movement
def movement( linear_velocity, angular_velocity):
        global pub

        msg = AckermannDriveStamped()
        msg.drive.speed = linear_velocity
        msg.drive.steering_angle = angular_velocity 
        pub.publish(msg)

        return

def error_history(error):
    global current_error
    global prev_error
    global prev_timestamp
    global current_timestamp

    prev_error = current_error
    prev_timestamp = current_timestamp

    if error != float("inf"):
        print "Infinite distance!"
        current_error = error

    current_timestamp = rospy.get_time()
  
def control(error, brake):
	global kp
	global kd
	global VELOCITY
        global current_error
        global prev_error
        global prev_timestamp
        global current_timestamp

        # TO-DO: Implement controller
	# ---

	# ---
        linear_velocity = VELOCITY
        steering_angle = kp * error + kd * (current_error-prev_error)/(current_timestamp-prev_timestamp) 

        # Apply brake, slow down and turn steering angel drastically
        if brake:
            linear_velocity = 0
            steering_angle = steering_angle * 10000
            print("Braking!")

	# Set maximum thresholds for steering angles
	if steering_angle > 0.25:
		steering_angle = 0.25
	elif steering_angle < -0.25:
		steering_angle = -0.25

        # make sharp turn while braking
        if brake:
            steering_angle = steering_angle * 3.5 

	print "Steering Angle is = %f" % steering_angle 

        if abs(steering_angle) > 0.2:
            linear_velocity = VELOCITY/4

	# TO-DO: Publish the message
	# ---

	# ---
        movement(linear_velocity, steering_angle)

def get_index(angle, data):
	# 	# TO-DO: For a given angle, return the corresponding index for the data.ranges array
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
        #print("Angel index: " + str(angle_index))

        return angle_index

def distance(angle_right, angle_lookahead, data):
	global ANGLE_RANGE
	global DISTANCE_RIGHT_THRESHOLD
        
	# TO-DO: Find index of the two rays, and calculate a, b, alpha and theta. Find the actual distance from the right wall.
	# ---

	# ---

        distance = 0
        distance_r  = 0

        # Calculate the angle theta and convert to radient. Get right and lookahead diatances
        theta = (angle_lookahead - angle_right) * math.pi/180
        dist_right = data.ranges[get_index(angle_right, data)]
        dist_lookahead = data.ranges[get_index(angle_lookahead, data)]
        #print( str(theta) + " " + str(dist_right) + " " + str(dist_lookahead))

        # Calculate slpha based on theta, dist_right and dist_lookahead
        alpha = math.atan((math.cos(theta) - dist_right/dist_lookahead) / math.sin(theta))
        distance_r = dist_right * math.cos(alpha)
        
        return distance_r 

def follow_right_wall(angle_right, angle_lookahead, data):
    global DISTANCE_RIGHT_THRESHOLD

    location_to_right_wall = distance(angle_right, angle_lookahead, data)

    # Calculate error to desired track
    error = (DISTANCE_RIGHT_THRESHOLD - location_to_right_wall)/1.5

    # Record error history
    error_history(error)

    #print "error: " + str(error) + " distance: " + str(distance)

    return error


def follow_center(angle_right,angle_lookahead_right, data):

	angle_left = 180 + angle_right
	angle_lookahead_left = 180 + angle_lookahead_right 

	dr = distance(angle_right, angle_lookahead_right, data)
	dl = distance(angle_left, angle_lookahead_left, data)

	# Find Centerline error
	# ---

	# ---
        center = (dr + dl)/2
        print "Center at: %f" % center

        centerline_error = (center - dr)/(dr+dl)
        error_history(centerline_error)

        #print "Centerline error = %f " % centerline_error

	return centerline_error

def callback(data):
        global cycle_count
        global following

	# Pick two rays at two angles
	angle_right = -90 
	angle_lookahead = -60 

        #print "Straight ahead distance: " + str(data.ranges[get_index(0, data)])

        cycle_count = cycle_count + 1

        if cycle_count % 2 ==0:
            if following == "right":
                # To follow right wall
	        e = follow_right_wall(angle_right,angle_lookahead, data)
            elif following == "center":
	        # To follow the centerline
	        e = follow_center(angle_right,angle_lookahead, data)
            else:
                print "Not following any wall"
                exit()

	    control(e, brake(data))

# Reorganize the code a bit into main function
def main():
        global pub
        global following

        # Set following wall
        try:
            following = rospy.get_param('~following')
        except KeyError:    # In case running standalone:
            following = "center"

        print("Wall following started")
        rospy.init_node('wall_following',anonymous = True)

        # TO-DO: Implement the publishers and subscribers
        # ---
        # Create publisher to Ackermann drive control and subscribe to scan message
        pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)
        rospy.Subscriber("/scan",LaserScan,callback)

        # ---

        rospy.spin()

if __name__ == '__main__':
	main()
