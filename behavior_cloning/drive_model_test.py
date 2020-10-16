#!/usr/bin/env python
# -*- coding: utf-8 -*-
#https://github.com/juano2310/CarND-Behavioral-Cloning-P3-Juan/blob/master/drive.py

import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
#sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy


import rospy
import time
import base64
from datetime import datetime
import os
import shutil
import numpy as np

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image,  CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import json
import tensorflow as tf
from tensorflow import keras
import h5py
from keras import __version__ as keras_version

from ackermann_msgs.msg import AckermannDriveStamped

print("Tensorflow Version:",tf.__version__)
print("Tensorflow Keras Version:",tf.keras.__version__)
print("Eager mode: ", tf.executing_eagerly())

GPU_OPTIONS = tf.GPUOptions(allow_growth=True)
CONFIG = tf.ConfigProto(gpu_options=GPU_OPTIONS)
tf.keras.backend.set_session(tf.Session(config=CONFIG))

model_path = '/home/adhitir/sae_ws/git_ws/bootcamp-assignments-master/behavior_cloning/'

class cmd_vel_node(object):
    def __init__(self):

        self.imgRcvd = False
        self.latestImage = None
        self.cmdvel = AckermannDriveStamped()
        self.bridge = CvBridge()

        self.ackermann_pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=10)
        self.image_sub = rospy.Subscriber("/camera/zed/rgb/image_rect_color",Image, self.image_callback )
        self.image_pub = rospy.Publisher("/image_converter/output_video",Image, queue_size=10)

    def image_callback(self,data):
      try:
        self.latestImage = self.bridge.imgmsg_to_cv2(data, "bgr8")	
      except CvBridgeError as e:
        print(e)
      if self.imgRcvd != True:
          self.imgRcvd = True   
          print("Image recieved")

    def publish(self, image,  bridge,  publisher):
        try:
            #Determine Encoding
            if np.size(image.shape) == 3: 
                imgmsg = bridge.cv2_to_imgmsg(image, "bgr8") 
            else:
                imgmsg = bridge.cv2_to_imgmsg(image, "mono8") 
            publisher.publish(imgmsg)  
        except CvBridgeError as e:
            print(e)

    def run(self):
        # check that model Keras version is same as local Keras version
        f = h5py.File(model_path + 'model.h5', mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version_installed = None
        keras_version_installed = str(keras_version).encode('utf8')

        if model_version != keras_version_installed:
            print('You are using Keras version ', keras_version_installed, ', but the model was built using ', model_version)

        with open(model_path + 'model.json', 'r') as f:
            model = tf.keras.models.model_from_json(f.read()) 

        model = tf.keras.models.load_model(model_path + 'model.h5')
        
        # Load weights into the new model
        print("Model loaded.")
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

             # Only run loop if we have an image
             if self.imgRcvd:
                 
                 # step 1: 
                 self.resized_image = cv2.resize(self.latestImage, (320,180)) 
                 
                 # step 2: 
                 image_array = np.asarray(self.resized_image)
                 
                 # step 3: 
                 self.cmdvel.drive.speed = 0.1
                 self.angle = float(model.predict(image_array[None, :, :, :], batch_size=1))*100
                 print("steering angle: %f" % self.angle)
                 self.angle = -0.25 if self.angle < -0.25 else 0.25 if self.angle > 0.25 else self.angle
                 self.cmdvel.drive.steering_angle = self.angle

                 
                 #print(self.cmdvel.angular.z)
                 self.ackermann_pub.publish(self.cmdvel)
                 
                 # Publish Processed Image
                 self.outputImage = self.latestImage
                 self.publish(self.outputImage, self.bridge,  self.image_pub)

             rate.sleep()

def main(args):

  rospy.init_node('model_control_node', anonymous=True)

  cmd = cmd_vel_node() 

  cmd.run() 

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
    main(sys.argv)
    
