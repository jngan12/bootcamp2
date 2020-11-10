#!/usr/bin/env python
# -*- coding: utf-8 -*-
# based on https://github.com/experiencor/keras-yolo3

import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
#sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospy

import rospy
import roslib
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from object_recognition.msg import Predictor
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
# ResNet import
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.python.keras.backend import set_session
from time import time

GPU_OPTIONS = tf.GPUOptions(allow_growth=True)
CONFIG = tf.ConfigProto(gpu_options=GPU_OPTIONS)
CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5

sess = tf.Session(config=CONFIG)
tf.keras.backend.set_session(sess)

# ResNet model load
model = ResNet50(weights='imagenet')
model._make_predict_function()
graph = tf.get_default_graph()
target_size = (224, 224)

bridge = CvBridge()

def callback(image_msg):
    print("In callback")
    time_start = time()
    #First convert the image to OpenCV image 
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

    cv_image_target = cv2.resize(cv_image, target_size)  # resize image
    np_image = np.asarray(cv_image_target)               # read as np array
    np_image = np.expand_dims(np_image, axis=0)   # Add another dimension for tensorflow
    np_image = np_image.astype(float)  # preprocess needs float64 and img is uint8
    np_image = preprocess_input(np_image)         # Regularize the data

    cv2.imshow('Camera',cv2.flip(cv_image,0))
    cv2.waitKey(1)

    global sess
    global graph                                  # This is a workaround for asynchronous execution
    with graph.as_default():
        set_session(sess)
        preds = model.predict(np_image)            # Classify the image
        pred_string = decode_predictions(preds, top=1)   # Decode top 1 predictions
        print(pred_string)
    print "Inference duration: " + str(time() - time_start) + " seconds"

rospy.init_node('classify', anonymous=True)

rospy.Subscriber("usb_cam/image_raw", Image, callback, queue_size = 1, buff_size = 16777216)

#pub = rospy.Publisher('object_detector', Predictor, queue_size = 1)

print("In main")

while not rospy.is_shutdown():
  rospy.spin()
