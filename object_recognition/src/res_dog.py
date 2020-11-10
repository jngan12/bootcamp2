#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')
model._make_predict_function()
graph = tf.get_default_graph()
target_size = (224, 224)


cv_image = cv2.imread("/home/sysver/sae_ws/ros_ws/src/bootcamp-assignments/object_recognition/images/dog.png",cv2.IMREAD_COLOR)
cv_image = cv2.resize(cv_image, target_size)  # resize image
np_image = np.asarray(cv_image)               # read as np array
np_image = np.expand_dims(np_image, axis=0)   # Add another dimension for tensorflow
np_image = np_image.astype(float)  # preprocess needs float64 and img is uint8
np_image = preprocess_input(np_image)         # Regularize the data
    
with graph.as_default():
    preds = model.predict(np_image)            # Classify the image
    pred_string = decode_predictions(preds, top=1)   # Decode top 1 predictions
    print(pred_string)
