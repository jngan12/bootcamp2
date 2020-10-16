#!/usr/bin/env python
# -*- coding: utf-8 -*-
# References:
#https://github.com/juano2310/CarND-Behavioral-Cloning-P3-Juan/blob/master/model.py
#https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/rambo/train.py
#https://github.com/wilselby/diy_driverless_car_ROS/blob/master/rover_ml/colab/RC_Car_End_to_End_Image_Regression_with_CNNs_(RGB_camera).ipynb

import os
import csv
import cv2
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import random

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D, MaxPooling2D, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Dropout, SpatialDropout2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array 

import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

print("Tensorflow Version: %s" % tf.__version__)
print("Tensorflow Keras Version: %s" % tf.keras.__version__)
print("Eager mode: %r" % tf.executing_eagerly())

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.keras.backend.set_session(tf.Session(config=config))

# Look for GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  #raise SystemError('GPU device not found')
  print('\n GPU device not found')
else:
  print('\n Found GPU at: {}'.format(device_name))
  
# Define paths
data_set = 'bag2csv/output'
csv_path = data_set + '/interpolated.csv'

# Read CSV files
df = pd.read_csv(csv_path, sep=",") # Load the CSV file into a pandas dataframe
#, nrows=2000

print("Dataset Dimensions: {}".format(df.shape)) # Print the dimensions
print("Dataset Summary: {}".format(df.head(5))) # Print the first 5 lines of the dataframe for review
df.drop(['index','frame_id'],axis=1,inplace=True) # Remove 'index' and 'frame_id' columns 

print("\nDataset Summary: {}".format(df.head(5))) # Verify new dataframe print the first 5 lines of the new dataframe for review

# Steering Command Statistics
print("\nSteering Command Statistics:")
print(df['angle'].describe())

print("\nThrottle Command Statistics:")
# Throttle Command Statistics
print(df['speed'].describe())


# Remove rows with 0 throttle values
print("\nRemoving {} 0 throttle values: ".format(df['speed'].eq(0).sum()))
if df['speed'].eq(0).any():
  df = df.query('speed != 0')
  df.reset_index(inplace=True,drop=True)   # Reset the index

#Remove excess steering values of the same type. Sort the steering data into bins and only keep 200 values in each bin. 

num_bins = 25 
hist, bins = np.histogram(df['angle'], num_bins)
center = (bins[:-1]+ bins[1:]) * 0.5

hist = True 
remove_list = []
samples_per_bin = 100

if hist:
  for j in range(num_bins):
    list_ = []
    for i in range(len(df['angle'])):
      if df.loc[i,'angle'] >= bins[j] and df.loc[i,'angle'] <= bins[j+1]:
        list_.append(i)
    random.shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

  print('removed:', len(remove_list))
  df.drop(df.index[remove_list], inplace=True)
  df.reset_index(inplace=True)
  df.drop(['index'],axis=1,inplace=True)
  print('remaining:', len(df))
  
  hist, _ = np.histogram(df['angle'], (num_bins))
  
print("\nNew Dataset Dimensions: {}".format(df.shape))
print("\nSteering Command Statistics: {}".format(df['angle'].describe()))
print("\nThrottle Command Statistics: {}".format(df['speed'].describe()))


# Create image data augmentation generator and choose augmentation types
datagen = ImageDataGenerator(#rotation_range=20,
                             zoom_range=0.15,
                             #width_shift_range=0.1,
                             #height_shift_range=0.2,
                             #shear_range=10,
                             brightness_range=[0.5,1.0],
                          	 #horizontal_flip=True,
                             #vertical_flip=True,
                             #channel_shift_range=100.0,
                             fill_mode="reflect")


# Split the dataset

samples = []
samples = df.values.tolist()

sklearn.utils.shuffle(samples)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("Number of traing samples: %f" % len(train_samples))
print("Number of validation samples: %f" % len(validation_samples))

# Image cropping data
index = random.randint(0,df.shape[0]-1)

img_name = data_set + '/' + df.loc[index,'filename']
angle = df.loc[index,'angle']

center_image = cv2.imread(img_name)
center_image_mod = cv2.resize(center_image, (320,180))
center_image_mod = cv2.cvtColor(center_image_mod,cv2.COLOR_RGB2BGR)

# Crop the image
height_min = 75 
height_max = center_image_mod.shape[0]
width_min = 0
width_max = center_image_mod.shape[1]

def generator(samples, batch_size=32, aug=0):
    num_samples = len(samples)

    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            #print(batch_samples)
            images = []
            angles = []
            for batch_sample in batch_samples:
                if batch_sample[5] != "filename":
                    name = data_set + '/' + batch_sample[3]
                    center_image = cv2.imread(name)
                    center_image = cv2.cvtColor(center_image,cv2.COLOR_RGB2BGR)
                    center_image = cv2.resize(
                        center_image,
                        (320, 180))  #resize from 720x1280 to 180x320
                    angle = float(batch_sample[4])
                    if not aug:
                      images.append(center_image)
                      angles.append(angle)
                    else:
                        data = img_to_array(center_image)
                        sample = expand_dims(data, 0)
                        it = datagen.flow(sample, batch_size=1)
                        batch = it.next()
                        image_aug = batch[0].astype('uint8')
                        if random.random() < .5:
                          image_aug = np.fliplr(image_aug)
                          angle = -1 * angle
                        images.append(image_aug)
                        angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


n_epoch = 5 # Define number of epochs
batch_size_value = 32
img_aug = 1

train_generator = generator(train_samples, batch_size=batch_size_value, aug=img_aug)
validation_generator = generator(validation_samples, batch_size=batch_size_value, aug=0)

# Initialize the model
model = Sequential()

# trim image to only see section with road
model.add(Cropping2D(cropping=((height_min,0), (width_min,0)), input_shape=(180,320,3))) # (top_crop, bottom_crop), (left_crop, right_crop)

# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Nvidia model
model.add(Convolution2D(24, (5, 5), activation="relu", name="conv_1", strides=(2, 2)))
model.add(Convolution2D(36, (5, 5), activation="relu", name="conv_2", strides=(2, 2)))
model.add(Convolution2D(48, (5, 5), activation="relu", name="conv_3", strides=(2, 2)))
model.add(SpatialDropout2D(.5, data_format=None))

model.add(Convolution2D(64, (3, 3), activation="relu", name="conv_4", strides=(1, 1)))
model.add(Convolution2D(64, (3, 3), activation="relu", name="conv_5", strides=(1, 1)))

model.add(Flatten())

model.add(Dense(1164))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(1))

tf.keras.backend.set_epsilon(1)
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mse','mae','mape'])

# Print model sumamry
model.summary()

# checkpoint
filepath = "/home/adhitir/sae_ws/ros_ws/src/bootcamp-assignments-master/behavior_cloning/bag2csv/output/weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='auto', period=1)

# Setup Early Stopping to Prevent Overfitting
# The patience parameter is the amount of epochs to check for improvement
early_stop = EarlyStopping(monitor='val_loss', patience=10)

# Reduce Learning Rate When a Metric has Stopped Improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

# # Train the model

# Define step sizes
STEP_SIZE_TRAIN = len(train_samples) / batch_size_value
STEP_SIZE_VALID = len(validation_samples) / batch_size_value

# Define callbacks
callbacks_list = [checkpoint, early_stop, checkpoint, reduce_lr]

# Fit the model
history_object = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    callbacks=callbacks_list,
    use_multiprocessing=True,
    epochs=n_epoch,
    verbose=2)

# Save model
model.save('model.h5')
with open('model.json', 'w') as output_json:
    output_json.write(model.to_json())

# Plot the training and validation loss for each epoch
print('Generating loss chart...')

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('model.png')


# Done
print('Done.')
