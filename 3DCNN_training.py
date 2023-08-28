#!/usr/bin/env python3
"""
While most functions in Keras and Tensorflow are unchanged, this code
is specific to Tensorflow1.15.0 and Keras2.3.1
"""


"""
Imported libraries and GPU Setup:
Importing libraries for creating and training the 3D CNN as well as setting 
the GPU for use while reserving some GPU memory for other processes
"""

# Imported Libraries
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json

# Loading a library specific to reading our data binaries but
# this code is adaptable to any 3D data
import idFHundReshape as F

# Importing the functions to create the model
from 3DCNN_model import create_model

# Importing the functions for generating the Multi-layer GradCAM from 
# a test sample and visualizing
from multiLayer_GradCAM import MultiLayer_GradCAM, plot_sample_GradCAM

# Configuring GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

"""
Loading data and interpolating:
This is specific to our data and our desired interpolations.
Will work as long as you have the data declared as 'inputs' and 'labels' arrays.
"""
dat,conf = F.datalist('/mnt/sda1/channel_re300/timeSeries/allData/')
inputs, labels, _ = F.inputDatInterp(dat, 'v', 0, 30, 40, 30, 1)

"""
Training/validation/testing split: 
The data should be a tensor of the shape (N, nx, ny, nz, 1) where N
is the number of data samples, nx,ny,nz are the dimensions of the 
3D samples and the 1 indicates a scalar value at each data point
within the 3D sample.
"""
# Configuration dictionary for training/validation/testing data split
train_param = {'test_pct': 0.15,
               'validation_test_split': 0.5,
               'random_state1': 14,
               'random_state2': 92 }

# Split the data into training and testing+validation
(x_train, x_val_test, y_train, y_val_test) = train_test_split(inputs, labels, 
                                              test_size = train_param['test_pct'], 
                                              random_state = train_param['random_state1'])

(x_test, x_valid, y_test, y_valid) = train_test_split(x_val_test, y_val_test, 
                                      test_size = train_param['validation_test_split'], 
                                      random_state = train_param['random_state2'])

def reshape_data(data):
    """
    Reshapes the array into the shape needed by the Keras model.
    In our case 9,000x[30,40,30] into (9,000, 30, 40, 30, 1)
    """
    data = np.array(data)
    data = np.expand_dims(data, -1)
    return data

x_train = reshape_data(x_train)
x_valid = reshape_data(x_valid)
x_test = reshape_data(x_test)

"""
Hyperparameters and architecture:
Reads the complimentary hyperparameters json file into a dictionary and 
assigns the architecture layout via number of filters for each convolutional
layer (f1, f2, etc.) and the size of each dense layer (d1, d2)
"""

# Loading hyperparameters from the json file
with open('hyperparameters.json', 'r') as file:
    hyperparam = json.load(file)

# Architecture layout, number of kernels and dense layer sizes
arch_sizes = {'f1': 32,
              'f2': 63,
              'f3': 128,
              'f4': 256,
              'd1': 256,
              'd2': 24 }

"""
Create the model and training:
Creates the model using the function from supplimentary_code_model.py and the
hyperparameter and architecture dictionaries and then trains the model
using the training and validation data sets
"""

# Creates the model
model = create_model(hyperparam, arch_sizes, x_train[0])

# Training the network
training_history = model.fit(x_train, y_train, 
                             batch_size = hyperparam['batch_size'], 
                             epochs=hyperparam['epochs'], 
                             verbose=1, 
                             validation_data=(x_valid, y_valid))


"""
Evaluation:
The trained model is evaluated on the test data set put aside earlier and the
mean percent error is calculated
"""
# Predict on the test set
predictions = model.predict(x_test)

# Calculating the mean percent error of the prediction
predictions, y_test = np.squeeze(predictions), np.squeeze(y_test)
percent_errors = 100*(predictions - y_test) / y_test
mean_percent_error = np.mean(abs(percent_errors))
print(f'Mean Percent Error of the Trained Model: {mean_percent_error}%')

"""
Inspect Network Focus:
Using the Multi-Layer GradCAM technique, inspect the trained networks focus on
a test sample and plot them. Functions found in supplementary_code_GradCAM.py
"""
# Select a test sample and calculate the Multi-layer GradCAM
test_sample = x_test[0]
ML_GradCAM = MultiLayer_GradCAM(model, test_sample)

# Plots the test sample and the Multi-Layer GradCAM side-by-side
plot_sample_GradCAM(test_sample, ML_GradCAM, 
			sample_cutoff = 0.1, gradcam_cutoff = 1.5)


