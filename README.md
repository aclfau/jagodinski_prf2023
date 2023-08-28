<img src=figures/graphicalAbstract.jpg width="750">

### Abstract

Near-wall regions in wall-bounded turbulent flows experience intermittent ejection of slow-moving fluid packets away from the wall and sweeps of faster moving fluid towards the wall. These extreme events play a central role in regulating the energy budget of the boundary layer, and are analyzed here with the help of three-dimensional Convolutional Neural Networks (CNNs) due to their inherent strength in extracting spatial structures and learning local patterns. A CNN is trained on Direct Numerical Simulation data from a periodic channel flow to deduce the intensity of such extreme events, and more importantly, to reveal localized 3D regions in the flow where the network focuses its attention to make an accurate determination of ejection intensity. These salient regions, identified autonomously using a modified multi-layer Gradient-weighted Class Activation Mapping (GradCAM) technique, correlate well with coherent fluid packets being ejected away from the wall and with regions involving high energy production. Statistical analysis further demonstrates the relationship between GradCAM values and high wall-normal velocity fluctuations and positive energy production. The results indicate that CNNs can identify three-dimensional spatial correlations in turbulent flow using a single scalar-valued metric provided as the quantity of interest, which in the present case is the ejection intensity. While the current work presents an alternate means of analyzing nonlinear spatial correlations associated with near-wall bursts, the techniques explored here may be used in other scenarios where the underlying spatial dynamics are not known a-priori.

# Code Structure

[hyperparameters.json](hyperparameters.json): For storing the hyperparameters as a dictionary to keep separate from the code.

[3DCNN_training.py](3DCNN_training.py): For loading the data, reshaping, reading hyperparameters.json, training the model, testing and using Multi-layer Gradient-weighted Class Activation Mapping (GradCAM) to inspect the trained network's focus.

[3DCNN_model.py](3DCNN_model.py): For creating the network architecture based on the hyperparameters dictionary and architecture sizes dictionary.

[multi-layer_GradCAM.py](multiLayer_GradCAM.py): Using the keras visualization library to create the functions for generating the GradCAM map and visualizing against a test data point.

# Training

## Environment and Libraries

The authors suggest using [Anaconda](https://www.anaconda.com/) for managing environments and version compatibility. Download by visiting their website for your respective OS. Specifically, Tensorflow v1.15.0 and Keras v2.3.1 with the keras visualize utilities were used here. These are necessary for using the visualize_cam function to generate the single-layer GradCAM.

```python
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf
import json

# Importing the functions to create the model
from 3DCNN_model import create_model

# Importing the functions for generating the Multi-layer GradCAM from 
# a test sample and visualizing
from multiLayer_GradCAM import MultiLayer_GradCAM, plot_sample_GradCAM
```

## GPU Usage

The code can be ran without using Tensorflow's GPU compatibility but this will make training significantly faster. The following code was used for configuring our GPU for usage.
```python
# Configuring GPU usage
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.95
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
```
Our training was done on a Nvidia Titan RTX graphics card with 4608 CUDA cores and 24 GB of GDDR6 VRAM. The training using 10,800 training samples of size `(30,40,30)` for 60 epochs took approximately one hour.

### For General Use With Any 3D Data Set

The following lines are for our specific data loading and interpolation. These can be replaced with any 3D data as long as the data is a tensor of the shape `(N, nx, ny, nz, 1)` where `N` is the number of data samples, `nx,ny,nz` are the dimensions of the 3D samples and the `1` indicates a scalar value at each data point within the 3D sample.

```python
# Loading data and interpolating
dat,conf = F.datalist('/mnt/sda1/channel_re300/timeSeries/allData/')
inputs, labels, _ = F.inputDatInterp(dat, 'v', 0, 30, 40, 30, 1)
```

### Training/Validation/Testing split

The scikit-learn `train_test_split` function is used for randomly splitting the data into the training, validation and testing sets. Validation is used for periodic checks during training while testing is set aside for after the training is complete. A dictionary called train_param is used for designating the data split percents and random states.

```python
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
```

The data is also reshaped into the shape needed by the Keras model using the following function. 
In our case `(10800, 30, 40, 30)` into `(10800, 30, 40, 30, 1)`

```python
def reshape_data(data):
    data = np.array(data)
    data = np.expand_dims(data, -1)
    return data
```

### Hyperparameters and CNN Architecture

In order to keep the code more orderly, hyperparameters are stored in a json file. Here, we read the complimentary [hyperparameters.json](hyperparameters.json) file into a dictionary and assign the architecture layout via number of filters for each convolutional layer (`f1`, `f2`, etc.) and the size of each dense layer (`d1`, `d2`).

```python
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
```

##  Model Build and Training

<img src=figures/CNN_arch.png width="1000">

Here, we create the model using the code [3DCNN_model.py](3DCNN_model.py), the hyperparameter and architecture dictionaries, and one training sample for reading the data size.
```python
# Creates the model
model = create_model(hyperparam, arch_sizes, x_train[0])

# Training the network
training_history = model.fit(x_train, y_train, 
                             batch_size = hyperparam['batch_size'], 
                             epochs=hyperparam['epochs'], 
                             verbose=1, 
                             validation_data=(x_valid, y_valid))
```
## Evaluation
The trained model is evaluated on the test data set put aside earlier and the
mean percent error is calculated.

<img src=figures/training_results.png width="350">
Training Results (plotting code not used here for simplicity)

```python
# Predict on the test set
predictions = model.predict(x_test)

# Calculating the mean percent error of the prediction
predictions, y_test = np.squeeze(predictions), np.squeeze(y_test)
percent_errors = 100*(predictions - y_test) / y_test
mean_percent_error = np.mean(abs(percent_errors))
print(f'Mean Percent Error of the Trained Model: {mean_percent_error}%')
```

## Using the Multi-layer GradCAM
Using the functions from [multi-layer_GradCAM.py](multi-layer_GradCAM.py) for inspecting the focus of the trained network on a selected test sample. `sample_cutoff` and `gradcam_cutoff` are used to select minimum values for visualization.

<img src=figures/multi_layer_gradcam.png width="1300">

```python
from supplementary_code_GradCAM import MultiLayer_GradCAM, plot_sample_GradCAM

# Select a test sample and calculate the Multi-layer GradCAM
test_sample = x_test[0]
ML_GradCAM = MultiLayer_GradCAM(model, test_sample)

# Plots the test sample and the Multi-Layer GradCAM side-by-side
plot_sample_GradCAM(test_sample, ML_GradCAM, 
			sample_cutoff = 0.1, gradcam_cutoff = 1.5)
```

Sample vs Multi-layer GradCAM post-processed in Paraview

<img src=figures/sample_gradcam.png width="500">

# Multi-layer GradCAM 

These are the libraries used for creating the CNN model and the Multi-layer GradCAM.

### Additional Libraries Used

```python
from vis.utils import utils
from vis.visualization import visualize_cam
```

### Specific Layer Names
Of note, if the layer names used in [3DCNN_model.py](3DCNN_model.py) are changed, than the following lines of the [multi-layer_GradCAM.py](multi-layer_GradCAM.py) must be changed to match.
```python
# 'out' and 'conv' are substrings of the output and convolution layer names, respectively
output_layer_name = find_layers(model, 'out')[0]
convolutional_layers = find_layers(model, 'conv')
```

# CNN Model Building

### Additional Libraries Used

```python
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout
from keras.optimizers import Adam
