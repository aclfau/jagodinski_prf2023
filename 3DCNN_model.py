#!/usr/bin/env python3
"""
"""

# Imported libraries for creating the CNN model
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout
from keras.optimizers import Adam

def create_model(hyperparam, arch_sizes, sample):
    """
    This creates the CNN model for training with 4 convolution + pooling layers
    followed by 2 fully connected layers
    """
    nx, ny, nz, _ = sample.shape
    
    model = Sequential()
    optimizer = Adam(hyperparam['learning_rate'], decay = hyperparam['decay'])	
    model.add(Conv3D(arch_sizes['f1'],
                     kernel_size = (hyperparam['kernel_size'],
                                  hyperparam['kernel_size'],
                                  hyperparam['kernel_size']), 
                     activation = hyperparam['activation'],
                     input_shape = (nx, ny, nz, 1), 
                     padding = hyperparam['padding'],
                     kernel_initializer = hyperparam['kernel_init'],
                     bias_initializer = hyperparam['bias_init'],
                     name = "conv1"))
    
    model.add(MaxPool3D(pool_size = (hyperparam['pool_size'],
                                     hyperparam['pool_size'],
                                     hyperparam['pool_size']),
                        name = "pool1"))
    
    model.add(Conv3D(arch_sizes['f2'], 
                     kernel_size = (hyperparam['kernel_size'],
                                  hyperparam['kernel_size'],
                                  hyperparam['kernel_size']),
                     activation = hyperparam['activation'],
                     padding = hyperparam['padding'],
                     name = "conv2"))
    
    model.add(MaxPool3D(pool_size = (hyperparam['pool_size'],
                                     hyperparam['pool_size'],
                                     hyperparam['pool_size']),
                        name = "pool2"))
    
    model.add(Conv3D(arch_sizes['f3'], 
                     kernel_size=(hyperparam['kernel_size'],
                                  hyperparam['kernel_size'],
                                  hyperparam['kernel_size']),
                     activation = hyperparam['activation'],
                     padding = hyperparam['padding'],
                     name = "conv3"))
    
    model.add(MaxPool3D(pool_size = (hyperparam['pool_size'],
                                     hyperparam['pool_size'],
                                     hyperparam['pool_size']),
                        name="pool3"))
    
    model.add(Conv3D(arch_sizes['f3'],                  
                     kernel_size=(hyperparam['kernel_size'],
                                  hyperparam['kernel_size'],
                                  hyperparam['kernel_size']),
                     activation = hyperparam['activation'],
                     padding = hyperparam['padding'],
                     name="conv4"))
    
    model.add(MaxPool3D(pool_size = (hyperparam['pool_size'],
                                     hyperparam['pool_size'],
                                     hyperparam['pool_size']),
                        name="pool4"))
    
    model.add(Flatten())
    
    model.add(Dense(arch_sizes['d1'], 
                    activation = hyperparam['activation'], 
                    name = "dens1"))
    
    model.add(Dropout(hyperparam['dropout'], name="drop1") )
    
    model.add(Dense(arch_sizes['d2'], 
                    activation = hyperparam['activation'],
                    name="dens2"))
    
    model.add(Dropout(hyperparam['dropout'],name="drop2"))
    
    model.add(Dense(1, 
                    activation='linear',
                    name="output"))
    
    model.compile(optimizer=optimizer, loss = hyperparam['loss_func'])
    model.summary()
    return model