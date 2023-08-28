#!/usr/bin/env python3
"""
"""
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_cam
import matplotlib.pyplot as plt

def find_layers(model, text):
    """ Find all layer names containing the string 'text' """
    layer_names = []
    for layer in reversed(model.layers):
        if (text in layer.name):
            layer_names.append(layer.name)
    return layer_names

def MultiLayer_GradCAM(model, test_sample):
    test_sample = np.expand_dims(test_sample,axis=0)
    output_layer_name = find_layers(model, 'out')[0]
    convolutional_layers = find_layers(model, 'conv')
    output_layer_index = utils.find_layer_idx(model, output_layer_name)
    total_GradCAM = np.zeros_like(test_sample[0,:,:,:,0])
    for conv_layer in convolutional_layers:
        print('layer: ' + conv_layer)
        conv_layer_index = utils.find_layer_idx(model, conv_layer)
        layer_GradCAM = visualize_cam(model,
                               output_layer_index,
                               filter_indices = None,
                               penultimate_layer_idx = conv_layer_index,
                               seed_input = test_sample,
                               backprop_modifier = 'guided') 
        layer_GradCAM = np.array(layer_GradCAM)
        total_GradCAM = total_GradCAM + layer_GradCAM

    return total_GradCAM

def plot_sample_GradCAM(test_sample, GradCAM, 
                        sample_cutoff = 0.1, gradcam_cutoff = 2):
    
    # Swap the y and z axis because Matplotlib plots the z axis vertically
    # while in our data, the y axis is the wall-normal axis of our channel flow
    test_sample = np.swapaxes(test_sample, 1, 2)
    GradCAM = np.swapaxes(GradCAM, 1, 2)
    
    # Sets up the plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Get the test_sample values greater than the selected cutoff
    vx, vy, vz, _ = np.where(test_sample > sample_cutoff)
    vc = [test_sample[vx[n], vy[n], vz[n], 0] for _,n in enumerate(vx)]
    
    # Get the GradCAM values greater than the selected cutoff
    gx, gy, gz = np.where(GradCAM > gradcam_cutoff)
    gc = [test_sample[gx[n], gy[n], gz[n]] for _,n in enumerate(gx)]
         
    # Plots, sets plot limits, adds titles
    ax1.scatter(vx, vy, vz, c=vc, cmap="RdPu")    
    ax2.scatter(gx, gy, gz, c=gc, cmap="YlOrBr")
    
    ax1.set_xlim(0, GradCAM.shape[0])
    ax1.set_ylim(0, GradCAM.shape[1])
    ax1.set_zlim(0, GradCAM.shape[2])
    ax2.set_xlim(0, GradCAM.shape[0])
    ax2.set_ylim(0, GradCAM.shape[1])
    ax2.set_zlim(0, GradCAM.shape[2])
    ax1.set_title('Test_sample')
    ax2.set_title('Multi-layer GradCAM')