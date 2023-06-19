# Author: Nicola Mendini
# functions to perform advanced analysis on the data

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import random
from maps_helper import *


# function to compute the grid score (chatgpt)
def grid_scores(pattern, spacing, printflag=False):
    # Compute the autocorrelogram
    autocorrelogram = F.conv2d(pattern, pattern, padding=pattern.shape[-1]//2)

    # Identify the peaks in the autocorrelogram
    #peaks = F.max_pool2d(autocorrelogram, kernel_size=spacing, padding=spacing//2) == autocorrelogram
    peaks = F.unfold(autocorrelogram, spacing, padding=spacing//2) * get_circle(spacing, spacing/2).view(1,-1,1)
    peaks = peaks.max(1)[0].view(autocorrelogram.shape) == autocorrelogram
    peaks = peaks * (autocorrelogram>1e-1)
    
    if printflag:
        plt.figure()
        plt.imshow(autocorrelogram[0,0])
        plt.show()
        plt.figure()
        plt.imshow(peaks[0,0])
        plt.show()

    # Compute the grid score metrics
    primary_peak = autocorrelogram[peaks].max()
    surrounding_peaks = autocorrelogram[peaks].sum() - primary_peak
    gridness_score = primary_peak - (surrounding_peaks / (peaks.sum() - 1))

    return gridness_score

# function to split activations into n tiles
def split_activations(activations, splits, threshold):

    act_tiles = activations[:,1:,1:]
    act_size = act_tiles.shape[-1]//splits
    act_tiles = F.unfold(act_tiles[:,None], kernel_size=act_size, stride=act_size)
    act_tiles = act_tiles.permute(0,2,1).reshape(-1, act_size**2)
    act_tiles = act_tiles[act_tiles.mean(1)>threshold]
    
    return act_tiles

# function to project to 3D from high dimensions using PCA and UMAP
def project_to_lower_dims(act_tiles, gridness_thresh, peak_spacing):

    act_size = int(np.sqrt(act_tiles.shape[-1]))
    
    if gridness_thresh:
        griddest_acts = []
        for i in tqdm(range(act_tiles.shape[0])):
            score = grid_scores(act_tiles[i].view(-1,1,act_size,act_size), peak_spacing)
            if score > gridness_thresh:
                griddest_acts.append(act_tiles[i][None])
        act_tiles = torch.cat(griddest_acts)

    pca = PCA(n_components=20)
    pca.fit(act_tiles)
    reduced_acts = pca.transform(act_tiles)
    embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.05,
                      metric='manhattan',
                      n_components=3,
                      init='spectral'
                     ).fit_transform(reduced_acts)
    
    return act_tiles, pca, embedding

# function to plot the pca and the activations
def draw_pca(pca, act_tiles, size):
    act_size = int(np.sqrt(act_tiles.shape[-1]))
    plt.figure(figsize=(size,size//2))
    s = random.randint(0, act_tiles.shape[0]-1)
    test = act_tiles[s]
    test_reco = pca.inverse_transform(pca.transform(test[None]))
    plt.subplot(1,2,1)
    plt.imshow(test.view(act_size,act_size))
    plt.subplot(1,2,2)
    plt.imshow(test_reco.reshape(act_size,act_size))
    plt.show()
    
# function to plot the umap 
def draw_umap(embedding, size, title='3D projection'):

    plt.title(title, fontsize=18)
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], s=10)
    plt.show()