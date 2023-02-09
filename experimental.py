import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import math
import torchvision.transforms.functional as TF


# Tries every single possible rotation of the RFs to see which one makes their STD the smallest
# experimental gradient descent method to find the orientations on a map
def aligner(model, grid_size, ksize, ori=None, steps=1000):
    
    if ori==None:
        ori = nn.Parameter(torch.zeros(grid_size**2))
        
    rfs = model.get_weights().detach()
    
    rotmat = torch.eye(3, device=rfs.device)[None,:2].repeat(grid_size**2, 1, 1)
    
    max_tracker = torch.ones((2,grid_size**2), device=rfs.device)
    
    progress_bar = tqdm(range(steps), position=0, leave=True)
    
    angrange = torch.linspace(0, torch.pi*(steps-1)/steps, steps)
        
    for i in progress_bar:
        
        rotmat[:,0,0] = torch.cos(angrange[i])
        rotmat[:,0,1] = torch.sin(angrange[i])
        rotmat[:,1,0] = -rotmat[:,0,1]
        rotmat[:,1,1] = rotmat[:,0,0]
        
        grid = F.affine_grid(rotmat, (grid_size**2, 1, ksize,ksize))
        rfs_t = F.grid_sample(rfs, grid)
        rfs_t = rfs_t / rfs_t.sum([-1,-2], keepdim=True)
        rfs_t = rfs_t.sum(-1)
        
        tempmax = rfs_t.std(2).squeeze()
                        
        max_tracker[1, max_tracker[0]<tempmax] = angrange[i]
        max_tracker[0, max_tracker[0]<tempmax] = tempmax[max_tracker[0]>tempmax]
                
    return max_tracker