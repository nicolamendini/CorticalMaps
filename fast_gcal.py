# Author: Nicola Mendini
# Main cortical map module

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import math
import torchvision.transforms.functional as TF

from maps_helper import *

# Cortical Map Module
class CorticalMap(nn.Module):
    
    def __init__(
        self, 
        device,
        sheet_units,
        aff_cf_units,
        aff_cf_channels,
        aff_cf_dilation,
        lat_cf_dilation,
        lat_iters,
        strength,
        homeo_target,
        homeo_timescale,
        exc_units,
        lat_cf_units,
        max_inh_fac,
        dtype
    ):
        
        super().__init__()
        
        self.type = dtype
        
        # defining the lateral interaction envelopes, connections cutoffs and masks
        aff_cutoff = get_circle(aff_cf_units, aff_cf_units/2)
        aff_cf_envelope = get_radial_cos(aff_cf_units,aff_cf_units)**2 * aff_cutoff
        aff_cf_envelope /= aff_cf_envelope.max()
        aff_cf_envelope = aff_cf_envelope.repeat(1,aff_cf_channels,1,1).view(1,-1,1)  
        self.aff_cf_envelope = aff_cf_envelope.to(device).type(self.type)
        
        exc_cutoff = get_circle(exc_units, exc_units/2)
        sre = get_radial_cos(exc_units, exc_units)**2 * exc_cutoff
        sre /= sre.sum()
        self.sre = sre.to(device).type(self.type)
        
        inh_cutoff = get_circle(lat_cf_units, lat_cf_units/2)
        inh_mask = get_radial_cos(lat_cf_units, exc_units/lat_cf_dilation)**2
        inh_mask *= get_circle(lat_cf_units, exc_units/(2*lat_cf_dilation))
        lri_envelope = get_radial_cos(lat_cf_units, lat_cf_units)**2
        lri_envelope *= inh_cutoff
        lri_envelope *= 1 - inh_mask    
        lri_envelope = lri_envelope.view(1,-1,1)
        lri_envelope /= lri_envelope.max()
        self.lri_envelope = lri_envelope.to(device).type(self.type)
        
        self.max_r = oddenise(exc_units*2)
        max_mask = get_circle(self.max_r, self.max_r/2).view(1,-1,1)
        self.max_mask = max_mask.to(device).type(self.type)
        
        
        # initialising the aff weights
        init_rfs = torch.ones(sheet_units**2, aff_cf_channels*aff_cf_units**2,1)
        init_rfs *= aff_cf_envelope
        init_rfs /= init_rfs.sum(1, keepdim=True) / (aff_cf_channels*aff_cf_units**2)
        self.rfs = nn.Parameter(init_rfs.type(self.type)) 
                
        # and the lateral inhibitory correlation weights
        init_lat = torch.ones(sheet_units**2,lat_cf_units**2,1)
        init_lat *= lri_envelope
        init_lat /= init_lat.sum(1, keepdim=True) / lat_cf_units**2
        self.lat_weights = nn.Parameter(init_lat.type(self.type))
        
        print('initialised means: ', init_lat.mean(), init_rfs.mean())
                
        # defining the variables of homeostatic adaptation
        thresh_init = torch.zeros((1,1,sheet_units,sheet_units), device=device) + 0.1
        self.adathresh = nn.Parameter(thresh_init,requires_grad=False).type(self.type)
        self.avg_acts = torch.zeros((1,1,sheet_units,sheet_units), device=device).type(self.type) + homeo_target
        self.lat_mean = torch.zeros((1,1,sheet_units,sheet_units), device=device).type(self.type)
        
        # storing some parameters
        self.homeo_lr = (1-homeo_timescale)/2
        self.lat_cf_units = lat_cf_units
        self.lat_cf_dilation = lat_cf_dilation
        self.aff_cf_units = aff_cf_units
        self.aff_cf_dilation = aff_cf_dilation
        self.aff_cf_channels = aff_cf_channels
        self.sheet_units = sheet_units
        self.lat_iters = lat_iters
        self.homeo_timescale = homeo_timescale
        self.homeo_target = homeo_target
        self.strength = strength
        self.max_inh_fac = max_inh_fac
                                
    def forward(self, x, reco_flag=False, learning_flag=True, print_flag=False):
        
        # unpacking some values
        lat_correlations = 0
        raw_aff = 0
        x_tiles = 0
                 
        # ensuring that the weights stay always non negative
        with torch.no_grad():
            #self.rfs *= (self.rfs>0)
            #self.lat_weights *= (self.lat_weights>0)
            self.rfs /= self.rfs.sum(1,keepdim=True) / self.rfs.shape[1]
            self.lat_weights /= self.lat_weights.sum(1,keepdim=True) / self.lat_weights.shape[1]
                                
        lat_weights = self.lat_weights
        rfs = self.rfs
                        
        if not learning_flag:
            rfs = rfs.detach()
        lat_w = lat_weights.detach() 
        lat_w = lat_w / lat_w.shape[1] #lat_w.sum(1, keepdim=True) + 1e-5
        
                                    
        # computing the afferent response
        if not reco_flag:
            self.lat_mean *= 0
            dilation = 1 if self.aff_cf_dilation<1 else round(self.aff_cf_dilation)
            x_tiles = F.unfold(x, self.aff_cf_units, dilation=dilation)    
            x_tiles = x_tiles * self.aff_cf_envelope
            x_tiles = x_tiles.permute(2,0,1)
            raw_aff = torch.bmm(x_tiles, rfs)
            raw_aff = raw_aff.view(1,1,self.sheet_units,self.sheet_units)
            aff = raw_aff.detach()
            aff /= rfs.shape[1] #rfs.detach().sum(1,keepdim=True).view(aff.shape) + 1e-5
                        
        # if feedback is coming, just take the feedback
        else:
            aff = x
            
        aff = aff - self.adathresh
        lat = torch.relu(aff)
                                
        # reaction diffusion
        for i in range(self.lat_iters):
                                    
            if print_flag:
                plt.imshow(lat.detach().cpu()[0,0])
                plt.show()
                print(lat.max(), lat.mean())
                
            # short range excitation is applied first
            pos_pad = self.sre.shape[-1]//2
            lat = F.pad(lat, (pos_pad,pos_pad,pos_pad,pos_pad), mode='reflect')
            lat = F.conv2d(lat, self.sre)
            
            # splitting into tiles and computing the lateral inhibitory component
            pad = self.lat_cf_units//2*self.lat_cf_dilation
            lat_tiles = F.pad(lat, (pad,pad,pad,pad), value=0)
            lat_tiles = F.unfold(lat_tiles, self.lat_cf_units, dilation=self.lat_cf_dilation)
            
            #print(lat_tiles.shape, neg_w.shape)
            lat_tiles = lat_tiles * self.lri_envelope            
            lat_tiles = lat_tiles.permute(2,0,1)
            lat_neg = torch.bmm(lat_tiles, lat_w).view(1,1,self.sheet_units,self.sheet_units)
            lat = lat - lat_neg*self.max_inh_fac
            
            lat = torch.relu(lat + aff)*self.strength

            max_pad = self.max_r//2
            lat_max = F.pad(lat, (max_pad, max_pad, max_pad, max_pad), mode='reflect')
            
            lat_max = F.unfold(lat_max, self.max_r) * self.max_mask
            lat_max = lat_max.max(1)[0]
            lat_max = lat_max.view(1,1,self.sheet_units,self.sheet_units)
            lat_max[lat_max<1] = 1
            lat = lat / (lat_max+1e-5)
        
            self.lat_mean += lat
                        
            #if i==4:
            #    print('reset')
            #    aff = -self.adathresh
                            
        if not reco_flag:
            if learning_flag:

                # learn the lateral correlations
                self.lat_mean /= self.lat_iters
                lat_tiles = F.pad(self.lat_mean, (pad,pad,pad,pad), value=0)
                lat_tiles = F.unfold(lat_tiles, self.lat_cf_units, dilation=self.lat_cf_dilation)
                lat_tiles = lat_tiles * self.lri_envelope 
                lat_tiles = lat_tiles.permute(2,0,1)
                weighted_lat_tiles = lat_tiles * self.lat_mean.view(self.sheet_units**2,1,1)
                lat_correlations = torch.bmm(weighted_lat_tiles, lat_weights).sum()
                    
            # update the homeostatic variables
            self.avg_acts = self.homeo_timescale*self.avg_acts + (1-self.homeo_timescale)*self.lat_mean
            gap = (self.avg_acts-self.homeo_target)
            self.adathresh += self.homeo_lr*gap
                                                                                
        return raw_aff, lat, lat_correlations, x_tiles
    
    def get_rfs(self):
        rfs = self.rfs.detach()
        return rfs / rfs.shape[1] #(rfs.sum(1,keepdim=True) + 1e-5)
    
    def get_lat_weights(self):
        lat_weights = self.lat_weights.detach()
        return lat_weights / lat_weights.shape[1] #(lat_weights.sum(1,keepdim=True) + 1e-5)