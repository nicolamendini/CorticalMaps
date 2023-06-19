# Author: Nicola Mendini
# Parameters of the model

import numpy as np
import torch
from maps_helper import *

# RETINA LGN PARAMETERS
# ---------------------------------
RET_LOG_STD = 1
LGN_LOG_STD = 1.
HARDGC = True
CG_EPS = 0.005 #4
CG_SATURATION = 0.5
LGN_ITERS = 0

# V1 SETTINGS
# ---------------------------------
KAPPA = 1.5
CROPSIZE = 38
EXPANSION = 0.67
GRID_SIZE = evenise(round(CROPSIZE*EXPANSION))
ITERS = 30
# relative to CROPSIZE [0,1]
HOMEO_TIMESCALE = 0.995
RF_STD = 4
EXC_STD = 0.01
# TARGET_STRENGTH = 0.8 is tested and proven to be the best
TARGET_ACT = 0.03
V1_SATURATION = 0.8

# TRAINING PARAMETERS
# -----------------------------------
N_BATCHES = 50000
LR = 1e-3
TARGET_LR_DEC = 10
# fast mode is torch.float16
DTYPE = torch.float32

# FLAGS
# ---------------------------------
PRINT = False
KEEP = False
RECO = True
RECOPRINT = False
LONGRUN_MODE = False
LEARNING = True
STATS = True

# SPARSITY
# dilation is twice the exc scale
DILATION = max(int(EXC_STD*2.5),1)
COMPRESSION = round(EXPANSION*3/2)
DILATION = 1
COMPRESSION = 1

# PLOTTING SETTINGS
# -----------------------------------
CORR_SAMPLES = 5
MAP_SAMPLS = 1000
MAPCHOP = 6 * (EXPANSION>1)

# CONTINUOUS TRAINING PARAMETERS
# -----------------------------------
FRAMES_PER_TOY = 1 #162 is the toy set size for snallNORB
LONG_STEP_P = 1
MIN_STEP = 10

