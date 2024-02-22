# Author: Nicola Mendini
# Parameters of the model

import numpy as np
import torch
from maps_helper import *

# RETINA LGN PARAMETERS
# ---------------------------------
RET_LOG_STD = 1
LGN_LOG_STD = 1
HARDGC = True
CG_EPS = 0.001
CG_SATURATION = 0.3
LGN_ITERS = 0

# V1 SETTINGS
# ---------------------------------
KAPPA = 1
CROPSIZE = 38
EXPANSION = 3
GRID_SIZE = evenise(round(CROPSIZE*EXPANSION))
ITERS = 30
# relative to CROPSIZE [0,1]
HOMEO_TIMESCALE = 0.995
RF_STD = 4
EXC_STD = 1.5
# TARGET_STRENGTH = 0.8 is tested and proven to be the best
TARGET_ACT = 0.06
V1_SATURATION = 1
#LRU = np.sqrt(2)*RF_STD
#LRU = None

# TRAINING PARAMETERS
# -----------------------------------
N_BATCHES = 8000
LR = 1e-3
TARGET_LR_DEC = 10
# fast mode is torch.float16
DTYPE = torch.float32

# FLAGS
# ---------------------------------
PRINT = False
KEEP = False
RECO = False
RECOPRINT = False
LONGRUN_MODE = False
LEARNING = True
EVALUATE = True
LAT_RESET = True

# SPARSITY
# dilation is twice the exc scale
DILATION = max(int(EXC_STD*2.5),1)
COMPRESSION = round(EXPANSION*3/2)
DILATION = 1
COMPRESSION = 1

NOISE = torch.linspace(0.5,-9, 2)
NOISE = list(torch.exp(NOISE))
NOISE = []

SPARSITY = [0.1]

RF_SPARSITY = [(i*0.1) for i in range(11)]
RF_SPARSITY = []

# PLOTTING SETTINGS
# -----------------------------------
CORR_SAMPLES = 5
MAPCHOP = 6 * (EXPANSION>1)
STATS_FREQ = 100
EVAL_BATCHES = 1000

# CONTINUOUS TRAINING PARAMETERS
# -----------------------------------
FRAMES_PER_TOY = 1 #162 is the toy set size for snallNORB
LONG_STEP_P = 1
MIN_STEP = 10

# NETWORK AREA
# -----------------------------------
BSIZE = 1024
NN_EPOCHS = 5000
BIAS = True