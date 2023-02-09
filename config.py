# Author: Nicola Mendini
# Parameters of the model

import numpy as np

# RETINA LGN PARAMETERS
# ---------------------------------

RET_LOG_STD = 1
RET_STRENGTH = 5
LGN_STRENGTH = 0
RET_LOG_STD = 1
HARDGC = True
CG_STD = 2
CG_SCALER = 0.5
CG_EPS = 0.03
SAME_MEAN = False

# PARAMETER CONSOLE
# ---------------------------------

EXPANSION = 2
ITERS = 16
KSIZE = 21
N_BATCHES = 20000
LR = 1e-4
HOMEO_TIMESCALE = 0.999
TARGET_LR_DEC = 20
SCALE = 3
INH_SCALE = 4.2
TARGET_STRENGTH = 0.8
COMPRESSION = 2
TARGET_ACT = 0.015
BASE_CORR = 0

# FLAGS
# ---------------------------------

PRINT = False
KEEP = False
RECO = True
RECOPRINT = False
LONGRUN_MODE = False

# MINOR PARAMETERS
# ---------------------------------

RECO_STRENGTH = 1
STRENGTH = 1
# nyquist argument
MINSIZE = 11
TRANSLATE = 1/3
LEARNING = True
LR_DECAY = (1-1/N_BATCHES*np.log(TARGET_LR_DEC))
DILATION = max(int(SCALE*2),1)
CORR_SAMPLES = 5
STATS_FREQ = 1
MAP_SAMPLS = 1000
CSTD = 1
MAPCHOP = 25
SIZE = round(INH_SCALE*5) 
RECO = False if SCALE<1 else RECO
INPUT_SIZE = 96
GRID_SIZE = round(INPUT_SIZE*(1-TRANSLATE)*EXPANSION)