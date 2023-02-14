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

CROPSIZE = 48
EXPANSION = 6
ITERS = 20
K_STD = 4
N_BATCHES = 40000
LR = 1e-6
HOMEO_TIMESCALE = 0.999
TARGET_LR_DEC = 1
EXC_STD = 3
TARGET_STRENGTH = 0.8
COMPRESSION = 2
TARGET_ACT = 0.01
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
MINSIZE = 11
LEARNING = True
LR_DECAY = (1-1/N_BATCHES*np.log(TARGET_LR_DEC))

# SCALES OF THE INTERACTIONS
SCALE = round((EXC_STD)*5)
SCALE = SCALE+1 if SCALE%2==0 else SCALE

# dilation is twice the exc scale
DILATION = max(int(EXC_STD*2),1)

#INH_SCALE = 21
INH_SCALE = round(K_STD*EXPANSION/DILATION*5)
INH_SCALE = INH_SCALE+1 if INH_SCALE%2==0 else INH_SCALE

KSIZE = round(K_STD*5)
KSIZE = KSIZE+1 if KSIZE%2==0 else KSIZE

CORR_SAMPLES = 5
STATS_FREQ = 1
MAP_SAMPLS = 1000
CSTD = 1
MAPCHOP = 25
RECO = False if SCALE<1 else RECO
INPUT_SIZE = 96
GRID_SIZE = round(CROPSIZE*EXPANSION)