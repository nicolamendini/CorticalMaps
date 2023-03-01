# Author: Nicola Mendini
# Parameters of the model

import numpy as np

# RETINA LGN PARAMETERS
# ---------------------------------

RET_LOG_STD = 1
LGN_LOG_STD = 1
HARDGC = True
CG_EPS = 1
SATURATION = 1/2
LGN_ITERS = 0

# PARAMETER CONSOLE
# ---------------------------------

CROPSIZE = 38
EXPANSION = 3.3
ITERS = 20
K_STD = 4
N_BATCHES = 1
LR = 1e-4
HOMEO_TIMESCALE = 0.999
TARGET_LR_DEC = 500
EXC_STD = 1.67
TARGET_STRENGTH = 0.8
TARGET_ACT = 0.01
AFF_STRENGTH = 1

# FLAGS
# ---------------------------------

PRINT = False
KEEP = False
RECO = True
RECOPRINT = False
LONGRUN_MODE = False

# MINOR PARAMETERS
# ---------------------------------

LEARNING = True
LR_DECAY = (1-1/N_BATCHES*np.log(TARGET_LR_DEC))

# SCALES OF THE INTERACTIONS
SCALE = round((EXC_STD)*5)
SCALE = SCALE+1 if SCALE%2==0 else SCALE

# dilation is twice the exc scale
DILATION = max(int(EXC_STD*3),1)
COMPRESSION = round(EXPANSION*3/2)

INH_SCALE = round(K_STD*EXPANSION/DILATION*5+1)
INH_SCALE = INH_SCALE+1 if INH_SCALE%2==0 else INH_SCALE

KSIZE = round(K_STD*5)
KSIZE = KSIZE+1 if KSIZE%2==0 else KSIZE

CORR_SAMPLES = 5
STATS_FREQ = 1
MAP_SAMPLS = 1000
CSTD = 1
MAPCHOP = 20
MAPCHOP = 0 if EXPANSION<1 else MAPCHOP
RECO = False if SCALE<1 else RECO
INPUT_SIZE = 96
GRID_SIZE = round(CROPSIZE*EXPANSION)
GRID_SIZE = GRID_SIZE if GRID_SIZE%2==0 else GRID_SIZE+1