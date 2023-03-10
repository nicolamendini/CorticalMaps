# Author: Nicola Mendini
# Parameters of the model

import numpy as np
from maps_helper import *

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
EXPANSION = 0.67
ITERS = 30
# relative to CROPSIZE [0,1]
RF_STD = 4
N_BATCHES = 3
LR = 1e-4
HOMEO_TIMESCALE = 0.999
TARGET_LR_DEC = 200
EXC_STD = 0.1
# TARGET_STRENGTH = 0.8 is tested and proven to be the best
TARGET_STRENGTH = 0.8
TARGET_ACT = 0.015
INH_EXC_BALANCE = 0.55

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
EXC_SCALE = max(round((EXC_STD)*5),1)
EXC_SCALE = oddenise(EXC_SCALE)

# dilation is twice the exc scale
DILATION = max(int(EXC_STD*2.5),1)
COMPRESSION = round(EXPANSION*3/2)
COMPRESSION = DILATION

INH_SCALE = round(RF_STD*EXPANSION/DILATION*5)
INH_SCALE = oddenise(INH_SCALE)

RF_SIZE = round(RF_STD*EXPANSION/round(EXPANSION)*5)
RF_SIZE = oddenise(RF_SIZE)

CORR_SAMPLES = 5
STATS_FREQ = 1
MAP_SAMPLS = 1000
MAPCHOP = 12
MAPCHOP = 0 if EXPANSION<1 else MAPCHOP
RECO = False if EXC_SCALE<1 else RECO
INPUT_SIZE = 96
GRID_SIZE = round(CROPSIZE*EXPANSION)
GRID_SIZE = evenise(GRID_SIZE)