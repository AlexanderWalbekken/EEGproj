import numpy as np

import mne

from pre import Event_ids, direct, list_files, common
from pre import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels
from joblib import Parallel, delayed
import itertools as it


##
Tresh_list = np.arange(1,5,step= 0.2)
##

##
G1_ids = ['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V']
G2_ids = G1_ids

G1_subgroup = Speech
g2_subgroi = Non_speech
##

