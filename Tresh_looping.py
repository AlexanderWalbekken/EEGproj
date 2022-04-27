import numpy as np

import mne

from pre import Event_ids, direct, list_files, common
from pre import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels
from joblib import Parallel, delayed
import itertools as it

#TODO: Fix file name of import
from permutation_test_computing_groups_different_groups import createGroupsFreq, permTestImpT, clustersPlo


##
Tresh_list = np.arange(1,5,step= 0.2)
##

##
G1_ids = ['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V']
G2_ids = G1_ids

G1_subgroup = Speech
G2_subgroup = Non_speech
##

X, tfr_epochs = createGroupsFreq(subgroups = [G1_subgroup , G2_subgroup], e_ids = [G1_ids,G2_ids])