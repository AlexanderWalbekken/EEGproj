import numpy as np

import mne

from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
from joblib import Parallel, delayed
import itertools as it

#TODO: Fix file name of import
from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot


##
Tresh_list = np.arange(1,5,step= 0.2)
##

##
G1_ids = ['visual/b']
G2_ids = ['auditory/b']

G1_subgroup = subjectIDs
G2_subgroup = subjectIDs
##

X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], allEpochs)