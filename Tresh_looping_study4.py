import numpy as np

import mne

from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
from joblib import Parallel, delayed
import itertools as it

# TO-DO: Fix file name of import
from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot

##
tresh_list = [3.8]
per_perm_n = 200
p_acc = 0.09
##

##
#G1_ids = ['visual/b']
#G2_ids = ['auditory/b']

G1_ids = ['audiovisual/high']
G2_ids = ['audiovisual/low']

G1_subgroup = subjectIDs
G2_subgroup = subjectIDs
##
freq_variables = freq_vars = {"freqs":np.arange(4, 38, 2), "n_cycles": 3}


X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], allEpochs, baseline=[0.4, 0.5], freq_vars = freq_variables)


loop_tail = 0
for i, loop_tresh in enumerate(tresh_list):
    if loop_tail == -1:
        loop_tresh = -loop_tresh
    
    T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=per_perm_n, 
                                                         thresh = loop_tresh, tail = loop_tail, seed = 4)
    clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, min_ch_num = 3, p_accept= p_acc, 
                 show=False, save = True, folder=f"Tresh{loop_tresh :.1f}_tail={loop_tail}" )


import matplotlib.pyplot as plt
plt.figure()
plt.hist(H0,bins=1000)
print(H0[0])
print("Minimum p-val: ",np.min(cluster_p_values))
plt.axvline(x = H0[0],linestyle = "dotted", color = 'r', label = 'First')
plt.show()

print("Minimum p-val: ",np.min(cluster_p_values))