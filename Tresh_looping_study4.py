import numpy as np

import mne

from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
from joblib import Parallel, delayed
import itertools as it

# TO-DO: Fix file name of import
from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot

##
tresh_list = [1.2,1.8,2.2,3]
per_perm_n = 200
tail_list = [-1,0,1]
p_acc = 0.10
##

##
#G1_ids = ['visual/b']
#G2_ids = ['auditory/b']

G1_ids = ['audiovisual/high']
G2_ids = ['audiovisual/low']

G1_subgroup = subjectIDs
G2_subgroup = subjectIDs
##
freq_variables = freq_vars = {"freqs":np.arange(4, 20+2, 2), "n_cycles": 5}


X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], allEpochs, crop_post= [0.58+0,0.58+0.500], baseline=[0.4, 0.5], freq_vars = freq_variables)


for loop_tail in tail_list:
    for i, loop_tresh in enumerate(tresh_list):
        if loop_tail == -1:
            loop_tresh = -loop_tresh
        
        T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=per_perm_n, 
                                                            thresh = loop_tresh, tail = loop_tail, seed = 4,
                                                            ttype="pairedT")
        
        clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, min_ch_num = 3, p_accept= p_acc, 
                    show=False, save = True, folder=f"Tresh{loop_tresh :.1f}_tail={loop_tail}" )
        
        if len(H0)>0:
            min_p = np.min(cluster_p_values)
            if min_p > p_acc:
                p_insig = np.sort(cluster_p_values)[:(min(10,len(cluster_p_values)))][-1]
                clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, min_ch_num = 3, p_accept= p_insig, 
                    show=False, save = True, folder=f"INSIG_Tresh{loop_tresh :.1f}_tail={loop_tail}" )
        
        


import matplotlib.pyplot as plt
plt.figure()
plt.hist(H0,bins=1000)
print(H0[0])
print("Minimum p-val: ",min_p)
plt.axvline(x = H0[0],linestyle = "dotted", color = 'r', label = 'First')
plt.show()