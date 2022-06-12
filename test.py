import numpy as np

import mne

from joblib import Parallel, delayed
import itertools as it

from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot, clustersSave

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#DANGEROUS but getting error here


############################
# Choosing data here
S2 = False
############################
if S2:
    from files_info_Study2 import Event_ids, direct, list_files, common
    from files_info_Study2 import Speech, Non_speech
    from Find_bads_and_interpolate import All_epochs, all_channels
else:
    from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
    All_epochs = allEpochs

###
test_type="corr"
tresh_list = [1.6,1.8,2,2.2,3]
tail_list = [1,-1]
per_perm_n = 200
p_acc = 0.10
###
f_vars = {"freqs":np.arange(4,8 +2,2),"n_cycles":5} # "+2" since the last step is excluded
###
G1_ids = ['audiovisual/low']
G2_ids = ['audiovisual/med']
G3_ids = ['audiovisual/high']

#
G1_subgroup = subjectIDs
G2_subgroup = subjectIDs
G3_subgroup = subjectIDs
###


if S2:
    X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids],
                                     All_epochs, crop_post= [0,0.500], freq_vars=f_vars)
else: #S4 (SNR) data tested
    X12, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids, G2_ids],
                                     allEpochs, crop_post= [0.58+0,0.58+0.500], baseline=[0.4, 0.5], freq_vars = f_vars)
    X23, tfr_epochs = createGroupsFreq([G2_subgroup, G3_subgroup], [G2_ids, G3_ids],
                                     allEpochs, crop_post= [0.58+0,0.58+0.500], baseline=[0.4, 0.5], freq_vars = f_vars)
assert np.all(X12[1] == X23[0]), "Nonequal MID"
X = [X12[0],X12[1],X23[1]]

#Testing starts
for loop_tail in tail_list:
    for i, loop_tresh in enumerate(tresh_list):
        # Flipping sign of treshold if the tail is negatative (insted of manually doing it)
        if loop_tail == -1:
            loop_tresh = -loop_tresh
        
        T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=per_perm_n, 
                                                            thresh = loop_tresh, tail = loop_tail, seed = 4, ttype=test_type)
        
        fol_name = f"Tresh{loop_tresh :.1f}_tail={loop_tail}__perms={per_perm_n}__type={test_type}"
        clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, 
                    p_accept= p_acc, min_ch_num = 3,
                    show=False, save = True, folder= fol_name )
        
        clustersSave(T_obs, clusters, cluster_p_values, H0, tfr_epochs, folder = fol_name)

        # If no significant clusters
        if len(H0)>0:
            min_p = np.min(cluster_p_values)
            if min_p > p_acc:
                p_insig = np.sort(cluster_p_values)[:(min(10,len(cluster_p_values)))][-1]
                clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, min_ch_num = 3, p_accept= p_insig, 
                    show=False, save = True, folder=f"INSIG_Tresh{loop_tresh :.1f}_tail={loop_tail}" )


#For plotting histogram after test
if len(H0)>0:
    H_ind = min(len(H0),10)
    print(np.sort(H0)[:H_ind])
    print(np.sort(H0)[-H_ind:])
    print(H0[0])
    print("Minimum p-val: ",np.min(cluster_p_values))
    
    import matplotlib.pyplot as plt
    plt.close('all') 
    plt.figure()
    plt.hist(H0,bins=1000)
    plt.axvline(x = H0[0],linestyle = "dotted", color = 'r', label = 'First')
    plt.show()
else:
    print("Last run no clusters")