import numpy as np

import mne

from files_info_Study2 import Event_ids, direct, list_files, common
from files_info_Study2 import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels
from joblib import Parallel, delayed
import itertools as it

from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot, clustersSave

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#DANGEROUS but getting error here

############################
# Choosing data here
S2 = True
############################
###
test_type="T"
tresh_list = [1.2,1.6,1.8,2,2.2,3]
tail_list = [1,-1]
per_perm_n = 200
p_acc = 0.10
###
f_vars = {"freqs":np.arange(4,8 +2,2),"n_cycles":5} # "+2" since the last step is excluded
###


if S2:
    from files_info_Study2 import Event_ids, direct, list_files, common
    from files_info_Study2 import Speech, Non_speech
    from Find_bads_and_interpolate import All_epochs, all_channels
    crop_post =[0,0.500]
    baseline = [-0.5,-0.2]
    ##
    G1_ids = ["audiovisual/congruent"] # ['Tabi_A_Tabi_V','Tagi_A_Tagi_V'] #+ ['Tagi_A_Tabi_V', 'Tabi_A_Tagi_V']
    G2_ids = G1_ids

    G1_subgroup = Speech
    G2_subgroup = Non_speech
    ##
else:
    from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
    All_epochs = allEpochs
    crop_post = [0.58+0,0.58+0.500]
    baseline = [0.58 -0.5,0.58 -0.2] #shifted to onset explicitly
    ##
    G1_ids = ['audiovisual/high']
    G2_ids = ['audiovisual/low']

    G1_subgroup = subjectIDs
    G2_subgroup = subjectIDs
    ##
    
#Groups created with subtreacted ERP
X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], All_epochs, 
                                 crop_post= crop_post, freq_vars=f_vars, baseline=baseline)

#Testing starts
for loop_tail in tail_list:
    for i, loop_tresh in enumerate(tresh_list):
        # Flipping sign of treshold if the tail is negatative (insted of manually doing it)
        if loop_tail == -1:
            loop_tresh = -loop_tresh
        
        T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=per_perm_n, 
                                                            thresh = loop_tresh, tail = loop_tail, seed = 4, 
                                                            ttype=test_type)
        
        fol_name = f"Tresh{loop_tresh :.1f}_tail={loop_tail}__perms={per_perm_n}__type={test_type}"
        clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, 
                    p_accept= p_acc, min_ch_num = 3,
                    show=False, save = True, folder= fol_name )

        # If no significant clusters
        if len(H0)>0:
            min_p = np.min(cluster_p_values)
            if min_p > p_acc:
                fol_name = "INSIG_" + fol_name
                
                p_insig = np.sort(cluster_p_values)[:(min(10,len(cluster_p_values)))][-1]
                clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, min_ch_num = 3, p_accept= p_insig, 
                    show=False, save = True, folder=fol_name )
                
        
        clustersSave(T_obs, clusters, cluster_p_values, H0, tfr_epochs, folder = fol_name)


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