import numpy as np

import mne

from files_info_Study2 import Event_ids, direct, list_files, common
from files_info_Study2 import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels
from joblib import Parallel, delayed
import itertools as it

from main_perm_test import createGroupsFreq, permTestImpT, clustersPlot

#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#DANGEROUS but getting error here

##
tresh_list = [None]
per_perm_n = 300
p_acc = 0.10
##

##
G1_ids = ['Tabi_A_Tabi_V','Tagi_A_Tagi_V'] + ['Tagi_A_Tabi_V', 'Tabi_A_Tagi_V']
G2_ids = G1_ids

G1_subgroup = Speech
G2_subgroup = Non_speech
##

X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], All_epochs)

loop_tail = 0#1#-1
for i, loop_tresh in enumerate(tresh_list):
    # Flipping sign of treshold if the tail is negatative (insted of manually doing it)
    if loop_tail == -1:
        loop_tresh = -loop_tresh
    
    T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=per_perm_n, 
                                                         thresh = loop_tresh, tail = loop_tail, seed = 4, ttype="F")
    
    clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, 
                 p_accept= p_acc, min_ch_num = 3,
                 show=False, save = True, folder="none")#f"Tresh{loop_tresh :.1f}_tail={loop_tail}" )

print(np.sort(H0)[:10])
print(np.sort(H0)[-10:])
print(H0[0])
print("Minimum p-val: ",np.min(cluster_p_values))

  
import matplotlib.pyplot as plt
plt.figure()
plt.hist(H0,bins=1000)
plt.axvline(x = H0[0],linestyle = "dotted", color = 'r', label = 'First')
plt.show()