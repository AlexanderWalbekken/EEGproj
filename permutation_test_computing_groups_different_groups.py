# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:57:02 2022

@author: alexa
"""

#Split non_speech and speech

import numpy as np

import mne

from pre import Event_ids, direct, list_files, common
from pre import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels
from joblib import Parallel, delayed
import itertools as it




# --Freq variables--
frequencies = np.arange(4, 38, 2)
n_cycles_morlet = 5 #2
decim_morlet = 3

#--------------------

# --Loading variables--


#channels_test = ["C1"]
crop_start = 0
crop_end = 0.7
baseline_tuple = (None,-0.1)

#-------------------

#or the N1/P2 dataset, between-subjects differences (SM vs. NSM) were tested 
#for each conidition (AV Congruent, AV Incongruent, Auditory, Visual), 
#and for the average of both AVconditions. Subsequently

# Dividing into two groups
Group1 = []
G1 = []

Group2 = []
G2 = []




def powerMinusERP(subject,id_in, All_epochs = All_epochs):
    spe = subject
    ids = id_in
    
    
    epochs = All_epochs[spe]
    ep_avg = epochs[ids].average()
    
    ep_done = epochs[ids].copy().subtract_evoked(ep_avg)
    power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                              return_itc=False,
                                              freqs=frequencies, 
                                              decim=decim_morlet)#.crop(-0.95,0.95)
    
    #change crop to 0.
    
    power = power.apply_baseline(mode="logratio", baseline = (-0.5,-0.2)).crop(-0.95,0.95)
    
    return [power.data, power]


 #['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V']

group_1_prod = it.product(Speech, ['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V'])   
group_2_prod = it.product(Non_speech, ['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V'])  


out1 = Parallel(n_jobs=-1)(delayed(powerMinusERP)(sub,ids)
                    for sub, ids in group_1_prod)

Group1 = [r[0] for r in out1]
G1 = [r[1] for r in out1]

print("Group1 DONE")

out2 = Parallel(n_jobs=-1)(delayed(powerMinusERP)(sub,ids)
                    for sub, ids in group_2_prod)

Group2 = [r[0] for r in out2]
G2 = [r[1] for r in out2]

print("Group2 DONE")

## Plot freq map to check for edge artefacts

tfr_epochs = G1[0]

X = [np.array(Group1), np.array(Group2)]

sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(tfr_epochs.info,"eeg")

adjacency = mne.stats.combine_adjacency(sensor_adjacency, 
                                        len(tfr_epochs.freqs), 
                                        len(tfr_epochs.times))



times_list = tfr_epochs.times
freqs_list = tfr_epochs.freqs


def permTestImp(X = X, thresh = 12, tail = 0, n_perm = 524):
    # X, Treshhold, tail, n_perm
    
    
    from scipy.stats import ttest_ind
    def testFun(*args):
        a, b = args
        t, _ = ttest_ind(a,b)

        return t
    
    # just try increasing the n_jobs number
    # import joblib #i want it to run in paralell more, but i am not well versed in joblib
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X, 
                                       threshold=thresh, tail=tail, #correct tail
                                       n_permutations= n_perm, adjacency = adjacency,
                                       n_jobs = -1)
    
    
    
    #stat_fun endre til t fra scipy remove the p
    
    print(cluster_p_values[cluster_p_values < 0.999999])
    
    print(min(cluster_p_values))
    min_p_indx = np.argmin(cluster_p_values)
    clust_min = clusters[min_p_indx]
    
    
    # We can use the non subtractred (ERP) adn see if the clusters ther make sense
    
    
    ########
    #Computing quantile?
    ## Assuming the test statistic observed is the first (same between different tests)
    obs_H0 = H0[0]
    
    #quanile of the test statistic
    quantil = (H0<obs_H0).mean()

    if __name__ == "__main__":
        print(1-quantil)
            
        def infoCluster(clust):
            
            ch_ret = list(np.array(all_channels)[np.unique(clust[0])])
            times_ret = times_list[np.unique(clust[2])]
            freq_ret = freqs_list[np.unique(clust[1])]
            
            return ch_ret, freq_ret, times_ret
        
        for clusts in np.array(clusters)[cluster_p_values < 0.5]:
            print(infoCluster(clusts))

    return T_obs, clusters, cluster_p_values, H0
    
    
T_obs, clusters, cluster_p_values, H0 = permTestImp()
    
    
    
    
    
    
    
    
    
    
    
      
    
    
    
#https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html    


    
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def clustersPLot(p_acc = 0.05, save = False, fol = "none"):
    
    p_accept = p_acc
    
    F_obs, p_values = T_obs, cluster_p_values
    freqs = freqs_list
    
    
    good_cluster_inds = np.where(p_values < p_accept)[0]
    p_values_good = p_values[p_values < p_accept]
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        
        space_inds,  freq_inds_all, time_inds_all = clusters[clu_idx]
        ### !!Check the order of these to match!!!
        
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds_all)
        freq_inds = np.unique(freq_inds_all)
    
        # get topography for F stat
        f_map = F_obs[:, freq_inds, :].mean(axis=1)
        f_map = f_map[:,time_inds].mean(axis=1)
    
        # get signals at the sensors contributing to the cluster
        sig_times = tfr_epochs.times[time_inds]
    
        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 6))
    
        # create spatial mask
        mask = np.zeros((f_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
    
        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(f_map[:, np.newaxis], tfr_epochs.info, tmin=0)
        f_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                              vmin=np.min, vmax=np.max, show=False,
                              colorbar=False, mask_params=dict(markersize=10))
        image = ax_topo.images[0]
    
        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)
    
        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged F-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))
    
        # add new axis for spectrogram
        ax_spec = divider.append_axes('right', size='300%', pad=1.2)
        #title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
        title = f'{p_values_good[i_clu] :.3f} p-value, Cluster #{i_clu+1}, with {len(ch_inds)} channels'
        if len(ch_inds) > 1:
            title += " (max over channels)"
        F_obs_plot = F_obs[ch_inds, :, :].max(axis=0) #axis = -1
        F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
        """
        F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
            F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]
            """
            
        F_obs_plot_sig[freq_inds_all,time_inds_all] = \
            F_obs_plot[freq_inds_all,time_inds_all]
        
        for f_image, cmap in zip([F_obs_plot, F_obs_plot_sig], ['gray', 'autumn']):
            c = ax_spec.imshow(f_image, cmap=cmap, aspect='auto', origin='lower',
                               extent=[tfr_epochs.times[0], tfr_epochs.times[-1],
                                       freqs[0], freqs[-1]])
        ax_spec.set_xlabel('Time (ms)')
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_title(title)
    
        # add another colorbar
        ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(c, cax=ax_colorbar2)
        ax_colorbar2.set_ylabel('F-stat')
    
        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        
    
        plt.show(block = True)
        
        if save:
            plt.savefig(f"clu{i_clu+1}_p{p_values_good[i_clu] :.3f}_time" +
                    "{:0.2f} - {:0.2f} s)".format(*sig_times[[0, -1]])
                    + ".png")

import os
def saving(name = "None"):
    wd = os.getcwd()
    
    if name == "None":
        inp1 = input("Save? ('No' or foldername)")
    else:
        inp1 = name
    
    if inp1.lower() != "no":
        folder_name = f"Tresh{thresh}_" + inp1
        #import os as os
        #os.ch_wrd(f"plots//{folder_name}")
        os.mkdir(wd + "\\plots\\" + folder_name)
        os.chdir(wd + "\\plots\\" + folder_name)
        clustersPLot(p_acc = p_acc, save = True)
    
    os.chdir(wd)
   

save = False
plot = True
p_acc = 0.1

if plot and __name__ == "__main__":
    clustersPLot(p_acc = p_acc)



###For saving plots###
#run just this part after viewing
if save:
    saving()
        
        
        
        
        
        
        
        
##
    
    
    #831.29401101
    
    # tresh = 12 (22 clusters)--- min p-cluster-val: 0.029 ~~ 0.02862595
    # tresh = 8 (105 clusters)--- min p-cluster-val: 0.09 ~~ 0.0916030534351145
    # tresh = 10 (52 clusters)--- min p-cluster-val: 0.04 ~~ 0.04007633587786259
    # tresh = 4 (286 clusters)--- min p-cluster-val: 0.46 ~~ 0.46564885496183206
    # tresh = 16 (9 clusters)--- min p-cluster-val: 0.15 ~~ 0.14694656488549618
    # tresh = ff (ff clusters)--- min p-cluster-val: ff ~~ ff 