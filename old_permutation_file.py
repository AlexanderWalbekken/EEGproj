# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:57:02 2022

@author: alexa
"""

#Split non_speech and speech

import numpy as np
#from scipy import stats as stats

import mne

from files_info_Study2 import Event_ids, direct, list_files, common
from files_info_Study2 import Speech, Non_speech

from Find_bads_and_interpolate import All_epochs, all_channels

#mne.spatial_src_adjacency
#mne.stats.spatio_temporal_cluster_test
#mne.stats.summarize_clusters_stc
#mne.datasets.sample


# --Freq variables--
frequencies = np.arange(4, 38, 2)
n_cycles_morlet = 2
decim_morlet = 3

#--------------------

# --Loading variables--
#ids = Event_ids[-1] #One Congruent plus [4]

ids_to_use = [4,7] #[1,4,6,7] #BOth congruent AV

channels_test = ["C1"]
baseline_tuple = (None,-0.1)

#-------------------

#or the N1/P2 dataset, between-subjects differences (SM vs. NSM) were tested 
#for each condition (AV Congruent, AV Incongruent, Auditory, Visual), 
#and for the average of both AVconditions. Subsequently

# Dividing into two groups
Group1 = []
G1 = []

Group2 = []
G2 = []




def powerMinusERP(subject,id_in):
    spe = subject
    ids = id_in
    
    
    epochs = All_epochs[spe]
    ep_avg = epochs[ids].average()
    
    ep_done = epochs.copy().subtract_evoked(ep_avg)
    power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                              return_itc=False,
                                              freqs=frequencies, 
                                              decim=decim_morlet)
    
    power = power.apply_baseline(mode="logratio", baseline = (-100,0)).crop(-0.95,0.95)
    
    return [power.data, power]


from joblib import Parallel, delayed
import itertools as it


out1 = Parallel(n_jobs=-1)(delayed(powerMinusERP)(sub,ids)
                    for sub, ids in it.product(Speech, ids_to_use ))

Group1 = [r[0] for r in out1]
G1 = [r[1] for r in out1]

print("Group1 DONE")

out2 = Parallel(n_jobs=-1)(delayed(powerMinusERP)(sub,ids)
                    for sub, ids in it.product(Non_speech, ids_to_use ))

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


if __name__ == "__main__":
    
    
    ##
    thresh = 10
    ##
    
    
    #831.29401101
    
    # tresh = 12 (22 clusters)--- min p-cluster-val: 0.029 ~~ 0.02862595
    # tresh = 8 (105 clusters)--- min p-cluster-val: 0.09 ~~ 0.0916030534351145
    # tresh = 10 (52 clusters)--- min p-cluster-val: 0.04 ~~ 0.04007633587786259
    # tresh = 4 (286 clusters)--- min p-cluster-val: 0.46 ~~ 0.46564885496183206
    # tresh = 16 (9 clusters)--- min p-cluster-val: 0.15 ~~ 0.14694656488549618
    # tresh = ff (ff clusters)--- min p-cluster-val: ff ~~ ff


    from scipy.stats import ttest_ind
    def testFun(*args):
        a, b = args
        t, _ = ttest_ind(a,b)

        return t

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X, 
                                       threshold=None, tail=0, 
                                       n_permutations=524, adjacency = adjacency,
                                       n_jobs = -1, 
                                       stat_fun = testFun)
    
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
    
    print(1-quantil)
        
    
    def infoCluster(clust):
        
        ch_ret = list(np.array(all_channels)[np.unique(clust[0])])
        times_ret = times_list[np.unique(clust[2])]
        freq_ret = freqs_list[np.unique(clust[1])]
        
        return ch_ret, freq_ret, times_ret
    
    for clusts in np.array(clusters)[cluster_p_values < 0.5]:
        print(infoCluster(clusts))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html    

plot = True
if plot and __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    p_accept = 0.9
    
    F_obs, clusters, p_values, _ = T_obs, clusters, cluster_p_values, H0
    freqs = freqs_list
    
    
    good_cluster_inds = np.where(p_values < p_accept)[0]
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        
        space_inds,  freq_inds, time_inds = clusters[clu_idx]
        ### !!Check the order of these to match!!!
        
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        freq_inds = np.unique(freq_inds)
    
        # get topography for F stat
        f_map = F_obs[:, freq_inds, :].mean(axis=1)
        f_map = f_map[:,time_inds].mean(axis=1)
    
        # get signals at the sensors contributing to the cluster
        sig_times = tfr_epochs.times[time_inds]
    
        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    
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
        title = f'Cluster #{i_clu+1}, with {len(ch_inds)} channels'
        if len(ch_inds) > 1:
            title += " (max over channels)"
        F_obs_plot = F_obs[ch_inds, :, :].max(axis=0) #axis = -1
        F_obs_plot_sig = np.zeros(F_obs_plot.shape) * np.nan
        F_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
            F_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]
    
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(sensor_adjacency.toarray(), cmap='gray', origin='lower',
              interpolation='nearest')
    ax.set_xlabel('{} ----meters'.format(len(ch_names)))
    ax.set_ylabel('{} ----meters'.format(len(ch_names)))
    #ax.xaxis.set_ticks(ch_names)
    #ax.yaxis.set_ticks(ch_names)
    ax.set_title('Between-sensor adjacency')
    fig.tight_layout()
    
    
    
    comb1 = mne.combine_evoked(G1, weights = [1]*len(G1))
    comb2 = mne.combine_evoked(G2, weights = [1]*len(G2))
    
    comb1.plot_joint(title= "G1")
    comb2.plot_joint(title= "G2")
    """
    
    """
    good_cluster_inds = np.arange(0,len(clusters))[cluster_p_values < 0.9]
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        space_inds, time_inds, freq_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)
        freq_inds = np.unique(freq_inds)
        
        t_map = T_obs[:, freq_inds, time_inds].mean(axis=0)
        
        
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        
        # get signals at the sensors contributing to the cluster
        sig_times = times_list[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        
        
        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))
    
        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], tfr_epochs.info, tmin=0)
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
        
        
        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.show()
        
        """
    


    
    
    
    








"""
#Add both event as individal samples
for spe in Speech:
    
    for ids in ids_to_use:
        
        epochs = All_epochs[spe]
        ep_avg = epochs[ids].average()
        
        ep_done = epochs.copy().subtract_evoked(ep_avg)
        power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                                  return_itc=False,
                                                  freqs=frequencies, 
                                                  decim=decim_morlet)
        
        power = power.apply_baseline(mode="logratio", baseline = (-100,0))
        
        
        G1.append(power)
        Group1.append(power.data)
    
for spe in Non_speech:
    
    for ids in ids_to_use:
        
        epochs = All_epochs[spe]
        ep_avg = epochs[ids].average()
        
        ep_done = epochs.copy().subtract_evoked(ep_avg)
        power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                                  return_itc=False,
                                                  freqs=frequencies, 
                                                  decim=decim_morlet)
        
        power = power.apply_baseline(mode="logratio", baseline = (-100,0))
        
        
        G2.append(power)
        Group2.append(power.data)

"""