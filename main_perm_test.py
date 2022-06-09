# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:57:02 2022

@author: alexa
"""

import numpy as np
import mne
from torch import layout

from files_info_Study2 import Speech, Non_speech

from joblib import Parallel, delayed
import itertools as it


#All_epochs are all the subjects providede as a dictionary
#subgroups [keys1,keys2], where keys* are then lists(or array) of the keys to be used in the groups
#e_ids is a list [ids1,ids2], where ids* are list of the event ids to use for each group
def createGroupsFreq(subgroups, e_ids, All_epochs, baseline = [-0.5,-0.2], 
                     freq_vars = {"freqs":np.arange(4, 38, 2), "n_cycles": 5}, crop_post = None, crop_pre = None):

    # Dividing into two groups
    Group1 = []
    G1 = []
    
    Group2 = []
    G2 = []
    
    
    def powerMinusERP(subject,id_in, All_epochs = All_epochs, baseline = baseline):
        # --Freq variables--
        frequencies = freq_vars["freqs"] #np.arange(4, 38, 2)
        n_cycles_morlet = freq_vars["n_cycles"]  #5 #2
        decim_morlet = 3
    
        spe = subject
        ids = id_in
        
        bas = baseline
        if crop_pre == None:
            epochs = All_epochs[spe]
        else:
            epochs = All_epochs[spe].crop(crop_pre[0],crop_pre[1])
        ep_avg = epochs[ids].copy().average()
        
        ep_done = epochs[ids].copy().subtract_evoked(ep_avg)
        power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                                  return_itc=False,
                                                  freqs=frequencies, 
                                                  decim=decim_morlet)
        
        power = power.apply_baseline(mode="logratio", baseline = (bas[0],bas[1]))
        if crop_post != None:
            power = power.crop(crop_post[0],crop_post[1])
        
        return [power.data, power]
    
    #Creting the groups to compute
    group_1_prod = it.product(subgroups[0], e_ids[0])   
    group_2_prod = it.product(subgroups[1], e_ids[1])  
    
    
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
    
    tfr_epochs = G1[0]
    
    X = [np.array(Group1), np.array(Group2)]

    return X, tfr_epochs


#-> X is the grouped data as it needs to be grouped in the mne.stats.permutation_cluster_test
#-> tfr_epochs is an example epoch (Evoked object) that has been morlet transformed
def permTestImpT(X, tfr_epochs, thresh = 12, tail = 0, n_perm = 524, ttype = "T", seed = None, num_categories = None):
    # X, Treshhold, tail, n_perm
    times_list = tfr_epochs.times
    freqs_list = tfr_epochs.freqs
    
    sensor_adjacency, ch_names = mne.channels.find_ch_adjacency(tfr_epochs.info,"eeg")

    adjacency = mne.stats.combine_adjacency(sensor_adjacency, 
                                            len(tfr_epochs.freqs), 
                                            len(tfr_epochs.times))
    
    # Changing statistical function to use for each "point"
    # Treshhold must be tailored to the given stat-funtion
    #num_categories = num_categories #Why????? Global vs local scope i guess in the testFun?
    if ttype == "corr":
        X_flat = [X[il].reshape(X[il].shape[0],-1) for il in range(len(X))]
        OUTnum_categories = list(range(1,1+len(X_flat)))
        OUTX_arr = np.hstack([OUTnum_categories[i]*np.ones(len(X_flat[i])) for i in range(len(OUTnum_categories))])
        import statsmodels.api as sm
        
        OUTX_arr_constantadd = sm.add_constant(OUTX_arr)
        def OLS_arr_func(a, X_arr = OUTX_arr_constantadd):
            model = sm.OLS(a,X_arr)
            results = model.fit()
            return results.tvalues[-1] #last, should be turned to "1" after testing, but works either way
        
            return
        
        def testFun(*args,X_arr = OUTX_arr, OLS_arr = OLS_arr_func):
            #if num_categories == None:
            #num_categories = list(range(1,1+len(args)))
            
            #X_arr = np.hstack([num_categories[i]*np.ones(len(args[i])) for i in range(len(num_categories))])
            
            Y_arr = np.vstack([arr for arr in args])
            """
            t_vals = np.empty(Y_arr.shape[1])
            for i in range(Y_arr.shape[1]):
                model = sm.OLS(Y_arr[:,i],X_arr)

                results = model.fit()
                slope = results.params
                t_vals[i] = results.tvalues
                """
            #Numpy implementation could be faster?
            t_vals_arr = np.apply_along_axis(OLS_arr,0,Y_arr)
            
            return t_vals_arr
        
    elif ttype == "F":
         testFun = None
    else:
        from scipy.stats import ttest_ind
        def testFun(*args):
            a, b = args
            t, _ = ttest_ind(a,b)

            return t
    
    # permittion test, runs in parallel
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(X, 
                                       threshold=thresh, tail=tail, #correct tail
                                       n_permutations= n_perm, adjacency = adjacency,
                                       n_jobs = -1, seed = seed,
                                       stat_fun = testFun, buffer_size=None) #changed buffer size to try and debug
    

    if __name__ == "__main__" and False:
        print(cluster_p_values[cluster_p_values < 0.999999])
    
        print(min(cluster_p_values))
        min_p_indx = np.argmin(cluster_p_values)
        
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

    return T_obs, clusters, cluster_p_values, H0
    
    
#https://mne.tools/stable/auto_tutorials/stats-sensor-space/75_cluster_ftest_spatiotemporal.html    
    
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#tfr_epochs is an example epoch (Evoked object) that has been morlet transformed
def clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs,
                 p_accept = 0.05, save = False, folder = None, ttype = "T", show = True, min_ch_num = None, topo_dim = [2,4]):

    
    F_obs, p_values = T_obs, cluster_p_values
    #F is used in the framework as a default
    #However, we use t-test for our permutation test
    freqs = tfr_epochs.freqs
    
    
    good_cluster_inds = np.where(p_values <= p_accept)[0]
    p_values_good = p_values[p_values <= p_accept]
    
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        
        space_inds,  freq_inds_all, time_inds_all = clusters[clu_idx]
        ### !!Check the order of these to match!!!
        
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds_all)
        freq_inds = np.unique(freq_inds_all)
        
        # Checking minimun channels before plotting
        if not min_ch_num == None:
            if len(ch_inds) < min_ch_num:
                continue #skipping cluster if not enough channels
    
        # get topography for F stat
        f_map_total = F_obs[:, freq_inds, :].mean(axis=1)
        f_map = f_map_total[:,time_inds].mean(axis=1)
    
        # get signals at the sensors contributing to the cluster
        sig_times = tfr_epochs.times[time_inds]
    
        # Initialize MAIN figure and subfigure
        plt.rcParams.update({'font.size': 8})
        fig_main = plt.figure(constrained_layout=True, figsize=(10, 5))
        try:
            subfigs = fig_main.subfigures(2, 1, hspace=0.1, #This fraction is not guaranteed to work with alll layouts
                              height_ratios= [1.1/topo_dim[1],topo_dim[0]/topo_dim[1]]) #adjusting sizes also
        except:
            raise Exception("Probably a VERSION ERROR \n Need matplotlib v3.4 or higher for subfigures")
        
        ax_topo = subfigs[0].subplots(1, 1)
    
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
            f"Avg. {ttype}-map"+' ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        #felxible spectrogram size
        prct = ((topo_dim[1]-1)*100) + 60
        sizeprct = str(prct) + "%"
        
        # add new axis for spectrogram
        ax_spec = divider.append_axes('right', size=sizeprct, pad=1.2)
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
                                       freqs[0], freqs[-1]+(freqs[-1] - freqs[-2])])
        ax_spec.set_xlabel('Time (s)')
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_title(title)
        
        #fixing frequency ticks (attempt 1)
        ##ax_spec.set_yticks(freqs)
        #fixing frequency ticks (attempt 2)
        
        freq_step = (freqs[-1] - freqs[-2])
        ax_spec.set_yticks(list(freqs))
        ax_spec.set_yticklabels('')
        ax_spec.set_yticks(freqs + (freq_step/2), minor=True)
        
        #This may be overkill just for the plot...
        """
        if len(freqs)//10 < 1:
            ylabs = [str(f_l) for f_l in freqs]
        else:"""
        ratio = len(freqs)//10
        ylabs = [str(freqs[j]) if j%(ratio+1) == 0 else '' for j in range(len(freqs))]
        ax_spec.set_yticklabels(ylabs, minor=True)
        
    
        # add another colorbar
        ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(c, cax=ax_colorbar2)
        ax_colorbar2.set_ylabel(f'{ttype}-stat')
    
        # clean up viz [Has trouble working with subfigures]
        #mne.viz.tight_layout()
        #fig.subplots_adjust(bottom=.05)
        
        
        ####
        # Topografic vizualizations for different sample times
        ####
        #topo_dim = [2,4] ## moved to arguments
        ax_snaps = subfigs[1].subplots(topo_dim[0],topo_dim[1])
        ax_snaps_flat = ax_snaps.flat
        
        snap_inds = np.linspace(0, (len(time_inds)-1),num = (topo_dim[0]*topo_dim[1]), dtype="int")
        for i, snap_shot in enumerate(time_inds[snap_inds]):
            f_map = f_map_total[:,snap_shot] #.mean(axis=1) #Assume the meam isn not needed for one time
            
            #Choosing only current channels
            this_ch_inds = np.unique(space_inds[time_inds_all == snap_shot])
            
            # create spatial mask
            mask = np.zeros((f_map.shape[0], 1), dtype=bool)
            mask[this_ch_inds, :] = True
            
            t_s = tfr_epochs.times[snap_shot] #TODO: DOuble check this is completly accurate
        
            f_evoked = mne.EvokedArray(f_map[:, np.newaxis], tfr_epochs.info, tmin=t_s)
            f_evoked.plot_topomap(times=t_s, mask=mask, axes=ax_snaps_flat[i], cmap='Reds',
                              vmin=np.min, vmax=np.max, show=False,
                              colorbar=False, mask_params=dict(markersize=10))
        
        if show:
            plt.show(block = False) #False in Spyder, True in VS-code for now...
        
        if save:
            import os
            wd = os.getcwd()
            
            # Creating plots folder if it doesnt exist
            if not os.path.exists(wd + "\\plots"):
                os.mkdir(wd + "\\plots")
            
            # Using datetime to name folder, if not spesified (should rather be spesified)
            if folder == None:
                import datetime
                folder_name = datetime.datetime.now()
            else:
                folder_name = folder
                
            curr_path = wd + "\\plots\\" + folder_name
            exists = os.path.exists(curr_path)
            #print(exists)
            if not exists:
                os.mkdir(curr_path)
            os.chdir(curr_path) #CHANGING!
            
            n_chan = len(ch_inds)
            percnt_p = (p_values_good[i_clu] * 100)
            n_clu = i_clu+1
            
            # Saving file with cluster information as name
            plt.savefig(f"clu{ n_clu }_p{percnt_p :.2f}%_max_{n_chan}chan" +
                        "(t="+"{:0.2f}-{:0.2f})".format(*sig_times[[0, -1]])
                    + ".png")
            
            
            os.chdir(wd) #CHANGING BACK!


##############
####      ####
##   MAIN   ##
####      ####
##############

#For the N1/P2 dataset, between-subjects differences (SM vs. NSM) were tested 
#for each conidition (AV Congruent, AV Incongruent, Auditory, Visual), 
#and for the average of both AVconditions. Subsequently

save = False
plot = True
p_acc = 0.1

#['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V']
if __name__ == "__main__":
    #NB!: Everything is multithreaded, so it will have a large impact on 
    # less powerful systems.
    
    #ALSO: need matplotliv v3.4 to plot
    
    ##
    G1_ids = ['Tabi_A_Tabi_V','Tagi_A_Tabi_V'] + ['Tagi_A_Tagi_V', 'Tabi_A_Tagi_V']
    G2_ids = G1_ids
    
    G1_subgroup = Speech
    G2_subgroup = Non_speech
    
    from Find_bads_and_interpolate import All_epochs, all_channels
    ##
    
    X, tfr_epochs = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], All_epochs)
        
    T_obs, clusters, cluster_p_values, H0 = permTestImpT(X, tfr_epochs, n_perm=300, thresh=2.1)
    
    if plot:
        clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, p_accept= p_acc)

        

        
##
    
    
    #831.29401101
    
    # tresh = 12 (22 clusters)--- min p-cluster-val: 0.029 ~~ 0.02862595
    # tresh = 8 (105 clusters)--- min p-cluster-val: 0.09 ~~ 0.0916030534351145
    # tresh = 10 (52 clusters)--- min p-cluster-val: 0.04 ~~ 0.04007633587786259
    # tresh = 4 (286 clusters)--- min p-cluster-val: 0.46 ~~ 0.46564885496183206
    # tresh = 16 (9 clusters)--- min p-cluster-val: 0.15 ~~ 0.14694656488549618
    # tresh = ff (ff clusters)--- min p-cluster-val: ff ~~ ff 