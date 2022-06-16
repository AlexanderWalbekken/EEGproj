
    
import numpy as np
import mne    
import matplotlib.pyplot as plt
from main_perm_test import createGroupsFreq
from joblib import Parallel, delayed
import itertools as it

def S4ERPplot(extra=False):
    
    ids_test = ["audiovisual/bg/high","audiovisual/bg/med","audiovisual/bg/low", "auditory/b"]
    labels = ["AV-high","AV-mid","AV-low","Aud ('B')"]

    erps = []

    #ch_pick = [f"E{i}" for i in [4,5,6,11,12,19]]
    #ch_pick = [f"E{i}" for i in [5,6,11,12,13,112]]

    ch_pick = [f"E{i}" for i in [3,4,5,6,7,10,11,12,13]]# ,19,20 ####,124,112,118
    
    styles_dict = {labels[0]: {"color": "blue"},
                labels[1]: {"color": "red"},
                labels[2]: {"color": "green"},
                labels[3]: {"color": "black", "linestyle": "dashed"},}

    for i in range(len(ids_test)):
        tots = []
        for spe_epoch in All_epochs.values():
            spe_avg = spe_epoch[ids_test[i]].apply_baseline(baseline = (0.58-0.1,0.58)).crop(0.58-0.1,0.58+0.5).average()
            tots.append(spe_avg)
        
        erps.append( mne.grand_average(tots).pick(ch_pick).filter(l_freq = None, h_freq = 40) )

    dict_all = {labels[i]:mne.channels.combine_channels(erps[i],dict(al = [i for i in range(len(ch_pick))])) for i in range(len(labels))}
    mne.viz.plot_compare_evokeds(dict_all, styles = styles_dict, ylim=dict(eeg=[-2, 2]),
                                vlines = [0.58],
                                legend='upper right', show_sensors='upper left')

    if extra:
        for erp in erps:
            erp.plot_joint()


def ThetaTopoPlot(tfr_mat, name_mat, times = [0,0.3],f=[3,9]):
    #Takes in nested lists
    
    x_dim = len(tfr_mat)
    y_dim = len(tfr_mat[0])
    
    #fig = plt.figure()
    #ax = fig.subplots(x_dim,y_dim)
    fig, ax = plt.subplots(x_dim,y_dim)
    
    for x in range(x_dim):
        for y in range(y_dim):
            tfr = tfr_mat[x][y]

            grand_avg = mne.grand_average(tfr)
            ga_data = grand_avg.crop(tmin=times[0],tmax=times[1], fmin=f[0], fmax=f[1]).data
            dB_ga_data = ga_data*10
            """
            mne.viz.plot_topomap(dB_ga_data, grand_avg.info,
                                vmin = -0.2, vmax = 0.6,title=name_mat[x][y], axes = ax[x][y], show = False)
            """
            dB_grand_avg = mne.time_frequency.AverageTFR(grand_avg.info,dB_ga_data, 
                                                         grand_avg.times, grand_avg.freqs, grand_avg.nave)
            dB_grand_avg.plot_topomap(tmin=times[0], tmax=times[1], fmin=f[0], fmax=f[1], mode='logratio',
                                vmin = -0.2, vmax = 0.6,title=name_mat[x][y], axes = ax[x][y], 
                                show = False, colorbar = False, cmap = "viridis")
            #cbar_fmt = "%.2f" #"%:.2f"
            """
            import matplotlib.ticker as ticker
            def myfmt(x, pos):
                return '{0:.5f}'.format(x)
            
            plt.colorbar(ax[x][y], format=ticker.FuncFormatter(myfmt))
            """
    
    image = ax[0][0].images
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image[0], cax=cbar_ax)            
            
    plt.show()

############################
# Choosing data here
S2 = True
S4 = not S2
############################
if S2:
    from files_info_Study2 import Event_ids, direct, list_files, common
    from files_info_Study2 import Speech, Non_speech
    from Find_bads_and_interpolate import All_epochs, all_channels
if S4:
    from loading import allEpochs, ch_names, event_dict, allFiles, directory, subjectIDs
    All_epochs = allEpochs


if __name__ == "__main__":
    if S4:
        #S4ERPplot()
        
        ##
        f_vars = {"freqs":np.arange(4,8 +2,2),"n_cycles":5} # "+2" since the last step is excluded
        ##
        G1_ids = ['audiovisual/high']
        G2_ids = ['audiovisual/low']

        G1_subgroup = subjectIDs
        G2_subgroup = subjectIDs
        baseline = [0.58 -0.5,0.58 -0.2] #shifted to onset explicitly
        ##
        
        settings = ["audiovisual/congruent","audiovisual/bg","auditory","visual"]

        tfr_data = [[0,0,0,0],
                    [0,0,0,0]]
        
        for i in range(len(settings)):
            Group_id_1 = [settings[i] + "/high"]
            Group_id_2 = [settings[i]+ "/low"]
            G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [Group_id_1,Group_id_2], All_epochs, baseline=baseline,
                                                freq_vars=f_vars, output_evoked=True)
            tfr_data[0][i] = G1
            tfr_data[1][i] = G2

        tfr_names = [["AV Congruent","AV Incongruent", "Audio", "Visual"],
                    ["AV Congruent","AV Incongruent", "Audio", "Visual"]]
        
        ThetaTopoPlot(tfr_data, tfr_names,times = [0 +0.58,0.3 + 0.58])
        
        ##
        
        
        G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [Group_ids,Group_ids], All_epochs, 
                                  baseline=[-0.5 +  0.58,-0.2 +  0.58],
                                    freq_vars=f_vars, output_evoked=True)
    
    if S2:
        f_vars = {"freqs":np.arange(4,8 +2,2),"n_cycles":5} # "+2" since the last step is excluded
        ##
        G1_ids = ["audiovisual/congruent"] # ['Tabi_A_Tabi_V','Tagi_A_Tagi_V'] #+ ['Tagi_A_Tabi_V', 'Tabi_A_Tagi_V']
        G2_ids = G1_ids

        G1_subgroup = Speech
        G2_subgroup = Non_speech
        ##
        """
        G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], All_epochs, baseline=[-0.5,-0.2],
                                        freq_vars=f_vars, output_evoked=True)
        
        tfr_data =[[G1,G2],
                   [G1,G2]]
        tfr_names = [["Speech","Non-speech"],
                     ["Speech","Non-speech"]]
                
        ThetaTopoPlot(tfr_data, tfr_names)
        """

        settings = [["audiovisual/congruent"],["audiovisual/incongruent"],["auditory"],["visual"]]

        tfr_data = [[0,0,0,0],
                    [0,0,0,0]]
        
        for i in range(len(settings)):
            Group_ids = settings[i]
            G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [Group_ids,Group_ids], All_epochs, baseline=[-0.5,-0.2],
                                                freq_vars=f_vars, output_evoked=True)
            tfr_data[0][i] = G1
            tfr_data[1][i] = G2

        tfr_names = [["AV Congruent","AV Incongruent", "Audio", "Visual"],
                    ["AV Congruent","AV Incongruent", "Audio", "Visual"]]
        
        ThetaTopoPlot(tfr_data, tfr_names)

#mne.combine_evoked(theta_tfr, 'equal')
#mne.grand_average(power_tots)