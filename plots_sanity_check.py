
    

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


def ThetaTopoPlot(tfr_mat, name_mat, times = [0,0.3],f=[3,9], dBscale = [-0.2,0.6], 
                  Spectogram = False, baseline = None, ytitles = None):
    #Takes in nested lists
    
    plt.rcParams.update({'font.size': 6})
    x_dim = len(tfr_mat)
    y_dim = len(tfr_mat[0])
    
    #fig = plt.figure()
    #ax = fig.subplots(x_dim,y_dim)
    fig, ax = plt.subplots(x_dim,y_dim)
    
    tfr_flat = [x for xs in tfr_mat for x in xs]
    name_flat = [x for xs in name_mat for x in xs]
    ax_flat = ax.flat
    
    for indx, ax_curr in enumerate(ax_flat):
        tfr = tfr_flat[indx]
        if Spectogram:
            y_lab = "Frequency (Hz)"
            
            freqs = tfr[0].freqs
            #dict_ch = {"all": [i for i in range(len(tfr[0].info.ch_names))]}
            avg_tfr = mne.combine_evoked(tfr, weights = "equal")
            avg_tfr = avg_tfr.crop(times[0], times[-1])
            #combined_tfr = mne.channels.combine_channels(avg_tfr,dict_ch)
            ch_avg = avg_tfr.data.mean(axis=0)
            info = avg_tfr.info
            info['ch_names'] = ['0']
            info['chs'] = [tfr[0].info["chs"][0]]
            info['nchan'] = 1
            combined_tfr = mne.time_frequency.AverageTFR(info,np.array([ch_avg]), 
                                                        avg_tfr.times, avg_tfr.freqs,avg_tfr.nave)
            if baseline!= None:
                combined_tfr = combined_tfr.apply_baseline(mode="logratio", baseline = (baseline[0],baseline[1]))
            dB_img = combined_tfr.data[0]*10
            
            ax_curr.imshow(dB_img, cmap="viridis", aspect='auto', origin='lower',
                            extent=[times[0], times[-1],
                                    freqs[0], freqs[-1]+(freqs[-1] - freqs[-2])],
                            vmin=dBscale[0], vmax=dBscale[1] )
            
            ax_curr.set_xlabel('Time (s)')
            #ax_curr.set_ylabel('Frequency (Hz)')
            ax_curr.set_title(r"$\bf{"+name_flat[indx]+"}$")
            
            #fixing frequency ticks (attempt 1)
            ##ax_spec.set_yticks(freqs)
            #fixing frequency ticks (attempt 2)
            
            freq_step = (freqs[-1] - freqs[-2])
            ax_curr.set_yticks(list(freqs))
            ax_curr.set_yticklabels('')
            ax_curr.set_yticks(freqs + (freq_step/2), minor=True)
            
            ratio = len(freqs)//10
            ylabs = [str(int(freqs[j])) if j%(ratio+1) == 0 else '' for j in range(len(freqs))]
            ax_curr.set_yticklabels(ylabs, minor=True)
            
            
        else:
            title_curr = r"$\bf{"+ name_flat[indx]+"}$" 
            grand_avg = mne.grand_average(tfr)
            if baseline!= None:
                grand_avg = grand_avg.apply_baseline(mode="logratio", baseline = (baseline[0],baseline[1]))
            ga_data = grand_avg.crop(tmin=times[0],tmax=times[1], fmin=f[0], fmax=f[1]).data
            dB_ga_data = ga_data*10
            
            dB_grand_avg = mne.time_frequency.AverageTFR(grand_avg.info,dB_ga_data, 
                                                        grand_avg.times, grand_avg.freqs, grand_avg.nave)
            dB_grand_avg.plot_topomap(tmin=times[0], tmax=times[1], fmin=f[0], fmax=f[1], mode='logratio',
                                vmin = dBscale[0], vmax = dBscale[1],title=title_curr, axes = ax_curr, 
                                show = False, colorbar = False, cmap = "viridis")
        #cbar_fmt = "%.2f" #"%:.2f"
        """
        import matplotlib.ticker as ticker
        def myfmt(x, pos):
            return '{0:.5f}'.format(x)
        
        plt.colorbar(ax_curr, format=ticker.FuncFormatter(myfmt))
        """
    
    image = ax_flat[0].images
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(image[0], cax=cbar_ax, label = 'power (dB)')
    cbar_ax.set_label('power (dB)')
    
    if ytitles != None:
        fig.subplots_adjust(left=0.05)
        r = 1/x_dim
        for y in range(x_dim):
            title_ax = ax[y][0]
            title_ax.set_ylabel(r"$\bf{"+ytitles[y]+"}$" +"\n" + y_lab)
    

    plt.show()

############################
# Choosing data here
S2 = False
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
        G1_ids = ['audiovisual/high'] #high (less theta band, more integration)
        G2_ids = ['audiovisual/low']

        G1_subgroup = subjectIDs
        G2_subgroup = subjectIDs
        baseline = [0.58 -0.5,0.58 -0.2] #shifted to onset explicitly
        ##
        
        settings = ["audiovisual/congruent","audiovisual/bg","auditory","visual"]
        subtracting = True
        
        tfr_names = [["AV Congruent","AV Incongruent", "Audio", "Visual"],
                    ["","", "", ""]]

        tfr_data = [[0,0,0,0],
                    [0,0,0,0]]
        
        if subtracting:
            tfr_data = [tfr_data[0]]
            tfr_names = [tfr_names[0]]
        
        for i in range(len(settings)):
            Group_id_1 = [settings[i] + "/high"]
            Group_id_2 = [settings[i]+ "/low"]
            G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [Group_id_1,Group_id_2], All_epochs, 
                                      baseline=None, #baseline=baseline
                                                freq_vars=f_vars, output_evoked=True)
            if subtracting:
                
                G_sub = []
                for idx in range(len(G1)):
                    g1_data = G1[idx].data
                    g2_data = G2[idx].data
                    
                    data_comp = np.log10(g2_data/g1_data)
                    
                    sub_ev = mne.time_frequency.AverageTFR(G1[idx].info, data_comp,G1[idx].times, G1[idx].freqs,G1[idx].nave)
                    G_sub.append(sub_ev)
                    
                tfr_data[0][i] = G_sub
                dBscale = [-0.4,0.4]
            else:
                tfr_data[0][i] = G1
                tfr_data[1][i] = G2
            
        
        ThetaTopoPlot(tfr_data, tfr_names,times = [0 +0.58,0.3 + 0.58], dBscale = dBscale, baseline=None)
        
        ##
    
    if S2:
        f_vars = {"freqs":np.arange(4,16 +2,2),"n_cycles":5} # "+2" since the last step is excluded
        ##
        G1_subgroup = Speech
        G2_subgroup = Non_speech
        ##
        baseline = [-0.5,-0.2]

        settings = [["audiovisual/congruent"],["audiovisual/incongruent"],["auditory"],["visual"]]
        #tfr_data = [[0,0],[0,0]]
        #tfr_names = [["AV_cong","c"],["2","c"]]
        
        tfr_data = [[0,0,0,0],
                    [0,0,0,0]]
        
        tfr_names = [["AV Congruent","AV Incongruent", "Audio", "Visual"],
                    ["","", "", ""]]
        
        for i in range(len(settings)):
            Group_ids = settings[i]
            G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [Group_ids,Group_ids], All_epochs, 
                                      baseline=None,
                                                freq_vars=f_vars, output_evoked=True)
            tfr_data[0][i] = G1
            tfr_data[1][i] = G2
        
        
        ThetaTopoPlot(tfr_data, tfr_names, Spectogram=True, ytitles=["Speech", "Non speech"],
                      dBscale=[-0.2,1.2], times=[-0.5,0.5], baseline=baseline)

#mne.combine_evoked(theta_tfr, 'equal')
#mne.grand_average(power_tots)