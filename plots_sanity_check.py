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


def ThetaTopoPlot(G1, G2):
    grand_avg = mne.grand_average(G1)
    grand_avg.plot_topomap(tmin=0, tmax=0.3, fmin=4, fmax=8, mode='logratio',
                           vmin = -0.06, vmax = 0.06,title='Theta band')
    #-100m
    grand_avg2 = mne.grand_average(G2)
    grand_avg2.plot_topomap(tmin=0, tmax=0.3, fmin=4, fmax=8, mode='logratio',
                            vmin = -0.06, vmax = 0.06, title='Theta band2')
    print("j")
    """
    times = np.arange(0.05, 0.151, 0.02)
    evoked.plot_topomap(times, ch_type='mag', time_unit='s')
    """
    

if __name__ == "__main__":
    if S4:
        S4ERPplot()
    
    if S2:
        f_vars = {"freqs":np.arange(4,8 +2,2),"n_cycles":5} # "+2" since the last step is excluded
        ##
        G1_ids = ["audiovisual/congruent"] # ['Tabi_A_Tabi_V','Tagi_A_Tagi_V'] #+ ['Tagi_A_Tabi_V', 'Tabi_A_Tagi_V']
        G2_ids = G1_ids

        G1_subgroup = Speech
        G2_subgroup = Non_speech
        ##
        G1, G2 = createGroupsFreq([G1_subgroup , G2_subgroup], [G1_ids,G2_ids], All_epochs, baseline=[-0.5,-0.2],
                                        freq_vars=f_vars, output_evoked=True)
        
        ThetaTopoPlot(G1, G2)
    

#mne.combine_evoked(theta_tfr, 'equal')
#mne.grand_average(power_tots)