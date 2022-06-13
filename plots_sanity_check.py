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
    
from unicodedata import name
import numpy as np
import mne    
import matplotlib.pyplot as plt
from main_perm_test import createGroupsFreq

def S4ERPplot(extra=False):
    
    ids_test = ["audiovisual/bg/high","audiovisual/bg/med","audiovisual/bg/low", "auditory/b"]
    labels = ["AV-high","AV-mid","AV-low","Aud ('B')"]

    erps = []

    #ch_pick = [f"E{i}" for i in [4,5,6,11,12,19]]
    #ch_pick = [f"E{i}" for i in [5,6,11,12,13,112]]

    ch_pick = [f"E{i}" for i in [3,4,5,6,7,10,11,12,13]]# ,19,20 ####,124,112,118

    for i in range(len(ids_test)):
        tots = []
        for spe_epoch in All_epochs.values():
            spe_avg = spe_epoch[ids_test[i]].apply_baseline(baseline = (0.58-0.1,0.58)).crop(0.58-0.1,0.58+0.5).average()
            tots.append(spe_avg)
        
        erps.append( mne.grand_average(tots).pick(ch_pick).filter(l_freq = None, h_freq = 40) )

    dict_all = {labels[i]:mne.channels.combine_channels(erps[i],dict(al = [i for i in range(len(ch_pick))])) for i in range(len(labels))}
    mne.viz.plot_compare_evokeds(dict_all,
                                legend='upper left', show_sensors='upper right')

    if extra:
        for erp in erps:
            erp.plot_joint()


def ThetaTopoPlot():
    smth = 0

if __name__ == "__main__":
    S4ERPplot()
    
    #

#mne.combine_evoked(theta_tfr, 'equal')
#mne.grand_average(power_tots)