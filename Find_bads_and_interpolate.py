# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:36:38 2022

@author: alexa
"""

import mne
import os
import numpy as np
from files_info_Study2 import Event_ids, direct, list_files, Speech, Non_speech, common
import copy


#overriding preprocessd directory to directory with all channels
direct_ic = "pipeline_data\\data_icmarked"

ic_channels = mne.io.read_epochs_eeglab(direct_ic +"\\" + list_files[0]).ch_names


epochs1 = mne.io.read_epochs_eeglab(direct +"\\" + list_files[0])
ch1 = epochs1.ch_names

epochs2 = mne.io.read_epochs_eeglab(direct +"\\" + list_files[1])
ch2 = epochs2.ch_names

for channel in ch2:
    if not (channel in ch1):
        miss_from_1 = channel



montage = epochs2.get_montage()

epochs1.add_reference_channels(miss_from_1)

std_montage = mne.channels.make_standard_montage("biosemi64")

epochs_out = epochs1.set_montage(std_montage)



# Vizualize 2 interpolated channels


epochs1.info['bads'].append(miss_from_1)

epochs_out = epochs1.copy().interpolate_bads()

single_test = False
if __name__ == "__main__" and single_test:
    
    #epochs1.plot_sensors(show_names=True)
    epochs1.info["bads"] = []
    epochs1.plot_sensors(show_names=True,kind='3d')
    avg = epochs1[Event_ids[0]].average()
    avg.pick(["FC4", miss_from_1, "FT8"]).plot_joint()
    
    avg2 = epochs_out[Event_ids[0]].average()
    avg2.pick(["FC4", miss_from_1, "FT8"]).plot_joint()
    
    epochs1[Event_ids[0]].pick(["FC4"]).average().plot(window_title = "Again average?")
    epochs1[Event_ids[0]].pick(["FC4"]).subtract_evoked(avg2.pick(["FC4"])).average().plot(window_title = "minus ERP")
    avg2.pick(["FC4"]).plot(window_title = "ERP?")



###############################
#Doing it for all
###############################

#Not a pretty solution, but functional for more detailed indexing of events
events_dict = epochs1.event_id
new_dict = {}
for key, value in events_dict.items():
    if "_A" in key and "_V" in key:
        new_key = key + "/audiovisual"
        if key[:4] == key[7:11]:
            new_key += "/congruent"
        elif key[:4] != key[7:11]:
            new_key += "/incongruent"
    elif "_A" in key:
        new_key = key + "/auditory"
    elif "_V" in key:
        new_key = key + "/visual"
    else:
        new_key = key
    
    new_dict[new_key] = value


all_channels = epochs1.ch_names

All_epochs = {}

list_of_bads_check = []
for file in list_files:
    
    epochs_loop = mne.io.read_epochs_eeglab(direct +"\\" + file)
    
    epochs_loop.event_id = new_dict
    
    curr_channels = epochs_loop.ch_names
    bads_check = []
    
    for ch in all_channels:
        if not (ch in curr_channels):
            epochs_loop.add_reference_channels(ch)
            epochs_loop.info['bads'].append(ch)
            bads_check.append(ch)
            
    xch = "F6" #"FC6"
    if not (xch in epochs_loop.info['bads']):
        epochs_loop.info['bads'].append(xch)
        bads_check.append(xch)
    
    epochs_loop.set_montage(std_montage) #biosemi64
    
    epochs_out = epochs_loop.copy().interpolate_bads()
    
    All_epochs[file[:4]] = epochs_out
    
    if __name__ == "__main__":
        if len(bads_check) > 4:
            list_of_bads_check.append([bads_check, file[:4]])
   

multi_test = True
if __name__ == "__main__" and multi_test:

    bads_list, name = list_of_bads_check[0]

    testing_epoch = All_epochs[name].copy()
    
    avg_test = testing_epoch[Event_ids[0]].average()
    
    avg_test.pick(bads_list).plot_joint()
    
    max_len, indx = 0, 0
    for i in range(len(list_of_bads_check)):
        if len(list_of_bads_check[i][0]) > max_len:
            max_len = len(list_of_bads_check[i][0])
            indx = i
    
    bads_list, name = list_of_bads_check[indx]

    testing_epoch = All_epochs[name].copy()
    
    avg_test = testing_epoch[Event_ids[0]].average()
    
    avg_test.pick(bads_list).plot_joint()   
            




