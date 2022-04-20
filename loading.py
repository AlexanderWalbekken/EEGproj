import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time as time
import copy as copy
from scipy.io import loadmat

directory = 'Data'
os.chdir(directory) # Change working directory

file = loadmat('AR47787.mat') # Raw file (dictionary) of one subject

trialtable = file['trialtable']
rej_idx = file['rej_idx']
data = file['data'] 
# data['label']
# data['fsample']
# data['elec'], which has data['elec']['pnt'] and data['elec']['label']   
# data['trial']
# data['time']
# data['cfg'], which has data['cfg']['version'], and then data['cfg']['version']['name'] and data['cfg']['version']['id']

trials = data['trial'][0,0] # Shape is (1, 1779)

ch_names = ["e{:02d}".format(x) for x in range(1,127)] # Channel names 
sfreq = int(data['fsample'][0,0]) # Sampling frequency (250)

"""
# Load data using RawArray from MNE toolbox 
for i in range(1779):
    trial_data = trials[0, i] * 10**(-6) # Convert values from microvolt to volt 
    
    info = mne.create_info(ch_names, sfreq, ch_types = 'eeg') # MNE Info object
    rawArray = mne.io.RawArray(trial_data, info, first_samp = 0.58) # MNE RawArray object 
"""

# Load data using EpochsArray from MNE toolbox 
allData = []

for i in range(1779):
    trial_data = trials[0, i] * 10**(-6) # Convert values from microvolt to volt 
    allData.append(trial_data)

allData = np.array(allData)
    
info = mne.create_info(ch_names, sfreq, ch_types = 'eeg') # MNE Info object
epochsArray = mne.EpochsArray(allData, info, tmin = -0.58) # MNE EpochsArray object 
    
# TO-DO:
# Create montage / sensor positions 
# Create dictionary of events and implement in epochsArray object 

# PLOTS: 
# epochsArray.copy().pick(ch_names[100:103]).average().plot()
# epochsArray.info





