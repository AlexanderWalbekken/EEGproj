import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time as time
import copy as copy
from scipy.io import loadmat

directory = 'Data'
os.chdir(directory) # Change working directory

# -------------------------- LOAD ONE SUBJECT ------------------------------ #

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

"""
# Load data using EpochsArray from MNE toolbox 
allData = []

for i in range(1804): # Loop over number of trials 
    trial_data = trials[0, i] * 10**(-6) # Convert values from microvolt to volt 
    allData.append(trial_data)

allData = np.array(allData)
    
info = mne.create_info(ch_names, sfreq, ch_types = 'eeg') # MNE Info object
epochsArray = mne.EpochsArray(allData, info, tmin = -0.58) # MNE EpochsArray object 
"""
     
# --------------------------- LOAD ALL SUBJECTS ---------------------------- #

# All subject files 
allFiles = [i for i in os.listdir() if i[-4:] == ".mat"]
allFiles.remove('Berlin_EEG_Head.mat')

# Dictionary for all subject epochs
allEpochs = {}

for subjectID in allFiles:
    subjectFile = loadmat(subjectID) # Subject file 
    
    # Create MNE Info object 
    subjectTrials = subjectFile['data']['trial'][0,0].shape[1] # Number of trials
    subjectChannels = [elec[0] for elec in subjectFile['data']['label'][0,0][0,:]] # Number of electrodes
    subjectInfo = mne.create_info(subjectChannels, sfreq, ch_types = 'eeg') # MNE Info object 

    subjectData = [] 
    subjectEventOnset = []
    
    numSamples = 550
    onset = 0.58 #### This should probably be changed!! ####
    
    for i in range(subjectTrials):
        subjectData.append(subjectFile['data']['trial'][0,0][0, i] * 10 **(-6))
        
        # Events for MNE EpochsArray object 
        trialSamples = subjectFile['data']['time'][0,0][0, i] 
        trialEventOnset = len(trialSamples[trialSamples < onset])
        subjectEventOnset.append(numSamples*i + trialEventOnset)
    
    
    subjectEventOnset = np.array(subjectEventOnset).reshape((-1,1))
    zeros = np.zeros((subjectTrials,1))
    ids = subjectFile['trialtable'][:,0].reshape((-1,1))
    
    subjectEvents = np.hstack((subjectEventOnset, zeros, ids)).astype(int) # Shape: (#trials, 3)
    subjectData = np.array(subjectData) # Shape: (#trials, #channels, #timepoints)
    
    subjectEpoch = mne.EpochsArray(subjectData, subjectInfo, events = subjectEvents, tmin = -0.52) # MNE EpochsArray object
    allEpochs[subjectID[:-4]] = subjectEpoch # Add to allEpochs dictionary 
    
# ------------------------------ EVENT IDs --------------------------------- #

eventIDs = { 'V_b_high' : 1, 
             'V_b_mid'  : 2, 
             'V_b_low'  : 3,
             'V_g_high' : 4, 
             'V_g_mid'  : 5, 
             'V_g_low'  : 6,
             'A_b_high' : 7,
             'A_b_mid'  : 8, 
             'A_b_low'  : 9,
             'A_g_high' : 10,
             'A_g_mid'  : 11, 
             'A_g_low'  : 12,
             'AV_b_high': 13, 
             'AV_b_mid' : 14,
             'AV_b_low' : 15,
             'AV_g_high': 16, 
             'AV_g_mid' : 17,
             'AV_g_low' : 18,
             'AV_TBD1_high' : 19,
             'AV_TBD1_mid'  : 20,
             'AV_TBD1_low'  : 21,
             'AV_TBD2_high' : 19,
             'AV_TBD2_mid'  : 20,
             'AV_TBD2_low'  : 21  }

# -------------------------------- TO DO ----------------------------------- #

# TO-DO:
# Create montage / sensor positions 
# Create dictionary of events and implement in epochsArray object 

# PLOTS: 
# epochsArray.copy().pick(ch_names[100:103]).average().plot()
# epochsArray.info





