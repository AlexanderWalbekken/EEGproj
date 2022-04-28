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
"""
file = loadmat('AR47787.mat') # Raw file (dictionary) of one subject
trialtable = file['trialtable']
rej_idx = file['rej_idx']
data = file['data'] 
trials = data['trial'][0,0] # Shape is (1, 1779)



# Load data using RawArray from MNE toolbox 
for i in range(1779):
    trial_data = trials[0, i] * 10**(-6) # Convert values from microvolt to volt 
    
    info = mne.create_info(ch_names, sfreq, ch_types = 'eeg') # MNE Info object
    rawArray = mne.io.RawArray(trial_data, info, first_samp = 0.58) # MNE RawArray object 

# Load data using EpochsArray from MNE toolbox 
allData = []

for i in range(1804): # Loop over number of trials 
    trial_data = trials[0, i] * 10**(-6) # Convert values from microvolt to volt 
    allData.append(trial_data)

allData = np.array(allData)
    
info = mne.create_info(ch_names, sfreq, ch_types = 'eeg') # MNE Info object
epochsArray = mne.EpochsArray(allData, info, tmin = -0.58) # MNE EpochsArray object 
"""

# --------------------------------- IDs ------------------------------------ #

# Subject IDs (from allEpochs.keys())
# dict_keys(['AM50127', 'AR47787', 'BJ50955', 'CJ47960', 'FR44603', 'GL49490', 
# 'HC41216', 'HD48764', 'KA48601', 'KN46410', 'KS49673', 'MA48066', 'ME45959', 
# 'PC49059', 'RP51494', 'SC46392', 'SC47790', 'SS40788', 'ST48974', 'WM45269'])

# Channel names
ch_names = [f"E{i}" for i in range(1,127)]

# Sampling freqency
sfreq = 250 

# Event IDs 
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
             'AV_bg_high' : 19,
             'AV_bg_mid'  : 20,
             'AV_bg_low'  : 21,
             'AV_async_high' : 22,   # Remove !!
             'AV_async_mid'  : 23,   # Remove !!
             'AV_async_low'  : 24  } # Remove !!

# Dictionary of events (link: https://mne.tools/dev/auto_tutorials/raw/20_event_arrays.html)
event_dict = { 'visual/b/high' : 1, 
               'visual/b/mid'  : 2, 
               'visual/b/low'  : 3,
               'visual/g/high' : 4, 
               'visual/g/mid'  : 5, 
               'visual/g/low'  : 6,
               'auditory/b/high' : 7,
               'auditory/b/mid'  : 8, 
               'auditory/b/low'  : 9,
               'auditory/g/high' : 10,
               'auditory/g/mid'  : 11, 
               'auditory/g/low'  : 12,
               'audiovisual/b/high': 13, 
               'audiovisual/b/med' : 14,
               'audiovisual/b/low' : 15,
               'audiovisual/g/high': 16, 
               'audiovisual/g/med' : 17,
               'audiovisual/g/low' : 18,
               'audiovisual/bg/high' : 19,
               'audiovisual/bg/med'  : 20,
               'audiovisual/bg/low'  : 21,
               'audiovisual/async/high' : 22,   # Remove !!
               'audiovisual/async/med'  : 23,   # Remove !!
               'audiovisual/async/low'  : 24  } # Remove !!
     
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
    subjectChannels = [f'E{elec}' for elec in range(1, subjectFile['data']['label'][0,0].shape[1] + 1)] # List of electrodes (E1, E2, ..)
    #subjectChannels = [elec[0] for elec in subjectFile['data']['label'][0,0][0,:]] # List of electrodes (e01, e02, ..)
    subjectInfo = mne.create_info(subjectChannels, sfreq, ch_types = 'eeg') # MNE Info object 

    subjectData = [] 
    subjectEventOnset = []
    
    numSamples = 550
    onset = 0.58 #### TODO: This should probably be changed!! ####
    
    for i in range(subjectTrials):
        subjectData.append(subjectFile['data']['trial'][0,0][0, i] * 10 **(-6)) # Convert values from microvolt to volt 
        
        # Events for MNE EpochsArray object 
        trialSamples = subjectFile['data']['time'][0,0][0, i] 
        trialEventOnset = len(trialSamples[trialSamples < onset])
        subjectEventOnset.append(numSamples*i + trialEventOnset)
    
    # Three columns of 'events' array 
    subjectEventOnset = np.array(subjectEventOnset).reshape((-1,1))
    zeros = np.zeros((subjectTrials,1))
    ids = subjectFile['trialtable'][:,0].reshape((-1,1))
    
    # 'data' and 'events' for EpochsArray 
    subjectEvents = np.hstack((subjectEventOnset, zeros, ids)).astype(int) # Shape: (#trials, 3)
    subjectData = np.array(subjectData) # Shape: (#trials, #channels, #timepoints)
    
    # Create EpochsArray and add to dictionary
    subjectEpoch = mne.EpochsArray(subjectData, subjectInfo, events = subjectEvents, event_id = event_dict, tmin = -0.52) # MNE EpochsArray object
    allEpochs[subjectID[:-4]] = subjectEpoch # Add to allEpochs dictionary 
    

# --------------------------- SENSOR POSITIONS ----------------------------- #

# Set sensor positions according to GSN HydroCel cap with 128 channels
HydroCel = mne.channels.make_standard_montage('GSN-HydroCel-128')

for subject in allEpochs.keys():
    allEpochs[subject].set_montage(HydroCel)

# Plot sensor positions
# allEpochs['KA48601'].plot_sensors()
# allEpochs['KA48601'].plot_sensors(kind = '3d', show_names = True)

# Plot ERP 
# allEpochs['KA48601'].copy().pick(ch_names[100:103]).average().plot()

# -------------------------------- TO DO ----------------------------------- #

# TO-DO:
# Create montage / sensor positions --DONE
# Create dictionary of events and implement in epochsArray object --DONE
# Creat event_id dictionary and implement in epochsArray object --TO DO
# Average over all subjects and plot ERP --TO DO
# Remove last three events (AV fusion, asynchronous 22:24)


