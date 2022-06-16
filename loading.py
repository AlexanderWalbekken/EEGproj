import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time as time
import copy as copy
from scipy.io import loadmat

# Get current working directory
workingDirectory = os.getcwd()

# --------------------------------- IDs ------------------------------------ #
# Set working directory to 'Data' folder 
directory = 'Data'
os.chdir(directory)

# Subject IDs (from allEpochs.keys())
subjectIDs = ['AM50127', 'AR47787', 'BJ50955', 'CJ47960', 'FR44603',
              'GL49490', 'HC41216', 'HD48764', 'KA48601', 'KN46410',
              'KS49673', 'MA48066', 'ME45959', 'PC49059', 'RP51494',
              'SC46392', 'SC47790', 'SS40788', 'ST48974', 'WM45269']

# dict_keys(['AM50127', 'AR47787', 'BJ50955', 'CJ47960', 'FR44603', 'GL49490', 
# 'HC41216', 'HD48764', 'KA48601', 'KN46410', 'KS49673', 'MA48066', 'ME45959', 
# 'PC49059', 'RP51494', 'SC46392', 'SC47790', 'SS40788', 'ST48974', 'WM45269'])

# Channel names
ch_names = [f"E{i}" for i in range(1,127)]

# Sampling freqency
sfreq = 250 

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
               'audiovisual/b/high/congruent': 13, 
               'audiovisual/b/med/congruent' : 14,
               'audiovisual/b/low/congruent' : 15,
               'audiovisual/g/high/congruent': 16, 
               'audiovisual/g/med/congruent' : 17,
               'audiovisual/g/low/congruent' : 18,
               'audiovisual/bg/high' : 19,
               'audiovisual/bg/med'  : 20,
               'audiovisual/bg/low'  : 21,
               'NA1'                  : 22,   # Remove !!
               'NA2'                  : 23,   # Remove !!
               'NA3'                  : 24  } # Remove !!
     
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
    subjectEpoch = subjectEpoch #.crop(0, 1.4) # Crop from 0.4 - 1 second
    allEpochs[subjectID[:-4]] = subjectEpoch # Add to allEpochs dictionary 
    

# -------------------------- SENSOR POSITIONS --------------------------- #

# Set sensor positions according to GSN HydroCel cap with 128 channels
HydroCel = mne.channels.make_standard_montage('GSN-HydroCel-128')

for subject in allEpochs.keys():
    allEpochs[subject].set_montage(HydroCel)
    
# ------------------------- ERP, EVOKED & PLOTS ------------------------- #

if __name__ == "__main__":
    # Plot sensor positions
    # allEpochs['KA48601'].plot_sensors()
    # allEpochs['KA48601'].plot_sensors(kind = '3d', show_names = True)
    
    # Plot specific conditions 
    # allEpochs['KA48601']['visual/b']
    # allEpochs['KA48601'][['visual/b', 'auditory/b']]
    # allEpochs['KA48601']['visual/b'].copy().pick(ch_names[0:5]).average().plot()
    
    # Plot ERP 
    # allEpochs['KA48601'].copy().pick(ch_names[100:103]).average().plot()
    
    # Average over all subjects 
    allEvoked = []
    allEvokedAV = []
    weights = list(np.ones(20))
    
    for subject in allEpochs.keys():
        allEvoked.append(allEpochs[subject].average())
        allEvokedAV.append(allEpochs[subject]['audiovisual'].average())
    
    combineEvoked = mne.combine_evoked(allEvoked, weights) # All conditions
    combineEvokedAV = mne.combine_evoked(allEvokedAV, weights) # Audiovisual conditions
    
    # Plot ERP over all subjects
    # combineEvoked.copy().pick(ch_names[0:10]).plot_joint()
    # combineEvokedAV.copy().pick(ch_names[0:20]).plot_joint()
    # combineEvokedAV.copy().pick(ch_names[0:20]).crop(0, 1.5).plot_joint()
    
    # Plot ERP over all subjects for 'audiovisual' conditions
    # combineEvokedAV.copy().plot_joint()
    
    # Subtracting the ERP 
    frequencies = np.arange(4, 38, 2)
    n_cycles_morlet = 5 #2
    decim_morlet = 3
    
    spe = 'KA48601'
    ids = 'audiovisual'
    
    epochs = allEpochs[spe]
    ep_avg = epochs[ids].average()
    
    ep_done = epochs[ids].copy().subtract_evoked(ep_avg)
    
    """
    power = mne.time_frequency.tfr_morlet(ep_done, n_cycles=n_cycles_morlet, 
                                              return_itc=False,
                                              freqs=frequencies, 
                                              decim=decim_morlet) # .crop(-0.95,0.95)
    """

# -------------------------------- TO DO ----------------------------------- #

# TO-DO:
# Create montage / sensor positions --DONE
# Create dictionary of events and implement in epochsArray object --DONE
# Creat event_id dictionary and implement in epochsArray object --DONE
# Average over all subjects and plot ERP --TO DO 
# Remove last three events (AV fusion, asynchronous 22:24)

# Replicate the behavioural precision graph (Figure 2, p. 9 in Study 4 Appendix) --TO DO 
# For the incongruent AV condition, the correct response is the spoken letter 
# Access subject responses for a given subject as: subjectFile['trialtable'][:, 2]

# Change back to working directory (not in the 'Data' folder)
os.chdir(workingDirectory)
