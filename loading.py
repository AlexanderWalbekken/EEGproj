import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import time as time
import copy as copy
from scipy.io import loadmat

directory = 'Data'
os.chdir(directory) # Change working directory

file = loadmat('AM50127.mat') # Raw file (dictionary) of one subject

data = file['data'] 
# data['label']
# data['fsample']
# data['elec'], which has data['elec']['pnt'] and data['elec']['label']   
# data['trial']
# data['time']
# data['cfg'], which has data['cfg']['version'], and then data['cfg']['version']['name'] and data['cfg']['version']['id']

trialtable = file['trialtable']
rej_idx = file['rej_idx']




