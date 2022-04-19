# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:15:44 2022

@author: alexa
"""
import os as os

import mne
import matplotlib.pyplot as plt
import numpy as np

#Default stuff

Event_ids =  ["Tagi_A", "Tagi_A_Tabi_V", "Tagi_V", "Tabi_V", 
              "Tagi_A_Tagi_V", "Tabi_A", "Tabi_A_Tagi_V", "Tabi_A_Tabi_V"]

direct = "pipeline_data\\data_preproc"

list_files = [i for i in os.listdir(direct) if i[-4:] == ".set"]


Speech = ["PP03","PP09", "PP10", "PP11", "PP12", "PP13", "PP14", "PP15", 
          "PP16", "PP17", "PP20", "PP25", "PP26", "PP28"]
Non_speech = ["PP02", "PP04", "PP05", "PP06", "PP07", "PP08", "PP18", "PP19", 
              "PP21", "PP22", "PP23", "PP24", "PP27", "PP29"]

common = ['AF4', 'AFz', 'C1', 'C2', 'C3', 'C4', 'CP1', 'CP2', 'CP3', 'CP4',
       'CP5', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'FC1', 'FC2', 'FCz',
       'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'PO3',
       'PO4', 'PO7', 'PO8', 'POz', 'Pz']

Speech_files = [ i + "_4adj.set" for i in Speech]
Non_speech_files = [i + "_4adj.set" for i in Non_speech]
# Test commit 
# Test commit 2

# Test commit 3
# Test new branch