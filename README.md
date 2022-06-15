# EEGproj
Project in BCs in AI and data



The code loads the data from the files, converts it into MNE if needed (SNR), and defines the event-ids in the correct format
SWS-data (Study 2) is loaded with use of files_info_Study2.py and Find_bads_and_interpolate.py
SNR-data (Study 4) is loaded with use of loading.py

The data is loaded from:
- (SWS) Folder named "pipeline_data" with data as .fdt and .set pairs must be in the top directory (is not tracked)
- (SNR) Folder named "Data" with data as .mat and a Berlin_EEG_Head.mat file must be in the top directory (is not tracked)

plots are outputted to a "plots" folder in top directory (is not tracked)

main_per_tets.py defines the main functions used:
- Subreacting ERP and computing average of time-frequency decomposition
- Running the clustered permutation test
- Plotting figures of clusters (and saving/loading of results)

All the test can be performed in the Tresh_looing.py script, where multiple tresholds can be tested in order to check for stability.
- "S2" variable controls which data is loaded
- "ttype", "n_perm", "thresh_list", "tail_list" control the parameters for the permutation test
- p-acc controls when the clusters are significant enough to be plotted (choosing something higher than the actual critical value allows exploration of results)
- Groups are defined by the "G_id" and "G_subgroup" variables which use the mne event ids and the subject key (defined in the loading scrips) respectivly





The "test.py" includes the code used to run the correlation t-test, but since a full implementation was not possible, with the limitations of the toolbox and project scope, the functions are not immidietly compatible with more than 2 groups. 