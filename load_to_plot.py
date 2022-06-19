
from main_perm_test import clustersLoad, clustersPlot, clustersSave



####
#folder in "plots"
fol_name = "TBDIncongT=-1.6"
###

p_acc = 0.5 #0.10

T_obs, clusters, cluster_p_values, H0, tfr_epochs = clustersLoad(folder = fol_name)

clustersPlot(T_obs, clusters, cluster_p_values, tfr_epochs, 
                p_accept= p_acc, min_ch_num = 3,
                    show=False, save = True, folder= fol_name )