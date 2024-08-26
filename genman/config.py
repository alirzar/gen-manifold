"""Configure file paths and parameters for project


"""
import os
pjoin = os.path.join

class Config(object):

    # project directories
    config_path = os.path.dirname(os.path.abspath(__file__))
    resources = pjoin(config_path, '../resources')
    fmriprep_dir = '/Raid6/raw/VMR-Learning-Complete/derivatives/2020/fmriprep'

    # parcellation
    atlas = pjoin(resources, 'atlases',
     'Schaefer2018_400Parcels_17Networks_order.dlabel.nii')

    adjacency = pjoin(resources, 'atlases', 
                      'Schaefer2018_400Parcels_17Networks_adjacency.tsv')
    
    # data directories
    data_dir = pjoin(config_path, '../data')
    dataset = 'Schaefer2018_400_17Networks_Tian_Subcortex_S2-final'
    dataset_dir = pjoin(data_dir, dataset)
    tseries = pjoin(dataset_dir, 'timeseries')
    cerebellum = pjoin(dataset_dir, 'nettekoven_cerebellar')
    connect = pjoin(dataset_dir, 'connectivity')
    connect_centered = pjoin(dataset_dir, 'connectivity-centered')
    
    net_connect = pjoin(dataset_dir, 'connectivity-network')
    net_connect_centered = pjoin(dataset_dir, 'connectivity-network-centered')

    # paths expected to change
    gradients = pjoin(dataset_dir, 'pca-gradients-centered')

    k = 3
    results = pjoin(config_path, f'../results/k{k}')
    figures = pjoin(config_path, f'../figures/fig-components-svgs-k{k}')

    os.makedirs(results, exist_ok=True)
    os.makedirs(figures, exist_ok=True)
