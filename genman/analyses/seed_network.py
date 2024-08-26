#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:11:12 2024

@author: ali
"""

import os, glob, re
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import pingouin as pg
import cmasher as cmr
from surfplot import Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from genman.config import Config
from genman.utils import (get_files, parse_roi_names,
                          schaefer400tian_nettekoven_roi_ix,
                          fdr_correct, 
                          )
from genman.analyses import plotting
from genman.analyses.plotting import (get_sulc, get_surfaces, yeo_cmap,
                                       weights_to_vertices)


plotting.set_plotting()

def get_epoch_name(fname):
    epochs = ['leftbaseline', 'rightbaseline', 
              'rightlearning', 'lefttransfer', 
              'early', 'late']
    name = [f for f in fname.split('_') if any(w in f for w in epochs)]
    if len(name) == 1:
        return name[0].split('-')[-1]
    elif len(name) == 2:
        return name[0].split('-')[-1] + '-' + name[1].split('.')[0]

def _epoch_order():
    ordered = {'leftbaseline': 'orange', 'rightbaseline': 'green', 'rightlearning-early': 'green', 
               'rightlearning-late': 'green', 'lefttransfer-early': 'orange', 'lefttransfer-late': 'orange'}
    return ordered

def compute_error(df, angle='hitAngle_hand_good', window=8):
    """Compute median error of the first non-NaN8 trials 

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level, trial-wise data of a specific epoch

    Returns
    -------
    pd.DataFrame
        Subject-level median error data
    """
    data = df.loc[:, ['sub', angle]].dropna()
    data = data.groupby('sub').head(window)
    return data.groupby("sub")[angle] \
             .agg('median') \
             .reset_index()

def map_network_to_rois(data, cols=['network', 'hemi', 'T', 'sig', 'sig_corrected']):
    gradients = pd.read_table(
        os.path.join(Config().results, 'subject_gradients.tsv')
    )
    target = gradients.iloc[:400, :][['roi', 'network', 'hemi', 'roi_ix']]
    target = target.merge(data[cols], 
                          on=['network', 'hemi'], how='left')
    return target
    
def connect_seed_net(cmats, seed_net, hemi):
    """Extract network seed connectivity by isolating row in connectivity matrix

    Parameters
    ----------
    cmats : list
        Connectivity matrices
    seed_region : str
        Seed region name

    Returns
    -------
    pandas.DataFrame
        Region connectivity profiles across subjects
    """
    seed = f'{hemi}_{seed_net}'
    list_ = []
    for i in cmats:
        cmat = pd.read_table(i, index_col=0)
        # isolate row of seed region
        res = parse_roi_names(cmat.loc[cmat.columns.str.contains(seed)].mean() \
                                .reset_index().rename(columns={'index': 'roi', 0:'r'}))
        res.loc[400:, 'network'] = res.loc[400:, 'roi'] 
        res = res.groupby(['network', 'hemi']).mean(numeric_only=True).reset_index()
        fname = os.path.basename(i)
        res['sub'] = fname.split('_')[0]
        res['epoch'] = get_epoch_name(fname)
        list_.append(res)
    connectivity = pd.concat(list_)
    connectivity['r'] = connectivity['r'].astype('float')
    return connectivity.reset_index(drop=True)

def seed_analysis(cmats, epochs, seed, hemi):
    """Perform seed connectivity contrast analysis

    Parameters
    ----------
    contrasts : pandas.DataFrame
        Eccentricity contrast results
    clust_num : int
        Eccentricity contrast cluster number
    cmats : List
        Connectivity matrices
    epochs : _type_
        Task epochs to compare connectivity, not necessarily the same task 
        epochs from the eccentricity contrast  

    Returns
    -------
    pandas.DataFrame, str, pandas.DataFrame
        Seed connectivity results (region and networks), and seed name
    """
    #seed = find_cluster_seed(contrasts.query("cluster == @clust_num"))
    connectivity = connect_seed_net(cmats, seed, hemi)

    df = connectivity.query('epoch in @epochs')

    res = df.groupby(['network', 'hemi'], sort=False) \
            .apply(pg.pairwise_tests, dv='r', within='epoch', subject='sub', 
                   include_groups=False) \
            .reset_index()
    # swap sign so that B condition is the positive condition
    res['T'] = -res['T']
    res['sig'] = (res['p-unc'] < .05).astype(float)

    return fdr_correct(res), connectivity

def get_seed_color():
    return {
        'LH': dict(color= '#8B008B', alpha=0.5),  # dark magneta
        'RH': dict(color='#008B8B', alpha=0.6)  # dark cyan
    }

def plot_spider(res, out_dir):
    colors = get_seed_color()
    custom_order = list(yeo_cmap(networks=19))
    
    # Create the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    all_values = []
    for h in res['hemi'].unique():
        data = res.query('hemi == @h').copy()
        # Perform the groupby operation and preserve the order
        data['network'] = pd.Categorical(data['network'], categories=custom_order, ordered=True)
        data = data.sort_values('network')

        # Extract values and labels
        values_net = data['r'].tolist()
        all_values += values_net
        network_labels = data['network'].tolist()
        
        # Number of variables
        num_vars = len(values_net)
        
        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Repeat the first data point to close the circular plot
        values_net += values_net[:1]
        angles += angles[:1]
    
        #whole brain
        cmap = colors[h]
        ax.fill(angles, values_net, **cmap, zorder=6)#color='#1E90FF', alpha=0.5 / color='#FF7F50', alpha=0.6
        ax.plot(angles, values_net, linestyle='-', color=cmap['color'], zorder=5)
    # Add a circle with a value of 1
    circle_radius = 0
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    circle_values_net = [circle_radius] * 100
    ax.plot(circle_angles, circle_values_net, linestyle='-', color='black', linewidth=3)
    # Add labels for each point
    label_angles = np.degrees(angles[:-1])  # Convert radians to degrees

    ax.set_thetagrids(label_angles, network_labels, fontsize=11, fontweight='bold')
    # Set the radial label position
    ax.set_rlabel_position(10)
       
    ax.set_facecolor('#e0e0e0') ##e0e0e0
    ax.xaxis.grid(color='white', linestyle='-', linewidth=5, alpha=.9)#white
    ax.yaxis.grid(color='white', linestyle='-', linewidth=5, alpha=.9)#white
    ax.spines['polar'].set_edgecolor('#FFF5E1')
    tmin = np.round(min(all_values), 1)
    tmax = np.round(max(all_values), 1)
    ax.set_rgrids(np.arange(tmin, tmax, .1))

    plt.show()
    fig.savefig(out_dir)


def plot_spider_legend(out_dir):
    # Define the colors and labels
    colors = [c['color'] for c in get_seed_color().values()]
    labels = list(get_seed_color().keys())
    # Create figure and axes
    patches = patches = [mpatches.Rectangle((0, 0), 1, 1, facecolor=color, 
                                            edgecolor='black', label=label) 
                         for color, label in zip(colors, labels)]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.legend(handles=patches, fontsize=14, ncol=1, 
              edgecolor='w', handlelength=1.5, handleheight=1.5)
    ax.axis('off')
    plt.show()
    # Save the figure
    out = os.path.join(out_dir, 'spider_legend')
    fig.savefig(out)
    
def correlation_map(errors, connectivity, seed_region, hemi, FIG_DIR, name, vmax=None,
                    sig_style=None, lateral_only=False, angle='hitAngle_hand_good', plot_maps=True):
    """Compute correlation between error and region eccentricity and display
    on surface

    Parameters
    ----------
    error : pd.DataFrame
        Subject-level median error data
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    name : str
        Name/label for analysis, as part of output filename
    plot_maps: bool
        Plot the correlation maps or not

    Returns
    -------
    pd.DataFrame
        Region-wise correlation results
    """
    cortical_nets = list(yeo_cmap(networks=17).keys())
    data = connectivity[['sub', 'network', 'hemi', 'r']] \
                .pivot(index='sub', columns=['network', 'hemi'], values='r')

    # double check!
    assert np.array_equal(errors['sub'], data.index.values)

    res = data.apply(lambda x: pearsonr(errors[angle], x), 
                     axis=0, )
    res = res.T.rename(columns={0: 'r', 1: 'p'}).reset_index()
    _, res['p_fdr'] = pg.multicomp(res['p'].values, method='fdr_bh')
    res_cortex = res.query('network in @cortical_nets')
    res_rois = map_network_to_rois(res_cortex, cols=res.columns.tolist())    
    if plot_maps:
        config = Config()
        rvals = plotting.weights_to_vertices(res_rois['r'], config.atlas)
        pvals = plotting.weights_to_vertices(res_rois['p'], config.atlas)
        pvals = np.where(pvals < .05, 1, 0)
        qvals = plotting.weights_to_vertices(res_rois['p_fdr'], config.atlas)
        qvals = np.where(qvals < .05, 1, 0)
        
        surfaces = get_surfaces()
        sulc = plotting.get_sulc()
        sulc_params = dict(data=sulc, cmap='gray', cbar=False)
        
        if vmax == None: 
            vmax = np.max(abs(res_rois['r']))    
        # cmap = cmr.get_sub_cmap('seismic', 0.2, .8)
        cmap = np.genfromtxt(os.path.join(config.resources, 'colormap.csv'), delimiter=',')
        cmap = ListedColormap(cmap)
        if lateral_only:
            p1 = Plot(surfaces['lh'], surfaces['rh'], views='lateral', 
                      layout='column', size=(250, 350), zoom=1.5)   
        else:
            p1 = Plot(surfaces['lh'], surfaces['rh'])
        
        p2 = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', 
                    size=(150, 200), zoom=3.3)
        seed = ((res_rois['network'] == seed_region) & (res_rois['hemi'] == hemi)).astype('float')
        z = weights_to_vertices(seed.values, Config().atlas, res_rois['roi_ix'].values)
        seed_cmap = LinearSegmentedColormap.from_list(
            'regions', ["yellow", 'k'], N=2
        )
        for p, suffix in zip([p1, p2], ['', '_dorsal']):
            cbar = True if suffix == '_dorsal' else False
            if sig_style is None:
                p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
                p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='viridis')
            elif sig_style == 'uncorrected':
                p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
                colors = ["#90EE90", "#008000"]  # Light cyan to pure cyan
                cyan_cmap = LinearSegmentedColormap.from_list("custom_cyan", colors, N=2)
                p.add_layer((np.nan_to_num(rvals*pvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='binary')
            elif sig_style == 'corrected':
                x = rvals * qvals
                vmin = np.nanmin(x[np.abs(x) > 0])
                p.add_layer(x, cbar=cbar, cmap=cmr.get_sub_cmap(cmap, .66, 1),  
                            color_range=(vmin, vmax))
                p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='binary')
            p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
            p.add_layer(np.nan_to_num(z), as_outline=True, cmap='binary', cbar=False)
            if suffix == '_dorsal':
                cbar_kws = dict(location='bottom', decimals=2, fontsize=10, 
                                n_ticks=2, shrink=.4, aspect=4, draw_border=False, 
                                pad=.05)
                fig = p.build(cbar_kws=cbar_kws)
            else:
                fig = p.build()
        
            if sig_style is None:
                suffix = suffix
            prefix = os.path.join(FIG_DIR, f'{name}_correlation_map{suffix}')
            fig.savefig(prefix)

    return res

def main():
    config = Config()
    cmats = get_files(
        os.path.join(
            config.dataset_dir, 'connectivity', 
            'sub*/*.tsv'
        )
    )

    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )
    behav_df = pd.read_csv(os.path.join(config.resources, 'subject_behavior.csv'))

    behav_df.loc[behav_df['blockNo'] > 2, 'hitAngle_hand_good'] += 45
    res_dir = os.path.join(config.results, 'seed-network')
    os.makedirs(res_dir, exist_ok =True)
    fig_dir = os.path.join(config.figures, 'seed-network')
    os.makedirs(fig_dir, exist_ok=True)
    seeds_df = pd.DataFrame(dict(epoch=[['rightbaseline', 'rightlearning-early']] * 5 \
                                 + [['leftbaseline', 'lefttransfer-early']] * 6,
                      seed=['DefaultA','DefaultA', 'DefaultB', 'DorsAttnA', 'VisCent', 
                            'DefaultA', 'DefaultC', 'VisPeri', 'DefaultA', 'DefaultC', 'VisPeri'],
                      hemi=['LH', 'RH', 'RH', 'RH', 'RH', 
                            'LH', 'LH', 'LH', 'RH', 'RH', 'RH']))
    cortical_nets = list(yeo_cmap(networks=17).keys())
    for i in seeds_df.index:
        epochs = seeds_df["epoch"][i]
        seed = seeds_df["seed"][i]
        hemi = seeds_df["hemi"][i]
        epoch_behav = behav_df.query('epoch == @epochs[1]')
        error = compute_error(epoch_behav, window=32)
        prefix = f'{epochs[1]}_vs_{epochs[0]}'
       
        res, connectivity = seed_analysis(cmats, seeds_df['epoch'][i], 
                               seeds_df['seed'][i], seeds_df['hemi'][i])
        res.to_csv(os.path.join(res_dir, f'{prefix}_{seed}-{hemi}.tsv'), sep='\t', index=False)
        
        epoch_connectivity = connectivity.query('epoch == @epochs[1]')
     
        base_connectivity = connectivity.query('epoch == @epochs[0]')
        epoch_connectivity.loc[:, 'r'] -= base_connectivity['r'].values
        correlation_name = f'{epochs[1]}_vs_{epochs[0]}-{seed}_{hemi}-error'
        res_correlation_corrected = correlation_map(error, epoch_connectivity, seed, hemi, vmax=.4,
                                          FIG_DIR=fig_dir, name=correlation_name, 
                                          angle='hitAngle_hand_good')
        res_correlation_corrected.to_csv(os.path.join(res_dir, 
                               f'{epochs[1]}_vs_{epochs[0]}-{seed}_{hemi}-error_correlations.tsv'),
                               index=False, sep='\t')
        df = res_correlation_corrected.query('network not in @cortical_nets') \
                                        .sort_values('network')
        df_sub = df.iloc[:32, :].groupby(['hemi']).mean(numeric_only=True).reset_index()
        df_sub['network'] = 'Subcortex'
        df_cereb = df.iloc[32:].groupby(['hemi']).mean(numeric_only=True).reset_index()
        df_cereb['network'] = 'Cerebellum'
        
        spider_data = res_correlation_corrected.query('network in @cortical_nets') \
                                            .query('network != @seed')
        spider_data = pd.concat([spider_data, df_cereb, df_sub])
        spider_name = os.path.join(fig_dir, f'{prefix}_{seed}-{hemi}-error_correlation_spider')
        plot_spider(spider_data, spider_name)
        
    plot_spider_legend(fig_dir)      

if __name__ == '__main__':
    main()

