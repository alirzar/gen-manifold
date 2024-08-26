"""Main analysis of eccentricity changes during Left Baseline, Right Baseline,
Right learning, and Left learning

Notes
-----
- Eccentricity is precomputed in the eccentricity.py module
- No Washout 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.cluster import KMeans
import cmasher as cmr
from surfplot import Plot

from genman.config import Config
from genman.utils import parse_roi_names, get_surfaces, test_regions, get_clusters
from genman.analyses import plotting

plotting.set_plotting()
config = Config()

def epoch_order():
    return {'rest': 1, 'leftbaseline': 2, 'rightbaseline': 3, 
             'rightlearning-early': 4, 'rightlearning-late': 5,
             'lefttransfer-early': 6, 'lefttransfer-late': 7,
             'baseline': 1, 'early': 2, 'late': 3,
             'left': 1, 'right': 2}

def eccentricity_analysis(data, method='anova', factor='epoch'):
    """Determine if regions show significant changes in eccentricity across
    task epochs

    Basic mass-univariate approach that performs an F-test across each region,
    followed by follow-up paired t-tests on significant regions

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        ANOVA and post-hoc stats tables, respectively
    """
    res = test_regions(data, method, factor)

    # post hoc analysis
    if method != 'ttest':
        if factor != 'epoch':
            sources = np.unique(res['Source'].values)
            posthoc = {}
            for source in sources:
                source_res = res.query('Source == @source')
                sig_regions = source_res.loc[source_res['sig_corrected'].astype(bool), 'roi'].tolist()
                if sig_regions:
                    post_data = data[data['roi'].isin(sig_regions)]
                    posthoc[source] = test_regions(post_data, 'ttest', factor=source)  
                else:
                    # no significant anova results
                    posthoc[source] = None
            posthoc = pd.concat([posthoc[i].fillna("-") for i in posthoc], ignore_index=True) \
                    .drop('index', axis=1) \
                    .sort_values(by='roi_ix').reset_index(drop=True)
            posthoc.insert(3, 'time', posthoc.pop('time'))
        else:   
            sig_regions = res.loc[res['sig_corrected'].astype(bool), 'roi'].tolist()
            if sig_regions:
                post_data = data[data['roi'].isin(sig_regions)]
                posthoc = test_regions(post_data, 'ttest', factor)  
            else:
                # no significant anova results
                posthoc = None
        return res, posthoc

    else: return res

def eccentricity_analysis_ttest(data):
    epochs = set(data[data['epoch'] != 'rest']['epoch'])
    res = {}
    for epoch in epochs:
        test_data = data[data['epoch'].isin([epoch, 'rest'])]
        res[epoch] = test_regions(test_data, method='ttest')
    return res

def plot_mean_scatters(data, out_dir, view_3d =(30, -120), eccentricity=False):
    """Plot scatter plot of mean gradients for each epoch

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    out_dir : str
        Figure save/output directory
     : tuple, optional
        Viewpoint as (evelation, rotation), by default (30, -110)
    eccentricity : bool, optional
        Whether plot region eccentricity via color scaling, by default False
    """
    k = [f'g{i}' for i in np.arange(3) + 1]
    mean_loadings = data.groupby(['epoch', 'roi', 'roi_ix'])[k + ['distance']] \
                        .mean() \
                        .reset_index()
    mean_loadings = parse_roi_names(mean_loadings)
    if eccentricity:
        c_col = 'distance'
        cmap = 'viridis'
        vmax = np.nanmax(mean_loadings['distance'])
        vmin = np.nanmin(mean_loadings['distance'])
        suffix = 'scatter_ecc'
    else:
        c_col='c'
        cmap = plotting.yeo_cmap(networks=19)
        mean_loadings['c'] = mean_loadings['network'].apply(lambda x: cmap[x])
        vmax, vmin = None, None
        suffix = 'scatter'

    for epoch in mean_loadings['epoch'].unique():
        df = mean_loadings.query("epoch == @epoch")
        
        x, y, z = k
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax = plotting.plot_3d(df[x], df[y], df[z], c=df[c_col], 
                                s=10, lw=0.5, ax=ax, view_3d=view_3d, 
                                vmax=vmax, vmin=vmin, alpha=.7)
        ax.set(xlim=(-3, 3), ylim=(-3, 3), zlim=(-3, 3))
        
        
        prefix = os.path.join(out_dir, f'mean_{epoch}_')
        fig.savefig(prefix + suffix, dpi=300, bbox_inches='tight')

   
def anova_stat_map(anova, out_dir, name='anova', vmax='auto',
                   vmin='auto', thresholded=True, outline=True):
    """Plot thresholded or unthresholded mass-univariate ANOVA results 

    Threshold set as q < .05, where q = FDR-corrected two-tailed p values

    Parameters
    ----------
    anova : pandas.DataFrame
        ANOVA results

    Returns
    -------
    pandas.DataFrame
        ANOVA results
    """
    df = anova.query("sig_corrected == 1") if thresholded else anova
    if len(df) == 0:
        return None
    fvals = df['F'].values
    if vmax == 'auto':
        vmax = int(np.nanmax(fvals))
    if vmin == 'auto':
        vmin = np.nanmin(fvals)

    # get orange (positive) portion. Max reduced because white tends to wash 
    # out on brain surfaces
    if thresholded:
        cmap = cmr.get_sub_cmap(plotting.stat_cmap(), .5, 1)
    else:
        cmap = cmr.get_sub_cmap('inferno_r', 0, 1)
    # get cmap that spans from stat threshold to max rather than whole range, 
    # which matches scaling of t-test maps
    cmap_min = vmin / vmax
    cmap = cmr.get_sub_cmap(cmap, cmap_min, 1)

    plotting.plot_cbar(cmap, vmin, vmax, 'horizontal', size=(1, .3), 
                         n_ticks=2)
    prefix = os.path.join(out_dir, name)
    plt.savefig(prefix + '_cbar')

    surfaces = get_surfaces()
    sulc = plotting.get_sulc()
    x = plotting.weights_to_vertices(fvals, Config().atlas, 
                                       df['roi_ix'].values)
    sulc_params = dict(data=sulc, cmap='gray', cbar=False)
    layer_params = dict(cmap=cmap, cbar=False, color_range=(vmin, vmax))
    outline_params = dict(data=(np.abs(x) > 0).astype(bool), cmap='binary', 
                          cbar=False, as_outline=True)

    # 2x2 grid
    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)

    cbar_kws = dict(n_ticks=2, aspect=8, shrink=.15, draw_border=False)
    fig = p.build()#(cbar_kws=cbar_kws)
    fig.savefig(prefix)

    # dorsal views
    p = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', size=(150, 200), 
             zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_dorsal')

    # posterior views
    p = Plot(surfaces['lh'], surfaces['rh'], views='posterior', 
             size=(150, 200), zoom=3.3)
    p.add_layer(**sulc_params)
    p.add_layer(x, **layer_params)
    if outline:
        p.add_layer(**outline_params)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + '_posterior')

    return x


def _epoch_order():
    ordered = ['leftbaseline', 'rightbaseline', 'rightlearning-early', 
               'rightlearning-late', 'lefttransfer-early', 'lefttransfer-late']
    return ordered


def plot_displacements(data, anova, k=3, ax=None, hue='network'):
    """Plot low-dimensional displacements of regions that show significant 
    ANOVA results (i.e. changes in eccentricity)

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data with distance column
    anova : pandas.DataFrame
        ANOVA results
    k : int, optional
        Number of gradients to include, by default 3
    ax : matplotlib.axes._axes.Axes, optional
        Preexisting matplotlib axis, by default None

    Returns
    -------
    matplotlib.figure.Figure and/or matplotlib.axes._axes.Axes
        Displacement scatterplot figure
    """
    if isinstance(k, int):
        k = [f'g{i}' for i in np.arange(k) + 1]

    mean_loadings = data.groupby(['epoch', 'roi', 'roi_ix'])[k].mean().reset_index()
    mean_loadings = parse_roi_names(mean_loadings)

    base = mean_loadings.query("epoch == 'leftbaseline'")
    sig_regions = anova.loc[anova['sig_corrected'].astype(bool), 'roi']
    sig_base = base[base['roi'].isin(sig_regions)]
    shifts = mean_loadings[mean_loadings['roi'].isin(sig_regions)]

    if hue == 'network':
        cmap = plotting.yeo_cmap(networks=19)

    if len(k) == 2:
        x, y = k
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3))
        
        # all regions  
        sns.scatterplot(x=x, y=y, data=base, color='k', alpha=.3, s=5, 
                        linewidths=0, legend=False, ax=ax)

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi").set_index('epoch').reindex(index=_epoch_order())
            xx = roi_df[x].values
            yy = roi_df[y].values
            val = roi_df[hue].iloc[0]
            ax.plot(xx, yy, lw=1, c=cmap[val])
            
            arrowprops = dict(lw=.1, width=.1, headwidth=4, headlength=3, 
                              color=cmap[val])
            ax.annotate(text='', xy=(xx[-1], yy[-1]), xytext=(xx[-2], yy[-2]), 
                        arrowprops=arrowprops)
        
        # plot color-coded markers of significant regions
        sns.scatterplot(x=x, y=y, data=sig_base, hue=hue, s=16, 
                        edgecolor='k', palette=cmap, linewidths=1, ax=ax, 
                        legend=False, zorder=20)
        sns.despine()
        return ax
    
    elif len(k) == 3:
        x, y, z = k
        sns.set(style='whitegrid')
        fig = plt.figure(figsize=(8, 4))
        gs = fig.add_gridspec(nrows=10, ncols=10)
        ax1 = fig.add_subplot(gs[:, :6], projection='3d')

        # remove sig regions so that their points don't obstruct their 
        # colour-coded points plotted below
        base_nonsig = base[~base['roi'].isin(sig_regions)]
        ax1 = plotting.plot_3d(base_nonsig[x], base_nonsig[y], base_nonsig[z],
                                color='gray', alpha=.3, s=1, ax=ax1, 
                                view_3d=(30, -120))
        ax1.set(xticks=range(-4, 6))

        # plot shifts/lines of significant regions
        for roi in shifts['roi'].unique():
            roi_df = shifts.query("roi == @roi").set_index('epoch').reindex(index=_epoch_order())
            xx = roi_df[x].values
            yy = roi_df[y].values
            zz = roi_df[z].values
            val = roi_df[hue].iloc[0]
            ax1.plot(xs=xx, ys=yy, zs=zz, lw=1, c=cmap[val])
        
        # color-coded significant regions
        sig_base['c'] = sig_base[hue].apply(lambda x: cmap[x])
        ax1 = plotting.plot_3d(sig_base[x], sig_base[y], sig_base[z], 
                                color=sig_base['c'], alpha=1, s=20,
                                ax=ax1, zorder=20, edgecolors='k', 
                                linewidths=.5)
        ax1.set(ylim=(-2, 3), xticks=np.arange(-2, 4, 1))
        sns.set(style='darkgrid')
        ax2 = fig.add_subplot(gs[:5, 6:9])
        ax2 = plot_displacements(data, anova, ['g1', 'g2'], ax=ax2)
        ax2.set(ylim=(-3, 3), xlim=(-3, 4), xticklabels=[], 
                xlabel='')
        ax2.set_ylabel('PC2', fontsize=12, fontweight='bold')
        ax3 = fig.add_subplot(gs[5:, 6:9])
        ax3 = plot_displacements(data, anova, ['g1', 'g3'], ax=ax3)
        ax3.set(ylim=(-3, 3), xlim=(-3, 4), xticks=np.arange(-3, 4, 1))
        ax3.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax3.set_ylabel('PC3', fontsize=12, fontweight='bold')

        fig.tight_layout()
        return fig, ax
    else:
        return None, None

def plot_sig_region_eccentricity(ttest_stats, gradients, fig_dir):
    '''
    plot eccentricity pattern for significant region for each contrast during all epochs for 
    all subjects
    
    Parameters
    ----------
    data : Pandas DataFrame 
        with eccentrity column for all rois and all subjects 
        and all the existing epochs in the experiment.
    seed : str
        roi name for the desired seed region.

    Returns
    -------
    plot of eccentricity pattern and mean eccentricity for input seed.

    '''
    ttest_stats = parse_roi_names(ttest_stats)
    ordered = {'leftbaseline': 'orange', 'rightbaseline': 'green', 'rightlearning-early': 'green', 
               'rightlearning-late': 'green', 'lefttransfer-early': 'orange', 'lefttransfer-late': 'orange'}
    effects = ['time', 'hand']
    cmap = list(ordered.values())
    for effect in effects:
        effect_sig_regions = ttest_stats.query('Contrast == @effect & sig_corrected == 1') \
                                    .groupby(['A', 'B'])['roi']
        for name, g in effect_sig_regions:
            for hemi in ['LH', 'RH']:
                data = gradients.query('roi in @g.values & hemi == @hemi').copy()
                data = data.groupby(['sub', 'epoch']).mean(numeric_only=True).reset_index()
                epoch_order = list(ordered.keys())
                data['epoch'] = pd.Categorical(data['epoch'], categories=epoch_order, ordered=True)
                data.sort_values('epoch', inplace=True)
                data.reset_index(drop=True, inplace=True)
                fig = plt.figure(figsize=(3, 3))
                sns.lineplot(data=data, x='epoch', y='distance', errorbar=None, 
                             marker='o', ms=6, lw=1.2, mfc='w', mec='k', color='k')
                sns.stripplot(data=data, x='epoch', y='distance', jitter=.1, 
                          zorder=-1, s=5, alpha=.5, palette=cmap, hue='epoch')
                plt.xlabel('', fontsize=12, fontweight='bold')
                plt.ylabel('Eccentricity', fontsize=12, fontweight='bold')
                ax = plt.gca()
                ticks = [0, 1, 2, 3, 4, 5]
                ax.set_xticks(ticks)
                
                # Set the tick labels with rotation and fontsize
                labels = ['Left: Baseline', 'Right: Baseline', 'Right Learning: Early',
                          'Right Learning: Late', 'Left Transfer: Early', 'Left Transfer: Late']
                ax.set_xticklabels(labels, rotation=90, fontsize=12)
                ax.set_yticks(np.arange(1, 4).astype(int))
                sns.despine()
                
                fig_name = f'{effect}_{name[1]}_vs_{name[0]}_{hemi}_sig-regions_ecc'
                fig.savefig(os.path.join(fig_dir, fig_name))
    effect = 'time * hand'
    effect_sig_regions = ttest_stats.query('Contrast == @effect & sig_corrected == 1')
    rois = effect_sig_regions['roi']
    data = gradients.query('roi in @rois').copy()
    data = data.groupby(['sub', 'epoch']).mean(numeric_only=True).reset_index()
    epoch_order = list(ordered.keys())
    data['epoch'] = pd.Categorical(data['epoch'], categories=epoch_order, ordered=True)
    data.sort_values('epoch', inplace=True)
    data.reset_index(drop=True, inplace=True)
    fig = plt.figure(figsize=(3, 3))
    sns.lineplot(data=data, x='epoch', y='distance', errorbar=None,
                 marker='o', ms=6, lw=1.2, mfc='w', mec='k', color='k', 
                 alpha=1, legend=False)
    sns.stripplot(data=data, x='epoch', y='distance', jitter=.1, 
                  zorder=-1, s=5, alpha=.7, palette=cmap, hue='epoch')
    plt.xlabel('', fontsize=12, fontweight='bold')
    plt.ylabel('Eccentricity', fontsize=12, fontweight='bold')
    ax = plt.gca()
    ticks = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks)
    
    # Set the tick labels with rotation and fontsize
    labels = ['Left: Baseline', 'Right: Baseline', 'Right Learning: Early',
              'Right Learning: Late', 'Left Transfer: Early', 'Left Transfer: Late']
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_yticks(np.arange(1, 4).astype(int))
    sns.despine()
    
    fig_name = f'{effect}_sig-regions_ecc'
    fig.savefig(os.path.join(fig_dir, fig_name))

def count_effects_sig_regions(anova_stats, fig_dir):
    """
    Count the number of significant regions in each network for each source and 
    plot the results as a bar chart.

    Parameters:
    anova_stats (pandas DataFrame): DataFrame containing ANOVA statistics
    fig_dir (str): Directory to save the output figures

    Returns:
    None

    This function first parses the ROI names in the ANOVA statistics, then 
    groups the data by source and network, and counts the number of significant 
    regions in each network for each source. The results are then plotted as a 
    bar chart, with the x-axis showing the networks, the y-axis showing the 
    proportion of significant regions, and the color of the bars indicating the 
    network. The figures are saved to the specified directory.
    """
    anova_stats = parse_roi_names(anova_stats)
    cmap = plotting.yeo_cmap(networks=19)
    network_counts = anova_stats.groupby(['Source', 'network'])['sig_corrected'] \
                                .sum().reset_index()
    for s in network_counts['Source'].unique():
        effect_count = network_counts.query('Source == @s')
        effect_count = effect_count.assign(sig_corrected_ratio = lambda x: 100 * (x['sig_corrected'] / x['sig_corrected'].sum()))
        effect_count.sort_values('sig_corrected_ratio', ascending=False, inplace=True)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(data=effect_count, x='network', y='sig_corrected_ratio', hue='network',
                    palette=cmap, ax=ax, saturation=.9)
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), ha='center', rotation=90, fontsize=12, fontweight='bold')
        ax.set_yticks(np.arange(0, 5 * (max(effect_count['sig_corrected_ratio']) // 5 + 1) + 1, 5))
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, fontweight='bold')
        sns.despine()
        fig.savefig(os.path.join(fig_dir, f'{s}_sig_regions_ratio'))
        
def main():

    config = Config()

    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )

    fig_dir = os.path.join(Config().figures, 'generalization')
    os.makedirs(fig_dir, exist_ok=True)
###############################################################################
#   two way anova stats + ttest posthoc    
    anova_stats, ttest_stats = eccentricity_analysis(gradients, 
                                                                 method='anova',
                                                                 factor=['time', 'hand'])
    plot_sig_region_eccentricity(ttest_stats, gradients, fig_dir)
    count_effects_sig_regions(anova_stats, fig_dir)

    anova_stats.to_csv(os.path.join(config.results, 'ecc_anova_stats.tsv'), 
                       sep='\t', index=False)
    ttest_stats.to_csv(os.path.join(config.results, 'ecc_ttest_stats.tsv'), 
                       sep='\t', index=False)
    #droping subcortical regions
    anova_stats_cor = anova_stats.query('roi_ix <= 400')
    ttest_stats_cor = ttest_stats.query('roi_ix <= 400')
    
    vmax_sig = int(np.nanmax(anova_stats.query('sig_corrected == 1')['F']))
    vmin_sig = np.nanmin(anova_stats.query('sig_corrected == 1')['F'])
    for f in anova_stats_cor['Source'].unique():
        effect_anova = anova_stats_cor[anova_stats_cor['Source']==f]
        anova_vertices = anova_stat_map(effect_anova,
                                          fig_dir, name=f+'_anova', 
                                          vmin=vmin_sig, vmax=vmax_sig)
        if anova_vertices is not None:
            np.savetxt(os.path.join(config.results, f'{f}_anova_vertices.tsv'),
                      anova_vertices)

    plt.close('all')
    vmax_sig = np.nanmax(ttest_stats.query('sig_corrected == 1')['T'])
    vmin_sig = np.nanmin(np.abs(ttest_stats.query('sig_corrected == 1')['T']))
    for f in ttest_stats_cor['Contrast'].unique():
        temp = ttest_stats[ttest_stats['Contrast'] == f]
        if f in ['time', 'hand']:
            plotting.pairwise_stat_maps(temp,
                                    os.path.join(fig_dir, f'{f}_ecc_ttests_'),
                                    vmax=vmax_sig, vmin=vmin_sig)
        elif f in ['time * hand']:
            for h in temp.iloc[:, 3].unique():
                plotting.pairwise_stat_maps(temp[temp.iloc[:, 3]==h], 
                                    os.path.join(fig_dir, f'{f}_{h}_ecc_ttests_'),
                                    vmax=vmax_sig, vmin=vmin_sig)
   
        # 3D plots
    if config.k == 3:
        fig_dir = os.path.join(Config().figures, 'displacements')
        os.makedirs(fig_dir, exist_ok=True)
        
        for i in anova_stats['Source'].unique():
            anova = anova_stats.query('Source == @i')
            fig, _ = plot_displacements(gradients, anova, config.k)
            fig.savefig(os.path.join(fig_dir, f'{i}_displacements'))             

if __name__ == '__main__':
    main()