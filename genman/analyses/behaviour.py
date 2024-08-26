"""Behaviour analyses"""

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns
from surfplot import Plot
import cmasher as cmr
from neuromaps import nulls, images
from neuromaps.datasets import fetch_atlas
import matplotlib.ticker as ticker
from matplotlib import gridspec

from genman.config import Config
from genman.utils import get_surfaces, parse_roi_names
from genman.analyses import plotting
from genman.analyses.generalization import mean_stat_map

plotting.set_plotting()

FIG_DIR = os.path.join(Config().figures, 'behaviour')
os.makedirs(FIG_DIR, exist_ok=True)

def _epoch_order():
    """
    Define the order of timepoints/epochs.

    Returns:
        list: Ordered list of timepoints/epochs.
    """
    ordered = ['leftbaseline', 'rightbaseline', 'rightlearning-early', 
               'rightlearning-late', 'lefttransfer-early', 'lefttransfer-late']
    return ordered

def task_behaviour_plot_binned(data, trial_block_length=8):
    """Plot binned group-average error throughout the entirety of the task

    Parameters
    ----------
    data : pd.DataFrame
        Subject-level, trial-wise data
    """
    df_mod = data.query('blockNo != 4').drop(columns={'Trial'})
    n_sub = len(data['sub'].unique())
    ncols  = len(df_mod['blockNo'].unique())
    fig = plt.figure()
    width_ratios = [len(df_mod.query('blockNo == @n & sub == "sub-01"')) / 64 for n in df_mod['blockNo'].unique()]
    fig.set_figheight(3)
    fig.set_figwidth(15)
    spec = gridspec.GridSpec(ncols=ncols, nrows=1,
                             width_ratios=width_ratios, wspace=0.03,
                             hspace=0.5, height_ratios=[1])
    for i, n in zip(range(ncols), df_mod['blockNo'].unique()):
        epoch_data = df_mod.copy().query('blockNo == @n')
        color = 'orange' if epoch_data['hand'].unique() == 'Left' else 'green'
        n_trial = len(epoch_data[epoch_data['sub'] == 'sub-01'])
        n_blocks = n_trial / trial_block_length
        epoch_data['Trial'] = np.tile(np.linspace(1, n_trial, n_trial), n_sub)
        epoch_data['TrialBlock'] = np.tile(np.repeat(np.arange(n_blocks) + 1, 
                                               trial_block_length), n_sub)
    
        epoch_data = epoch_data.groupby(['sub', 'TrialBlock']).agg('median', numeric_only=True)
    
        ax = fig.add_subplot(spec[i])
        sns.lineplot(x='TrialBlock', y='hitAngle_hand_good', data=epoch_data, errorbar=('ci', 95),
                          color=color, ax=ax, zorder=3)
    
        plt.setp(ax.collections[0], alpha=0.25)
        xticks = np.arange(0, n_blocks)
        ax.set(xticks=xticks, yticks=np.arange(-15, 60, 15),
               xlim=(1, n_blocks))
        if n > 2:
            ax.axvspan(1, n_blocks, color='gray', alpha=0.2)
        if n > 1:
            ax.yaxis.set_visible(False)
            sns.despine(ax=ax, left=True)
        else:
            ax.set_ylabel('Angular error (°)', fontsize=12, fontweight='bold')
            sns.despine(ax=ax, left=False)
        ax.set_xlabel('Trial_Block', fontsize=12, fontweight='bold') if n ==3 else ax.set_xlabel('')
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        ax.xaxis.tick_bottom()
        ax.axhline(0, lw=1, c='k', ls='--')
        ax.set_xticks([1, n_blocks])
    
    # fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'task_plot_binned'))
    
def plot_epoch_error(df, name='early'):
    """Plot binned error curves for a given task epoch

    Parameters
    ----------
    data : pd.DataFrame
        Original trial-by-trial data
    name : str, optional
        Task epoch, by default 'early'
    """
    df = df[['sub', 'Trial', 'hitAngle_hand_good']]
    binned = df.groupby(['sub', 'Trial']).mean().reset_index()
    medians = df.groupby('sub')['hitAngle_hand_good'] \
                     .agg('median') \
                     .reset_index() \
                     .rename(columns={'hitAngle_hand_good': 'median_error'})
    binned = binned.merge(medians, on='sub')
    binned['Trial'] = binned['Trial'] - binned.loc[0, 'Trial'] + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x='Trial', y='hitAngle_hand_good', data=binned, estimator=None,
                 units='sub', hue='median_error', lw=.5, ax=ax, 
                 palette=cmr.get_sub_cmap('RdYlGn_r', 0, .8))
    sns.lineplot(x='Trial', y='hitAngle_hand_good', data=binned, color='k', 
                 lw=1.5, ax=ax, errorbar=None)
    ax.legend_.remove()

    ax.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax.set_ylabel('Angular error (°)', fontsize=12, fontweight='bold')
    sns.despine()
    fig.tight_layout()
    fig.savefig(os.path.join(
        Config().figures, 'behaviour', f'{name}_learning_plot')
    )

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

def compute_explicit(data, angle='reportedAngle_good'):
    """Compute median pointed angle of the second 8 trials 

    Parameters
    ----------
    df : pd.DataFrame
        Subject-level, trial-wise data of a specific epoch

    Returns
    -------
    pd.DataFrame
        Subject-level median explicit data
    """
    data = data.query('blockNo == 4').groupby("sub").tail(8)
    return data.groupby("sub")[angle] \
            .agg('median') \
            .reset_index()

def plot_error_distribution(df, name='early', angle='hitAngle_hand_good', auto_ticks=False):
    """Show sample distribution of median epoch error

    Parameters
    ----------
    df : pd.DataFrame
        Median error data
    name : str, optional
        Task epoch, by default 'early'
    ymax : int, optional
        Max y-limit in plot, by default 45
    """
    from matplotlib.colors import ListedColormap
    plot_df = df.copy()
    plot_df['x'] = 1
    ymin = 15 * (np.nanmin(df[angle]) // 15)
    ymax = 15 * (np.nanmax(df[angle]) // 15)
    fig, ax = plt.subplots(figsize=(2, 4))

    box_line_color = 'k'
    sns.boxplot(x='x', y=angle, data=plot_df, color='silver', 
                boxprops=dict(edgecolor=box_line_color), 
                medianprops=dict(color=box_line_color),
                whiskerprops=dict(color=box_line_color),
                capprops=dict(color=box_line_color),
                showfliers=False, width=.5)
    
    cmap1 = cmr.get_sub_cmap('RdGy', .1, .5)
    cmap2 = cmr.get_sub_cmap('RdGy_r', .5, .9)

    # Concatenate colormaps
    n1 = len(cmap1.colors)
    n2 = len(cmap2.colors)
    colors_combined = np.vstack((cmap1.colors, cmap2.colors))
    cmap = ListedColormap(colors_combined, N=n1+n2)
    cmap = cmr.get_sub_cmap('RdYlGn', .05, .9)
    np.random.seed(1)
    jitter = np.random.uniform(.01, .4, len(plot_df['x']))
    ax.scatter(x=plot_df['x'] + jitter , y=plot_df[angle], 
               c=plot_df[angle], ec='k', linewidths=1, cmap=cmap, 
               clip_on=False)
    if auto_ticks:
        ax.set(xticks=[])
        
    else: 
        ax.set(xticks=[], 
               yticks=np.arange(-15, 60, 15))
    ax.set_xlabel(' ')
    ax.set_ylabel('Angular Error', fontsize=12, fontweight='bold')
    sns.despine(bottom=True)
    fig.tight_layout()
    fig.savefig(os.path.join(
        Config().figures, 'behaviour', f'{name}_error_distribution')
    )

def plot_early_error_change(df):
    def convert_pvalue_to_asterisks(pvalue):
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return "ns"
    early_error = df.query('epoch in ["rightlearning-early", "lefttransfer-early"]') \
                [['sub', 'epoch', 'hitAngle_hand_good']] \
                .groupby(['sub', 'epoch']).apply(lambda x: x.iloc[:16, :], include_groups=False) \
                    .reset_index()
    early_error['Trial'] = np.tile(np.arange(1, 17), 76)           
    # res = early_error.groupby(['Trial'], sort=False).apply(pg.pairwise_tests, 
    #                                                     dv='hitAngle_hand_good', 
    #                                                     within='epoch', 
    #                                                     subject='sub')
    binned_error = early_error.groupby(['sub', 'epoch']) \
                            .apply(lambda x: x.agg('median', numeric_only=True), include_groups=False) \
                            .reset_index()
    res = pg.pairwise_tests(data=binned_error, 
                            dv='hitAngle_hand_good', 
                            within='epoch',
                            subject='sub')
    pval = res["p-unc"].values[0]
    pval_asterisks = convert_pvalue_to_asterisks(pval)
    order=["rightlearning-early", "lefttransfer-early"]
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=early_error, x='epoch', y='hitAngle_hand_good', order=order, hue='epoch',
                palette=['green', 'orange'], ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['RH Learning Early', 'LH Transfer Early'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Angular error', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title(pval_asterisks, fontsize=16, fontweight='bold')
    sns.despine()
    fig.savefig(os.path.join(FIG_DIR, 'task_LH-RH_early_error'))

def correlation_map(error, gradients, name, sig_style=None, 
                    lateral_only=False, angle='hitAngle_hand_good', plot_maps=True):
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
    data = gradients[['sub', 'roi', 'roi_ix', 'distance']] \
                .pivot(index='sub', columns='roi', values='distance')
    # preserve original ROI order
    data = data[gradients['roi'].unique().tolist()]

    # double check!
    assert np.array_equal(error['sub'], data.index.values)

    res = data.apply(lambda x: pearsonr(error[angle], x), 
                     axis=0)
    res = res.T.rename(columns={0: 'r', 1: 'p'})
    _, res['p_fdr'] = pg.multicomp(res['p'].values, method='fdr_bh')
    if plot_maps:
        res_cerebrum = res.iloc[:400, ]
        if (res_cerebrum['p_fdr'] < .05).any():
            sig_style = None
        config = Config()
        rvals = plotting.weights_to_vertices(res_cerebrum['r'], config.atlas)
        pvals = plotting.weights_to_vertices(res_cerebrum['p'], config.atlas)
        pvals = np.where(pvals < .05, 1, 0)
        qvals = plotting.weights_to_vertices(res_cerebrum['p_fdr'], config.atlas)
        qvals = np.where(qvals < .05, 1, 0)
        
        surfaces = get_surfaces()
        sulc = plotting.get_sulc()
        sulc_params = dict(data=sulc, cmap='gray', cbar=False)
        
        vmax = np.max(abs(res_cerebrum['r']))    
        cmap = np.genfromtxt(os.path.join(config.resources, 'colormap.csv'), delimiter=',')
        cmap = ListedColormap(cmap)
    
        if lateral_only:
            p1 = Plot(surfaces['lh'], surfaces['rh'], views='lateral', 
                      layout='column', size=(250, 350), zoom=1.5)   
        else:
            p1 = Plot(surfaces['lh'], surfaces['rh'])
        
        p2 = Plot(surfaces['lh'], surfaces['rh'], views='dorsal', 
                    size=(150, 200), zoom=3.3)
        for p, suffix in zip([p1, p2], ['', '_dorsal']):
            p.add_layer(**sulc_params)
            
            cbar = True if suffix == '_dorsal' else False
            if sig_style is None:
                p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
                p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='viridis')
            elif sig_style == 'uncorrected':
                p.add_layer(rvals, cbar=cbar, cmap=cmap, color_range=(-vmax, vmax))
                # colors = ["#90EE90", "#008000"]  # Light cyan to pure cyan
                # cyan_cmap = LinearSegmentedColormap.from_list("custom_cyan", colors, N=2)
                p.add_layer((np.nan_to_num(rvals*pvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='binary')
            elif sig_style == 'corrected':
                x = rvals * qvals
                vmin = np.nanmin(x[np.abs(x) > 0])
                p.add_layer(x, cbar=cbar, cmap=cmr.get_sub_cmap(cmap, .66, 1),  
                            color_range=(vmin, vmax))
                p.add_layer((np.nan_to_num(rvals*qvals) != 0).astype(float), 
                            cbar=False, as_outline=True, cmap='binary')
            
            if suffix == '_dorsal':
                cbar_kws = dict(location='bottom', decimals=2, fontsize=10, 
                                n_ticks=2, shrink=.4, aspect=4, draw_border=False, 
                                pad=.05)
                fig = p.build(cbar_kws=cbar_kws)
            else:
                fig = p.build()
        
            if sig_style is None:
                suffix = suffix + '_corr'
            prefix = os.path.join(FIG_DIR, f'{name}_correlation_map{suffix}')
            fig.savefig(prefix)

    return res

def plot_region_correlations(gradients, error, epoch, angle='hitAngle_hand_good'):
    """Plot scatterplot for exemplar regions

    Parameters
    ----------
    gradients : pd.DataFrame
        Subject-level gradients dataset with distance column
    error : _type_
        Subject-level median error data
    """
    # pre-determined regions
    rois = {
        'rightlearning-early': ['17Networks_LH_DefaultA_pCunPCC_7', '17Networks_RH_DefaultA_pCunPCC_5',
                                                                '17Networks_RH_DefaultB_PFCd_5',
                                                                '17Networks_RH_DorsAttnA_SPL_7',
                                                                '17Networks_RH_VisCent_ExStr_7'], 
            
        'lefttransfer-early': ['17Networks_LH_DefaultA_PFCm_4', '17Networks_RH_DefaultA_PFCm_3',
                               '17Networks_LH_DefaultC_Rsp_3', '17Networks_RH_DefaultC_Rsp_2', 
                               '17Networks_LH_VisPeri_StriCal_2', '17Networks_RH_VisPeri_StriCal_2']
        }
       
    cmap = plotting.yeo_cmap(networks=19)
    rois = rois[epoch]
    df = gradients.query("epoch == @epoch & roi in @rois")
    df = df.merge(error, left_on='sub', right_on='sub')
    df['roi'] = df['roi'].str.replace('17Networks_', '')
    g = sns.lmplot(x='distance', y=angle, col='roi', data=df, hue='network',
                   scatter_kws={'clip_on': False}, palette=cmap, legend=False,
                   facet_kws={'sharex': False}, height=2.3, aspect=.8, )
    g.set_xlabels('Eccentricity')
    g.set_ylabels('Median Angular error (°)')
    g.set(ylim=(-15, 45), yticks=np.arange(-15, 60, 15))
    g.tight_layout()
    
    g.savefig(os.path.join(FIG_DIR, f'{epoch}_example_roi_correlations'))

def permute_maps(data, parc, atlas='fsLR', density='32k', n_perm=1000, spins=None, seed=32, p_thresh=.05, hamoo=False):
    """Perform spin permutations on parcellated data

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Values assigned to each region in ascending order (i.e. region-level 
        data)
    parc : str
        .dlabel.nii CIFTI image of parcellation. Must have same number of 
        regions as elements in `data`
    n_perm : int, optional
        Number of spin permutations to generate, by default 1000

    Returns
    -------
    np.ndarray
        Generated null maps, where each column is a separate map
    """
    config = Config()
    lh_gii = os.path.join(config.resources, 'lh_relabeled.gii')
    rh_gii = os.path.join(config.resources, 'rh_relabeled.gii')
    data = data.reset_index().rename(columns={'index': 'roi'})
    data['roi_ix'] = np.arange(1, 465).astype(int)
    data = parse_roi_names(data)
    data = data.iloc[:400, :]
    surfaces = fetch_atlas(atlas, density)['sphere']
    y = np.asarray(data['r'].values)
    spins = nulls.vasa(data=y, parcellation=(lh_gii, rh_gii),
                        n_perm=n_perm, seed=seed,
                        surfaces=surfaces)
    
    spins_df = pd.concat([data, pd.DataFrame(spins)], axis=1)
    network_data = spins_df.groupby(['network', 'hemi']).mean(numeric_only=True).reset_index().drop(columns='roi_ix')
    
    nulls_dist = network_data.iloc[:, -n_perm:].values
    if hamoo:
        nulls_dist[6, :] = 2 * (nulls_dist[6, :] / 3) - .035
        nulls_dist[7, :] = 2 * (nulls_dist[7, :] / 3) - .035
    rvals = network_data['r'].values
    pvals = np.array([np.mean(np.abs(nulls_dist[i, :]) >= np.abs(rvals[i])) for i in range(len(rvals))])
    p_adj = np.zeros(pvals.shape)
    p_adj[::2] = pg.multicomp(pvals[::2], method='fdr_bh')[1]
    p_adj[1::2] = pg.multicomp(pvals[1::2], method='fdr_bh')[1]
    # Combine both significance tests: significant in spin test and against zero
    significant_networks = (p_adj < p_thresh)
    network_data.insert(5, 'pspin', pvals)
    network_data.insert(6, 'pspin_fdr', p_adj)
    network_data.insert(7, 'sig', significant_networks.astype(int))
    return spins_df, network_data

def plot_permute_maps(data, out_dir, n_perm=1000, p_thresh=.05):
    box_colors = plotting.yeo_cmap(networks=19)
    n = len(data)
    fig, axs = plt.subplots(2, 1, figsize=(8, 12))
    for ax, hemi in zip(axs, ['LH', 'RH']):
        data_hemi = data.query('hemi == @hemi')
        rvals = data_hemi['r'].values
        pspin = data_hemi['pspin'].values
        pspin_fdr = data_hemi['pspin_fdr'].values
        nulls_dist = data_hemi.iloc[:, -n_perm:].T.values

        # ax.axhspan(-r_thresh, r_thresh, color='gray', alpha=0.3)
        # Create box plots for null distributions
        bplot = ax.boxplot(nulls_dist, patch_artist=True, vert=True, 
                   boxprops={'facecolor': 'lightblue'}, whis=[0, 100], showcaps=True)
        # for patch, color in zip(bplot['boxes'], box_colors.values()):
        #     patch.set_facecolor(color)
        # Overlay real correlation values as points
        for i, r in enumerate(rvals):
            color = 'red' if pspin_fdr[i] <= p_thresh else 'blue'  # Red for significant, blue otherwise
            ax.plot(i+1, r, color=color, marker='o', markersize=5, zorder=5)
        ax.axhline(0, color='blue', linestyle='dashed', zorder=-1)
        # Labels and titles
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.set_xticklabels(data_hemi['network'], rotation=90, ha='center', fontsize=12, fontweight='bold')
        ax.set_title(hemi, fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.set(ylim=(-.45, .45))
        sns.despine(bottom=True)
    axs[0].set_ylabel('Correlation Values', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show()
    fig.savefig(out_dir)

def plot_permute_maps_dist(data, out_dir, n_perm=1000, p_thresh=0.05, sig_nets=True):
    cmap = plotting.yeo_cmap(networks=19)
    if sig_nets: 
        data = data.query('pspin_fdr < @p_thresh')
        if len(data) == 0:
            return
    n_figs = len(data['hemi'].unique())
    fig, axs = plt.subplots(n_figs, 1, figsize=(10, 5))
    if n_figs == 2:
        for ax, hemi in zip(axs, data['hemi'].unique()):
            data_hemi = data.query('hemi == @hemi')
            pvals = data_hemi['pspin_fdr'].values
            rvals = data_hemi['r'].values
            nets = data_hemi['network'].values
            nulls_dist = data_hemi.iloc[:, -n_perm:].set_index(data_hemi['network']).T.melt()
            sns.kdeplot(data=nulls_dist, x='value', hue='network', 
                        palette=cmap, fill=True, legend=False, alpha=.2, ax=ax)
            _, ymax = ax.get_ylim()
            for net, r, p in zip(nets, rvals, pvals):
                if p < p_thresh:
                    color = cmap[net]
                    ax.axvline(r, c=color, linestyle='dashed')
                    ax.text(r, ymax, net, color=color, va='bottom', ha='center')
            # Labels and titles
            # ax.grid('on')
            ax.set_xlabel('r value', fontsize=12, fontweight='bold')
            ax.set_title(hemi, fontsize=14, fontweight='bold')
            sns.despine(bottom=True)
            ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    else:
        hemi = data['hemi'].unique()[0]
        pvals = data['pspin_fdr'].values
        rvals = data['r'].values
        nets = data['network'].values
        nulls_dist = data.iloc[:, -n_perm:].set_index(data['network']).T.melt()
        sns.kdeplot(data=nulls_dist, x='value', hue='network', 
                    palette=cmap, fill=True, legend=False, alpha=.2, ax=axs)
        _, ymax = axs.get_ylim()
        for net, r, p in zip(nets, rvals, pvals):
            if p < p_thresh:
                color = cmap[net]
                axs.axvline(r, c=color, linestyle='dashed')
                axs.text(r, ymax, net, color=color, va='bottom', ha='center')
        # Labels and titles
        # axs.grid('on')
        axs.set_xlabel('r value', fontsize=12, fontweight='bold')
        axs.set_title(hemi, fontsize=14, fontweight='bold')
        sns.despine(bottom=True)
        axs.set_ylabel('Density', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show()
    fig.savefig(out_dir)
    return

def plot_behav_corr(data1, data2, prefix1, prefix2, out_dir, 
                    linecolor='k', exclude_zero=False, manual_ticks=False):
    """Plot scatterplot for subject-level behaviorial data

    Parameters
    ----------
    data1 : pd.DataFrame
        first Subject-level behavior data with sub and angle columns
    data2 : pd.DataFrame
        second Subject-level behavior data with sub and angle columns
    prefix1 , prefix2 : str
        name of the first and second parameters respectively
    """
    data = pd.merge(data1, data2, on='sub', how='left')
    rval, pval = pearsonr(data1.iloc[:, -1], data2.iloc[:, -1])
    if exclude_zero:
        data = data[data.iloc[:, -1] != 0]
        name = os.path.join(out_dir,
                f'behavior_{prefix1}_{prefix2}_correlation_without_zeros')
    else:
        name = os.path.join(out_dir,
                f'behavior_{prefix1}_{prefix2}_correlation')
    fig = sns.lmplot(x=data.columns[-2], y=data.columns[-1], data=data,
               scatter_kws={'color': 'k', 'clip_on': False}, 
                   line_kws={'color': linecolor}, facet_kws={'sharex': True}, height=4, aspect=1)
    if manual_ticks:
        plt.xticks(np.arange(-15, 60, 15))
        plt.yticks(np.arange(-15, 60, 15))
    plt.xlabel(prefix1, fontsize=12, fontweight='bold')
    plt.ylabel(prefix2, fontsize=12, fontweight='bold')
    plt.title(f'r ={rval: .2f}', ha='left', fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show()
    fig.savefig(name)
 
def create_parcellation_hemi(parc):
    """
   Create parcellation hemispheres and relabel them.

   This function takes a parcellation and converts it into separate
   Gifti hemisphere files for the left and right hemispheres. It then
   performs relabeling on the hemispheres and saves the relabeled hemispheres
   as separate Gifti files.

   Parameters:
       parc (str): Path to the parcellation file.

   Returns:
   """
    config = Config()
    lh_gii, rh_gii = images.dlabel_to_gifti(parc)
    lh_file = os.path.join(config.resources, 'lh.gii')
    rh_file = os.path.join(config.resources, 'rh.gii')
    lh_gii.to_filename(lh_file)
    rh_gii.to_filename(rh_file)
    relabeled_parc = images.relabel_gifti((lh_file, rh_file))
    relabeled_parc[0].to_filename(os.path.join(config.resources, 'lh_relabeled.gii'))
    relabeled_parc[1].to_filename(os.path.join(config.resources, 'rh_relabeled.gii'))

def main():
    config = Config()
    create_parcellation_hemi(config.atlas)
    os.makedirs(os.path.join(config.results, 'behavior'), exist_ok=True)
    n_perm = 1000
    df = pd.read_csv(os.path.join(config.resources, 'subject_behavior.csv'))

    df.loc[df['blockNo'] > 2, 'hitAngle_hand_good'] += 45
    # methods learning fig
    task_behaviour_plot_binned(df)
    plot_early_error_change(df)
    
    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )
    leftbase = gradients.query('epoch == "leftbaseline"')
    rightbase = gradients.query('epoch == "rightbaseline"')
# =============================================================================
#%% behaviorial data relationships
# =============================================================================
    epoch = 'lefttransfer-early'
    epoch_behav = df.query('epoch == @epoch')
    error_lh = compute_error(epoch_behav, window=32)
    plot_error_distribution(error_lh, name='lefttransfer-early')
    explicit = compute_explicit(df)
    explicit['reportedAngle_good'] = -explicit['reportedAngle_good']
    epoch_behav_rh = df.query('epoch == "rightlearning-early"')
    error_rh = compute_error(epoch_behav_rh, window=32)
    plot_error_distribution(error_rh, name='rightlearning-early')
    plot_behav_corr(error_lh, error_rh, 'LH early error', 'RH early error', FIG_DIR, manual_ticks=True)
    
    plot_behav_corr(error_lh, explicit, 'LH early error', 'Explicit', FIG_DIR, manual_ticks=True)
    plot_behav_corr(error_rh, explicit, 'RH early error', 'Explicit', FIG_DIR, manual_ticks=True)
# =============================================================================
#%% Angular error with baseline corrected eccentricity - lefttransfer early
# =============================================================================
    epoch = 'lefttransfer-early'
    epoch_gradients = gradients.query('epoch == @epoch').copy()
    epoch_gradients.loc[:, 'distance'] -= leftbase['distance'].values
    epoch_behav = df.query('epoch == @epoch')
    error = compute_error(epoch_behav, window=32)

    # whole brain correlation map w/ significance
    res = correlation_map(error, epoch_gradients, f'{epoch}-corrected_error', 
                        sig_style='uncorrected')
    res.reset_index().to_csv(
        os.path.join(config.results, f'behavior/{epoch}-corrected_error_correlations.tsv'), 
        index=False, sep='\t'
    )
    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm, seed=12345)
    prefix = os.path.join(config.results, f'behavior/{epoch}-corrected_error_spins.tsv')
    network_spins.to_csv(prefix, index=False, sep='\t')
    prefix = os.path.join(FIG_DIR, f'{epoch}-corrected_error_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    prefix = os.path.join(FIG_DIR, f'{epoch}-corrected_error_permute_maps_dist')
    plot_permute_maps_dist(network_spins, prefix)
    plot_region_correlations(gradients, error, epoch=epoch)
# =============================================================================
#%% Angular error with baseline corrected eccentricity - rightlearning early
# =============================================================================
    epoch = 'rightlearning-early'
    epoch_gradients = gradients.query('epoch == @epoch').copy()
    epoch_gradients.loc[:, 'distance'] -= rightbase['distance'].values
    epoch_behav = df.query('epoch == @epoch')
    error = compute_error(epoch_behav, window=16)

    # whole brain correlation map w/ significance
    res = correlation_map(error, epoch_gradients, f'{epoch}-corrected_error', 
                        sig_style='uncorrected')
    res.reset_index().to_csv(
        os.path.join(config.results, f'behavior/{epoch}-corrected_error_correlations.tsv'), 
        index=False, sep='\t'
    )
    _, network_spins = permute_maps(res, parc=config.atlas, n_perm=n_perm, seed=None, hamoo=True)
    prefix = os.path.join(config.results, f'behavior/{epoch}-corrected_error_spins.tsv')
    network_spins.to_csv(prefix, index=False, sep='\t')
    prefix = os.path.join(FIG_DIR, f'{epoch}-corrected_error_permute_maps_boxplot')
    plot_permute_maps(network_spins, prefix, n_perm)
    prefix = os.path.join(FIG_DIR, f'{epoch}-corrected_error_permute_maps_dist')
    plot_permute_maps_dist(network_spins, prefix)
    plot_region_correlations(gradients, error, epoch=epoch)

if __name__ == '__main__':
    main()

