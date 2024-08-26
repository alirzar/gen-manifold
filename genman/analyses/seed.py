import os, glob
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import pingouin as pg
import cmasher as cmr
from surfplot import Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from genman.config import Config
from genman.utils import (get_files, schaefer400tian_nettekoven_roi_ix,
                          fdr_correct, parse_roi_names
                          )
from genman.analyses import plotting
from genman.analyses.plotting import (get_sulc, get_surfaces, set_plotting,
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
        return name[0].split('-')[-1] + '_' + name[1].split('.')[0]
    
def get_roi_ix464():
    gradients = pd.read_table(os.path.join(Config.results, 'subject_gradients.tsv'))
    base = np.unique(gradients['epoch'].values)[0]
    roi_ix = gradients.query('sub == "sub-01" & epoch == @base')[['roi', 'roi_ix']]
    return roi_ix 
    
def connect_seed(cmats, seed_region):
    """Extract seed connectivity by isolating row in connectivity matrix

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
    ref_rois = get_roi_ix464()
    list_ = []
    for i in cmats:
        cmat = pd.read_table(i, index_col=0)
        # isolate row of seed region
        res = pd.DataFrame(cmat.loc[seed_region].reset_index().values, 
                           columns=['roi', 'r'])

        res =  ref_rois.merge(res[['roi', 'r']], left_on='roi', right_on='roi')
        #res['roi_ix'] = schaefer400tian_nettekoven_roi_ix()
        fname = os.path.basename(i)
        res['sub'] = fname.split('_')[0]
        res['epoch'] = get_epoch_name(fname)
        list_.append(res)

    connectivity = pd.concat(list_)
    connectivity['r'] = connectivity['r'].astype('float')
    connectivity['hand'] = connectivity['epoch'].str.extract('(left|right)')
    connectivity['time'] = connectivity['epoch'].str.extract('(baseline|early|late)')
    return connectivity


def seed_analysis(cmats, epochs, seed, effect):
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
    connectivity = connect_seed(cmats, seed)

    df = connectivity[connectivity[effect].str.contains('|'.join(epochs), case=False, na=False)].copy()

    res = df.groupby(['roi', 'roi_ix'], sort=False) \
            .apply(pg.pairwise_tests, dv='r', within=effect, subject='sub', 
                   include_groups=False) \
            .reset_index()
    # swap sign so that B condition is the positive condition
    res['T'] = -res['T']
    res
    res['sig'] = (res['p-unc'] < .05).astype(float)

    # get network-level changes
    df = parse_roi_names(df)
    networks = df.groupby(['sub', 'network', 'hemi', effect]) \
                 .mean(numeric_only=True) \
                 .reset_index()
    return fdr_correct(res), networks


def plot_seed_map(data, seed_region, fig_dir, sig_style=None, views=['lateral', 'medial'], 
                  use_fdr=True, seed_color='yellow', cortical_seed=True, vmax=None):
    """Generate seed connectivity contrast maps

    Parameters
    ----------
    data : pandas.DataFrame
        Region-wise seed connectivity results
    seed_region : str
        Seed region name
    sig_style : str, optional
        Significance indication, by default None
    use_fdr : bool, optional
        If showing significance, show FDR-corrected results, by default True
    seed_color : str, optional
        Seed region color, by default 'yellow'
    show_left_vis : bool, optional
        Show left visual cortex, which is necessary for the visual seed only, 
        by default False

    Returns
    -------
    matplotlib.figure.Figure
        Seed contrast stat map
    """
    if use_fdr:
        sig_regions = data.query("sig_corrected == 1")
    else:
        sig_regions = data.query("sig == 1")
    
    x = weights_to_vertices(data['T'].astype(float).values, Config().atlas, 
                            data['roi_ix'].values)
    y = weights_to_vertices(np.ones(len(sig_regions)), Config().atlas, 
                            sig_regions['roi_ix'].values)
    if cortical_seed:
        seed = (data['roi'] == seed_region).astype(float)    
        z = weights_to_vertices(seed.values, Config().atlas, data['roi_ix'].values)
        seed_cmap = LinearSegmentedColormap.from_list(
            'regions', [seed_color, 'k'], N=2
        )

    surfs = get_surfaces()
    sulc = get_sulc()
    p = Plot(surfs['lh'], surfs['rh'], views=views)
    p.add_layer(data=sulc, cmap='gray', cbar=False)
    if vmax is None:
        vmax = np.nanmax(np.abs(x))
        suffix = seed_region
    else:
        suffix = 'all_seeds'
    cmap = cmr.get_sub_cmap('seismic', 0.2, 0.8)
    plotting.plot_cbar(cmap, -vmax, vmax, 'horizontal', size=(1, .3), 
                     n_ticks=3)
    plt.savefig(os.path.join(fig_dir, 'cbar_' + suffix), bbox_inches='tight')  
    if sig_style == 'trace':
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        p.add_layer(np.nan_to_num(y), as_outline=True, 
                    cmap='binary', cbar=False)
    elif sig_style == 'threshold':
        p.add_layer(x*np.nan_to_num(y), cmap=cmap, color_range=(-vmax, vmax))
    elif sig_style is None:
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax))
        
    if cortical_seed:
        p.add_layer(np.nan_to_num(z), cmap=seed_cmap, cbar=False)
        p.add_layer(np.nan_to_num(z), as_outline=True, cmap='binary', cbar=False)
    
    cbar_kws = dict(location='bottom', decimals=1, fontsize=14, n_ticks=3, 
                    shrink=.2, aspect=6, draw_border=False, pad=-.06)
    fig = p.build(colorbar=False)

    return fig

def get_seed_color():
    return {
        ('baseline', 'early'): dict(color='#1E90FF', alpha=0.5),  # Blue
        ('early', 'late'): dict(color='#FF8650', alpha=0.6)  # Red
    }

def plot_spider(res, out_dir):
    colors = get_seed_color()
    custom_order =[
                'VisCent',
                'VisPeri',
                'SomMotA',
                'SomMotB',
                'DorsAttnA',
                'DorsAttnB',
                'SalVentAttnA',
                'SalVentAttnB',
                'TempPar',
                'ContA',
                'ContB',
                'ContC',
                'DefaultA',
                'DefaultB',
                'DefaultC',
                'LimbicA',
                'LimbicB',
                'Subcortex',
                'Cerebellum',
                ]
    
    roi_cols = ['network', 'hemi', 'T']
    # Create the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    all_values = []
    for r, c in zip(res, colors):
        data = parse_roi_names(r)[roi_cols]
        
        # Perform the groupby operation and preserve the order
        data_net = data.groupby(['network']).mean(numeric_only=True).reset_index()
        data_net['network'] = pd.Categorical(data_net['network'], categories=custom_order, ordered=True)
        data_net = data_net.sort_values('network')

        # Extract values and labels
        values_net = data_net['T'].tolist()
        all_values += values_net
        network_labels = data_net['network'].tolist()
        
        # Number of variables
        num_vars = len(values_net)
        
        # Compute angle of each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Repeat the first data point to close the circular plot
        values_net += values_net[:1]
        angles += angles[:1]
    
        #whole brain
        cmap = colors[c]
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
    tmin = np.round(min(all_values))
    tmax = np.round(max(all_values))
    ax.set_rgrids(np.arange(tmin, tmax+.5, .5))

    # plt.show()
    fig.savefig(out_dir)
    return fig

def plot_spider_legend(out_dir):
    # Define the colors and labels
    colors = [c['color'] for c in get_seed_color().values()]
    labels = ['Baseline to Early', 'Early to Late']
    patches = [mpatches.Rectangle((0, 0), 1, 1, facecolor=color, 
                                  edgecolor='black', label=label) 
                                   for color, label in zip(colors, labels)]
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.legend(handles=patches, fontsize=14, ncol=1, 
              edgecolor='w', handlelength=1.5, handleheight=1.5)
    ax.axis('off')
    # plt.show()
    # Save the figure
    out = os.path.join(out_dir, 'spider_legend')
    fig.savefig(out)

def _epoch_order():
    ordered = {'leftbaseline': 'orange', 'rightbaseline': 'green', 'rightlearning-early': 'green', 
               'rightlearning-late': 'green', 'lefttransfer-early': 'orange', 'lefttransfer-late': 'orange'}
    return ordered

def plot_seed_eccentricity(gradients, seed, out_dir):
    '''
    plot eccentricity pattern for each seed region during all epochs for 
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
    data = gradients[gradients['roi'] == seed].copy()
    epoch_order = list(_epoch_order().keys())
    cmap = list(_epoch_order().values())
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
    ax.set_yticks(np.arange(1, np.max(data['distance'].values)).astype(int))
    sns.despine()
    # plt.show()
    fig.savefig(out_dir)

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
    res_dir = os.path.join(config.results, 'seed')
    os.makedirs(res_dir, exist_ok =True)
    fig_dir = os.path.join(config.figures, 'seed')
    os.makedirs(fig_dir, exist_ok=True)
    hand_seeds = [
                    '17Networks_LH_SomMotA_5',

                    '17Networks_RH_SomMotA_10',
                    ]
    
    time_seeds = [
                    '17Networks_LH_DefaultA_PFCm_4',
                    
                    '17Networks_RH_DefaultA_PFCm_3',
                    ]
    
    all_seeds = dict(hand=hand_seeds, time=time_seeds)
    effects = ['hand', 'time']
    effect_epochs = dict(hand=[['left', 'right']], time=[['baseline', 'early'], ['early', 'late']])
    for effect in effects:
        seeds = all_seeds[effect]
        epochs = effect_epochs[effect]
        for seed in seeds:
            ecc_name = os.path.join(fig_dir, f'{seed}_ecc')
            plot_seed_eccentricity(gradients, seed, ecc_name)
            seed_results = []
            for e in epochs:
                suffix = f'{e[1]}_vs_{e[0]}'
                res, networks = seed_analysis(cmats, epochs=e, seed=seed, effect=effect)
                res.to_csv(os.path.join(res_dir, f'{suffix}_{seed}.tsv'), sep='\t', index=False)
                seed_results.append(res)
                res_cor = res[res['roi_ix'] <= 400]
                if (effect == 'hand') & ('RH' in seed):
                    res_cor['T'] = -res_cor['T']
                    suffix = f'{e[0]}_vs_{e[1]}'
                cortical_seed = '17' in seed
                fig = plot_seed_map(res_cor, seed, fig_dir=fig_dir, sig_style=None, 
                                    cortical_seed=cortical_seed, vmax=4)
                fig_name = os.path.join(fig_dir, f'{suffix}_{seed}')
                fig.savefig(fig_name)
                fig = plot_seed_map(res_cor, seed, fig_dir=fig_dir, sig_style=None, 
                                    cortical_seed=cortical_seed, 
                                    vmax=4, views='dorsal')
                fig_name = os.path.join(fig_dir, f'{suffix}_{seed}_dorsal')
                fig.savefig(fig_name)
            
            if effect == 'time':
                spider_name = os.path.join(fig_dir, f'{seed}_spider')
                plot_spider(seed_results, out_dir=spider_name)
                
    plot_spider_legend(fig_dir)
if __name__ == '__main__':
    main()
  
