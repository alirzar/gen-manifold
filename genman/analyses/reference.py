"""Reference creation and analysis module"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cmasher as cmr
from surfplot import Plot
from genman.utils import load_table, load_gradients, get_surfaces
from genman.config import Config
from genman.analyses import plotting

FIG_DIR = os.path.join(Config().figures, 'reference')
os.makedirs(FIG_DIR, exist_ok=True)


plotting.set_plotting()


def plot_eigenvalues():
    """Plot variance explained and cumulative variance explained
    """
    fname = os.path.join(Config().results, 'ref_eigenvalues.tsv')
    k = 10
    eigenvals = pd.read_table(fname)['proportion'][:k]
    cum_eigenvals = pd.read_table(fname)['cumulative'][:k]

    fig, ax = plt.subplots( figsize=(3, 4))
    ax.plot(np.arange(k) + 1, cum_eigenvals, marker='o', markerfacecolor='k', 
             color='k')
    ax.grid('on')
    ax.bar(np.arange(k) + 1, eigenvals, color='c', zorder=2)
    ax.set_xlabel(xlabel='Principal Component (PC)', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel='Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set(xticks=np.arange(0, k + 1, 2), yticks=np.arange(0, 1, .1))
    sns.despine()
    fig.tight_layout(w_pad=3)
    return fig


def show_eigenvalues():
    config = Config()
    fname = os.path.join(config.gradients, 'reference_eigenvalues.tsv')
    df = load_table(fname)
    df['cumulative'] = df['proportion'].cumsum()
    return df


def plot_ref_brain_gradients(k=3):
    """Plot region gradient weights on brain surface

    Parameters
    ----------
    k : int
        Number of gradients to plot (e.g., 2)

    Returns
    -------
    Ipython figure
        Surface gradient figure
    """
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    #filtering out subcortex and cerebellum for visualization
    filtered_networks = ['Cerebellum']
    gradients = load_gradients(fname, k).query("network not in @filtered_networks")
    prefix = os.path.join(FIG_DIR, 'gradients_')
    cmap = cmr.get_sub_cmap('twilight_shifted', 0.05, .95)

    grads = gradients.filter(like='g').values
    vmax = np.around(grads.max(), decimals=1)
    plotting.plot_cbar(cmap, -vmax, vmax, 'horizontal', size=(1, .3), 
                         n_ticks=3)
    plt.savefig(prefix + 'cbar')        
    
    surfaces = get_surfaces()
    for i in range(k):
        x = plotting.weights_to_vertices(grads[:, i], Config().atlas)

        p = Plot(surfaces['lh'], surfaces['rh'])
        p.add_layer(x, cmap=cmap, color_range=(-vmax, vmax), cbar=False)
        fig = p.build(colorbar=False)
        fig.savefig(prefix + f'PC{i + 1}_brain')
                                       

def plot_loading_distributions(k=3, view_3d=(30, -120)):
    """Plot Jointplot of top dimensions, labelled according to Yeo networks

    Parameters
    ----------
    k : int, optional
        Number of gradients to include, by default 3
    view_3d : tuple, optional
        Viewing orientation when plotting 3 dimensions. By default (30, -110)

    Returns
    -------
    seaborn.axisgrid.JointGrid or matplotlib.axes._subplots.Axes3DSubplot
        Network loading/distribution plot
    """
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, k)

    cmap = plotting.yeo_cmap(networks=19)
    if k == 2:
        g = sns.jointplot(x='g1', y='g2', hue='network', data=df, 
                        palette=cmap, legend=False, height=4.5, 
                        marginal_kws=dict(alpha=.7), 
                        joint_kws=dict(linewidth=0, s=15),)
        g.ax_joint.set(xlabel='PC1', ylabel='PC2')
        return g

    if k == 3:
        df['c'] = df['network'].apply(lambda x: cmap[x])
        sns.set(style='whitegrid')
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(nrows=16, ncols=16) 
        # 3D scatter plot
        ax1 = fig.add_subplot(gs[:, :7], projection='3d')
        plotting.plot_3d(df['g1'], df['g2'], df['g3'], color=df['c'], ax=ax1, 
                           view_3d=view_3d, s=15, lw=0.5, alpha=.8)
        ax1.set(zlim=(-2, 2), ylim=(-1, 3))
        sns.set(style='darkgrid')
        # 2D views
        ax2 = fig.add_subplot(gs[:8, 7:11])
        sns.scatterplot(x='g1', y='g2', hue='network', data=df, 
                    palette=cmap, legend=False, ax=ax2, linewidths=0, s=15, 
                    edgecolor='none')
        ax2.set(xlabel='', xticklabels=[], xlim=(-2.5, 3.5),
                ylim=(-2.5, 3))
        ax2.set_ylabel('PC2',fontsize=12, fontweight='bold')
        ax2 = fig.add_subplot(gs[8:, 7:11])
        sns.scatterplot(x='g1', y='g3', hue='network', data=df, 
                    palette=cmap, legend=False, ax=ax2, linewidths=0, s=15,
                    edgecolor='none')
        ax2.set(xlim=(-2.5, 3.5), ylim=(-2.5, 3))
        ax2.set_xlabel('PC1', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PC3', fontsize=12, fontweight='bold')

        # distribution plots
        for i, g in zip([2, 7, 12], ['g1', 'g2', 'g3']):
            ax = fig.add_subplot(gs[i:i+3, 12:])
            sns.kdeplot(x=g, hue='network', data=df, palette=cmap, 
                        fill=True, ax=ax, legend=False, alpha=.6)
            
            ax.set(xlabel='', ylim=(0, .2), yticks=(0, .2), ylabel='', 
                   xlim=(-3, 4), xticks=range(-3, 5, 1), 
                   yticklabels=(['0', .2]))
            if g == 'g3':
                ax.set_xlabel('Loading', fontsize=12, fontweight='bold')
            else:
                ax.set_xticklabels([])
            num = g[1]
            ax.set_title(f'PC{num}', loc='right', y=.5, fontsize=12, fontweight='bold')
            sns.despine()

        return fig

    if k == 4:
        # plot single distribution for PC4 only
        fig, ax = plt.subplots(figsize=(3, 1.5))
        sns.kdeplot(x='g4', hue='network', data=df, palette=cmap, 
                    fill=True, ax=ax, legend=False, alpha=.7)
        
        ymax =.20
        ax.set(xlabel='Loading', ylim=(0, ymax), yticks=(0, ymax), ylabel='', 
                yticklabels=(['0', ymax]))
        sns.despine()
        return fig


def plot_eccentricity_calc(view_3d=(30, -120)):
    """Plot vectors of example regions in manifold 

    Parameters
    ----------
    view_3d : tuple, optional
        Viewing orientation, by default (30, -120)
    """
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, 3)

    cmap = plotting.yeo_cmap(networks=19)
    df['c'] = df['network'].apply(lambda x: cmap[x])
    # networks = df['network'].unique()
    # color_dict = {network: cmap(i / len(networks)) for i, network in enumerate(networks)}
    
    # Create a new column 'color' in df using the color dictionary
    # df['c'] = df['network'].map(color_dict)
    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection='3d')
    plotting.plot_3d(df['g1'], df['g2'], df['g3'], color=df['c'], ax=ax1, 
                        view_3d=view_3d, s=20, lw=0.5, alpha=.5)
    min_gval = df.loc[:, ['g1', 'g2', 'g3']].min()
    centroid = np.array([0, 0, 0])
    for i in [87, 66, 455]:
        data = np.vstack([centroid, df.loc[i, ['g1', 'g2', 'g3']].values])
        ax1.plot(data[:, 0], data[:, 1], data[:, 2], c='k', ls='--', linewidth=2)
        
        plotting.plot_3d(data[1, 0], data[1, 1], data[1, 2], color=df.loc[i, 'c'], 
                ax=ax1, alpha=1, edgecolor='k', s=40, view_3d=view_3d, zorder=600)
        plotting.plot_3d(data[0, 0], data[0, 1], data[0, 2], color='k', 
                ax=ax1, alpha=1, view_3d=view_3d, zorder=600)
        ax1.scatter([0],[0], [0], color='k', marker='s', s=100, alpha=1, zorder=600)
    ax1.set(zlim=(-2, 2), ylim=(-1, 3))
    # plt.tight_layout(pad=0)
    fig.savefig(os.path.join(FIG_DIR, 'ecc_calculation'))

def reference_eccentricity(k=3, view_3d=(30, -120)):
    """Plot reference eccentricity brain map and scatterplot

    Parameters
    ----------
    k : int, optional
        Number of gradients to include, by default 3
    view_3d : tuple, optional
        Viewing orientation when plotting 3 dimensions. By default (30, -110)

    Returns
    -------
    pandas.DataFrame
        Eccentricity values of reference
    """
    config = Config()
    fname = os.path.join(config.gradients, 'reference_gradient.tsv')
    df = load_gradients(fname, k)
    grads = df.filter(like='g').values
    
    centroid = np.mean(grads, axis=0)
    # should be 0, 0, 0
    # assert np.allclose(np.around(centroid, decimals=16), np.zeros(k))

    ecc = np.linalg.norm(grads - centroid, axis=1)
    #filtering out subcortex and cerebellum for visualization
    filtered_networks = ['Subcortex', 'Cerebellum']    
    ecc_cortex = ecc[df.query("network not in @filtered_networks").index]
    vmax = np.nanmax(ecc)
    vmin = np.nanmin(ecc)
    cmap = 'viridis'

    prefix = os.path.join(FIG_DIR, 'ref_ecc_')
    if k == 2:
        fig, ax = plt.subplots()
        ax = sns.scatterplot('g1', 'g2', c=ecc, data=df, palette=cmap, 
                             ax=ax)
        ax.set(xlabel='PC1', ylabel='PC2', xlim=(-3, 4), ylim=(-3, 4))
        sns.despine()
        fig.savefig(prefix + 'scatter')
    elif k == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plotting.plot_3d(grads[:, 0], grads[:, 1], grads[:, 2], 
                           c=ecc, ax=ax, cmap=cmap, lw=0.5, s=20, 
                           view_3d=view_3d)
        ax.set(zlim=(-2, 2), ylim=(-1, 3))
        fig.savefig(prefix + 'scatter')
    
    plotting.plot_cbar(cmap, vmin, vmax, orientation='horizontal', 
                       size=(1, .2), fontsize=8)
    plt.savefig(prefix + 'cbar')      

    surfaces = get_surfaces()
    x = plotting.weights_to_vertices(ecc_cortex, Config().atlas)

    p = Plot(surfaces['lh'], surfaces['rh'])
    p.add_layer(x, color_range=(vmin, vmax), cmap=cmap)
    cbar_kws = dict(location='bottom', decimals=2, fontsize=12, 
                            n_ticks=2, shrink=.4, aspect=4, draw_border=False, 
                            pad=-.06)
    fig = p.build(colorbar=False)
    fig.savefig(prefix + 'brain')

    return pd.DataFrame({'roi': df['roi'], 'distance': ecc})

def main():
    
    config = Config()

    df = show_eigenvalues()
    df.to_csv(os.path.join(config.results, 'ref_eigenvalues.tsv'), 
              sep='\t', index=False)

    fig = plot_eigenvalues()
    fig.savefig(os.path.join(FIG_DIR, 'var_explained'))
    
    plot_ref_brain_gradients(config.k)
    plot_eccentricity_calc()
    
    ecc = reference_eccentricity(config.k)
    ecc.to_csv(os.path.join(config.results, 'ref_ecc.tsv'), 
                            sep='\t', index=False)
    
    fig = plot_loading_distributions(k=config.k)
    fig.savefig(os.path.join(FIG_DIR, 'ref_networks'))


if __name__ == '__main__':
    main()
