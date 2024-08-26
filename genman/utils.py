"""General utilities"""

import os
import glob
import subprocess
from datetime import datetime
import natsort
import numpy as np
import pandas as pd
import pingouin as pg
import nibabel as nib
from brainspace.mesh.mesh_io import read_surface
import bct
from neuromaps import images
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from genman.config import Config


pjoin = os.path.join

def get_files(pattern, force_list=False):
    """Extracts files in alphanumerical order that match the provided glob 
    pattern.

    Parameters
    ----------
    pattern : str or list
        Glob pattern or a list of strings that will be joined together to form 
        a single glob pattern.  
    force_list : bool, optional
        Force output to be a list. If False (default), a string is returned in
        cases where only one file is detected.

    Returns
    -------
    str or list
        Detected file(s).

    Raises
    ------
    FileNotFoundError
        No files were detected using the input pattern.
    """
    if isinstance(pattern, list):
        pattern = pjoin(*pattern)
    
    files = natsort.natsorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError('Pattern could not detect file(s)')

    if (not force_list) & (len(files) == 1):
        return files[0] 
    else:
        return files


def check_img(img):
    """Load image if not already loaded"""
    return nib.load(img) if isinstance(img, str) else img


def display(msg):
    """Print timestamped message to terminal

    Parameters
    ----------
    msg : str
        Message to print
    """
    t = datetime.now().strftime("%H:%M:%S")
    print(f'[{t}] {msg}')


def parse_roi_names(x, col='roi'):
    if 'roi_ix' in x.columns:
        roi_cols = ['hemi', 'network', 'name']
        x[roi_cols] = x[col].str.split('_', n=3, expand=True).iloc[:, 1:]
        x.loc[(401 <= x['roi_ix']) & (x['roi_ix'] <= 416), 'hemi'] = 'LH'
        x.loc[(417 <= x['roi_ix']) & (x['roi_ix'] <= 432), 'hemi'] = 'RH'
        x.loc[(433 <= x['roi_ix']) & (x['roi_ix'] <= 448), 'hemi'] = 'LH'
        x.loc[(449 <= x['roi_ix']), 'hemi'] = 'RH'
        x.loc[(401 <= x['roi_ix']) & (x['roi_ix'] <= 432), 'network'] = 'Subcortex'
        x.loc[(433 <= x['roi_ix']), 'network'] = 'Cerebellum'
        x.loc[(401 <= x['roi_ix']) & (x['roi_ix'] <= 432), 'name'] = x.loc[(401 <= x['roi_ix']) & (x['roi_ix'] <= 432), 'roi'].str.replace('-rh', '').str.replace('-lh', '')
        x.loc[(433 <= x['roi_ix']), 'name'] = x.loc[(433 <= x['roi_ix']), 'roi']
    elif len(x) == 464:
        roi_cols = ['hemi', 'network', 'name']
        x[roi_cols] = x[col].str.split('_', n=3, expand=True).iloc[:, 1:]
        x.loc[400:416, 'hemi'] = 'LH'
        x.loc[416:432, 'hemi'] = 'RH'
        x.loc[432:448, 'hemi'] = 'LH'
        x.loc[448:, 'hemi'] = 'RH'
        x.loc[400:432, 'network'] = 'Subcortex'
        x.loc[432:, 'network'] = 'Cerebellum'
        x.loc[400:432, 'name'] = x.loc[400:432, 'roi'].str.replace('-rh', '').str.replace('-lh', '')
        x.loc[432:, 'name'] = x.loc[432:, 'roi']
    else:
        raise ValueError('No Info To Parse the ROI Names')
    return x
def load_gradients(fname, k=None):
    """Read gradient file and parse region information

    Parameters
    ----------
    fname : str
        Gradient file name
    k : int, optional
        Number of gradients to keep. If None, all gradients are loaded. By 
        default None

    Returns
    -------
    pd.DataFrame
        Gradient dataframe
    """
    df = pd.read_table(fname, index_col=0)
    if k is not None:
        df = df.iloc[:, :k]
        
    df = df.reset_index().rename(columns={'index': 'roi'})
    df = parse_roi_names(df)
    return df


def load_table(fname):
    return pd.read_table(fname, index_col=0)


def get_surfaces(style='inflated', load=True):
    """Fetch surface files of a given surface style

    Parameters
    ----------
    style : str, optional
        Type of surface to return, by default 'inflated'

    Returns
    -------
    dict
        Dictionary of left (lh) and right (rh) surface files
    """
    config = Config()
    surf_path = os.path.join(config.resources, 'surfaces')
    surfaces = get_files([surf_path, f'*.{style}_*'])
    
    if load:
        surfs = [read_surface(i) for i in surfaces]
        return dict(zip(['lh', 'rh'], surfs))
    else:
        return surfaces


def schaefer1000_roi_ix():
    x = np.arange(1000) + 1
    # drop indices of missing regions in Schaefer 1000. These values/regions 
    # appear in the dlabel.nii labels, but not in the actual vertex array, as
    # they have been 'upsampled-out' of the atlas
    return x[~np.isin(x, [533, 903])]

def schaefer400_roi_ix():
    return np.arange(400) + 1

def schaefer400tian_roi_ix():
    return np.arange(432) + 1

def schaefer400tian_nettekoven_roi_ix():
    return np.arange(464) + 1

def fdr_correct(x, colname='p-unc'):
    """Apply FDR correction across all rows of dataframe"""
    corrected = pg.multicomp(x[colname].values, method='fdr_bh')
    x[['sig_corrected', 'p_fdr']] = np.array(corrected).T
    return x


def parcellation_adjacency(dlabel, lh_surf, rh_surf, min_vertices=1):

    tmp = 'adjacency.pconn.nii'
    cmd = f"wb_command -cifti-label-adjacency {dlabel} {tmp} " \
          f"-left-surface {lh_surf} -right-surface {rh_surf}"
    subprocess.run(cmd.split())
    
    adj = (nib.load(tmp).get_fdata() >= min_vertices).astype(float)
    os.remove(tmp)

    # check symmetric
    assert np.array_equal(adj, adj.T)
    # remove nodes with 0 adjacent nodes; not in actual parcellation
    remove_ix = np.where(np.sum(np.abs(adj), axis=1) == 0)[0]
    if len(remove_ix) > 0:
        adj = np.delete(adj, remove_ix, axis=0)
        adj = np.delete(adj, remove_ix, axis=1)
    return adj


def get_clusters(data, adjacency, sort=True, yuh=False):
    data_cortex = data.query('roi_ix <= 400')
    ix = data_cortex.query("sig_corrected == 1")['roi_ix'].values
    
    adjacency.columns = adjacency.columns.astype(int)
    x = adjacency.loc[ix, ix].values
    assignments, sizes = bct.get_components(x)

    cluster_table = pd.DataFrame({
        'cluster': np.arange(len(sizes)) + 1, 
        'size': sizes
    })

    res = data_cortex.copy()
    res.index = res['roi_ix'].values
    res['cluster'] = 0
    res.loc[ix, 'cluster'] = assignments
    res = res.merge(cluster_table, on='cluster', how='left')
    res['size'] = np.nan_to_num(res['size'])

    # relabel clusters based on size (descending order)
    if sort:
        labels = res.sort_values('size', ascending=False)['cluster'].unique()
        # 0 (no cluster) stays 0 in remapping
        new_labels = np.concatenate([np.arange(len(labels[:-1])) + 1, [0]])
        relabel_map = dict(zip(labels, new_labels))
        res['cluster'] = res['cluster'].apply(lambda x: relabel_map[x])
    data['cluster'] = 0
    data['size'] = 0
    res = pd.concat([res, data.query('roi_ix > 400')])
    return res

def get_hand_learn_phase(x):
    #expand each hand learning pattern to baseline, early and late
    res = x.copy()
    res['hand'] = x['epoch'].str.extract('(left|right)')
    res['time'] = x['epoch'].str.extract('(baseline|early|late)')
    return res

def test_regions(data, method='anova', factor='epoch', p_thresh=.05):

    test = dict(anova=pg.rm_anova, ttest=pg.pairwise_tests)
    if method not in test.keys():
        raise ValueError(f"method must be one of {list(test.keys())}")
    
    test_data = data[['sub', 'roi', 'roi_ix', 'epoch', 'distance']]

    kwargs = dict(correction=True) if method in ['anova'] else {}
    
    if factor == 'epoch':    
        res = test_data.groupby(['roi', 'roi_ix'], sort=False) \
                       .apply(test[method], 
                              dv='distance', 
                              within=factor, 
                              subject='sub', **kwargs,
                              include_groups=False) \
                       .reset_index() \
                       .drop('level_2', axis=1)
    
    else:
        test_data = get_hand_learn_phase(test_data)
        if factor == 'time * hand':
            res = {}
            for time in ['baseline', 'early', 'late']:
                 time_test_data = test_data.query('time == @time')
                 res[time] = time_test_data.groupby(['roi', 'roi_ix'], sort=False) \
                               .apply(test[method], 
                                      dv='distance', 
                                      within='hand', 
                                      subject='sub', **kwargs,
                                      include_groups=False) \
                               .reset_index() \
                                .drop('level_2', axis=1)
                 res[time]['Contrast'] = 'time * hand'
                 res[time].insert(3, 'time', time, allow_duplicates=False)
            res = pd.concat([res[i] for i in res], ignore_index=True).sort_values(by='roi_ix').reset_index()
            
        else:
            res = test_data.groupby(['roi', 'roi_ix'], sort=False) \
                           .apply(test[method], 
                                  dv='distance', 
                                  within=factor, 
                                  subject='sub', **kwargs,
                                  include_groups=False) \
                           .reset_index() \
                           .drop('level_2', axis=1)        
                       
    res['sig'] = (res['p-unc'] < p_thresh).astype(float)
    res = fdr_correct(res)
    # adj = pd.read_table(Config().adjacency, index_col=0)
    # if method == 'anova':
    #     res = res.groupby(['Source'], sort=False) \
    #                 .apply(get_clusters, adj) \
    #                     .reset_index(drop=True)
    return res
    
def _cornblath(data, atlas='fsaverage', density='10k', parcellation=None,
              n_perm=1000, seed=None, spins=None, surfaces=None):
    """Temporary bug-free cornblath function to replace 
    neuromaps.nulls.cornblath
    """
    from neuromaps.datasets import fetch_atlas
    from neuromaps import nulls

    if parcellation is None:
        raise ValueError('Cannot use `cornblath()` null method without '
                         'specifying a parcellation. Use `alexander_bloch() '
                         'instead if working with unparcellated data.')
    y = np.asarray(data)
    if surfaces is None:
        surfaces = fetch_atlas(atlas, density)['sphere']
    nulls = nulls.cornblath(y, surfaces, parcellation,
                      n_perm=n_perm, spins=spins, seed=seed)
    return nulls


def permute_map(data, atlas='fslr', density='32k', parcellation=None, 
                n_perm=1000, surfaces=None, spins=None, seed=1234):
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
    from neuromaps.datasets import fetch_atlas
    from neuromaps import nulls

    if parcellation is None:
        raise ValueError('Cannot use `cornblath()` null method without '
                         'specifying a parcellation. Use `alexander_bloch() '
                         'instead if working with unparcellated data.')
    y = np.asarray(data)
    if surfaces is None:
        surfaces = fetch_atlas(atlas, density)['sphere']
    nulls = nulls.cornblath(y, atlas, density, parcellation,
                      n_perm=n_perm, spins=spins, seed=seed, surfaces=surfaces)
    return nulls

def optimal_k(x, out_dir, prefix):
    '''find the optimal k for k-means clustering and plot the results of elbow
    and silhouette analysis
    '''
    range_n_clusters = list(range(2,12)) 
    elbow = []
    silhouette = []
    
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=1234)
        preds = clusterer.fit_predict(x)
        centers = clusterer.cluster_centers_
    
        elbow.append(clusterer.inertia_) 
        silhouette.append(silhouette_score(x, preds)) 

    k_opt = []
    for i in range(1, len(silhouette) - 1):
        if silhouette[i - 1] < silhouette[i] > silhouette[i + 1]:
            k_opt.append(i + 2)
            
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('WCSS (Inertia)', color=color)
    ax1.plot(range_n_clusters, elbow, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(range_n_clusters, silhouette, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title('Elbow and Silhouette Analysis')
    plt.grid(True)
    plt.show()

    fig.savefig(os.path.join(out_dir, f'{prefix}elbow_silhouette_analysis.png'), dpi=300)
    return k_opt