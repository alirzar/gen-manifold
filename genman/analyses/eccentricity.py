"""Compute subject-level manifold eccentricity across all epochs"""
import os
import numpy as np
import pandas as pd

from genman.config import Config
from genman.utils import (get_files, load_gradients,
                          schaefer400tian_nettekoven_roi_ix)

FIG_DIR = os.path.join(Config().figures, 'eccentricity')
os.makedirs(FIG_DIR, exist_ok=True)


def _read_gradients(fname, k, sub, ses, epoch):
    """Load gradient files"""
    df = load_gradients(fname, k)
    # set constant/identifier columns 
    for col, val in zip(['sub', 'ses', 'epoch'], [sub, ses, epoch]):
        df[col] = val
    df['roi_ix'] = schaefer400tian_nettekoven_roi_ix()
    return df


def load_subject_gradients(k=3, session=None):
    """Load subject gradients into a single DataFrame

    Parameters
    ----------
    k : int, optional
        Number of gradients to load, by default 2
    session : {'ses-01', 'ses-02'}, optional
        Specify only one session, by default None
    washout : bool, optional
        Include washout epoch, by default True

    Returns
    -------
    pandas.DataFrame
        Subjet gradients in long/tidy data format
    """
    config = Config()
    gradient_dir = config.gradients
    task_files = get_files([gradient_dir, '*/*_gradient.tsv'])

    subject_gradients = []
    for g in task_files:
        sub, ses, epoch = os.path.basename(g).split('_')[:3]
        epoch = epoch[5:]
        if epoch in ['rightlearning', 'lefttransfer']:
            epoch = epoch + '-' + os.path.basename(g).split('_')[4] 
        subject_gradients.append(_read_gradients(g, k, sub, ses, epoch))

    subject_gradients = pd.concat(subject_gradients)

    if session is not None:
        return subject_gradients.query("ses == @session")
    else:
        return subject_gradients


def compute_eccentricity(data):
    """Compute Euclidean distance of each region from each manifold centroid

    Parameters
    ----------
    data : pandas.DataFrame
        Subject gradient data
    
    Returns
    -------
    pandas.DataFrame
        Gradient data with distance column
    """
    def _get_dists(x):
        grads = x.filter(like='g')
        centroid = grads.mean().values
        x['distance'] = np.linalg.norm(grads.values - centroid, axis=1)
        return x

    return data.groupby(['sub', 'epoch']).apply(_get_dists, 
                                        include_groups=False) \
                .reset_index().drop(columns={'level_2'})


def main():

    config = Config()
    
    gradients = load_subject_gradients(config.k)
    gradients = compute_eccentricity(gradients)
    gradients.to_csv(os.path.join(config.results, 'subject_gradients.tsv'), 
                     sep='\t', index=False)

        
if __name__ == '__main__':
    main()
