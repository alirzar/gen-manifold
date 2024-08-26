#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational Similarity Analysis
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from scipy.spatial.distance import squareform
import matplotlib.cm as cm
from genman.config import Config
from genman.analyses import plotting
from sklearn.manifold import MDS
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr

plotting.set_plotting()

FIG_DIR = os.path.join(Config().figures, 'rsa')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(os.path.join(Config().figures, 'rsa/anova'), exist_ok=True)
RES_DIR = os.path.join(Config().results, 'rsa')
os.makedirs(RES_DIR, exist_ok=True)
def _epoch_order():
    """
    Define the order of timepoints/epochs.

    Returns:
        list: Ordered list of timepoints/epochs.
    """
    ordered = ['leftbaseline', 'rightbaseline', 'rightlearning-early', 
               'rightlearning-late', 'lefttransfer-early', 'lefttransfer-late']
    return ordered

def custom_sort(group):
    epoch_order = _epoch_order()
    group['epoch'] = pd.Categorical(group['epoch'], categories=epoch_order, ordered=True)
    return group.sort_values(by=['epoch'])

def time_model(order):
    vals = pd.DataFrame([[0, 0, 0, 1, 0, 0], 
                         [0, 0, 1, 0, 0, 1], 
                         [0, 1, 0, 0, 0, 0], 
                         [1, 0, 0, 0, 1, 0], 
                         [0, 0, 0, 1, 0, 1], 
                         [0, 1, 0, 0, 1, 0]],
                        index=order, columns=order)
    return vals

class ModelMatrix:
    def __init__(self, order):
        self.epoch_order = order
        self.model = pd.DataFrame(np.zeros((6, 6)), index=self.epoch_order, columns=self.epoch_order)
        self.time_model = time_model(order)

    def _validate_key(self, key):
        possible_keys = ['epoch', 'hand', 'baseVSlearing', 'early', 'late', 'null', 'time']
        assert key in possible_keys, f"Invalid key '{key}'. Possible keys are: {', '.join(possible_keys)}"
        
    def _update_model(self, i, c, key):
        if key == 'epoch':
            if i != c and re.search(r'base|early|late', i).group() == re.search(r'base|early|late', c).group():
                self.model.loc[i, c] = 1
        elif key == 'hand':
            if i != c and re.search(r'left|right', i).group() == re.search(r'left|right', c).group():
                self.model.loc[i, c] = 1
        elif key == 'baseVSlearing':
            if i != c and ((re.search(r'base|early|late', i).group() == 'base' and re.search(r'base|early|late', c).group() == 'base') or (re.search(r'base|early|late', i).group() != 'base' and re.search(r'base|early|late', c).group() != 'base')):
                self.model.loc[i, c] = 1
        elif key == 'early':
            if i != c and re.search(r'base|early|late', i).group() == 'early' and re.search(r'base|early|late', c).group() == 'early':
                self.model.loc[i, c] = 1
        elif key == 'late':
            if i != c and re.search(r'base|early|late', i).group() == 'late' and re.search(r'base|early|late', c).group() == 'late':
                self.model.loc[i, c] = 1
        elif key == 'time':
            self.model.loc[i, c] = self.time_model.loc[i, c]
        # elif key == 'null':
        #     if i == c:
        #         self.model.loc[i, c] = 1

    def generate_model_matrix(self, key):
        self._validate_key(key)
        for i in self.model.index:
            for c in self.model.columns:
                self._update_model(i, c, key)
        return self.model

def plot_rdms(rdms, names, fig_name):
    mapping = [2, 3, 4, 0, 1, 12, 13, 6, 9, 14, 7, 10, 8, 11, 5]
    descriptors = _epoch_order()
    rdms = rdms[:, mapping]
    num_rdms = rdms.shape[0]
    if num_rdms > 1:
        fig, axs = plt.subplots(1, num_rdms, figsize=(4 * num_rdms, 8))
    else:
        fig, axs = plt.subplots(figsize=(4, 8))
        axs = [axs] 
    for i, name in enumerate(names):
        # Create the heatmap using seaborn
        sns.heatmap(squareform(rdms[i]), annot=True,
                    cbar=False, ax=axs[i], linewidth=0.5, square=True)
        axs[i].set_xticklabels('')
        axs[i].set_yticklabels('')
        axs[i].set_title(f'{name} Model', fontsize=14, fontweight='bold')
        axs[i].set_xticklabels(descriptors, fontsize=12, fontweight='bold', rotation=90)
    axs[0].set_yticklabels(descriptors, fontsize=12, fontweight='bold', rotation=0)
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, fig_name))

def plot_epoch_umap(gradients):
    """1. Plot mean eccentericity and centered mean eccentericity for all epochs
    2.Compute subject-level correlation values between rightlearning-early and lefttransfer-early
    eccentericity

    Parameters
    ----------
    gradients : pd.DataFrame
        Subject-level, trial-wise data of all epochs

    Returns
    -------
    pd.DataFrame
        Subject-level correlations
    """
    def custom_sort(group):
        epoch_order = _epoch_order()
        group['epoch'] = pd.Categorical(group['epoch'], categories=epoch_order, ordered=True)
        return group.sort_values(by=['epoch'])
    mean_ecc = gradients.groupby(['epoch', 'roi', 'roi_ix'])['distance'] \
                   .mean() \
                   .reset_index() \
                   .sort_values(['epoch', 'roi_ix'])
    
    # roi by epoch (in chronological order) data for plotting
    epoch_ecc = pd.DataFrame({name: g['distance'].values 
                              for name, g in mean_ecc.groupby('epoch')}).T
    reducer = UMAP(random_state=32)
    
    embedding = pd.DataFrame(reducer.fit_transform(epoch_ecc), index=epoch_ecc.index) \
                .reindex(_epoch_order())
    labels = embedding.index.to_list()
    #colors = [cmap(i/len(labels)) for i in task_ids[:len(labels)]]
    colors = ['k', 'r', 'g', 'b', 'cyan', 'm']  # Colors for each task
    
    # Plot before centering
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(labels)):  # Replace 6 with the number of unique tasks
        if 'left' in labels[i]:
            marker = 'o'
        else:
            marker = 's'
        ax.scatter(embedding.iloc[i, 0], embedding.iloc[i, 1], c=colors[i], 
                   label=labels[i].capitalize(), s=100, marker=marker)
        # ax.errorbar(embedding.iloc[i, 0], embedding.iloc[i, 1], xerr=std_embedding[i, 0], 
        #             yerr=std_embedding[i, 1], color=colors[i], alpha=.4, linewidth=3, capsize=1)
    ax.legend()
    # ax.set_xticks(np.arange(14, 16, .5))
    # ax.set_yticks(np.arange(0, 2.5, .5))
    ax.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'mean_ecc_umap'))    

def main():
    config = Config()
    gradients = pd.read_table(
        os.path.join(config.results, 'subject_gradients.tsv')
    )
    plot_epoch_umap(gradients)
    # create a dataset object
    sub_data = pd.DataFrame({name: g['distance'].values
                            for name, g in gradients.groupby(['sub', 'epoch'])}).T
    subs = gradients['sub'].unique()
    data = []
    for sub in subs:
        measurements = sub_data.loc[sub].reindex(_epoch_order())
        des = {'sub': sub}
        obs_des = {'epoch': measurements.index.values}
        chn_des = {'region': measurements.columns}
        data.append(rsd.Dataset(measurements=measurements.values,
                    descriptors=des,
                    obs_descriptors=obs_des,
                    channel_descriptors=chn_des))
        
    diss_measure = 'correlation'
    data_rdms = rsr.calc_rdm(data, descriptor='epoch', method=diss_measure)
    plot_rdms(1 - data_rdms.dissimilarities, subs, 'subject_rsm')
    mean_data_rsm = 1 - data_rdms.mean().dissimilarities
    plot_rdms(mean_data_rsm, [f'Mean RSM {diss_measure}'], f'group_average_{diss_measure}')   
    # data_rdms = rsatoolbox.rdm.sqrt_transform(data_rdms)
    data_rdms.rdm_descriptors
    data_rdms.pattern_descriptors['epoch']
    fig, ax, ret_val = rsatoolbox.vis.show_rdm(data_rdms, rdm_descriptor='sub', 
                                            figsize=(10,10), cmap='Blues', show_colorbar=None)
    fig.savefig(os.path.join(FIG_DIR, 'subject_rdms'))

    print(data_rdms.subsample('sub', ['sub-01', 'sub-02']))
    
    model_names = sorted(['Epoch', 'Hand', 'Baseline vs Learing', 'Time'])#, 'Early', 'Late', 'Performance']
    keys = sorted(['epoch', 'hand', 'baseVSlearing', 'time'])#, 'early', 'late']
    models = []
    for key in keys:
        models.append(squareform(ModelMatrix(order=data_rdms.pattern_descriptors['epoch']) \
                                .generate_model_matrix(key)))
    # models.append(get_performance_model())
    models = np.array(models)
    model_rdms = rsatoolbox.rdm.RDMs(models,
                                rdm_descriptors={'model_name': model_names},
                                dissimilarity_measure=diss_measure)
    plot_rdms(model_rdms.dissimilarities, model_names, 'models_rsatoolbox')

    models_fixed = []
    for i_model in np.unique(model_names):
        rdm_m = model_rdms.subset('model_name', i_model)
        m = rsatoolbox.model.ModelFixed(i_model, rdm_m)
        models_fixed.append(m)

    print('created the following models:')
    for i in range(len(models_fixed)):
        print(models_fixed[i].name)
        print(models_fixed[i].rdm)

    p_thresh = .05
    colors = cm.tab10(np.arange(len(models)))
    eval_methods = ['cosine']#, 'corr', 'spearman', 'corr_cov', 'rho-a', 'tau-a']
    for eval_method in eval_methods:
        results_2a = rsatoolbox.inference.eval_bootstrap_rdm(models_fixed, data_rdms, method=eval_method)
        fig, _, _ = rsatoolbox.vis.plot_model_comparison(results_2a, alpha=p_thresh, 
                                            test_below_noise_ceil='icicles', colors=colors)
        fig.savefig(os.path.join(FIG_DIR, f'rsa_bootstrap-rdm_fixed-models_{eval_method}'))
        print(results_2a)
        results_2a.save(os.path.join(RES_DIR, 
                                     f'rsa_bootstrap-rdm_fixed_model_{eval_method}'), 
                                        overwrite=True)

if __name__ == '__main__':
    main()
