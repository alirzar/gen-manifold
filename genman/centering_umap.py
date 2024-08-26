
'''UMAP representation of connectivity matrices before and after Riemmanian cenreting'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import seaborn as sns
import matplotlib.patches as mpatches

from genman.config import Config
from genman.utils import get_files
from genman.analyses import plotting

plotting.set_plotting()

FIG_DIR = os.path.join(Config().figures, 'centering_umap')
os.makedirs(FIG_DIR, exist_ok=True)

def _read_and_flatten(x):
    res = pd.DataFrame()
    data_frames = []
    
    for i in x:
        if os.path.basename(i).split('_')[2][5:] not in ['rightlearning', 'lefttransfer']:
            name = os.path.basename(i).split('_')[0] + '_' \
            + os.path.basename(i).split('_')[2][5:]
        else: 
            name = os.path.basename(i).split('_')[0] + '_' \
            + os.path.basename(i).split('_')[2][5:] + '-' \
            + os.path.basename(i).split('_')[-1].replace('.tsv', '')
        
        data_frame = pd.read_table(i, index_col=0)
        flattened_values = data_frame.values.flatten()
        data_frames.append(pd.DataFrame(flattened_values, columns=[name]))
    
    res = pd.concat(data_frames, axis=1)
    
    return res

def cmat_umap(dataset_dir):
    cmats = get_files([dataset_dir, '*/*.tsv'])
    flat_cmats = _read_and_flatten(cmats)
    labels = list(flat_cmats.columns.str.split('_', expand=True))
    subject_labels = [label[0] for label in labels]
    task_labels = [label[1] for label in labels]
    
    subject_ids = pd.factorize(subject_labels)[0]
    task_ids = pd.factorize(task_labels)[0]

    reducer = UMAP(random_state=23)
    embedding = reducer.fit_transform(flat_cmats.T)
    return embedding, subject_ids, task_ids, task_labels


def main():
    config = Config()
    embedding_before, subject_ids, task_ids, task_labels = cmat_umap(config.connect)
    embedding_after, subject_ids, task_ids, task_labels = cmat_umap(config.connect_centered)
    
    cmap = plt.get_cmap('tab20')
    
    labels = np.unique(task_labels)  # Labels for each task
    colors = ['k', 'cyan', 'm', 'r', 'g', 'b']  # Colors for each task    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot before centering
    axs[0].scatter(embedding_before[:, 0], embedding_before[:, 1], c=subject_ids, cmap=cmap)
    axs[0].set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    axs[0].set_title('Subjects', fontsize=16)
    # axs[0].set(xticks=np.arange(2, 10, 2), yticks=np.arange(4, 11, 2))    
    for i in range(len(labels)):
        if 'left' in labels[i]:
            marker = 'o'
        else:
            marker = 's'
        axs[1].scatter(embedding_before[task_ids == i, 0], embedding_before[task_ids == i, 1], 
                       marker=marker, color=colors[i], label=labels[i].capitalize())

    axs[1].set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    axs[1].set_title('Tasks', fontsize=16)
    # axs[1].set(xticks=np.arange(2, 10, 2), yticks=np.arange(4, 11, 2))
    sns.despine()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'umap-before_centering'))
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    # Plot after centering
    axs[0].scatter(embedding_after[:, 0], embedding_after[:, 1], c=subject_ids, cmap=cmap)
    axs[0].set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    axs[0].set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
    axs[0].set_title('Subjects', fontsize=16)
    # axs[0].set(xticks=np.arange(10, 15, 2), yticks=np.arange(2, 9, 2))
    for i in range(len(labels)):
        if 'left' in labels[i]:
            marker = 'o'
        else:
            marker = 's'
        axs[1].scatter(embedding_after[task_ids == i, 0], embedding_after[task_ids == i, 1], 
                       marker=marker, color=colors[i], label=labels[i].capitalize)
    # axs[1].set(xticks=np.arange(10, 15, 2), yticks=np.arange(2, 9, 2))
    axs[1].set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
    axs[1].set_title('Tasks', fontsize=16)    
    sns.despine()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'umap-after_centering'))
    
    epoch_ids = [0, 3, 4, 5, 1, 2]
    patches = [mpatches.Patch(color=colors[i], label=labels[i].capitalize()) for i in epoch_ids]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.legend(handles=patches, fontsize=14, ncol=1, 
              edgecolor='gray', handlelength=1, handleheight=1)
    ax.axis('off')
    plt.show()
    fig.savefig(os.path.join(FIG_DIR, 'umap_legend'))
if __name__ == '__main__':
    main()