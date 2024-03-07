import torch
from torch import Tensor
import numpy as np
import pandas as pd
import sys
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
from sklearn.manifold import MDS

sys.path.append('../')
from models import get_model_from_name
from data import get_data_sample
from activations import get_residual_stream_activations_for_layer_wise_dmd



def apply_dmd(models: List,
              model_types: List,
              nmodels: int = 1,
              n_delays: int = 10,
              delay_interval: int = 1 ,
              rank: int = 10,
              device: int = None,
              fig_file_name: str = 'test'
):
    
    models_Ai = []

    for x in models:
        if len(x.shape) == 3:
            x = x[0,:,:] # -> x.shape [timepoints, dimensions]
        dmd = DMD(x, 
                  n_delays=n_delays, 
                  rank=rank, 
                  delay_interval=delay_interval,
                  device=device, 
                  verbose=True)
        dmd.fit()
        Ai = dmd.A_v #extract DMD matrix
        models_Ai.append(Ai.numpy())

    nmodels_tot = len(models_Ai)
    sims_dmd = np.zeros((nmodels_tot,nmodels_tot))
    sims_mtype = np.zeros((nmodels_tot,nmodels_tot))
    #notice how we are initializing the similarity transform separately here
    comparison_dmd = SimilarityTransformDist(device=device,iters=2000,lr=1e-3)

    for i,mi in enumerate(models_Ai):
        for j,mj in enumerate(models_Ai):
            smtype = int(model_types[i] == model_types[j])
            sims_mtype[i,j] = sims_mtype[j,i] = smtype
            if i == j:
                sims_mtype[i,i] = 2
            if j < i:
                continue
            sdmd = comparison_dmd.fit_score(mi,mj)
            print(i,j,sdmd)

            sims_dmd[i,j] = sims_dmd[j,i] = sdmd

    # DMD Heatmap
    plot_dmd_heatmap(sims_dmd, 
                     models_names=model_types,
                     fig_file_name=fig_file_name)

    # DMD MDS
    mds_df = plot_dmd_mds(sims_dmd,
                 model_types,
                 fig_file_name=fig_file_name)
    #mds_df.to_pickle(f"{fig_file_name}.pkl")
    
    return sims_dmd, model_types, mds_df



def plot_dmd_heatmap(sims_dmd,
                     models_names,
                     palette: str = 'Greens',
                     fig_file_name: str = 'test'):

    sns_fig = sns.heatmap(sims_dmd, yticklabels=models_names, xticklabels=models_names, cmap=palette)
    plt.yticks(rotation=0) 
    sns_fig.figure.savefig(f'../figures/{fig_file_name}_dmd_heatmap.png')
    print(f'DMD heatmap saved at: ../figures/{fig_file_name}_dmd_heatmap.png')
    return 


def plot_dmd_mds(sims_dmd,
                 model_types,
                 fig_file_name: str = 'test'):

    df = pd.DataFrame()
    df['Model Type'] = model_types
    lowd_embedding = MDS(dissimilarity='precomputed').fit_transform(sims_dmd)
    df[f"DMD:0"] = lowd_embedding[:,0] 
    df[f"DMD:1"] = lowd_embedding[:,1]

    _, ax = plt.subplots(1,1,sharex=True,sharey=True)
    g = sns.scatterplot(data=df,
                        x=f"DMD:0",
                        y=f"DMD:1",
                        hue="Model Type",
                        ax=ax,palette='plasma')
    
    ax.set_xlabel(f"MDS 1")
    ax.set_ylabel(f"MDS 2")
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    plt.savefig(f"../figures/{fig_file_name}_mds.png") 
    print(f'Saved DMD MDS at: ../figures/{fig_file_name}_mds.png')
    return df