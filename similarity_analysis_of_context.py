from transformer_lens import utils, HookedTransformer, ActivationCache, patching, evals
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
from tqdm import tqdm
import torch
from torch import Tensor
import numpy as np
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
from einops import repeat
import pickle
import argparse
import os.path
import multiprocessing

from DSA.dsa import DSA
from DSA.dmd import DMD
from DSA.simdist import SimilarityTransformDist
from sklearn.manifold import MDS


#torch.set_grad_enabled(False) raises an error during MDS

# List all available GPUs
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")

else:
    print("CUDA is not available. Listing CPUs instead.", multiprocessing.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-model', type=int)
opt = parser.parse_args()


def get_model(model_name: str = 'gpt2-small', 
              checkpoint_index: int = None,
              device=None) -> HookedTransformer:
    """
    Loads source or target model.
    Pythia models have 154 checkpoints.

    model_name: ['gpt2-small', 'pythia-6.9b'] see transformer_lens list
    """
    if 'model_name' != 'bert-base-cased':
        return HookedTransformer.from_pretrained(model_name, 
                                             checkpoint_index=checkpoint_index,
                                             device=device)
    else:
        return HookedTransformer.from_pretrained(model_name, device=device)


def store_activations_in_array(model: HookedTransformer,
                               prompt: str,
                               activation_type: str = 'resid_pre'):

    _, cache = model.run_with_cache(prompt)
    
    x = torch.zeros([1, model.cfg.n_layers, model.cfg.d_model])

    for l in range(0, model.cfg.n_layers):
        activations_last_token = cache[activation_type, l][0, -1, :]
        x[0, l, :] = activations_last_token

    return x


models_names = ['gpt2-small', 'gpt2-medium', 'pythia-14m', 'pythia-31m', 'pythia-70m', 'tiny-stories-1M', 'tiny-stories-3M', 'tiny-stories-8M']
models_names = models_names[opt.model:opt.model+1]
print(models_names)

if os.path.isfile("data/similarity_analysis_of_contest_activations.pkl"):
    with open("data/similarity_analysis_of_contest_activations.pkl","rb") as file:
        activations = pickle.load(file)
else:
    activations = []

if os.path.isfile("data/similarity_analysis_of_contest_model_types.pkl"): 
    with open("data/similarity_analysis_of_contest_model_types.pkl", 'rb') as file:
        model_types = pickle.load(file)
else:
    model_types = []

if os.path.isfile("data/similarity_analysis_of_contest_models.pkl"):
    with open("data/similarity_analysis_of_contest_models.pkl", 'rb') as file:
        models = pickle.load(file)
else:
    models = []


for i, model_name in enumerate(models_names):
    print(f'Model number {i}/{len(models_names)-1}')
    model = get_model(model_name, device=device)
    x = store_activations_in_array(model, prompt="When John and Mary went to the store, John gave a bottle of milk to ")
    activations.append(x)
    model_types.append(f'{model_name}')


nmodels = 1
n_delays = 10
delay_interval = 1
rank = 10

for x in activations:
    x = x[0,:,:] # x.shape [timepoints, dimensions]
    print('x.shape', x.shape)
    dmd = DMD(x,n_delays=n_delays,rank=rank,delay_interval=delay_interval,device=device, verbose=True)
    dmd.fit()
    Ai = dmd.A_v #extract DMD matrix
    models.append(Ai.numpy())

nmodels_tot = len(models)

with open("data/similarity_analysis_of_context_models.pkl", 'wb') as file:
    pickle.dump(models, file)
with open("data/similarity_analysis_of_context_activations.pkl", 'wb') as file:
    pickle.dump(activations, file)
with open("data/similarity_analysis_of_context_model_types.pkl", 'wb') as file:
    pickle.dump(model_types, file)


sims_dmd = np.zeros((nmodels_tot,nmodels_tot))
sims_mtype = np.zeros((nmodels_tot,nmodels_tot))
#notice how we are initializing the similarity transform separately here
comparison_dmd = SimilarityTransformDist(device=device,iters=2000,lr=1e-3)

for i,mi in enumerate(models):
    for j,mj in enumerate(models):
        smtype = int(model_types[i] == model_types[j])
        sims_mtype[i,j] = sims_mtype[j,i] = smtype
        if i == j:
            sims_mtype[i,i] = 2
        if j < i:
            continue
        sdmd = comparison_dmd.fit_score(mi,mj)
        print(i,j,sdmd)

        sims_dmd[i,j] = sims_dmd[j,i] = sdmd


df = pd.DataFrame()
df['Model Type'] = model_types
lowd_embedding = MDS(dissimilarity='precomputed').fit_transform(sims_dmd)
df[f"DMD:0"] = lowd_embedding[:,0] 
df[f"DMD:1"] = lowd_embedding[:,1]

with open("data/similarity_analysis_of_context_mds.pkl", 'wb') as file:
    pickle.dump(df, file)
    

fig,ax = plt.subplots(1,1,sharex=True,sharey=True)
palette = 'plasma'
sns.scatterplot(data=df,x=f"DMD:0",y=f"DMD:1",hue="Model Type",ax=ax,palette=palette)
ax.set_xlabel(f"MDS 1")
ax.set_ylabel(f"MDS 2")

plt.savefig("figures/similarity_analysis_of_context.png") 


#playing around with optimization here, we don't necessarily need the metric to converge to 
#get good clustering!
dsa = DSA(models,n_delays=n_delays,rank=rank,delay_interval=delay_interval,verbose=True,device=device,iters=1000,lr=1e-2)
similarities = dsa.fit_score()

with open("data/similarity_analysis_of_context_dsa.pkl", 'wb') as file:
    pickle.dump(similarities, file)

sns_heatmap = sns.heatmap(similarities)
sns_heatmap.figure.savefig("figures/similarity_analysis_of_context_heatmap.png")

