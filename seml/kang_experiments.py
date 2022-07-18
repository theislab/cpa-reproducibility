import cpa
from cpa.data import load_dataset_splits
import numpy as np
import scanpy as sc
import torch
import seml
from sacred import Experiment

np.random.seed(420)

ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )

@ex.automain
def run(
    ct: int,
    loss_ae: str,
    max_epochs: int,
    max_minutes: int,
    save_dir: str,

):
    adata = sc.read('/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/datasets/kang.h5ad')
    adata.obs['split'] = adata.obs['split'].astype(str)
    adata.obs['split'][~(adata.obs['cell_type'] == ct) & (adata.obs['condition'] == 'stimulated')] = 'ood'

    #create a CPA API instance
    cpa_api = cpa.api.API(
        adata,
        perturbation_key='condition',
        covariate_keys=['cell_type'],
        dose_key='dose_val',
        split_key='split',
        control='control',
        patience=500,
        loss_ae=loss_ae,
        save_dir=save_dir,
        device='cuda',
    )
    #train model
    cpa_api.train(
        filename=f"{loss_ae}_{ct}.pt",
        run_eval=False,
        max_epochs=max_epochs,
        max_minutes=max_minutes,
    )

    #get interesting metrics
    datasets, _ = load_dataset_splits(
        data=adata,
        control='control',
        perturbation_key='condition',
        dose_key='dose_val',
        covariate_keys=['cell_type'],
        split_key = 'split',
        return_dataset=True
    )  
    
    genes_control = datasets['training_control'].genes
    preds = cpa_api.predict(
        genes_control, 
        cov={'cell_type':[ct]}, 
        pert=['stimulated'],
        dose=['1'],
        sample=False
    )
    preds.write(f'/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/seml/kang_adatas/{ct}_{loss_ae}_pred.h5ad')
    if loss_ae != 'mse':
        preds_sampled = cpa_api.predict(
            genes_control, 
            cov={'cell_type':[ct]}, 
            pert=['stimulated'],
            dose=['1'],
            sample=True
        )
        preds_sampled.write(f'/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/seml/kang_adatas/{ct}_{loss_ae}_pred_sampled.h5ad')
    return None
