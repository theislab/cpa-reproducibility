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
    run: int,
):
    run = run + 1
    adata = sc.read('/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/datasets/Norman2019_prep_new.h5ad')
    save_dir = '/storage/groups/ml01/workspace/carlo.dedonno/cpa-reproducibility/notebooks/'
    cpa_api = cpa.api.API(
        adata,
        perturbation_key='condition',
        covariate_keys=['cell_type'],
        dose_key='dose_val',
        split_key='split',
        loss_ae='gauss',
        doser_type='linear',
        decoder_activation='ReLU',
        patience=500,
        only_parameters=False,
        save_dir=save_dir,
        hparams={
            'adversary_depth': 1,
            'adversary_width': 128,
            'adversary_lr': 1e-3,
            'adversary_wd': 1e-8,
            'adversary_steps': 3,
            'autoencoder_depth': 2,
            'autoencoder_width': 512,
            'autoencoder_lr': 6e-4,
            'autoencoder_wd': 1e-8,
            'batch_size': 256,
            'dim': 64,
            'dosers_depth': 3,
            'dosers_width': 128,
            'dosers_lr': 6e-4,
            'dosers_wd': 1e-8,
            'penalty_adversary': 33,
            'reg_adversary': 45,
            'step_size_lr': 45.
        },  
    )
    #train model
    cpa_api.train(
        filename=f"fig4_model.pt",
        run_eval=True,
        max_epochs=2000,
        max_minutes=2*120,
    )
    return None