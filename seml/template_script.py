import numpy as np
import pandas as pd
import scanpy as sc
import torch
import seml

import cpa

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
    save_dir: int,
    adata_path: str,
    perturbation_key: str,
    covariate_keys: list,
    dose_key: str,
    split_key: str,
    max_epochs: int,
    max_minutes: int,
    patience: int,
    loss_ae: str,
    checkpoint_freq: int,
    run_eval: bool,
    batch_size: int,
    step_size_lr: int,
    dim: int,
    autoencoder_width: int,
    autoencoder_depth: int,
    autoencoder_lr: float,
    autoencoder_wd: float,
    adversary_width: int,
    adversary_depth: int,
    adversary_lr: float,
    adversary_wd: float,
    adversary_steps: int,
    reg_adversary: float,
    penalty_adversary: float,
    dosers_width: int,
    dosers_depth: int,
    dosers_lr: float,
    dosers_wd: float
):
    #create a CPA API instance
    cpa_api = cpa.api.API(
        adata_path,
        perturbation_key=perturbation_key,
        covariate_keys=[covariate_keys],
        dose_key=dose_key,
        split_key=split_key,
        loss_ae=loss_ae,
        patience=patience, 
        save_dir=save_dir,
        hparams = {
            'dim': dim,
            'batch_size': batch_size,
            'step_size_lr': step_size_lr,
            'autoencoder_width': autoencoder_width,
            'autoencoder_depth': autoencoder_depth,
            'autoencoder_lr': autoencoder_lr,
            'autoencoder_wd': autoencoder_wd,
            'adversary_width': adversary_width,
            'adversary_depth': adversary_depth,
            'adversary_lr': adversary_lr,
            'adversary_wd': adversary_wd,
            'adversary_steps': adversary_steps,
            'reg_adversary': reg_adversary,
            'penalty_adversary': penalty_adversary,
            'dosers_width': dosers_width,
            'dosers_depth': dosers_depth,
            'dosers_lr': dosers_lr,
            'dosers_wd': dosers_wd,
        }
    )

    #train model
    cpa_api.train(
        filename=None,
        run_eval=run_eval,
        checkpoint_freq=checkpoint_freq,
        max_epochs=max_epochs,
        max_minutes=max_minutes,
    )

    #get interesting metrics
    history = cpa_api.history
    results = {
        "history": history,
    }
    return results
