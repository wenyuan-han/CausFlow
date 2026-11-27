import torch
import numpy as np
import random
from .Modules import Denoise_net, FlowGenerator, FlowTrainer
from .Dataset import Dataset, Generation_Dataset
import scanpy as sc
from torch.utils import data
import joblib
import os
import pandas as pd
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import anndata as ad
import itertools
import math

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sampling(model, data_pwd, profile_size, factor_list, batch_size, device=None, debug=False):
    """
    Robust sampling: ensure we always pass a BxD tensor to the encoder.
    - model: FlowGenerator instance (model.flow_fn is the Denoise_net)
    - data_pwd: path for Generation_Dataset
    - profile_size: number of features
    - factor_list: list of factors
    - batch_size: dataloader batch size
    - device: torch.device or string (fallback to model device)
    - debug: if True prints shapes
    Returns: numpy array shaped (n_factors+1, n_cells, emb_dim)
    """
    if device is None:
        try:
            device = next(model.flow_fn.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    with torch.no_grad():
        dataset = Generation_Dataset(data_pwd, profile_size, factor_list)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        disentanglement_embs = []

        enc = model.flow_fn.DisentanglementEncoder  # encoder module

        for batch_idx, batch in enumerate(dataloader):
            # --- normalize batch type ---
            # If dataset yields tuple (x, labels, ...), pick x
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # --- move to device ---
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if not torch.is_tensor(x):
                x = torch.tensor(x)

            # At this point x should be a tensor of shape (B, D) or (D,)
            if debug:
                print(f"[sampling] batch_idx={batch_idx} raw x.shape before any op: {x.shape}  type={type(x)}")

            # If somehow we accidentally indexed into the batch earlier (e.g., x = batch[batch_idx]), guard here:
            # If x has first dim equal to batch_size but we mistakenly indexed, we would see x.dim()==1
            # Ensure x is 2D: if 1D, add batch dim
            if x.dim() == 1:
                # this means we have a single sample vector (D,) -> make it (1, D)
                x = x.unsqueeze(0)

            # If x is >2d (rare), flatten leading dims into batch
            if x.dim() > 2:
                # e.g., (1, D) ok, but if (B,1,D) collapse middle dim
                x = x.view(x.shape[0], -1)

            x = x.to(device, dtype=torch.float32)

            if debug:
                print(f"[sampling] after fix x.shape -> {x.shape}")

            # Call encoder extractor which expects (B, D)
            batch_exo = enc.extract_exogenous_embs(x)  # -> (B, num_factors, emb_dim)
            # keep on CPU for aggregation
            disentanglement_embs.append(batch_exo.cpu())

        # aggregate per-factor lists
        n_factors = len(factor_list) + 1
        concept_embs = [[] for _ in range(n_factors)]
        for block in disentanglement_embs:          # block shape: (B, n_factors, emb_dim)
            for b in block:                         # iterate over batch elements
                for f in range(n_factors):
                    concept_embs[f].append(b[f])

        concept_embs_stacked = []
        for arr in concept_embs:
            if len(arr) == 0:
                concept_embs_stacked.append(np.zeros((0, 0), dtype=np.float32))
            else:
                # stack list of tensors -> tensor (N_cells, emb_dim) then convert to np
                stacked = torch.stack(arr, dim=0)  # (N_cells, emb_dim)
                concept_embs_stacked.append(np.array(stacked))

    return np.array(concept_embs_stacked)



def extract_concept_embs(data_pwd, model: FlowGenerator, profile_size, factor_list, factor_counts, factor_cdag, batch_size):
    """
    Repeatedly call sampling and average the results (10 repeats by default) to reduce sampling noise.
    """
    # sample 10 times and average (keeps backward compatibility)
    repeats = 10
    sum_embs = None
    for i in range(repeats):
        cur = sampling(model, data_pwd, profile_size, factor_list, batch_size)
        if sum_embs is None:
            sum_embs = cur
        else:
            sum_embs += cur
    sum_embs /= float(repeats)
    return sum_embs

def causality_based_concept_embs(exo_concept_embs, factor_cdag):
    """
    Given exogenous embeddings (n_factors+1, n_cells, emb_dim),
    apply structural equation: z = (I - C)^(-1) * exo
    Returns same shape as input (float32).
    """
    C = np.array(factor_cdag)
    I = np.eye(C.shape[0])
    inv = np.linalg.inv(I - C)
    # exo_concept_embs shape: (F, N, D)
    exo = np.array(exo_concept_embs)
    # transpose to (N, F, D) for matmul convenience
    exo_t = exo.transpose(1, 0, 2)
    z = np.matmul(inv, exo_t)  # (F, N, D)?? careful: inv shape (F,F) matmul (F,N,D) -> (F,N,D) broadcasting over D
    # If broadcasting doesn't work as desired, do explicit loop:
    # z = np.stack([inv.dot(exo_t[:, :, d]) for d in range(exo_t.shape[2])], axis=2)
    # transpose back to (F, N, D)
    z = z.transpose(1, 0, 2).astype(np.float32)
    return z

def multi_target_generation(ori_data_pwd,
                            training_data_pwd,
                            train_concept_embs,
                            model_save_pwd,
                            factor_list,
                            factor_counts,
                            factor_cdag,
                            multi_target_list,
                            retain=True,
                            name=''):
    """
    Adapted multi-target generation using FlowGenerator / FlowTrainer.
    This function:
      - modifies train_concept_embs according to multi_target_list
      - loads a FlowGenerator checkpoint (saved by FlowTrainer)
      - calls trainer.model.sample_with_factor to generate expression data
      - assemble an AnnData and save to disk
    """
    ori_data = sc.read_h5ad(ori_data_pwd)
    train_concept_embs = np.array(train_concept_embs)
    copyed_train_concept_embs = train_concept_embs.copy()

    # apply multi-target changes on the factor embeddings (same logic as original)
    for target_factor_dict in multi_target_list:
        target_factor = target_factor_dict["target_factor"]
        ref_val = target_factor_dict["ref_factor_value"]
        tgt_val = target_factor_dict["tgt_factor_value"]
        idx = factor_list.index(target_factor)
        target_embeddings = train_concept_embs[idx]
        if ref_val != 'all':
            ref_mask = np.array(ori_data.obs[target_factor] == ref_val)
            tgt_embeddings = target_embeddings[np.array(ori_data.obs[target_factor] == tgt_val)]
            if tgt_embeddings.shape[0] == 0:
                raise ValueError(f"No examples for target value {tgt_val} of factor {target_factor}")
            sampled_indices = np.random.choice(tgt_embeddings.shape[0], size=ref_mask.sum(), replace=True)
            generated_tcell_embs = tgt_embeddings[sampled_indices]
            train_concept_embs[idx][ref_mask] = generated_tcell_embs
        else:
            tgt_embeddings = target_embeddings[np.array(ori_data.obs[target_factor] == tgt_val)]
            sampled_indices = np.random.choice(tgt_embeddings.shape[0], size=target_embeddings.shape[0], replace=True)
            train_concept_embs[idx] = tgt_embeddings[sampled_indices]

    # prepare FlowGenerator + FlowTrainer and load checkpoint
    training_output_pwd = model_save_pwd
    training_profile_size = min(ori_data.X.shape[1], 2000)
    profile_size = training_profile_size

    num_factor = len(factor_counts) + 1
    causal_dag_tensor = torch.tensor(factor_cdag).float()
    denoise_fn = Denoise_net(profile_size, profile_size, num_factor, causal_dag_tensor, factor_counts).to(_device)

    flow_model = FlowGenerator(
        flow_fn=denoise_fn,
        encoder=denoise_fn.DisentanglementEncoder,  # 建议显式传入 encoder
        profile_size=profile_size,
        time_steps=100
    ).to(_device)
    # build a FlowTrainer (we only use it to load the saved checkpoint which was saved by FlowTrainer.save)
    trainer = FlowTrainer(flow_model, training_data_pwd, factor_list, profile_size=profile_size, results_folder=training_output_pwd, train_log=False, device=_device)
    # load checkpoint (milestone 10 as original)
    trainer.load(10)
    # trainer.model is the loaded FlowGenerator
    loaded_model = trainer.model
    loaded_model.to(_device)
    loaded_model.eval()

    # selection if retain
    if retain:
        selected_ids = np.array([True] * train_concept_embs.shape[1])
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_val = target_factor_dict["ref_factor_value"]
            tgt_val = target_factor_dict["tgt_factor_value"]
            if ref_val != "all":
                selected_ids = selected_ids & np.array((ori_data.obs[target_factor] == ref_val))
            else:
                selected_ids = selected_ids & np.array((ori_data.obs[target_factor] != tgt_val))
        train_concept_embs = train_concept_embs[:, selected_ids, :]
        ori_train_concept_embs = copyed_train_concept_embs[:, selected_ids, :]
        train_concept_embs = np.concatenate([train_concept_embs, ori_train_concept_embs], axis=1)

    # generate in batches
    with torch.no_grad():
        generated_samples = []
        batch_sz = 1024
        total = train_concept_embs.shape[1]
        steps = math.ceil(total / batch_sz)
        for ii in range(steps):
            batch_concept_embs = train_concept_embs[:, ii * batch_sz:(ii + 1) * batch_sz, :]
            if batch_concept_embs.shape[1] == 0:
                continue
            # convert from shape (F, B, D) -> (B, F, D)
            batch_concept_embs = np.transpose(batch_concept_embs, (1, 0, 2))
            # to torch on device
            tcb = torch.tensor(batch_concept_embs, dtype=torch.float32, device=_device)
            samples_with_cross_attention = trainer.model.sample_with_factor(concept_embs=tcb, batch_size=tcb.shape[0])
            generated_samples.append(samples_with_cross_attention.cpu())

        if len(generated_samples) == 0:
            generated_samples = np.zeros((0, profile_size))
        else:
            generated_samples = torch.cat(generated_samples, dim=0).cpu().numpy()

    new_generated_data = ad.AnnData(generated_samples)

    # assemble obs (retain behavior)
    if retain:
        generated_df = (ori_data[selected_ids].obs).copy()
        ori_df = (ori_data[selected_ids].obs).copy()
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_factor_value = str(target_factor_dict["ref_factor_value"])
            tgt_factor_value = str(target_factor_dict["tgt_factor_value"])
            generated_df[target_factor] = generated_df[target_factor].astype(str)
            ori_df[target_factor] = ori_df[target_factor].astype(str)
            if ref_factor_value != "all":
                generated_df.loc[generated_df[target_factor] == ref_factor_value, target_factor] = str(tgt_factor_value)
            else:
                mask = generated_df[target_factor] != tgt_factor_value
                generated_df.loc[mask, target_factor] = str(tgt_factor_value)
        merged_df = pd.concat([generated_df, ori_df], axis=0).reset_index(drop=True)
        new_generated_data.obs = merged_df
        half = int(len(new_generated_data) / 2)
        new_generated_data.obs['Type'] = ['Generated'] * half + ['Original'] * (len(new_generated_data) - half)
    else:
        new_generated_data.obs = ori_data.obs

    new_generated_data.write(f"{model_save_pwd}/generated_data_{name}.h5ad")
    return new_generated_data


def factor_value_pool(train_concept_embs, factor_list, ori_data_pwd):
    ori_data = sc.read_h5ad(ori_data_pwd)
    unexplained_factor_pool = train_concept_embs[-1]
    factor_dict = {}
    for factor in factor_list:
        concept_embs = train_concept_embs[factor_list.index(factor)]
        factor_values = np.unique(ori_data.obs[factor])
        factor_value_dict = {}
        for factor_value in factor_values:
            factor_value_dict[factor_value] = concept_embs[ori_data.obs[factor] == factor_value]
        factor_dict[factor] = factor_value_dict
    factor_dict['unexplained_variables'] = unexplained_factor_pool
    return factor_dict


def concept_embs_sampling_based_factor_dict(ori_data_pwd, factor_dict, factor_list):
    ori_data = sc.read_h5ad(ori_data_pwd)
    new_concept_embs = []
    for factor in factor_list:
        tmp_concept_embs = []
        for factor_value in ori_data.obs[factor]:
            pool = factor_dict[factor][factor_value]
            if pool.shape[0] == 0:
                raise ValueError(f"No embeddings for factor value {factor_value} of factor {factor}")
            tmp_concept_embs.append(pool[np.random.choice(pool.shape[0])])
        new_concept_embs.append(np.array(tmp_concept_embs))
    new_concept_embs.append(factor_dict['unexplained_variables'])
    new_concept_embs = np.array(new_concept_embs)
    return new_concept_embs


def target_embs_generation(factor_dict, factor_list, multi_target_list, ori_data_pwd, ref_concept_embs):
    ori_data = sc.read_h5ad(ori_data_pwd)
    for target_factor_dict in multi_target_list:
        target_factor = target_factor_dict["target_factor"]
        tgt_factor_value = target_factor_dict["tgt_factor_value"]
        print("========== DEBUG factor_dict ==========")
        print("target_factor:", target_factor)
        print("available factors:", factor_dict.keys())
        print("available values for factor:", factor_dict[target_factor].keys())
        print("requested tgt_factor_value:", tgt_factor_value, type(tgt_factor_value))
        print("=======================================")

        try:
            target_embeddings = factor_dict[target_factor][tgt_factor_value]
        except KeyError:
            alt_key = str(tgt_factor_value) if not isinstance(tgt_factor_value, str) else int(tgt_factor_value)
            target_embeddings = factor_dict[target_factor][alt_key]
        sampled_indices = np.random.choice(target_embeddings.shape[0], size=ref_concept_embs[factor_list.index(target_factor)].shape[0], replace=True)
        generated_tcell_embs = target_embeddings[sampled_indices]
        ref_concept_embs[factor_list.index(target_factor)] = generated_tcell_embs
    return ref_concept_embs


def generation_based_concept_embs(model: FlowGenerator, concept_embs, save_pwd,
                                  factor_list, factor_counts, factor_cdag,
                                  factor_df, name, batch_size):
    """
    Generate expression profiles from concept embeddings using provided FlowGenerator (model).
    concept_embs shape expected (F, N, D).
    """
    train_concept_embs = np.array(concept_embs)
    with torch.no_grad():
        generated_samples = []
        batch_sz = batch_size
        total = train_concept_embs[0].shape[0]
        steps = math.ceil(total / batch_sz)
        for ii in range(steps):
            batch_concept_embs = train_concept_embs[:, ii * batch_sz:(ii + 1) * batch_sz, :]
            if batch_concept_embs.shape[1] == 0:
                continue
            batch_concept_embs = np.transpose(batch_concept_embs, (1, 0, 2))  # (B, F, D)
            tcb = torch.tensor(batch_concept_embs, dtype=torch.float32, device=_device)
            samples_with_cross_attention = model.sample_with_factor(concept_embs=tcb, batch_size=tcb.shape[0])
            generated_samples.append(samples_with_cross_attention.cpu())
        if len(generated_samples) == 0:
            generated_samples = np.zeros((0, model.profile_size))
        else:
            generated_samples = torch.cat(generated_samples, dim=0).cpu().numpy()

    new_generated_data = ad.AnnData(generated_samples)
    new_generated_data.obs = factor_df.reset_index(drop=True)
    os.makedirs(save_pwd, exist_ok=True)
    new_generated_data.write(f"{save_pwd}/{name}.h5ad")
    return new_generated_data


def Simulated_RCT(factor_dict, factor_list, target_factor, number_of_each_factor=500):
    all_randomized_factor_value_embs = []
    target_factor_value_dict = factor_dict[target_factor]
    target_factor_values = np.array(list(target_factor_value_dict.keys()))
    num_of_target_values = len(target_factor_values)
    obs_df = {}
    for factor in factor_list:
        if factor == target_factor:
            factor_value_dict = factor_dict[factor]
            factor_values = np.array(list(factor_value_dict.keys()))
            selected_factor_value_embs = []
            factor_names = []
            for selected_factor_value in factor_values:
                tmp_factor_value_embs = factor_value_dict[selected_factor_value]
                selected_factor_value_embs.append(
                    tmp_factor_value_embs[np.random.choice(len(tmp_factor_value_embs), number_of_each_factor, replace=True)]
                )
                factor_names += [selected_factor_value] * number_of_each_factor
            selected_factor_value_embs = np.concatenate(selected_factor_value_embs, axis=0)
            all_randomized_factor_value_embs.append(selected_factor_value_embs)
            obs_df[factor] = np.array(factor_names).astype(str)
        else:
            factor_value_dict = factor_dict[factor]
            factor_values = np.array(list(factor_value_dict.keys()))
            sampled_indices = np.random.choice(len(factor_values), size=number_of_each_factor, replace=True)
            selected_factor_values = factor_values[sampled_indices]
            selected_factor_value_embs = []
            factor_names = list(selected_factor_values) * num_of_target_values
            obs_df[factor] = np.array(factor_names).astype(str)
            for selected_factor_value in selected_factor_values:
                tmp_factor_value_embs = factor_value_dict[selected_factor_value]
                selected_factor_value_embs.append(
                    tmp_factor_value_embs[np.random.choice(len(tmp_factor_value_embs), 1, replace=True)][0]
                )
            selected_factor_value_embs = np.array(selected_factor_value_embs)
            selected_factor_value_embs = np.concatenate([selected_factor_value_embs] * num_of_target_values, axis=0)
            all_randomized_factor_value_embs.append(selected_factor_value_embs)

    selected_factor_value_embs = factor_dict["unexplained_variables"][
        np.random.choice(len(factor_dict["unexplained_variables"]), number_of_each_factor * num_of_target_values, replace=True)
    ]
    all_randomized_factor_value_embs.append(selected_factor_value_embs)
    all_randomized_factor_value_embs = np.array(all_randomized_factor_value_embs)
    return all_randomized_factor_value_embs, pd.DataFrame(obs_df)
