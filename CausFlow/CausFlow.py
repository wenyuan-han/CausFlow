import torch
import numpy as np
import pandas as pd
import anndata as ad
import random
import scipy.sparse as sp
from sympy import false
from .Modules import Denoise_net, FlowGenerator, FlowTrainer
from .Dataset import Dataset
import scanpy as sc
from torch.utils import data
import joblib
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from copy import deepcopy
from .Metrics_calculation import cal_acc_precision_recall_f1_factors
from pathlib import Path
from .Generation_utils import extract_concept_embs, factor_value_pool, causality_based_concept_embs, \
    target_embs_generation, generation_based_concept_embs


class CausFlow:

    def __init__(self,
                 device='cuda',
                 ema_decay=0.995,
                 gradient_accumulate_every=2,
                 fp16=False,
                 step_start_ema=2000,
                 update_ema_every=1000,
                 save_and_sample_every=10000):

        self.device = device
        self.ema_decay = ema_decay
        self.gradient_accumulate_every = gradient_accumulate_every
        self.fp16 = fp16
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.save_and_sample_every = save_and_sample_every
        self.model = None  # will hold FlowGenerator instance

    def data_transformation(self,
                            data_pwd,
                            save_pwd,
                            concept_list,
                            log_norm=False,
                            reuse_mapping=False):
        data = sc.read_h5ad(data_pwd)

        if sp.issparse(data.X):
            data.X = data.X.toarray()

        if log_norm:
            normed_data = data.X / data.X.sum(axis=1)[:,None] * 10000
            exp_data = np.log(normed_data + 1)
        else:
            exp_data = data.X
        new_obs = []
        label_categories = []

        for idx, factor_name in enumerate(concept_list):
            # if test set
            if reuse_mapping:
                val2idx = joblib.load(f"{save_pwd}/{factor_name}_dict.pkl")
            # if training set
            else:
                factor_vals = list(data.obs[factor_name].unique())
                val2idx = dict(zip(factor_vals, range(len(factor_vals))))
                joblib.dump(val2idx, f"{save_pwd}/{factor_name}_dict.pkl")
            new_obs.append([val2idx.get(i, -1) for i in data.obs[factor_name]])
            label_categories.append(len(val2idx))
        new_df = pd.DataFrame(list(zip(*new_obs)), columns = concept_list)
        new_data = ad.AnnData(exp_data)
        new_data.obs = new_df
        data_name = data_pwd.split("/")[-1]
        new_data.write(f"{save_pwd}/transformed_{data_name}")
        return new_data,label_categories

    def train(self,
              training_data_pwd,
              model_save_pwd,
              concept_list,
              concept_counts,
              concept_cdag,
              *,
              loss_type="l2",
              training_num_steps=100000,
              training_batch_size=64,
              training_lr=5e-6,
              max_profile_size=1000,
              timesteps=100,  # recommend ~100, your Modules.FlowGenerator default is 100
              seed=888,
              train_log=True):
        # random seed setting
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        # read training dataset
        train_data = sc.read_h5ad(training_data_pwd)
        training_output_pwd = model_save_pwd
        if train_data.X.shape[1] > max_profile_size:
            training_profile_size = max_profile_size
        else:
            training_profile_size = train_data.X.shape[1]

        # -----------------------------
        # build Denoise_net (velocity network) and FlowGenerator
        # -----------------------------
        # note: Denoise_net signature: (dim, out_dim, num_factor, causal_dag, label_categories, ...)
        num_factor = len(concept_counts) + 1
        causal_dag_tensor = torch.tensor(concept_cdag).float()
        model_fn = Denoise_net(training_profile_size,
                               training_profile_size,
                               num_factor,
                               causal_dag_tensor,
                               concept_counts).to(self.device)

        # FlowGenerator signature: FlowGenerator(flow_fn, profile_size, time_steps=100, denoiser_like=None)
        diffusion_model = FlowGenerator(flow_fn=model_fn, encoder=model_fn.DisentanglementEncoder,profile_size=training_profile_size, time_steps=timesteps
                                        ).to(self.device)

        # -----------------------------
        # create FlowTrainer (note API differs from old Trainer)
        # FlowTrainer(diffusion_model, folder, factor_list, *, ema_decay=..., profile_size=200, ...)
        # -----------------------------
        trainer = FlowTrainer(diffusion_model,
                              training_data_pwd,
                              concept_list,
                              ema_decay=self.ema_decay,
                              profile_size=training_profile_size,
                              train_batch_size=training_batch_size,
                              train_lr=training_lr,
                              train_num_steps=training_num_steps,
                              gradient_accumulate_every=self.gradient_accumulate_every,
                              fp16=self.fp16,
                              step_start_ema=self.step_start_ema,
                              update_ema_every=self.update_ema_every,
                              save_and_sample_every=self.save_and_sample_every,
                              results_folder=training_output_pwd,
                              train_log=train_log,
                              device=self.device)

        trainer.train()
        # keep a copy of the trained generator (FlowGenerator)
        self.model = deepcopy(diffusion_model)

    def load_trained(self,
                     concept_list,
                     concept_counts,
                     concept_cdag,
                     results_folder,
                     *,
                     trained_profile_size=1000,
                     milestone=10,
                     timesteps=100):
        """
        Load a saved FlowGenerator model (the checkpoint was saved by FlowTrainer.save
        with keys {'step','model','ema'}). We create architecture then load the state dict.
        """
        num_factor = len(concept_counts) + 1
        causal_dag_tensor = torch.tensor(concept_cdag).float()

        # instantiate denoiser (flow_fn) and FlowGenerator with same hyperparams used in training
        denoise_fn = Denoise_net(trained_profile_size, trained_profile_size, num_factor, causal_dag_tensor,
                                 concept_counts).to(self.device)
        flow_model = FlowGenerator(flow_fn=denoise_fn, encoder=denoise_fn.DisentanglementEncoder,profile_size=trained_profile_size, time_steps=timesteps,
                                   ).to(self.device)

        ckpt = torch.load(str(Path(results_folder) / f'model-{milestone}.pt'), map_location=self.device ,weights_only=True)
        flow_model.load_state_dict(ckpt['model'],strict = True)
        self.model = deepcopy(flow_model)

    def sampling_cells(self,
                       testing_data_pwd,
                       concept_list,
                       concept_counts,
                       concept_cdag,
                       *,
                       profile_size=1000,
                       sample_batch_size=128):
        assert self.model is not None, "Model not loaded. Call train() or load_trained() first."
        print(">>> sampling using:", type(self.model))
        device = next(self.model.flow_fn.parameters()).device

        with torch.no_grad():
            dataset = Dataset(testing_data_pwd, profile_size, concept_list)
            dataloader = data.DataLoader(dataset, batch_size=sample_batch_size, shuffle=False, pin_memory=True)
            generated_samples_list = []

            for idx, data_ in enumerate(dataloader):
                x_batch = data_[0].to(device)
                labels_batch = data_[1].to(device)
                res = self.model.flow_fn.DisentanglementEncoder.eval()(x_batch, labels_batch)
                batch_concept_embs = res[0]
                samples_raw = self.model.sample_with_factor(
                    concept_embs=batch_concept_embs.to(device),
                    batch_size=len(x_batch),
                    labels=labels_batch,
                    steps=self.model.time_steps
                )
                batch_np = samples_raw.cpu().numpy()
                batch_np = np.clip(batch_np, a_min=0, a_max=None)
                batch_np[batch_np < 0.5] = 0
                generated_samples_list.append(batch_np)

            if len(generated_samples_list) >= 2:
                generated_samples = np.concatenate(generated_samples_list, axis=0)
            else:
                generated_samples = np.array(generated_samples_list[0], dtype=float)

        return generated_samples

    def sampling_concepts(self,
                          testing_data_pwd,
                          concept_list,
                          concept_counts,
                          concept_cdag,
                          *,
                          profile_size=1000):
        with torch.no_grad():
            dataset = Dataset(testing_data_pwd, profile_size, concept_list)
            dataloader = data.DataLoader(dataset, batch_size=1280, shuffle=False, pin_memory=True)
            disentanglement_embs = []
            for idx, data_ in enumerate(dataloader):
                batch_concept_embs = \
                self.model.flow_fn.DisentanglementEncoder.eval()(data_[0].cuda(), data_[1].cuda())[0].cpu()
                disentanglement_embs.append(batch_concept_embs)

            concept_embs = [[] for _ in range(len(concept_list) + 1)]
            for factor in range(len(concept_list) + 1):
                for idx, i in enumerate(disentanglement_embs):
                    for b in i:
                        concept_embs[factor].append(b[factor])

            concept_embs_stacked = []
            for i in concept_embs:
                concept_embs_stacked.append(np.array(torch.stack(i)))
        return np.array(concept_embs_stacked)

    def disentanglement(self,
                        testing_data_pwd,
                        saved_pwd,
                        concept_list,
                        concept_counts,
                        concept_cdag,
                        *,
                        sampling_counts=10,
                        profile_size=1000):
        concept_embs = self.sampling_concepts(testing_data_pwd, concept_list, concept_counts, concept_cdag,
                                              profile_size=profile_size)
        for i in range(sampling_counts - 1):
            concept_embs += self.sampling_concepts(testing_data_pwd, concept_list, concept_counts, concept_cdag,
                                                   profile_size=profile_size)
        concept_embs /= sampling_counts
        if not os.path.exists(saved_pwd):
            os.mkdir(saved_pwd)
        joblib.dump(concept_embs, saved_pwd + f"/factors_embs.pkl")
        return concept_embs

    def concept_prediction(self, testing_data_pwd, concept_embs, concept_list, concept_counts, concept_cdag):
        # multi-label classification: use label_predictor in DisentanglementEncoder
        factor_scores = []
        device = next(self.model.flow_fn.parameters()).device
        with torch.no_grad():
            for idx, predictor in enumerate(self.model.flow_fn.DisentanglementEncoder.label_predictor):
                scores = predictor.eval()(torch.tensor(concept_embs[idx], dtype=torch.float32).to(device))
                factor_scores.append(np.array(scores.cpu()))
        pred_labels = [i.argmax(axis=1) for i in factor_scores]
        test_data = sc.read_h5ad(testing_data_pwd)
        ground_labels = [np.array(test_data.obs[i]) for i in concept_list]

        accs, precisions, recalls, f1s = cal_acc_precision_recall_f1_factors(pred_labels, ground_labels)
        for idx in range(len(accs)):
            print(
                f"Factor_{concept_list[idx]}: ACC:{accs[idx]}\tPrecision:{precisions[idx]}\tRecall:{recalls[idx]}\tF1:{f1s[idx]}")
        results = []
        for idx in range(len(accs)):
            results.append({
                "Factor": concept_list[idx],
                "ACC": accs[idx],
                "Precision": precisions[idx],
                "Recall": recalls[idx],
                "F1": f1s[idx]
            })
        df = pd.DataFrame(results)
        save_path = "factor_classification_metrics.csv"
        df.to_csv(save_path, index=False)
        print(f"Metrics saved to {save_path}")

    def get_generated_cells(self,
                            testing_data_pwd,
                            saved_pwd,
                            concept_list,
                            concept_counts,
                            concept_cdag,
                            *,
                            sampling_counts=10,
                            profile_size=1000,
                            sample_batch_size=128):
        generated_samples = self.sampling_cells(testing_data_pwd, concept_list, concept_counts, concept_cdag,
                                                profile_size=profile_size, sample_batch_size=sample_batch_size)
        for _ in range(sampling_counts - 1):
            generated_samples += self.sampling_cells(testing_data_pwd, concept_list, concept_counts, concept_cdag,
                                                     profile_size=profile_size, sample_batch_size=sample_batch_size)
        generated_samples /= sampling_counts
        if not os.path.exists(saved_pwd):
            os.mkdir(saved_pwd)
        joblib.dump(generated_samples, saved_pwd + f"/generated_cells.pkl")
        return generated_samples

    def counterfactual_generation(self,
                                  data_pwd,
                                  save_pwd,
                                  concept_list,
                                  concept_counts,
                                  concept_cdag,
                                  multi_target_list,
                                  file_name,
                                  *,
                                  batch_size=32):
        encoder_in_dim = self.model.flow_fn.DisentanglementEncoder.exogenous_encoder_m_v[0].in_features

        concept_embs = extract_concept_embs(data_pwd, self.model, encoder_in_dim, concept_list, concept_counts,
                                            concept_cdag, batch_size)
        factor_dict = factor_value_pool(concept_embs, concept_list, data_pwd)
        ori_data = sc.read_h5ad(data_pwd)
        selected_ids = np.array([True] * len(concept_embs[0]))
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            ref_factor_value = target_factor_dict["ref_factor_value"]
            tgt_factor_value = target_factor_dict["tgt_factor_value"]
            selected_ids = selected_ids & np.array((ori_data.obs[target_factor] == ref_factor_value))
        ref_concept_embs = concept_embs[:, selected_ids, :].copy()
        final_embs_ori = causality_based_concept_embs(ref_concept_embs, concept_cdag)

        target_embs = target_embs_generation(factor_dict, concept_list, multi_target_list, data_pwd, ref_concept_embs)
        final_embs_target = causality_based_concept_embs(target_embs, concept_cdag)
        new_df = ori_data[selected_ids].obs.copy()
        ori_df = ori_data[selected_ids].obs.copy()
        for target_factor_dict in multi_target_list:
            target_factor = target_factor_dict["target_factor"]
            tgt_factor_value = target_factor_dict["tgt_factor_value"]
            new_df[target_factor] = [str(tgt_factor_value)] * len(new_df[target_factor])
            ori_df[target_factor] = ori_df[target_factor].astype(str)

        final_embs = np.concatenate([final_embs_target, final_embs_ori], axis=1)
        final_df = pd.concat([new_df, ori_df])
        final_df['Type'] = ['Generated'] * len(new_df) + ['Original'] * len(ori_df)
        new_generated_data = generation_based_concept_embs(self.model, final_embs, save_pwd,
                                                           concept_list, concept_counts, concept_cdag,
                                                           final_df, file_name, batch_size)
        return new_generated_data
