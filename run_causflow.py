from statsmodels.stats.descriptivestats import pd_ptp

from CausFlow import CausFlow
import numpy as np
import anndata as ad
import pandas as pd
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from CausFlow.Metrics_calculation import (cal_pearson_correlation_with_CT, cal_MSE_CT,
                                          cal_marker_gene_matching_score_CT, cal_ARI_NMI)
import scanpy as sc
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

model = CausFlow(save_and_sample_every=10000)

CFM_INFERENCE_STEPS = 100
concept_list = ["Age", "Domain", "Celltype"]
concept_counts = [3, 8, 10]
concept_cdag = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]

# set up an output directory of model training
results_folder = "./Output"

transformed_data = model.data_transformation(data_pwd="./Data/MERFISH_Brain.h5ad",
                                                   save_pwd="./Data",
                                                   concept_list=concept_list,
                                                   log_norm=False
                                                   )

# train dataset format transformation for CausFlow training
transformed_train_data = model.data_transformation(data_pwd="./Data/MERFISH_ID_train.h5ad",
                                                   save_pwd="./Data",
                                                   concept_list=concept_list,
                                                   log_norm=False,
                                                   )

# # model training
model.train(training_data_pwd="./Data/transformed_MERFISH_Brain_ID_train.h5ad",
            model_save_pwd="./Output",
            concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag,
            training_num_steps=100000,
            timesteps = CFM_INFERENCE_STEPS)

# load trained model parameters from previous training
model.load_trained(concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag,
                   results_folder=results_folder,
                   trained_profile_size=374,
                   milestone=10,
                   timesteps = CFM_INFERENCE_STEPS)

# # test dataset format transformation for CausFlow training
transformed_test_data = model.data_transformation(data_pwd="./Data/MERFISH_ID_test.h5ad",
                                                save_pwd="./Data",
                                                concept_list=concept_list,
                                                log_norm=False,
                                                reuse_mapping=True)

testing_data_pwd = "./Data/transformed_MERFISH_ID_test.h5ad"

# obtained the concept representations of all cells in test dataset
concept_embs = model.disentanglement(testing_data_pwd=testing_data_pwd,
                                     saved_pwd="./Output",
                                     concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag)

model.concept_prediction(testing_data_pwd, concept_embs, concept_list, concept_counts, concept_cdag)

# obtained the reconstructed gene expression profiles of all cells in test dataset
generated_cells = model.get_generated_cells(testing_data_pwd=testing_data_pwd, saved_pwd="./Output",
                                            concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag)


adata_test = sc.read_h5ad(testing_data_pwd)
real_samples = adata_test.X

# 如果是稀疏矩阵
if not isinstance(real_samples, np.ndarray):
    real_samples = real_samples.toarray()
if hasattr(generated_cells, "detach"):
    generated_samples = generated_cells.detach().cpu().numpy()
else:
    generated_samples = generated_cells


# ==========================================================
# 🔥 Save predicted & real transcriptomes for visualization
# ==========================================================
print("\nSaving transcriptomes for fidelity visualization...")

adata_pred = ad.AnnData(
    X=generated_samples,
    obs=adata_test.obs.copy(),
    var=adata_test.var.copy()
)

adata_real = ad.AnnData(
    X=real_samples,
    obs=adata_test.obs.copy(),
    var=adata_test.var.copy()
)

adata_pred.write("./Output/predicted_expression_MERFISH_Brain_ID.h5ad")
adata_real.write("./Output/real_expression_MERFISH_Brain_ID.h5ad.h5ad")

print("Saved predicted & real expression to ./Output/")

# PCC
pcc, pcc_ct = cal_pearson_correlation_with_CT(
    real_samples, generated_samples
)

# MSE
mse, mse_ct = cal_MSE_CT(
    real_samples, generated_samples
)

print(f"PCC (aligned): {pcc:.4f}")
print(f"PCC (CT): {pcc_ct:.4f}")
print(f"MSE (aligned): {mse:.4f}")
print(f"MSE (CT): {mse_ct:.4f}")

# --- 新加的诊断代码 ---
print("\n" + "="*30)
print("Model Generation Diagnostic:")
print(f"Real data range:      {real_samples.min():.2f} to {real_samples.max():.2f}")
print(f"Generated data range: {generated_samples.min():.2f} to {generated_samples.max():.2f}")

# 检查零值占比（基因表达数据通常很稀疏）
real_sparsity = (real_samples == 0).mean()
gen_sparsity = (generated_samples <= 1e-3).mean() # 给生成值一点容差
print(f"Real Sparsity:        {real_sparsity:.2%}")
print(f"Generated Sparsity:   {gen_sparsity:.2%}")
print("="*40)

# ==========================================================
# 7. Disentanglement Evaluation (ARI / NMI)
#    Celltype-level semantic correctness
# ==========================================================
true_celltypes = adata_test.obs["cell_type"].values

kmeans = KMeans(n_clusters=len(np.unique(true_celltypes)), random_state=0)
pred_celltypes = kmeans.fit_predict(generated_samples)

ari = adjusted_rand_score(true_celltypes, pred_celltypes)
nmi = normalized_mutual_info_score(true_celltypes, pred_celltypes)

# ==========================================================
# 8. Marker Gene Matching Score (Celltype only)
# ==========================================================
obs_df = adata_test.obs[["cell_type"]].reset_index(drop=True)

marker_score, marker_score_ct = cal_marker_gene_matching_score_CT(
    real_samples,
    generated_samples,
    obs_df,
    top_rank=50
)

# ==========================================================
# 9. Final Metric Summary (Paper-ready)
# ==========================================================
print("\n📊 Final Evaluation Metrics")
print("-" * 70)
print(f"{'Metric':<30} | {'Aligned':<15} | {'CT / OOD':<15}")
print("-" * 70)

print(f"{'PCC':<30} | {pcc:<15.4f} | {pcc_ct:<15.4f}")
print(f"{'MSE':<30} | {mse:<15.4f} | {mse_ct:<15.4f}")
print(f"{'ARI (Celltype)':<30} | {ari:<15.4f} | {'N/A':<15}")
print(f"{'NMI (Celltype)':<30} | {nmi:<15.4f} | {'N/A':<15}")
print(f"{'Marker Gene Match':<30} | {marker_score:<15.4f} | {marker_score_ct:<15.4f}")

print("-" * 70)