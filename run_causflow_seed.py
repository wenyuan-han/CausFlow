import os
import random
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from causcell import CausCell
from causcell.Metrics_calculation import (
    cal_pearson_correlation_with_CT,
    cal_MSE_CT,
    cal_marker_gene_matching_score_CT
)

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from scipy.stats import t as student_t

# Keep the original MERFISH model, split, and metric implementations.
concept_list = ["Age", "Domain", "Celltype"]
concept_counts = [3, 8, 10]
concept_cdag = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[0,0,0,0]]
CFM_INFERENCE_STEPS = 100
METRICS = ["PCC", "MSE", "ARI", "NMI", "MarkerScore"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def state_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================
# Automatically find label column
# ==========================================================
def find_label_column(adata):
    candidate_cols = [
        "cell_type",
        "CellType",
        "celltype",
        "Celltype",
        "annotation",
        "label"
    ]

    for col in candidate_cols:
        if col in adata.obs.columns:
            return col

    return None


# ==========================================================
# Evaluation function
# ==========================================================
def evaluate_one_setting(model, testing_data_pwd, save_tag, output_dir):
    print(f"\nEvaluating {save_tag}")

    generated_cells = model.get_generated_cells(
        testing_data_pwd=testing_data_pwd,
        saved_pwd=str(output_dir),
        concept_list=concept_list,
        concept_counts=concept_counts,
        concept_cdag=concept_cdag,
        profile_size=374, sampling_counts=10, sample_batch_size=128
    )

    adata_test = sc.read_h5ad(testing_data_pwd)
    real_samples = adata_test.X

    if not isinstance(real_samples, np.ndarray):
        real_samples = real_samples.toarray()

    if hasattr(generated_cells, "detach"):
        generated_samples = generated_cells.detach().cpu().numpy()
    else:
        generated_samples = generated_cells

    # ======================================================
    # Save transcriptomes
    # ======================================================
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

    adata_pred.write(f"{output_dir}/predicted_{save_tag}.h5ad")
    adata_real.write(f"{output_dir}/real_{save_tag}.h5ad")

    # ======================================================
    # PCC / MSE (always computed)
    # ======================================================
    pcc, pcc_ct = cal_pearson_correlation_with_CT(
        real_samples,
        generated_samples
    )

    mse, mse_ct = cal_MSE_CT(
        real_samples,
        generated_samples
    )

    # default values if no labels
    ari = np.nan
    nmi = np.nan
    marker_score = np.nan
    marker_score_ct = np.nan

    # ======================================================
    # Find label column
    # ======================================================
    label_key = find_label_column(adata_test)

    if label_key is not None:
        print(f"Found label column: {label_key}")

        true_celltypes = adata_test.obs[label_key].values

        # ==================================================
        # ARI / NMI
        # ==================================================
        kmeans = KMeans(
            n_clusters=len(np.unique(true_celltypes)),
            random_state=0
        )

        pred_celltypes = kmeans.fit_predict(generated_samples)

        ari = adjusted_rand_score(true_celltypes, pred_celltypes)
        nmi = normalized_mutual_info_score(
            true_celltypes,
            pred_celltypes
        )

        # ==================================================
        # Marker Score
        # ==================================================
        obs_df = adata_test.obs[[label_key]].copy()
        obs_df.columns = ["cell_type"]
        obs_df = obs_df.reset_index(drop=True)

        marker_score, marker_score_ct = cal_marker_gene_matching_score_CT(
            real_samples,
            generated_samples,
            obs_df,
            top_rank=50
        )

    else:
        print(
            f"[Warning] No label column found in {save_tag}. "
            f"Available columns: {adata_test.obs.columns.tolist()}"
        )
        print("Skip ARI / NMI / MarkerScore.")

    # ======================================================
    # Print diagnostics
    # ======================================================
    print("=" * 60)
    print(f"PCC:         {pcc:.4f}")
    print(f"MSE:         {mse:.4f}")
    print(f"ARI:         {ari}")
    print(f"NMI:         {nmi}")
    print(f"MarkerScore: {marker_score}")
    print("=" * 60)

    return {
        "PCC": pcc,
        "PCC_CT": pcc_ct,
        "MSE": mse,
        "MSE_CT": mse_ct,
        "ARI": ari,
        "NMI": nmi,
        "MarkerScore": marker_score,
        "MarkerScore_CT": marker_score_ct
    }



def main():
    ap = argparse.ArgumentParser(description="Independent MERFISH seed rerun; uses existing transformed splits and never overwrites a previous run.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--train-only", action="store_true", help="Skip evaluation (for smoke tests only).")
    args = ap.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds) or any(x < 0 for x in args.seeds):
        ap.error("Use positive steps and distinct nonnegative seeds.")
    root = Path(args.root).resolve()
    out = (root / args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --out directory.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the current Python environment.")
    if "seed" not in inspect.signature(CausCell.train).parameters:
        raise RuntimeError("Unexpected CausCell.train signature; inspect the actual import.")
    paths = {k:root/"Data"/f"transformed_MERFISH_{v}.h5ad" for k,v in {"train":"ID_train","ID":"ID_test","OOD":"OOD_test"}.items()}
    data_info = {}
    for key, path in paths.items():
        a = sc.read_h5ad(path)
        if a.n_vars != 374 or a.n_obs == 0:
            raise ValueError(f"Unexpected shape {a.shape}: {path}")
        for label, count in zip(concept_list, concept_counts):
            v = pd.to_numeric(a.obs[label], errors="raise").to_numpy()
            if not np.isfinite(v).all() or not ((v >= 0) & (v < count) & (v == np.floor(v))).all():
                raise ValueError(f"Invalid encoded labels for {key}/{label}")
        data_info[key] = {"path":str(path), "shape":list(a.shape), "sha256":sha256(path)}
        del a
    out.mkdir(parents=True, exist_ok=False)
    save_interval = 10000 if args.steps % 10000 == 0 else args.steps
    milestone = args.steps // save_interval
    manifest = {"source":inspect.getfile(CausCell), "source_sha256":sha256(inspect.getfile(CausCell)), "train_steps_argument":args.steps, "train_seeds":args.seeds, "evaluation_seed":2026, "train_batch_size":64, "gradient_accumulate_every":2, "training_lr":5e-6, "inference_steps":100, "sampling_counts":10, "sample_batch_size":128, "checkpoint_key":"model", "concept_list":concept_list, "concept_counts":concept_counts,"concept_cdag":concept_cdag,"data":data_info,"train_only":args.train_only,"torch_version":torch.__version__}
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    results, audits = [], []
    for seed in args.seeds:
        sd = out/f"seed_{seed}"
        sd.mkdir()
        print(f"TRAIN seed={seed} steps={args.steps} -> {sd}",flush=True)
        set_seed(seed)
        model = CausCell(save_and_sample_every=save_interval, gradient_accumulate_every=2)
        model.train(training_data_pwd=str(paths["train"]), model_save_pwd=str(sd), concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, training_num_steps=args.steps, training_batch_size=64, training_lr=5e-6, timesteps=100, seed=seed)
        ckpt = sd/f"model-{milestone}.pt"
        model.load_trained(concept_list=concept_list, concept_counts=concept_counts, concept_cdag=concept_cdag, results_folder=str(sd), trained_profile_size=374, milestone=milestone, timesteps=100)
        record = {"Seed":seed,"Checkpoint":str(ckpt),"FileSHA256":sha256(ckpt),"ModelStateSHA256":state_fingerprint(model.model)}
        audits.append(record)
        pd.DataFrame(audits).to_csv(out/"checkpoint_audit.csv",index=False)
        if len({x["ModelStateSHA256"] for x in audits}) != len(audits):
            raise RuntimeError("Duplicate loaded model tensors across seeds. Stop and investigate.")
        if not args.train_only:
            # Fix the evaluation RNG across models to isolate training-seed variation.
            # Preserve the original metric functions, KMeans settings and sampling code.
            model.model.eval()
            for scenario in ("ID","OOD"):
                ed = sd/scenario; ed.mkdir()
                set_seed(2026)
                row = evaluate_one_setting(model,str(paths[scenario]),f"seed{seed}_{scenario}",ed)
                row.update(Seed=seed,Scenario=scenario)
                if not np.isfinite([row[k] for k in METRICS]).all():
                    raise ValueError(f"Nonfinite metrics: {row}")
                results.append(row)
                pd.DataFrame(results).to_csv(out/"MERFISH_Brain_5seed_ID_OOD_results.csv",index=False)
        del model
        torch.cuda.empty_cache()
    if results:
        rows=[]
        df=pd.DataFrame(results)
        for scenario, group in df.groupby("Scenario"):
            for metric in METRICS:
                x=group[metric]; n=len(x); sd=x.std(ddof=1)
                ci=student_t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
                rows.append({"Scenario":scenario,"Metric":metric,"N":n,"Mean":x.mean(),"SD":sd,"CI95_halfwidth":ci})
        pd.DataFrame(rows).to_csv(out/"MERFISH_Brain_summary.csv",index=False)
    print(f"Completed. Audit: {out/'checkpoint_audit.csv'}")

if __name__ == "__main__":
    main()
