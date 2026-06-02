import pandas as pd
import numpy as np
import anndata as ad
import os
from run_causflow import model, concept_list, concept_counts, concept_cdag

file_map = {
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\LoVo.count_mtx.tsv.txt': 'LOVO',
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\HCT116.count_mtx.tsv.txt': 'HCT-116',
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\SW620.count_mtx.tsv.txt': 'SW620',
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\DLD-1.count_mtx.tsv.txt': 'DLD-1',
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\HT29.count_mtx.tsv.txt': 'HT-29',
    'D:\PycharmProjects\CausFlow\Data\CRC_raw_6cellline_scRNA-seq\SW480.count_mtx.tsv.txt': 'SW480'
}

adatas = []

# 1. 循环读取（修正了分隔符）
for file_path, line_name in file_map.items():
    if not os.path.exists(file_path): continue

    print(f"读取 {line_name}...")
    # 强制使用制表符读取，因为诊断显示 sep='\t' 才是正确的
    df = pd.read_csv(file_path, sep='\t')

    if 'ID' not in df.columns:
        df.rename(columns={df.columns[0]: 'ID'}, inplace=True)

    df['ID'] = df['ID'].astype(str).str.strip().str.split('.').str[0].str.upper()
    df = df.groupby('ID').mean()

    temp_adata = ad.AnnData(X=df.T.values.astype('float32'))
    temp_adata.obs_names = [f"{line_name}_C{i}" for i in range(df.shape[1])]
    temp_adata.var_names = df.index.astype(str)
    temp_adata.obs['cell_line'] = line_name
    temp_adata.obs['cell_type'] = 'CRC'
    temp_adata.obs['drug'] = 'control'
    temp_adata.obs['dose'] = '0'
    adatas.append(temp_adata)

# 2. 合并
common_genes = sorted(list(set.intersection(*[set(a.var_names) for a in adatas])))
adata_all = ad.concat([a[:, common_genes] for a in adatas], join='inner')
print(f"\n合并成功，维度: {adata_all.shape}")

# 3. 保存
if not os.path.exists("./Data"): os.makedirs("./Data")
save_raw_path = "./Data/CRC_all_combined_raw.h5ad"
adata_all.write(save_raw_path)

#----------------------------------------------------------------------------------------------------------------------------------
# 1. 基因空间裁剪 (3109, 16611) -> (3109, 1000)
print("\n--- 正在执行基因裁剪与归一化 ---")
transformed_crc_pwd = model.data_transformation(
    data_pwd=save_raw_path,
    save_pwd="./Data",
    concept_list=concept_list,
    log_norm=True,      # CSV通常是原始Count，这里需要True
    reuse_mapping=True  # 复用A549的1000基因
)

# 2. 提取潜变量 (z_cell)
print("--- 正在提取初始因果潜变量 ---")
# 这一步会生成 concept_embs，代表细胞在解耦空间中的初始坐标
concept_embs_crc = model.disentanglement(
    testing_data_pwd=transformed_crc_pwd,
    saved_pwd="./Output",
    concept_list=concept_list,
    concept_counts=concept_counts,
    concept_cdag=concept_cdag
)

# 3. 计算每个细胞系的“基准坐标”
adata_crc = ad.read_h5ad(transformed_crc_pwd)
baseline_latents = {}

for lineage in adata_crc.obs['cell_line'].unique():
    mask = (adata_crc.obs['cell_line'] == lineage).values
    # 取该细胞系所有细胞潜变量的均值，代表其稳态起点
    lineage_avg_latent = concept_embs_crc[mask].mean(axis=0)
    baseline_latents[lineage] = lineage_avg_latent
    print(f"细胞系 {lineage} 基准坐标提取完成。")

# 保存基准坐标供下一步使用
np.save('./Output/crc_baseline_latents.npy', baseline_latents)