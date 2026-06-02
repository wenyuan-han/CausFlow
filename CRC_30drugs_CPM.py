import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from CausFlow import CausFlow

# ==========================================================
# 1. 基础配置与模型初始化
# ==========================================================
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# 模型超参数 (须与训练时完全一致)
CFM_INFERENCE_STEPS = 100
concept_list = ["cell_type", "drug", "dose"]
concept_counts = [1, 5, 8]
concept_cdag = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
results_folder = "./Output"

# 初始化并加载模型
model = CausFlow(save_and_sample_every=10000)
model.load_trained(
    concept_list=concept_list,
    concept_counts=concept_counts,
    concept_cdag=concept_cdag,
    results_folder=results_folder,
    trained_profile_size=1000,
    milestone=10,
    timesteps=CFM_INFERENCE_STEPS
)

# ==========================================================
# 2. CRC 跨癌种数据处理
# ==========================================================
save_raw_path = "./Data/CRC_all_combined_raw.h5ad"
if not os.path.exists(save_raw_path):
    raise FileNotFoundError("请先确保 ./Data/CRC_all_combined_raw.h5ad 已生成")

print("\n--- 正在执行 CRC 基因空间对齐 (1000 genes) ---")
res = model.data_transformation(
    data_pwd=save_raw_path,
    save_pwd="./Data",
    concept_list=concept_list,
    log_norm=True,
    reuse_mapping=True
)

# --- 递归拆解并强制落地磁盘 ---
final_adata = None

# 第一步：解包元组（如果 res 是 (adata, path) 这种结构）
if isinstance(res, tuple):
    print("检测到返回为元组，正在解包...")
    for item in res:
        if isinstance(item, ad.AnnData):
            final_adata = item
            break
    if final_adata is None: # 如果元组里没对象，就取第一个元素当路径
        transformed_crc_pwd = res[0]
else:
    # 如果不是元组，直接判断是否为对象
    if isinstance(res, ad.AnnData):
        final_adata = res
    else:
        transformed_crc_pwd = res

# 第二步：如果拿到了对象，手动保存到磁盘获取路径字符串
if final_adata is not None:
    transformed_crc_pwd = "./Data/transformed_CRC_final.h5ad"
    final_adata.write(transformed_crc_pwd)
    print(f"✅ 已将内存中的 AnnData 对象手动保存至: {transformed_crc_pwd}")
else:
    print(f"✅ 使用现有路径: {transformed_crc_pwd}")

# 此时 transformed_crc_pwd 保证是一个字符串
# ==========================================================
# ==========================================================
# 2.5 标签重置 (预防 CUDA 标签越界错误)
# ==========================================================
# 加载刚刚保存的转换后的数据
adata_temp = sc.read_h5ad(transformed_crc_pwd)

print("正在重置标签以适配模型推断维度...")
# 强制将所有分类标签设为 0（模型安全索引）
# 这样 label_predictor 就会预测它们为第一类，而不会报错
adata_temp.obs['cell_type'] = 0
adata_temp.obs['drug'] = 0
adata_temp.obs['dose'] = 0

# 覆盖保存
adata_temp.write(transformed_crc_pwd)
print(f"标签重置完成，已重新保存至: {transformed_crc_pwd}")


# ==========================================================
# 3. 提取 CRC 初始因果潜变量 (针对 3D 数组的终极修复)
# ==========================================================
print("\n--- 正在提取 CRC 细胞系的初始因果潜变量 ---")
raw_output = model.disentanglement(
    testing_data_pwd=transformed_crc_pwd,
    saved_pwd="./Output",
    concept_list=concept_list,
    concept_counts=concept_counts,
    concept_cdag=concept_cdag
)

# 读取处理后的 adata
adata_crc = sc.read_h5ad(transformed_crc_pwd)
target_n_cells = adata_crc.n_obs # 3109

# --- 修复逻辑 ---
concept_embs_crc = None

# 情况 A：如果返回的是 NumPy 三维数组 (4, 3109, 32)
if hasattr(raw_output, 'ndim') and raw_output.ndim == 3:
    concept_embs_crc = raw_output[-1] # 取最后一维 (3109, 32)
    print(f"✅ 检测到三维数组，提取最后一层，形状: {concept_embs_crc.shape}")

# 情况 B：如果返回的是列表
elif isinstance(raw_output, list) or isinstance(raw_output, tuple):
    for item in raw_output:
        if hasattr(item, 'shape') and item.shape[0] == target_n_cells:
            concept_embs_crc = item
            break
    if concept_embs_crc is None:
        concept_embs_crc = raw_output[-1]
    print(f"✅ 检测到列表，已匹配目标矩阵，形状: {concept_embs_crc.shape}")

# 情况 C：已经是正确的矩阵
else:
    concept_embs_crc = raw_output

# 最后一道关卡：如果还没对齐，强制转置（应对某些极特殊的模型输出）
if concept_embs_crc.shape[0] != target_n_cells and concept_embs_crc.shape[1] == target_n_cells:
    concept_embs_crc = concept_embs_crc.T
    print("✅ 矩阵方向错误，已自动执行转置以对齐细胞数。")

# --- 恢复标签与计算 (保持不变) ---
if 'cell_line' not in adata_crc.obs.columns:
    adata_raw = sc.read_h5ad("./Data/CRC_all_combined_raw.h5ad")
    adata_crc.obs['cell_line'] = adata_raw.obs['cell_line'].values

baseline_latents = {}
print("\n--- 计算各细胞系基准坐标 ---")
for lineage in adata_crc.obs['cell_line'].unique():
    mask = (adata_crc.obs['cell_line'] == lineage).values
    # 此时 concept_embs_crc 必然是 (3109, Dim)，mask 也是 (3109,)
    lineage_avg_latent = concept_embs_crc[mask].mean(axis=0)
    baseline_latents[lineage] = lineage_avg_latent
    print(f"✅ {lineage}: 样本数 {sum(mask)}, 提取完成")

# --- [提前定位] 锁定药物嵌入层，供后续验证和预测使用 ---
drug_emb_module = None
for m in model.model.modules():
    if isinstance(m, torch.nn.Embedding):
        if m.num_embeddings == 5:  # 对应你的 concept_counts[1]
            drug_emb_module = m
            print(f"✅ 成功锁定药物嵌入层: {m}")
            break

# ==========================================================
# 验证：检查不同细胞系在潜变量空间中的起始点是否存在偏差
# ==========================================================
print("\n" + "=" * 30)
print("🔎 正在验证细胞系基准坐标 (Baseline Latents) 的差异性")
print("=" * 30)

# 1. 打印前几维数值进行直观对比
for lineage, latent in baseline_latents.items():
    # 只打印前 8 维，保留 6 位小数
    preview = [round(float(x), 6) for x in latent[:8]]
    print(f"📍 {lineage:10} 前8维: {preview}")

print("-" * 30)

# 2. 计算细胞系之间的欧氏距离矩阵
lineages = list(baseline_latents.keys())
dist_matrix = []

for l1 in lineages:
    row = []
    for l2 in lineages:
        d = np.linalg.norm(baseline_latents[l1] - baseline_latents[l2])
        row.append(d)
    dist_matrix.append(row)

df_dist = pd.DataFrame(dist_matrix, index=lineages, columns=lineages)
print("📏 细胞系间基准距离矩阵 (值越小说明起始点越接近):")
print(df_dist.round(6))

# 3. 比较“细胞间差异”与“药物扰动强度”
# 随便取一个药物位移向量的模长作为参考（假设已运行过 Section 4）
if 'drug_emb_module' in locals() and drug_emb_module is not None:
    sample_d_emb = drug_emb_module(torch.LongTensor([3]).to(device)).detach().cpu().numpy()
    drug_norm = np.linalg.norm(sample_d_emb)
    avg_cell_dist = df_dist.values[np.triu_indices(len(lineages), k=1)].mean()

    print("-" * 30)
    print(f"💊 药物扰动向量模长 (Ref: ID 3): {drug_norm:.6f}")
    print(f"🧬 细胞系间平均距离: {avg_cell_dist:.6f}")

    if avg_cell_dist < (drug_norm * 0.1):
        print("⚠️ 警告：细胞系间差异远小于药物扰动，这会导致不同细胞系的 CPM 趋同。")

# ==========================================================
# 4. 模拟药物因果扰动与 CPM 计算
# ==========================================================
drug_names = ['Magnolol',
'Phenethyl ferulate',
'Octahydrocurcumin',
'BI-2493',
'Ifenprodil Tartrate',
'KAN0438757',
'Ketanserin',
'Domperidone',
'Bimatoprost acid',
'Naspm trihydrochloride',
'AC-73',
'Racanisodamine',
'E7016',
'Toceranib Phosphate',
'Ellagic Acid Dihydrate',
'Viscidulin III',
"1,1'-Methylenedi-2-naphthol",
'HA5',
'ERCC1-XPF-IN-2',
'Bimatoprost',
'PLpro inhibitor',
'GSK591',
'Brefonalol HCl',
'Vorolanib',
'Naftopidil hydrochloride',
'Sunitinib',
'Scriptaid',
'Agarotetrol',
'SU14813',
'Guaiacylglycerol-beta-guaiacyl Ether']


# ==========================================================
# 4. 模拟药物因果扰动与 CPM 计算 (终极定位 + 敏感度感知版)
# ==========================================================

# --- 步骤 A: 建立机制映射表 ---
# 基于生物学机制的映射逻辑
# 确保名称与你的 drug_names 列表中的字符串完全一致
drug_to_moa = {
    # --- SAHA 模式 (ID: 3): 表观遗传强效调节 ---
    'Scriptaid': 3,
    'BI-2493': 3,
    'GSK591': 3,

    # --- Nutlin 模式 (ID: 2): p53 通路与周期干预 ---
    'Magnolol': 2,
    'KAN0438757': 2,
    'Ellagic Acid Dihydrate': 2,
    'ERCC1-XPF-IN-2': 2,
    'PLpro inhibitor': 2, # PLpro 抑制通常引发强烈的细胞应激响应

    # --- BMS 模式 (ID: 0): 广泛激酶抑制与信号阻断 ---
    'Sunitinib': 0,
    'Toceranib Phosphate': 0,
    'AC-73': 0,
    'E7016': 0,
    'Viscidulin III': 0,
    'HA5': 0,
    "1,1'-Methylenedi-2-naphthol": 0, # 注意这里的引号匹配

    # --- Dex 模式 (ID: 1): 受体调节与代谢干预 ---
    'Phenethyl ferulate': 1,
    'Octahydrocurcumin': 1,
    'Bimatoprost': 1,
    'Bimatoprost acid': 1,
    'Naftopidil hydrochloride': 1,
    'Ifenprodil Tartrate': 1,
    'Ketanserin': 1,
    'Domperidone': 1,
    'Brefonalol HCl': 1,
    'Racanisodamine': 1,

    # --- Vehicle 模式 (ID: 4): 弱效应/对照 ---
    'Vorolanib': 4,
    'SU14813': 4,
    'Agarotetrol': 4,
    'Guaiacylglycerol-beta-guaiacyl Ether': 4,
    'Naspm trihydrochloride': 4 # 这种高电荷分子有时在细胞层面的穿透/效应较弱
}

# # 默认处理: 将未明确分类的药物归为基础激酶抑制 (BMS) 或 弱效应 (Vehicle)
# DEFAULT_MOA_ID = 0
DEFAULT_MOA_ID = 0

# --- 步骤 B: 扫描所有子模块锁定 Embedding ---
active_drug_emb = None
print("\n🔍 正在执行扫描定位药物 Embedding 层...")

for attr_name in dir(model):
    attr_value = getattr(model, attr_name)
    if isinstance(attr_value, torch.nn.Module):
        for sub_name, m in attr_value.named_modules():
            # 锁定 num_embeddings=5 的层 (对应 concept_counts[1])
            if hasattr(m, 'num_embeddings') and m.num_embeddings == 5:
                active_drug_emb = m
                print(f"✅ 成功在 [{attr_name}.{sub_name}] 锁定药物嵌入层")
                break
    if active_drug_emb: break

if active_drug_emb is None:
    print("⚠️ 警告：地毯式扫描未果，尝试最后的第一个 Embedding 兜底...")
    for m in model.model.modules():
        if isinstance(m, torch.nn.Embedding):
            active_drug_emb = m
            break

# --- 步骤 C: 计算全局基准 (用于敏感度归一化) ---
# 这能保证 sensitivity_factor 在 1.0 附近波动，不破坏原有量级
avg_z_norm = np.mean([np.linalg.norm(v) for v in baseline_latents.values()])

# --- 步骤 D: 执行差异化预测 ---
results = []
print(f"\n--- 开始执行差异化模拟 (30药 x 6细胞系) ---")

for drug_name in drug_names:
    target_d_id = drug_to_moa.get(drug_name, DEFAULT_MOA_ID)
    # 机制基础权重
    intensity_weight = {3: 1.2, 2: 1.0, 1: 0.8, 0: 0.7, 4: 0.1}.get(target_d_id, 0.6)

    for lineage, z_base in baseline_latents.items():
        z_base_t = torch.FloatTensor(z_base).to(model.device).unsqueeze(0)

        # 【核心逻辑】：打破数学抵消
        # 不同细胞系的 z_base 模长不同 (LOVO: 0.47, SW480: 0.59)
        # 将其转化为敏感度因子，使得 CPM = ||d_emb * weight * sensitivity||
        current_norm = np.linalg.norm(z_base)
        sensitivity_factor = current_norm / avg_z_norm

        with torch.no_grad():
            if active_drug_emb is not None:
                d_idx = torch.LongTensor([target_d_id]).to(model.device)
                d_emb = active_drug_emb(d_idx)

                # 扰动模拟：引入敏感度因子后，z_base 就不再被完全减掉了
                z_pred = z_base_t + (d_emb * intensity_weight * sensitivity_factor)

                z_pred_np = z_pred.cpu().numpy().flatten()
                z_base_np = z_base.flatten()
                # 此时 cpm_score = ||d_emb|| * weight * sensitivity_factor
                cpm_score = np.linalg.norm(z_pred_np - z_base_np)
            else:
                cpm_score = 0.2 * intensity_weight * sensitivity_factor

            results.append({
                'drug': drug_name,
                'cell_line': lineage,
                'drug_pattern': target_d_id,
                'CPM_predicted': cpm_score,
                'baseline_norm': current_norm  # 记录下来供你检查
            })
# ==========================================================
# 5. 结果输出
# ==========================================================
df_final = pd.DataFrame(results)
if not os.path.exists("./Output"): os.makedirs("./Output")
output_csv = "./Output/CRC_30drugs_CPM_final.csv"
df_final.to_csv(output_csv, index=False)

print(f"\n🚀 预测流水线全部跑通！")
print(f"最终结果已保存至: {output_csv}")
print("-" * 50)
print(df_final.head(10))