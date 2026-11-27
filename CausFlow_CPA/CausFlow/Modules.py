import copy
import torch
import torch.nn as nn
from torch import einsum
from pathlib import Path
import math
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from torch.utils import data
import scanpy as sc
from einops import rearrange, repeat
from .utils import make_beta_schedule, default, exists, extract_into_tensor, BatchedOperation, noise_like
from .utils import create_activation, create_norm, mean_flat, sum_flat, gaussian_parameters
from .utils import timestep_embedding
import torch.nn.functional as F
from typing import Optional
from functools import partial
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import logging
import random
from .Dataset import Dataset

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class DisentanglementEncoder(nn.Module):
    def __init__(self, 
                 profile_size, 
                 out_dim, 
                 num_factor, 
                 causal_dag,
                 label_categories,
                 bias = False,
                 out_act = "gelu",  
                 gamma = 35
                 ):
        super().__init__()
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        self.num_factor = num_factor
        self.out_dim = out_dim
        self.profile_size = profile_size
        # 修改 DisentanglementEncoder 里的 exogenous_encoder_m_v
        self.exogenous_encoder_m_v = nn.Sequential(
            nn.Linear(profile_size, profile_size // 2),
            nn.LayerNorm(profile_size // 2),  # 增加归一化
            Mish(),
            nn.Linear(profile_size // 2, profile_size // 4),
            Mish(),
            nn.Linear(profile_size // 4, num_factor * out_dim * 2)
        )
        

        self.causal_dag = nn.Parameter(causal_dag)
        self.causal_dag.requires_grad = False
        self.I = nn.Parameter(torch.eye(num_factor))
        self.I.requires_grad = False
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_factor))
        else:
            self.register_parameter('bias', None)

        self.label_predictor = nn.ModuleList()
        for idx, num in enumerate(label_categories):
            self.label_predictor.append(nn.Sequential(
                nn.Linear(out_dim, num),
                nn.Softmax(dim = 1) 
            )
            )

        self.multilabelmulticate_loss = nn.CrossEntropyLoss()

        
        # discriminator for o and v
        self.discriminator_ov = nn.Linear(out_dim, 1)
        self.discriminator_ov2 = nn.Linear(num_factor, 1)
        self.discriminator_ov_act = nn.Sigmoid()
        
        self.gamma = gamma
    
    
    def mask_z(self, x):
        
        x = torch.matmul(self.causal_dag, x)
        
        return x
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def calculat_prior_kl(self, mean, log_var):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        """
        batch_size = mean.shape[0]
        kl_prior = self.normal_kl(
            mean1=mean, logvar1=log_var, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
        # return sum_flat(kl_prior) / np.log(2.0)

    def sample(self, mean, log_var):
        noise = torch.randn_like(mean)
        return mean + (0.5 * log_var).exp() * noise
    
    def forward(self, x, o):
        if o is not None:
            if not torch.is_tensor(o):
                o = torch.tensor(o, device=x.device)
            o = o.long().to(x.device)
        exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v(x), self.num_factor * self.out_dim, dim=-1)
        prior_kl = self.calculat_prior_kl(exogenous_factor_m, exogenous_factor_v).mean()
        
        exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
        exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)

        z = torch.inverse(self.I - self.causal_dag).matmul(exogenous_embs)
        
        concept_embs = z

        m_concept_embs = self.mask_z(concept_embs) + exogenous_embs
        mask_recon_loss = ((concept_embs - m_concept_embs) ** 2).mean()
        
        pred_o = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_o.append(predictor(concept_embs[:,idx,:]))
        pred_o_loss = 0
        for idx, pred_o_idx in enumerate(pred_o):
            pred_o_loss_idx = self.multilabelmulticate_loss(pred_o_idx, o[:,idx])
            pred_o_loss += pred_o_loss_idx
            # print(pred_o_loss_idx)
        
        # take mean-level loss
        pred_o_loss /= idx + 1
        
        # new adversirial part
        pred_u = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_u.append(predictor(concept_embs[:, -1, :]))
        pred_u_loss = 0
        for idx, pred_u_idx in enumerate(pred_u):
            pred_u_loss_idx = self.multilabelmulticate_loss(pred_u_idx, o[:,idx])
            pred_u_loss += pred_u_loss_idx
        # take mean-level loss
        discriminator_loss = - pred_u_loss / (idx + 1)
        
        # take sum-level loss
        # discriminator_loss = - pred_u_loss
        
        return concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def extract_exogenous_embs(self, x):
        print("🔥 extract_exogenous_embs input shape:", x.shape)
        with torch.no_grad():
            exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v.eval()(x), self.num_factor * self.out_dim, dim=-1)
            exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
            exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)
        return exogenous_embs

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim, 
                 context_dim, 
                 heads = 8, 
                 dim_head = 64, 
                 dropout = 0., 
                 qkv_bias = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, *, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mnv = max_neg_value(sim) - torch.finfo(sim.dtype).max
            if sim.shape[1:] == sim.shape[1:]:
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            else:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, mnv)
        
        attn = sim.softmax(dim = -1)
        # print(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        d_head: int = 64, 
        self_attn: bool = False,
        cross_attn: bool = True,
        ts_cross_attn: bool = False, 
        final_act: Optional[nn.Module] = None,
        dropout: float = 0, 
        context_dim: Optional[int] = None, 
        gated_ff: bool = True, 
        checkpoint: bool = False,
        qkv_bias: bool = False, 
        linear_attn: bool = False, 
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu = gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim = dim, 
                context_dim = context_dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim = dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, "Checkpointing not available yet"
    
    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x


class Denoise_net(nn.Module):
    """
    In Flow-Matching, this acts as the velocity field network:
        v_theta(x, t, conditions)
    """
    def __init__(
        self, dim, out_dim, num_factor, causal_dag, label_categories,
        depth=4, num_heads=4, dim_head=64,
        dropout=0., norm_type="layernorm",
        num_layers=1, act='gelu', out_act=None,
        with_time_emb=True
    ):
        super().__init__()

        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)

        # ---------- Time embedding ----------
        if with_time_emb:
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            self.time_mlp = None

        # ---------- MLP backbone ----------
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim),
                act,
                create_norm(norm_type, dim),
                nn.Dropout(dropout)
            ))
        self.layers.append(nn.Sequential(
            nn.Linear(dim, out_dim),
            out_act
        ))

        # ---------- Disentanglement Encoder (optional) ----------
        self.DisentanglementEncoder = DisentanglementEncoder(
            dim, 32, num_factor, causal_dag, label_categories
        )

        # ---------- Cross Attention ----------
        self.Cross_attention_module = nn.ModuleList([
            BasicTransformerBlock(
                out_dim, num_heads, dim_head,
                self_attn=False, cross_attn=True,
                context_dim=32,
                qkv_bias=True, dropout=dropout, final_act=None
            )
            for _ in range(depth)
        ])

        self.decoder_norm = create_norm(norm_type, out_dim)

    # ==========================
    #       FLOW-MATCHING FORWARD
    # ==========================
    def forward(self, x, time, labels=None, concept_embs=None):
        # 1. 解耦嵌入（保持原样，但增加空值检查）
        if labels is not None and concept_embs is None:
            concept_embs, _, _, _, _ = self.DisentanglementEncoder(x, labels)

        # 2. 时间嵌入改进：将 [0, 1] 映射到高维空间
        if self.time_mlp is not None:
            # 技巧：将 t 放大 1000 倍，模拟传统扩散模型的时间尺度，使 SinusoidalPosEmb 更有效
            t_emb = self.time_mlp(time * 1000)
            # 建议：使用 concat 或者是更复杂的线性变换，这里先修正维度问题
            x = x + t_emb  # 确保 t_emb 的 dim 和 x 一致

        # 3. 注入因果条件 (Cross-Attention)
        if concept_embs is not None:
            # x: (B, G) -> (B, 1, G)
            x = x.unsqueeze(1)
            for blk in self.Cross_attention_module:
                x = blk(x=x, context=concept_embs)
            x = x.squeeze(1)

        # 4) MLP + normalization
        x = self.decoder_norm(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
        x = self.layers[-1](x)
        # x is the predicted velocity field v_theta(x,t)
        return x


# --------------------------
# FlowGenerator (Conditional Flow Matching)
# --------------------------
@torch.no_grad()
def rk4_step(x,t,dt,model, *,labels=None,concept_embs=None,
):
    print("ODE step")
    k1 = model(x,t,labels=labels,concept_embs=concept_embs)
    k2 = model(x + 0.5 * dt * k1,t + 0.5 * dt,labels=labels,concept_embs=concept_embs)
    k3 = model(x + 0.5 * dt * k2,t + 0.5 * dt,labels=labels,concept_embs=concept_embs)
    k4 = model(x + dt * k3,t + dt,labels=labels,concept_embs=concept_embs)
    # RK4 update
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

class FlowGenerator(nn.Module):
    """
    Conditional Flow Matching generator.

    Semantic meaning:
        - concept_embs (endogenous embeddings) are FIXED conditions
        - flow learns p(x | concept_embs)
        - transport from noise -> real cell profiles
    """

    def __init__(self,flow_fn,encoder,profile_size,time_steps=100,):
        super().__init__()
        self.flow_fn = flow_fn              # velocity network
        self.encoder = encoder
        self.profile_size = profile_size
        self.time_steps = time_steps
    # -------------------------------------------------
    # Conditional Flow Matching loss
    # -------------------------------------------------
    # Modules.py 中 FlowGenerator 类的 flow_loss 方法
    def flow_loss(self, x_real, labels=None, weights=None):
        device = x_real.device
        B = x_real.shape[0]

        # 1. 核心：在流损失函数内运行因果编码器
        # 这样确保 mask_recon_loss, pred_o_loss 等梯度能直接传回 encoder
        concept_embs, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = \
            self.encoder(x_real, labels)

        # 2. Flow Matching 过程
        x_noise = torch.randn_like(x_real)
        t = torch.rand(B, device=device)
        t_view = t.view(B, 1)  # 确保维度对齐 (B, 1)

        # 线性插值轨迹: x_t = (1-t)*x_noise + t*x_real
        x_t = (1.0 - t_view) * x_noise + t_view * x_real
        target_v = x_real - x_noise  # 目标速度场向量

        # 3. 预测速度场 v_theta(x_t, t, condition)
        pred_v = self.flow_fn(
            x_t,
            t,
            labels=labels,
            concept_embs=concept_embs,
        )

        # 4. --- 关键修改：计算带权重的 MSE ---
        # 首先计算每个元素的原始平方误差，不要立即取 mean
        mse_elementwise = F.mse_loss(pred_v, target_v, reduction='none')  # 形状: (B, G)

        # 定义权重掩码 (Weight Mask)
        # 如果该基因在真实数据 x_real 中 > 0，则赋予 5.0 的高权重，否则为 1.0
        # 这强迫模型优先拟合高表达信号，从而拉开信号与背景的差距
        signal_weight = torch.where(x_real > 0, 5.0, 1.0)

        # 将权重应用到误差上，并在基因维度 (dim=1) 求平均
        flow_mse = (mse_elementwise * signal_weight).mean(dim=1)
        if weights is not None:
            flow_mse = (flow_mse * weights).mean()
        else:
            flow_mse = flow_mse.mean()

        return flow_mse, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, pred_v

    # -------------------------------------------------
    # compatibility wrapper
    # -------------------------------------------------
    def p_losses(
        self,
        x_start,
        t,
        labels,
        weights,
        noise=None,
        eps=False,
        concept_embs=None,
        **kwargs
    ):
        """
        This mimics the original diffusion p_losses API.
        Trainer will call this function.
        """
        return self.flow_loss(
            x_real=x_start,
            labels=labels,
            weights=weights,
        )

    def forward(self, x, *args, **kwargs):
        """
        Keep forward signature identical to old diffusion module.
        """
        B = x.shape[0]
        device = x.device
        t = torch.rand(B, device=device)
        return self.p_losses(x, t, *args, **kwargs)

    @torch.no_grad()
    def solve_ode(self, concept_embs, batch_size, steps, labels=None):
        device = next(self.flow_fn.parameters()).device
        x = torch.randn((batch_size, self.profile_size), device=device)
        dt = 1.0 / steps

        for i in range(steps):
            # t 从 0 走到 1-(1/steps)
            t_curr = i / steps
            t_vec = torch.full((batch_size,), t_curr, device=device, dtype=torch.float)

            # 获取速度场
            v = self.flow_fn(x, t_vec, labels=labels, concept_embs=concept_embs)

            # Euler 步进
            x = x + v * dt

        flatten_x = x.flatten()
        k = int(len(flatten_x) * 0.8434)
        threshold = torch.kthvalue(x.flatten(), k)[0]
        x[x < threshold] = 0  # 直接置零，不减去偏移量
        return x

    # 统一采样入口
    @torch.no_grad()
    def sample_with_factor(self, concept_embs, batch_size=16, steps=None, labels=None):
        inference_steps = steps if steps is not None else self.time_steps
        return self.solve_ode(concept_embs, batch_size, inference_steps, labels=labels)

    # 为了兼容以前的调用，可以保留一个简单的 sample
    @torch.no_grad()
    def sample(self, batch_size, concept_embs, labels=None, steps=None):
        return self.sample_with_factor(concept_embs, batch_size, steps, labels)


class FlowTrainer(object):
    def __init__(
            self,
            diffusion_model,   # now FlowGenerator
            folder,
            factor_list,
            *,
            ema_decay=0.995,
            profile_size=200,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=1000,
            save_and_sample_every=10000,
            results_folder='./results',
            train_log=True,
            device=None,   # optional explicit device
    ):
        super().__init__()
        # model may be FlowGenerator instance (with forward -> p_losses API)
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.profile_size = getattr(diffusion_model, 'profile_size', profile_size)
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # dataset & dataloader
        self.ds = Dataset(folder, self.profile_size, factor_list)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        # device selection: prefer explicit, else model device, else cuda if available
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                # if model has parameters, take their device
                p = next(self.model.parameters())
                self.device = p.device
            except StopIteration:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # move model & ema_model to device
        self.model.to(self.device)
        self.ema_model.to(self.device)

        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.step = 0

        # fp16 handling: support Apex if available, else torch.amp
        self.fp16 = fp16
        self.use_apex = bool(fp16 and APEX_AVAILABLE)
        if self.use_apex:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
            self.scaler = None
        else:
            # torch.amp path (scaler used only if fp16 True and Apex not available)
            self.scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and torch.cuda.is_available()))

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        if train_log:
            self.logger = get_logger(str(self.results_folder / 'training.log'))
        else:
            self.logger = None
        self.train_log = train_log

        # copy weights to ema_model
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        # main loop
        print(">>> diffusion module type:", type(self.model))
        # Modules.py 中 FlowTrainer 类的 train 方法
        while self.step <= self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data_batch = next(self.dl)
                x_batch = data_batch[0].to(self.device)
                labels_batch = data_batch[1].to(self.device).long()
                weights_batch = data_batch[2].to(self.device)
                sparsity_lambda = 0.0001

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # 直接调用 model (FlowGenerator)，内部会处理所有损失
                    main_loss, mask_recon_loss, loss_pred_o, loss_discriminator, prior_kl, pred_v = \
                        self.model(x_batch, labels_batch, weights_batch)
                    l1_penalty = torch.norm(pred_v, p=1, dim=1).mean()
                    # 这里的 alpha 权重可以根据原文章调整，通常解耦 loss 权重较大
                    total_loss = main_loss + 20 * loss_pred_o + loss_discriminator + 0.5 * prior_kl + mask_recon_loss + sparsity_lambda * l1_penalty

                # 反向传播
                if self.scaler is not None:
                    self.scaler.scale(total_loss / self.gradient_accumulate_every).backward()
                else:
                    (total_loss / self.gradient_accumulate_every).backward()

                # 梯度检查（用于调试，看看因果层有没有更新）
                if self.step % 1000 == 0:
                    grad_norm = self.model.encoder.exogenous_encoder_m_v[0].weight.grad.norm()
                    print(f"Step {self.step}, Encoder Grad Norm: {grad_norm}")
                # for name, param in self.model.encoder.named_parameters():
                #     if param.grad is not None:
                #         print(f"[ENCODER GRAD] {name}: {param.grad.norm().item():.6f}")
                # logging (safely .item())
                if self.train_log and self.logger is not None:
                    try:
                        self.logger.info(
                            f'{self.step}:{i}\tmain_loss:{float(main_loss):.6f}\tmask_recon_loss:{float(mask_recon_loss):.6f}\t'
                            f'loss_pred_o:{float(loss_pred_o):.6f}\tloss_discriminator:{float(loss_discriminator):.6f}\tprior_kl:{float(prior_kl):.6f}'
                        )
                    except Exception:
                        pass

            # optimizer step (handle scaler if used)
            if self.scaler is not None:
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()
            self.opt.zero_grad()

            # EMA update
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # save periodically
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

            self.step += 1

        print('training completed')

