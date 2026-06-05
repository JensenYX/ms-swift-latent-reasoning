"""
LatentPolicy: CoLaR 的 latent head（从 colar/src/modules/projector.py 移植）。

它是一个小 MLP，输入 LLM 最后一层在“思考段”各位置的 hidden state，
输出一个高斯分布 N(mean, std)，用来预测“下一个压缩 embedding”。
训练时用 NLL（或 MSE）监督它去拟合真实的（压缩后）embedding 序列。

与 LoRA adapter 不同，这是一组全新的、全精度可训练参数，需要：
  - 显式 requires_grad=True
  - 进 optimizer（本实现把它注册为 model 的子模块 model.latent_policy，
    由 HF optimizer 自动收集；参见 colar_trainer.py）
  - 在 _save_model 里额外 torch.save（PEFT 只存 adapter）
"""
import torch
import torch.nn as nn


class LatentPolicy(nn.Module):

    def __init__(self, feature_size: int, intermediate_size: int = 512, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )
        self.mean = nn.Linear(intermediate_size, feature_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, feature_size)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_std = self.log_std(x)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)


@torch.no_grad()
def compute_embeds_std(model) -> float:
    """
    实测模型 input embedding 矩阵的标准差，替代 CoLaR 里硬编码的 MODEL_EMB_STD。

    CoLaR 用这个常数把 gold embedding 归一化到 ~单位方差空间再算 NLL/MSE，
    并在 latent 生成时把采样结果乘回去。不同模型 embedding 尺度不同，必须各自实测。

    兼容 Qwen3-Omni：input embedding 在 thinker.model.embed_tokens。
    """
    emb = None
    # 优先走标准接口
    try:
        emb = model.get_input_embeddings()
    except Exception:
        emb = None
    if emb is None or not hasattr(emb, 'weight'):
        # Qwen3-Omni 兜底
        base = getattr(model, 'thinker', model)
        inner = getattr(base, 'model', base)
        emb = inner.get_input_embeddings() if hasattr(inner, 'get_input_embeddings') else inner.embed_tokens
    w = emb.weight
    return w.float().reshape(-1).std().item()
