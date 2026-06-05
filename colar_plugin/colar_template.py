"""
自定义 Template：colar_qwen3_omni

继承 Qwen3OmniTemplate，几乎不改编码逻辑（音频/mrope/思考块都沿用父类）。
存在的意义：
  1. 提供一个独立注册名 `colar_qwen3_omni`，作为“CoLaR 模式”的开关标记。
  2. 暴露 <think>/</think> 的 token id 常量，供 Trainer 在 input_ids 上定位“要压缩的思考段”。
  3. 里程碑2 接入音频时，音频相关的微调有一个落点。

关键事实（已确认）：
  - Qwen3OmniTemplate._post_encode 是 no-op，因此 input_ids 会原样流到模型 forward，
    Trainer 在 compute_loss 里能直接拿到 input_ids 来定位思考段（无需在 encode/collator
    里手动追踪 padding 偏移，最稳）。
  - <think>=151667, </think>=151668 都是单 token。
"""
from typing import Any, Dict

from swift.template.register import register_template
from swift.template.templates.qwen import Qwen3OmniTemplate, QwenTemplateMeta
from swift.template.constant import MLLMTemplateType
from swift.utils import get_logger

logger = get_logger()

COLAR_QWEN3_OMNI_TEMPLATE = 'colar_qwen3_omni'

# Qwen3-Omni tokenizer 中的 thinking 标记（单 token）
THINK_OPEN_ID = 151667   # <think>
THINK_CLOSE_ID = 151668  # </think>


class ColarQwen3OmniTemplate(Qwen3OmniTemplate):
    """与 Qwen3OmniTemplate 完全一致的编码行为；仅作为 CoLaR 的注册入口。"""

    # 供 Trainer 读取
    think_open_id = THINK_OPEN_ID
    think_close_id = THINK_CLOSE_ID

    def _encode(self, inputs) -> Dict[str, Any]:
        encoded = super()._encode(inputs)
        # 仅做一次性健全性检查（不修改任何字段，避免 padding 偏移问题）
        input_ids = encoded.get('input_ids')
        if input_ids is not None and self.is_training:
            n_open = sum(1 for t in input_ids if t == self.think_open_id)
            n_close = sum(1 for t in input_ids if t == self.think_close_id)
            if n_open != 1 or n_close != 1:
                logger.info_once(
                    f'[colar] 样本 think 标记数量异常 (<think>={n_open}, </think>={n_close})；'
                    f'此类样本将退化为不压缩（整段按普通 SFT 处理）。')
        return encoded


def register_colar_template():
    register_template(
        QwenTemplateMeta(
            COLAR_QWEN3_OMNI_TEMPLATE,
            template_cls=ColarQwen3OmniTemplate,
            default_system=None,
            thinking_prefix='<think>\n',
        ),
        exist_ok=True,
    )
    logger.info(f'[colar] registered template: {COLAR_QWEN3_OMNI_TEMPLATE}')
