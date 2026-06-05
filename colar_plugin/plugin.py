"""
CoLaR 插件入口：被 `--external_plugins` 在训练启动前 import。

职责：
  1. 注册自定义 Template `colar_qwen3_omni`。
  2. 把 TrainerFactory 的 'causal_lm' 映射改写为我们的 ColarSeq2SeqTrainer，
     这样 swift sft 走标准 causal_lm 流程时会实例化 CoLaR Trainer。

用法（在 swift sft 命令里加）：
  --external_plugins <abs_path>/colar_plugin/plugin.py
  --template colar_qwen3_omni
"""
from swift.utils import get_logger

logger = get_logger()


def _register():
    # 1) 注册 template
    from .colar_template import register_colar_template
    register_colar_template()

    # 2) 改写 trainer 工厂映射 -> CoLaR Trainer
    from swift.trainers import trainer_factory
    target = 'colar_plugin.colar_trainer.ColarSeq2SeqTrainer'
    trainer_factory.TrainerFactory.TRAINER_MAPPING['causal_lm'] = target
    logger.info(f"[colar] TRAINER_MAPPING['causal_lm'] -> {target}")


# external_plugins 用 import_module 加载本文件，import 时即注册。
# 兼容两种 import 方式：作为包模块（colar_plugin.plugin）或单文件（plugin）。
try:
    _register()
except ImportError:
    # 当 import_external_file 以单文件方式加载（sys.path 插入了 colar_plugin 目录、
    # 模块名为 'plugin'）时，相对 import 会失败，这里改用绝对/直接 import 再注册。
    import os
    import sys
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(_this_dir)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from colar_plugin.colar_template import register_colar_template
    from swift.trainers import trainer_factory
    register_colar_template()
    trainer_factory.TrainerFactory.TRAINER_MAPPING['causal_lm'] = \
        'colar_plugin.colar_trainer.ColarSeq2SeqTrainer'
    logger.info('[colar] registered via fallback import path')
