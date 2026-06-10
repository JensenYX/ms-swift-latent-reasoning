"""Plugin entry for query-anchored CoLaR experiments.

Use this file instead of ``colar_plugin/plugin.py`` when running the
query-anchor experiment.  It registers the same template but maps causal_lm to
``QueryAnchoredColarSeq2SeqTrainer``.  The baseline plugin is untouched.
"""
from swift.utils import get_logger

logger = get_logger()


def _register():
    from .colar_template import register_colar_template
    register_colar_template()

    from swift.trainers import trainer_factory
    target = 'colar_plugin.query_anchor_trainer.QueryAnchoredColarSeq2SeqTrainer'
    trainer_factory.TrainerFactory.TRAINER_MAPPING['causal_lm'] = target
    logger.info(f"[query-anchor-colar] TRAINER_MAPPING['causal_lm'] -> {target}")


try:
    _register()
except ImportError:
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
        'colar_plugin.query_anchor_trainer.QueryAnchoredColarSeq2SeqTrainer'
    logger.info('[query-anchor-colar] registered via fallback import path')
