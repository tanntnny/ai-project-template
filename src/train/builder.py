from __future__ import annotations

from ..interfaces.protocol import TrainerProtocol
from .hf_trainer import HFTrainer

# ---------------- Build trainer ----------------
def build_trainer(cfg) -> TrainerProtocol:
    trainer_type = cfg.train.trainer
    if trainer_type == "hf":
        return HFTrainer(cfg)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")