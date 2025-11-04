from __future__ import annotations

from .mnist import MNISTClassifier

# ---------------- Build model ----------------
def build_model(cfg):
    if cfg.model.name == "mnist_classifier":
        return MNISTClassifier(cfg)
    elif cfg.model.name == "resnet":
        from .resnet import Resnet
        return Resnet(cfg)
    else:
        raise ValueError(f"Model {cfg.model.name} not recognized.")