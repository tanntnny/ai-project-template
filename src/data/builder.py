from __future__ import annotations

from ..interfaces.protocol import DataProtocol

# ---------------- Build data ----------------
def build_data(cfg):
    if cfg.data.module == "mnist":
        from .mnist import MNISTData
        return MNISTData(cfg)