from ..interfaces.protocol import InferencerProtocol

def build_inferencer(cfg) -> InferencerProtocol:
    inferencer = cfg.inference.inferencer