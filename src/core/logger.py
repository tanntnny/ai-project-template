from typing import Dict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# ---------------- Configuration ----------------
_BREAK_LINE = "=" * 100

# ---------------- Logger ----------------
class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = Path(self.cfg.output_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.log_dir.as_posix())
    
    def log_info(self, tag: str, log: str, break_section: bool = False):
        if break_section:
            print(f"\n{_BREAK_LINE}")
        print(f"[{tag}] {log}")

    def log_tb_scalar(self, scalars: Dict[str, float], step: int):
        for key, value in scalars.items():
            self.tb.add_scalar(key, value, step)
    
    def log_tb_text(self, tag: str, text: str, step: int):
        self.tb.add_text(tag, text, step)

    def close(self):
        self.tb.close()

# ---------------- Singleton Logger ----------------
global_logger = None
def get_logger(cfg) -> Logger:
    global global_logger
    if global_logger is None:
        if cfg is None:
            raise ValueError("Logger has not been initialized and no configuration was provided.")
        if not hasattr(cfg, 'output_dir'):
                raise ValueError("Configuration must have 'output_dir' attribute to initialize the Logger.")
        global_logger = Logger(cfg)
    return global_logger