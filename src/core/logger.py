import os
from typing import Dict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

_BREAK_LINE = "=" * 100

def is_main_process() -> bool:
    if "RANK" in os.environ:
        try:
            return int(os.environ["RANK"]) == 0
        except ValueError:
            pass
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

class Logger:
    def __init__(self):
        base = os.environ.get("PROJECT_ROOT", os.getcwd())
        self.cfg = None
        self.log_dir = Path(base) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(self.log_dir.as_posix())
        self.is_main_process = is_main_process()

    def setup(self, cfg):
        self.cfg = cfg
        out_dir = getattr(cfg, "output_dir", None)
        new_dir = Path(out_dir) / "logs" if out_dir else self.log_dir
        if new_dir != self.log_dir:
            try:
                self.tb.close()
            except Exception:
                pass
            new_dir.mkdir(parents=True, exist_ok=True)
            self.tb = SummaryWriter(new_dir.as_posix())
            self.log_dir = new_dir
    
    def log_info(self, tag: str, log: str, break_section: bool = False, only_rank0: bool = True):
        if self.is_main_process or not only_rank0:
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

logger = Logger()