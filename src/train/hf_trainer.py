from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, List

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_callback import TrainerCallback
import time

from ..core.protocol import TrainerProtocol, DataProtocol
from ..core.logger import logger
from ..models.builder import build_model
from ..data.builder import build_data
from ..core.metrics import build_metrics


# ---------------- Trainer ----------------
class HFTrainer(TrainerProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        # Use module-level logger and reconfigure it with cfg
        self.logger = logger
        self.logger.setup(self.cfg)

        self.logger.log_info("HFTrainer", "Initializing Hugging Face Trainer.")
        cfg_str = str(asdict(self.cfg)) if (is_dataclass(self.cfg) and not isinstance(self.cfg, type)) else str(self.cfg)
        self.logger.log_info("HFTrainer/Cfg", cfg_str)

        self.model = build_model(self.cfg)
        self.data = build_data(self.cfg)
        self.data_collator = self.data.get_collator()
        self.train_dataset = self.data.get_train_dataset()
        self.eval_dataset = self.data.get_eval_dataset()
        train_section = getattr(self.cfg, "train", self.cfg)
        metrics_list = getattr(train_section, "metrics", [])
        self.compute_metrics = build_metrics(metrics_list, self.logger)
        self.callbacks: Optional[list[TrainerCallback]] = _build_callbacks(self.cfg)
        self.tokenizer = None

        self.args = self._build_training_args()
        self.trainer: Optional[Trainer] = None

    def fit(self) -> None:
        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        if getattr(self.cfg, "verbose", False):
            self.logger.log_info("HFTrainer/Args", str(self.args))

        resume = getattr(self.cfg.train, "resume_from_checkpoint", None)
        self.logger.log_info("HFTrainer", "Starting training...")
        self.trainer.train(resume_from_checkpoint=resume)

        self.trainer.save_state()
        self.trainer.save_model(self.args.output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.args.output_dir)

        self.logger.log_info("HFTrainer", f"Training complete. Artifacts saved to: {self.args.output_dir}")
        self.logger.close()


    def _build_training_args(self) -> TrainingArguments:
        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logging_dir = Path(self.cfg.output_dir) / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.log_info("HFTrainer", f"Output directory set to: {out_dir}")
        self.logger.log_info("HFTrainer", f"Logging directory set to: {logging_dir}")
        
        def g(key: str, default: Any = None):
            train_cfg = self._to_plain_dict(self.cfg.train)
            return train_cfg.get(key, default)

        def gs(key: str, default: str) -> str:
            v = g(key, default)
            return str(v) if v is not None else str(default)

        def gi(key: str, default: int) -> int:
            v = g(key, default)
            try:
                return int(v) if v is not None else int(default)
            except Exception:
                return int(default)

        def gi_opt(key: str) -> Optional[int]:
            v = g(key, None)
            try:
                return int(v) if v is not None else None
            except Exception:
                return None

        def gf(key: str, default: float) -> float:
            v = g(key, default)
            try:
                return float(v) if v is not None else float(default)
            except Exception:
                return float(default)

        def gb(key: str, default: bool) -> bool:
            v = g(key, default)
            return bool(v) if v is not None else bool(default)

        backend = g("backend", "single")  # single | ddp | deepspeed | fsdp
        deepspeed_config = g("deepspeed_config", None) if backend == "deepspeed" else None
        fsdp_policy = g("fsdp", "") if backend == "fsdp" else ""

        # Precision
        fp16 = bool(g("fp16", False))
        bf16 = bool(g("bf16", False))

        # DDP niceties: only set when using DDP/FSDP to avoid warnings
        # Sample script sets this True for unused SigLIP layers
        ddp_find_unused = (
            bool(g("ddp_find_unused_parameters", True)) if backend in {"ddp", "fsdp"} else None
        )
        ddp_bucket_cap_mb = gi_opt("ddp_bucket_cap_mb")

        # Logging & evaluation strategies
        # Sample script doesn't evaluate during training
        eval_strategy = gs("eval_strategy", "no") # "no" | "steps" | "epoch"
        eval_steps = gi_opt("eval_steps")
        save_strategy = gs("save_strategy", "no")
        log_strategy = gs("logging_strategy", "steps")
        log_steps = gi("logging_steps", 10)

        # Gradient checkpointing
        grad_ckpt = bool(g("gradient_checkpointing", False))
        grad_ckpt_kwargs = g("gradient_checkpointing_kwargs", {"use_reentrant": False})

        # Max steps (optional override)
        max_steps = gi("max_steps", -1)

        args = TrainingArguments(
            output_dir=str(out_dir),
            overwrite_output_dir=gb("overwrite_output_dir", True),
            num_train_epochs=gf("epochs", 3.0),
            per_device_train_batch_size=gi("batch_size", 8),
            per_device_eval_batch_size=gi("eval_batch_size", gi("batch_size", 8)),
            gradient_accumulation_steps=gi("grad_accum_steps", 1),
            learning_rate=gf("lr", 5e-5),
            weight_decay=gf("weight_decay", 0.01),
            max_grad_norm=gf("max_grad_norm", 1.0),
            warmup_steps=gi("warmup_steps", 50),
            lr_scheduler_type=gs("lr_scheduler_type", "linear"),
            optim=gs("optim", "adamw_torch"),
            adam_beta1=gf("adam_beta1", 0.9),
            adam_beta2=gf("adam_beta2", 0.95),
            adam_epsilon=gf("adam_epsilon", 1e-7),
            
            dataloader_drop_last=gb("dataloader_drop_last", False),
            dataloader_pin_memory=gb("dataloader_pin_memory", True),

            # Precision / speed
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs=grad_ckpt_kwargs,

            # Evaluation / saving / logging
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_total_limit=gi("save_total_limit", 10),
            save_only_model=gb("save_only_model", True),
            logging_strategy=log_strategy,
            logging_steps=log_steps,
            logging_dir=os.path.join(logging_dir, "tb"),
            report_to=g("report_to", "none"),   # 'none' to match sample script
            disable_tqdm=gb("disable_tqdm", False),

            # Distributed backends
            deepspeed=deepspeed_config,   # path to ds config json OR dict
            fsdp=fsdp_policy,             # e.g. "full_shard auto_wrap"
            ddp_find_unused_parameters=ddp_find_unused,
            ddp_bucket_cap_mb=ddp_bucket_cap_mb,

            # Misc
            dataloader_num_workers=gi("num_workers", 4),
            remove_unused_columns=gb("remove_unused_columns", False),
            seed=gi("seed", 42),
            load_best_model_at_end=gb("load_best_model_at_end", False),
            metric_for_best_model=g("metric_for_best_model", None),
            greater_is_better=g("greater_is_better", None),
            max_steps=int(max_steps) if max_steps is not None else -1,

            # Resume
            resume_from_checkpoint=g("resume_from_checkpoint", None),
        )

        backend_note = "deepspeed" if deepspeed_config else ("fsdp" if fsdp_policy else ("ddp" if backend == "ddp" else "single/auto"))
        self.logger.log_info("HFTrainer", f"TrainingArguments ready (backend={backend_note}).")
        return args

    @staticmethod
    def _to_plain_dict(obj: Any) -> Dict[str, Any]:
        """Support dataclass or dict-like cfg.train."""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)  # type: ignore[arg-type]
        if isinstance(obj, dict):
            return obj
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}
    
# ---------------- Callbacks ----------------
class TimingCallback(TrainerCallback):
    def __init__(self):
        self.last_time = time.time()

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.last_time = time.time()

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        dt = time.time() - self.last_time
        logger.log_info("HFTrainer/Time", f"Step {state.global_step} took {dt:.4f} sec")

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.log_info("HFTrainer/Time", f"Epoch {state.epoch} finished at {time.time():.2f}")

    def on_evaluate(self, args, state, control, **kwargs):
        logger.log_info("HFTrainer/Time", f"Evaluation at step {state.global_step} started {time.ctime()}")

def _build_callbacks(cfg) -> Optional[list[TrainerCallback]]:
    callbacks = []
    for cb in cfg.train.get("callbacks", []):
        if cb == "timing":
            callbacks.append(TimingCallback())
        else:
            raise ValueError(f"Unknown callback: {cb}")
    return callbacks if callbacks else None