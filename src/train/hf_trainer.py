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
from ..core.logger import get_logger, Logger
from ..models.builder import build_model
from ..data.builder import build_data
from ..core.metrics import build_metrics


# ---------------- Trainer ----------------
class HFTrainer(TrainerProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger(self.cfg)

        self.logger.log_info("HFTrainer", "Initializing Hugging Face Trainer.", break_section=True)
        self.logger.log_info("HFTrainer/Cfg", str(asdict(self.cfg)) if is_dataclass(self.cfg) else str(self.cfg))

        self.model = build_model(self.cfg)
        self.data: Any = build_data(self.cfg)
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
        out_dir = Path(self.cfg.output_dir) / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        logging_dir = Path(self.cfg.output_dir) / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.log_info("HFTrainer", f"Output directory set to: {out_dir}")
        self.logger.log_info("HFTrainer", f"Logging directory set to: {logging_dir}")
        
        def g(key: str, default: Any = None):
            train_cfg = self._to_plain_dict(self.cfg.train)
            return train_cfg.get(key, default)

        backend = g("backend", "single")  # single | ddp | deepspeed | fsdp
        deepspeed_config = g("deepspeed_config", None) if backend == "deepspeed" else None
        fsdp_policy = g("fsdp", "") if backend == "fsdp" else ""

        # Precision
        fp16 = bool(g("fp16", False))
        bf16 = bool(g("bf16", False))

        # DDP niceties: only set when using DDP/FSDP to avoid warnings
        ddp_find_unused = False if backend in {"ddp", "fsdp"} else None
        ddp_bucket_cap_mb = g("ddp_bucket_cap_mb", None)

        # Logging & evaluation strategies
        eval_strategy = g("eval_strategy", "epoch") # "no" | "steps" | "epoch"
        eval_steps = g("eval_steps", None)
        save_strategy = g("save_strategy", "epoch")
        log_strategy = g("logging_strategy", "epoch")
        log_steps = g("logging_steps", 50)

        # Gradient checkpointing
        grad_ckpt = bool(g("gradient_checkpointing", False))

        # Max steps (optional override)
        max_steps = g("max_steps", -1)

        args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=bool(g("overwrite_output_dir", True)),
            num_train_epochs=float(g("epochs", 3)),
            per_device_train_batch_size=int(g("batch_size", 8)),
            per_device_eval_batch_size=int(g("eval_batch_size", g("batch_size", 8))),
            gradient_accumulation_steps=int(g("grad_accum_steps", 1)),
            learning_rate=float(g("lr", 5e-5)),
            weight_decay=float(g("weight_decay", 0.0)),
            max_grad_norm=float(g("max_grad_norm", 1.0)),
            warmup_steps=int(g("warmup_steps", 0)),
            lr_scheduler_type=g("lr_scheduler_type", "linear"),
            
            dataloader_drop_last=bool(g("dataloader_drop_last", False)),
            dataloader_pin_memory=bool(g("dataloader_pin_memory", True)),

            # Precision / speed
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=grad_ckpt,

            # Evaluation / saving / logging
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_total_limit=g("save_total_limit", 3),
            logging_strategy=log_strategy,
            logging_steps=log_steps,
            logging_dir=os.path.join(logging_dir, "tb"),
            report_to=g("report_to", ["tensorboard"]),   # wandb/tensorboard/none

            # Distributed backends
            deepspeed=deepspeed_config,   # path to ds config json OR dict
            fsdp=fsdp_policy,             # e.g. "full_shard auto_wrap"
            ddp_find_unused_parameters=ddp_find_unused,
            ddp_bucket_cap_mb=ddp_bucket_cap_mb,

            # Misc
            dataloader_num_workers=int(g("num_workers", 4)),
            remove_unused_columns=bool(g("remove_unused_columns", False)),
            seed=int(g("seed", 42)),
            load_best_model_at_end=bool(g("load_best_model_at_end", False)),
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
        if is_dataclass(obj):
            return asdict(obj)
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
        print(f"[TIME] Step {state.global_step} took {dt:.4f} sec")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[TIME] Epoch {state.epoch} finished at {time.time():.2f}")

    def on_evaluate(self, args, state, control, **kwargs):
        print(f"[TIME] Evaluation at step {state.global_step} started {time.ctime()}")

def _build_callbacks(cfg) -> Optional[list[TrainerCallback]]:
    callbacks = []
    for cb in cfg.train.get("callbacks", []):
        if cb == "timing":
            callbacks.append(TimingCallback())
        else:
            raise ValueError(f"Unknown callback: {cb}")
    return callbacks if callbacks else None