from __future__ import annotations

import os
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, List

from transformers import (
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.trainer_callback import TrainerCallback

from ..interfaces.protocol import TrainerProtocol, DataProtocol
from ..core.logger import get_logger, Logger
from ..models.builder import build_model
from ..data.builder import build_data

# ---------------- Trainer ----------------
class HFTrainer(TrainerProtocol):
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = get_logger(self.cfg)

        self.logger.log_info("HFTrainer", "Initializing Hugging Face Trainer.", break_section=True)
        self.logger.log_info("HFTrainer/Cfg", str(asdict(self.cfg)) if is_dataclass(self.cfg) else str(self.cfg))

        self.model = build_model(self.cfg)
        self.data = build_data(self.cfg)
        self.data_collator = self.data.get_collator()
        self.train_dataset = self.data.get_train_dataset()
        self.eval_dataset = self.data.get_eval_dataset()
        self.compute_metrics = _build_metrics(self.cfg.train.metrics, self.logger)
        self.callbacks: Optional[list[TrainerCallback]] = None
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
        eval_strategy = g("evaluation_strategy", "epoch") # "no" | "steps" | "epoch"
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

            # Precision / speed
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=grad_ckpt,

            # Evaluation / saving / logging
            eval_strategy=eval_strategy,
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

# ---------------- Metrics ----------------
def _build_metrics(metrics: List[str], logger: Logger) -> Optional[Any]:
    metrics_dict = {}
    for m in metrics:
        if m in ["accuracy"]:
            from sklearn.metrics import accuracy_score
            def compute(p: EvalPrediction) -> Dict[str, float]:
                preds = p.predictions.argmax(-1)
                return {"accuracy": accuracy_score(p.label_ids, preds)}
            metrics_dict.update({"accuracy": compute})
        elif m in ["f1", "f1_score"]:
            from sklearn.metrics import f1_score
            def compute(p: EvalPrediction) -> Dict[str, float]:
                preds = p.predictions.argmax(-1)
                return {"f1": f1_score(p.label_ids, preds, average="weighted")}
            metrics_dict.update({"f1": compute})
        elif m in ["precision"]:
            from sklearn.metrics import precision_score
            def compute(p: EvalPrediction) -> Dict[str, float]:
                preds = p.predictions.argmax(-1)
                return {"precision": precision_score(p.label_ids, preds, average="weighted")}
            metrics_dict.update({"precision": compute})
        elif m in ["recall"]:
            from sklearn.metrics import recall_score
            def compute(p: EvalPrediction) -> Dict[str, float]:
                preds = p.predictions.argmax(-1)
                return {"recall": recall_score(p.label_ids, preds, average="weighted")}
            metrics_dict.update({"recall": compute})
        elif m in ["mse"]:
            from sklearn.metrics import mean_squared_error
            def compute(p: EvalPrediction) -> Dict[str, float]:
                preds = p.predictions.squeeze()
                return {"mse": mean_squared_error(p.label_ids, preds)}
            metrics_dict.update({"mse": compute})
        else:
            raise ValueError(f"Unknown metric: {m}")
            
    def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        returning_dict = {}
        for m, func in metrics_dict.items():
            try:
                returning_dict.update(func(p))
            except Exception as e:
                logger.log_info(f"HFTrainer/Metrics/{m}", f"Error computing metric: {e}")
        return returning_dict
    return compute_metrics