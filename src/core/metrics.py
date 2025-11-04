from __future__ import annotations

from typing import Any, Dict, List, Optional

# transformers' EvalPrediction is optional; we type as Any to avoid hard dependency
EvalPrediction = Any


# Note: We depend only on scikit-learn metrics. To avoid importing unused
# metrics eagerly, we import them inside each metric builder branch.


def build_metrics(metrics: List[str], logger) -> Optional[Any]:
	"""
	Build a Hugging Face-compatible compute_metrics callable from a list of metric names.

	Inputs
	- metrics: list of metric identifiers, e.g. ["accuracy", "f1", "precision", "recall", "mse", "auc", "cm"].
	- logger: object exposing logger.log_info(tag, message) used to report metric errors.

	Output
	- A function compute_metrics(p: EvalPrediction) -> Dict[str, float] suitable for HF Trainer,
	  or None if metrics is empty.

	Behavior mirrors the previous inline implementation in `hf_trainer.py` so existing configs
	continue to work unchanged.
	"""

	if not metrics:
		return None

	metrics_dict: Dict[str, Any] = {}

	for m in metrics:
		key = m.lower()

		if key in ["accuracy"]:
			from sklearn.metrics import accuracy_score

			def _compute_accuracy(p: Any) -> Dict[str, float]:
				preds = p.predictions.argmax(-1)
				return {"accuracy": float(accuracy_score(p.label_ids, preds))}

			metrics_dict["accuracy"] = _compute_accuracy

		elif key in ["f1", "f1_score"]:
			from sklearn.metrics import f1_score

			def _compute_f1(p: Any) -> Dict[str, float]:
				preds = p.predictions.argmax(-1)
				return {"f1": float(f1_score(p.label_ids, preds, average="weighted"))}

			metrics_dict["f1"] = _compute_f1

		elif key in ["precision"]:
			from sklearn.metrics import precision_score

			def _compute_precision(p: Any) -> Dict[str, float]:
				preds = p.predictions.argmax(-1)
				return {"precision": float(precision_score(p.label_ids, preds, average="weighted"))}

			metrics_dict["precision"] = _compute_precision

		elif key in ["recall"]:
			from sklearn.metrics import recall_score

			def _compute_recall(p: Any) -> Dict[str, float]:
				preds = p.predictions.argmax(-1)
				return {"recall": float(recall_score(p.label_ids, preds, average="weighted"))}

			metrics_dict["recall"] = _compute_recall

		elif key in ["mse"]:
			from sklearn.metrics import mean_squared_error

			def _compute_mse(p: Any) -> Dict[str, float]:
				preds = p.predictions.squeeze()
				return {"mse": float(mean_squared_error(p.label_ids, preds))}

			metrics_dict["mse"] = _compute_mse

		elif key in ["auc", "roc_auc"]:
			from sklearn.metrics import roc_auc_score

			def _compute_auc(p: Any) -> Dict[str, float]:
				# Mirrors original behavior: uses class argmax labels as scores
				preds = p.predictions.argmax(-1)
				return {"auc": float(roc_auc_score(p.label_ids, preds))}

			metrics_dict["auc"] = _compute_auc

		elif key in ["cm", "confusion_matrix"]:
			from sklearn.metrics import confusion_matrix

			def _compute_cm(p: Any) -> Dict[str, float]:
				preds = p.predictions.argmax(-1)
				cm = confusion_matrix(p.label_ids, preds)
				results = {}
				for i in range(cm.shape[0]):
					for j in range(cm.shape[1]):
						results[f"cm_{i}_{j}"] = float(cm[i, j])

				return results

			metrics_dict["cm"] = _compute_cm

		else:
			raise ValueError(f"Unknown metric: {m}")

	def compute_metrics(p: Any) -> Dict[str, float]:
		results: Dict[str, float] = {}
		for name, func in metrics_dict.items():
			try:
				results.update(func(p))
			except Exception as e:  # robust metric computation
				# Expect logger.log_info(tag, message)
				try:
					logger.log_info(f"HFTrainer/Metrics/{name}", f"Error computing metric: {e}")
				except Exception:
					# Fallback if logger signature differs
					pass
		return results

	return compute_metrics

