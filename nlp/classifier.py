"""Artifact-backed inference with the notebook's preprocessing and calibration."""
from __future__ import annotations
import json
import threading
import time
from pathlib import Path
from .artifacts import file_sha256, resolve_active, validate_metadata
from .calibration import calibrated_prediction
from .head_tail import encode_head_tail
from .preprocessing import prepare_text


class ArtifactBackedClassifier:
    def __init__(self, artifact_dir: Path, model_name: str = "vinai/phobert-base-v2", auto_approve_threshold: float = .75, review_threshold: float = .68):
        self._active_path = Path(artifact_dir).resolve()
        self._lock = threading.RLock()
        self._signature = None
        self._snapshot = None
        self._load_artifacts()

    def _manifest_signature(self):
        manifest = self._active_path if self._active_path.suffix == ".json" else self._active_path / "active.json"
        return manifest.read_bytes() if manifest.is_file() else None

    def _load_artifacts(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        directory, run_id, manifest = resolve_active(self._active_path)
        classes, temperature, thresholds = validate_metadata(directory)
        if manifest.get("model_sha256") and file_sha256(directory / "model.safetensors") != manifest["model_sha256"]:
            raise ValueError("Active model checksum does not match manifest")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        model, loading = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Model weights do not completely match the declared architecture")
        model = model.to(device).eval()
        # Publish a complete snapshot. A running request holds its own reference.
        self._snapshot = (torch, tokenizer, model, device, classes, temperature, thresholds, run_id)
        self._signature = self._manifest_signature()

    def predict(self, request: dict) -> dict:
        started = time.perf_counter()
        with self._lock:
            if self._manifest_signature() != self._signature:
                self._load_artifacts()
            snapshot = self._snapshot
        torch, tokenizer, model, device, classes, temperature, thresholds, run_id = snapshot
        text = prepare_text(request.get("title", ""), request.get("content", ""))
        if not text:
            raise ValueError("Article has no usable text after preprocessing")
        encoded = encode_head_tail(text, tokenizer)
        inputs = {key: torch.tensor([value], dtype=torch.long, device=device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float().cpu().numpy()
        result = calibrated_prediction(logits, classes, temperature, thresholds, request.get("top_k", 3))
        result.update({
            "request_id": request.get("request_id", ""),
            "model_version": run_id,
            # Editorial routing uses the API's database policy.
            "auto_decision": "",
            "rationale_keywords": [],
            "latency_ms": int((time.perf_counter() - started) * 1000),
        })
        return result
