"""Resolve immutable model packages using a portable active-model manifest."""
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("VNN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
from nlp.taxonomy import CANONICAL_CLASSES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_active(path: Path) -> tuple[Path, str, dict]:
    path = Path(path)
    manifest_path = path if path.suffix == ".json" else path / "active.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        relative = Path(manifest["artifact_path"])
        if relative.is_absolute():
            raise ValueError("Artifact path must be relative to the project")
        directory = (PROJECT_ROOT / relative).resolve()
        if not directory.is_relative_to(PROJECT_ROOT):
            raise ValueError("Artifact path leaves the project")
        return directory, str(manifest["run_id"]), manifest
    if path.is_dir() and (path / "config.json").is_file():
        return path.resolve(), "phobert_headtail_seed42", {}
    raise ValueError(f"No valid active-model manifest or model directory at {path}")


def validate_metadata(directory: Path) -> tuple[list[str], float, list[float]]:
    directory = Path(directory)
    for name in ("config.json", "label_config.json", "thresholds.json", "model.safetensors", "tokenizer_config.json", "bpe.codes", "vocab.txt"):
        if not (directory / name).is_file():
            raise ValueError(f"Missing model file: {name}")
    labels = json.loads((directory / "label_config.json").read_text(encoding="utf-8"))
    model = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads((directory / "thresholds.json").read_text(encoding="utf-8"))
    count = len(CANONICAL_CLASSES)
    if set(labels.get('id2label',{})) != {str(i) for i in range(count)} or set(model.get('id2label',{})) != {str(i) for i in range(count)}:
        raise ValueError('Model label IDs must cover the current taxonomy exactly')
    classes = [labels["id2label"][str(i)] for i in range(count)]
    if classes != CANONICAL_CLASSES or classes != [model["id2label"][str(i)] for i in range(count)]:
        raise ValueError("Model and label taxonomy do not match")
    expected_mapping = {label: i for i, label in enumerate(classes)}
    if labels.get("classes") != classes or labels.get("n_classes") != count or labels.get("label2id") != expected_mapping or model.get("label2id") != expected_mapping:
        raise ValueError("Label mapping must be complete and ordered consistently")
    if labels.get("max_length") != 256 or labels.get("strategy") != "head_tail":
        raise ValueError("Checkpoint requires head_tail with max_length 256")
    required = {"title_weight": 1, "lowercase": True, "remove_punct": True, "remove_digits": True, "tokenizer": "pyvi.ViTokenizer", "stopwords": False}
    if any(labels.get("preprocessing", {}).get(k) != v for k, v in required.items()):
        raise ValueError("Unsupported preprocessing configuration")
    if set(calibration.get('thresholds',{})) != set(classes):
        raise ValueError('Calibration thresholds must cover the current taxonomy exactly')
    temperature = float(calibration["temperature"])
    thresholds = [float(calibration["thresholds"][c]) for c in classes]
    import math
    if not all(math.isfinite(v) and v > 0 for v in [temperature, *thresholds]):
        raise ValueError("Invalid temperature or class thresholds")
    if (directory / "model.safetensors").stat().st_size < 1_000_000:
        raise ValueError("Model weights are incomplete")
    return classes, temperature, thresholds
