from __future__ import annotations

from copy import deepcopy


from nlp.artifacts import CANONICAL_CLASSES
LABELS = list(CANONICAL_CLASSES)


def create_seed_state() -> dict:
    state = {
        "active_model": "No active model package",
        "thresholds": {
            "auto_approve": 0.75,
            "review_floor": 0.68,
        },
        "editor_articles": [],
        "admin_ops": {
            "users": [
                {"name": "Editor Admin", "role": "Editor", "queue": "All queues", "status": "Active"},
                {"name": "System Admin", "role": "Admin", "queue": "All queues", "status": "Active"},
                {"name": "Data Scientist", "role": "Data Scientist", "queue": "Monitoring", "status": "Active"},
                {"name": "Education Editor", "role": "Editor", "queue": "Education", "status": "Active"},
            ],
            "audit_log": [],
        },
        "model_versions": {
            "runs": [],
            "confusion_matrix": [],
            "package_details": [],
        },
        "dataset_lab": {
            "hard_samples": [],
            "priority_labels": [],
        },
    }

    return deepcopy(state)
