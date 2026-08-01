"""
Lightweight prediction logger. Appends each prediction as a JSON line
to a log file, and provides an aggregation function for the dashboard.

This is intentionally simple (a JSON-lines file, not a database) since
the goal is to demonstrate observability/logging practices without
adding infrastructure overhead to a portfolio project.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Use abspath so this works correctly on Windows regardless of working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
LOG_PATH = os.path.join(_PROJECT_ROOT, "checkpoints", "predictions_log.jsonl")

logger.info(f"[prediction_logger] Log path: {LOG_PATH}")

_lock = threading.Lock()


def log_prediction(pred_class: str, confidence: float, uncertain: bool, source: str = "predict"):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class": pred_class,
        "confidence": confidence,
        "uncertain": uncertain,
        "source": source,
    }
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with _lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"[prediction_logger] Failed to write log: {e}")


def read_all_predictions():
    if not os.path.exists(LOG_PATH):
        return []
    records = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_stats():
    """Aggregates the full prediction log into dashboard-ready stats."""
    records = read_all_predictions()

    total = len(records)
    if total == 0:
        return {
            "total": 0,
            "class_counts": {"cat": 0, "dog": 0},
            "avg_confidence": {"cat": 0, "dog": 0, "overall": 0},
            "uncertain_count": 0,
            "uncertain_rate": 0,
            "confidence_histogram": [0] * 10,  # 10 bins, 0-10%, 10-20%, ... 90-100%
            "recent": [],
        }

    class_counts = {"cat": 0, "dog": 0}
    conf_sums = {"cat": 0.0, "dog": 0.0}
    uncertain_count = 0
    histogram = [0] * 10
    overall_conf_sum = 0.0

    for r in records:
        cls = r.get("class")
        conf = r.get("confidence", 0)
        if cls in class_counts:
            class_counts[cls] += 1
            conf_sums[cls] += conf
        if r.get("uncertain"):
            uncertain_count += 1
        overall_conf_sum += conf

        bin_idx = min(int(conf * 10), 9)
        histogram[bin_idx] += 1

    avg_confidence = {
        "cat": round(conf_sums["cat"] / class_counts["cat"], 4) if class_counts["cat"] else 0,
        "dog": round(conf_sums["dog"] / class_counts["dog"], 4) if class_counts["dog"] else 0,
        "overall": round(overall_conf_sum / total, 4),
    }

    recent = list(reversed(records))[:20]  # most recent 20, newest first

    return {
        "total": total,
        "class_counts": class_counts,
        "avg_confidence": avg_confidence,
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_count / total, 4),
        "confidence_histogram": histogram,
        "recent": recent,
    }