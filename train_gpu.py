from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from dataset import export_dataset, iter_examples
from guards import relabel_outputs_with_guards
from local_model import FEATURE_NAMES, MODEL_PATH, _flags, vector
from project_state import OUT, SOURCE_JOB, load_manifest
from train_local import RESOLVED, audit_labels, group
from training_report import write_training_report

ROOT = Path(__file__).resolve().parent
SEED = 42
GPU_DEVICE = "0"


def _selected_rows() -> tuple[list[dict], dict]:
    export_dataset(OUT)
    manifest = load_manifest()
    allowed = set(int(n) for n in manifest["training_sheets"])
    rows = [
        row
        for row in iter_examples(OUT)
        if row.get("folder") == SOURCE_JOB
        and int(row.get("sheet_no") or 0) in allowed
        and row.get("decision") in RESOLVED
        and row.get("value_mm") is not None
    ]
    return rows, manifest


def _sample_weight(row: dict) -> float:
    """Штраф за дорогие ошибки: ложный include габарита/вложенного/штурвала ломает сумму."""
    flags = _flags(row)
    value = int(row.get("value_mm") or 0)
    decision = str(row.get("decision") or "")
    weight = 1.0
    if row.get("locked") or row.get("reviewed_by_human"):
        weight *= 8.0
    if "overall_mark" in flags or "overall_pair" in flags:
        weight *= 5.0
    if "nested_offset" in flags:
        weight *= 4.0
    if "handwheel" in flags or "insulation" in flags:
        weight *= 3.5
    if "support_repeat" in flags:
        weight *= 2.5
    nearby = str(row.get("nearby") or "")
    nums = [int(n) for n in nearby.replace("|", " ").split() if n.isdigit()]
    larger = [n for n in nums if n > value > 0]
    if larger and 0.12 <= value / max(larger) <= 0.55:
        weight *= 3.0
    if decision == "exclude_nested":
        weight *= 2.2
    elif decision == "not_length":
        weight *= 1.6
    if value >= 2000:
        weight *= 1.8
    if value <= 25:
        weight *= 2.0
    return float(weight)


def _xy(rows: list[dict]):
    X = np.asarray([vector(row) for row in rows], dtype=np.float32)
    y = np.asarray([str(row["decision"]) for row in rows])
    groups = np.asarray([group(row) for row in rows])
    w = np.asarray([_sample_weight(row) for row in rows], dtype=np.float32)
    return X, y, groups, w


def _model(iterations: int = 1200) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=12.0,
        random_strength=1.2,
        bagging_temperature=0.7,
        min_data_in_leaf=20,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        class_weights={
            "include": 1.2,
            "exclude_nested": 2.2,
            "not_length": 1.8,
        },
        task_type="GPU",
        devices=GPU_DEVICE,
        random_seed=SEED,
        verbose=50,
        allow_writing_files=False,
        use_best_model=True,
        od_type="Iter",
        od_wait=100,
    )


def _metrics(rows: list[dict], pred: np.ndarray) -> dict:
    _, y, groups, _ = _xy(rows)
    values = np.asarray([int(r["value_mm"]) for r in rows])
    errors = []
    for sheet in sorted(set(groups)):
        at = groups == sheet
        true_sum = int(values[at & (y == "include")].sum())
        pred_sum = int(values[at & (pred == "include")].sum())
        errors.append(abs(pred_sum - true_sum))
    return {
        "rows": len(rows),
        "sheets": len(set(groups)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=RESOLVED, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, pred, labels=RESOLVED, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y, pred, labels=RESOLVED, output_dict=True, zero_division=0
        ),
        "confusion_labels": list(RESOLVED),
        "confusion_matrix": confusion_matrix(y, pred, labels=RESOLVED).tolist(),
        "sheet_sum_mae_mm": float(np.mean(errors)),
        "sheet_sum_median_ae_mm": float(np.median(errors)),
        "sheet_sum_max_ae_mm": int(max(errors, default=0)),
    }


def predict_with_abstain(model: CatBoostClassifier, X: np.ndarray, threshold: float) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    classes = np.asarray(model.classes_)
    indices = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(X)), indices]
    labels = classes[indices].astype(object)
    labels[confidence < threshold] = "ambiguous"
    return labels.astype(str)


def _choose_threshold(model: CatBoostClassifier, rows: list[dict]) -> tuple[float, list[dict]]:
    X, y, _, _ = _xy(rows)
    trials = []
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        pred = predict_with_abstain(model, X, threshold)
        accepted = pred != "ambiguous"
        trials.append(
            {
                "threshold": threshold,
                "coverage": float(np.mean(accepted)),
                "accuracy_accepted": (
                    float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else 0.0
                ),
                "macro_f1_with_abstain": float(
                    f1_score(y, pred, labels=RESOLVED, average="macro", zero_division=0)
                ),
            }
        )
    eligible = [t for t in trials if t["accuracy_accepted"] >= 0.90]
    best = max(eligible or trials, key=lambda t: (t["macro_f1_with_abstain"], t["coverage"]))
    return float(best["threshold"]), trials


def _holdout_check(rows: list[dict], manifest: dict) -> dict:
    used = {int(r.get("sheet_no") or 0) for r in rows}
    holdout = manifest.get("holdout_test_pages") or {}
    tests = {name: {int(n) for n in pages} for name, pages in holdout.items()}
    leak_tests = {name: sorted(used & pages) for name, pages in tests.items() if used & pages}
    if leak_tests:
        raise RuntimeError(f"Утечка holdout в train: tests={leak_tests}")
    return {
        "ok": True,
        "train_sheets": len(used),
        "excluded_tests": {k: sorted(v) for k, v in tests.items()},
        "overlap_tests": {},
    }


def train() -> dict:
    audit = audit_labels(OUT)
    relabeled = relabel_outputs_with_guards(OUT)
    rows, manifest = _selected_rows()
    if len(rows) < 100:
        n_raw = len(list((OUT / SOURCE_JOB / "llm_raw").glob("sheet_*.json"))) if (OUT / SOURCE_JOB / "llm_raw").exists() else 0
        if MODEL_PATH.exists() and n_raw >= 100:
            raise RuntimeError(
                f"Не собрались примеры для обучения (rows={len(rows)}, json={n_raw}), "
                f"но модель уже есть: {MODEL_PATH.name}. Можно сразу идти в Расчёт."
            )
        raise RuntimeError(
            f"Нет данных для обучения (rows={len(rows)}, json-листов={n_raw}). "
            "Дождись окончания prepare, потом снова Обучить."
        )
    holdout = _holdout_check(rows, manifest)

    groups = np.asarray([group(row) for row in rows])
    idx = np.arange(len(rows))
    split = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=SEED)
    train_idx, val_idx = next(split.split(idx, groups=groups))
    train_rows = [rows[int(i)] for i in train_idx]
    val_rows = [rows[int(i)] for i in val_idx]

    X_train, y_train, _, w_train = _xy(train_rows)
    X_val, y_val, _, w_val = _xy(val_rows)
    model = _model()
    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
    )
    best_iter = int(model.get_best_iteration() or model.tree_count_ or 400)
    threshold, threshold_trials = _choose_threshold(model, val_rows)
    train_pred = predict_with_abstain(model, X_train, threshold)
    val_pred = predict_with_abstain(model, X_val, threshold)
    train_metrics = _metrics(train_rows, train_pred)
    val_metrics = _metrics(val_rows, val_pred)
    overfit_gap = float(train_metrics["macro_f1"] - val_metrics["macro_f1"])
    # Не доучиваем на всём пуле: берём модель, остановленную по validation.

    holdout_pages = manifest.get("holdout_test_pages") or {}
    report = {
        "model_kind": "catboost_gpu_v1",
        "device": "GPU 0 (NVIDIA CUDA)",
        "features": list(FEATURE_NAMES),
        "data_source": SOURCE_JOB,
        "relabeled_sheets": relabeled,
        "excluded_test_pages": holdout_pages,
        "holdout_check": holdout,
        "best_iteration": best_iter,
        "hyperparams": {
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 12.0,
            "min_data_in_leaf": 20,
            "class_weights": {"include": 1.2, "exclude_nested": 2.2, "not_length": 1.8},
            "early_stopping_rounds": 100,
        },
        "overfit_gap_macro_f1": overfit_gap,
        "overfit_ok": overfit_gap < 0.05,
        "train": {
            **train_metrics,
            "class_counts": dict(Counter(r["decision"] for r in train_rows)),
            "kind": "train",
        },
        "validation": {
            **val_metrics,
            "kind": "validation",
        },
        "threshold": threshold,
        "threshold_trials": threshold_trials,
        "audit": {
            "sheets": audit["sheets"],
            "candidates": audit["candidates"],
            "problems_count": audit["problems_count"],
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "kind": "catboost_gpu_v1",
                "model": model,
                "threshold": threshold,
                "features": FEATURE_NAMES,
                "report": report,
            },
            f,
        )
    report_path = MODEL_PATH.parent / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_training_report(report)
    return report


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, ensure_ascii=False, indent=2))
