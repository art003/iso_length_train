from __future__ import annotations

import argparse
import html
import json
import pickle
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import export_dataset, iter_examples
from local_model import FEATURE_NAMES, MODEL_PATH, vector

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MODELS = ROOT / "models"
RESOLVED = ("include", "exclude_nested", "not_length")
VALID = set(RESOLVED) | {"ambiguous"}
SEED = 42


def audit_labels(out_root: Path = OUT) -> dict:
    problems: list[str] = []
    jobs = sheets = candidates = locked_sheets = 0
    classes: Counter[str] = Counter()
    sources: dict[str, dict] = {}

    for job in sorted(out_root.iterdir()):
        metrics_path = job / "metrics.json"
        if not job.is_dir() or not metrics_path.exists():
            continue
        jobs += 1
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        source = sources.setdefault(job.name, {"sheets": 0, "candidates": 0, "locked": 0})
        for sheet in metrics.get("sheets") or []:
            sheets += 1
            source["sheets"] += 1
            n = int(sheet["sheet_no"])
            line_id = sheet.get("line_id") or ""
            raw_path = job / "llm_raw" / f"sheet_{n:02d}.json"
            hits_path = job / "overlays" / f"sheet_{n:02d}_{line_id}_hits.json"
            if not raw_path.exists():
                problems.append(f"{job.name}/{n}: нет llm_raw")
                continue
            if not hits_path.exists():
                matches = list((job / "overlays").glob(f"sheet_{n:02d}_*_hits.json"))
                if len(matches) == 1:
                    hits_path = matches[0]
                else:
                    problems.append(f"{job.name}/{n}: нет hits.json")
                    continue

            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            hits = json.loads(hits_path.read_text(encoding="utf-8"))
            items = raw.get("items") or []
            hit_items = hits.get("candidates") or []
            raw_ids = [str(x.get("id") or "") for x in items]
            hit_ids = [str(x.get("id") or "") for x in hit_items]
            if len(raw_ids) != len(set(raw_ids)):
                problems.append(f"{job.name}/{n}: дубли id в json")
            if len(hit_ids) != len(set(hit_ids)):
                problems.append(f"{job.name}/{n}: дубли id в hits")
            missing = sorted(set(hit_ids) - set(raw_ids))
            extra = sorted(set(raw_ids) - set(hit_ids))
            if missing:
                problems.append(f"{job.name}/{n}: нет решений {missing}")
            if extra:
                problems.append(f"{job.name}/{n}: лишние решения {extra}")
            for item in items:
                decision = str(item.get("decision") or "ambiguous")
                classes[decision] += 1
                if decision not in VALID:
                    problems.append(f"{job.name}/{n}/{item.get('id')}: класс {decision}")
            for hit in hit_items:
                if hit.get("value_mm") is None:
                    problems.append(f"{job.name}/{n}/{hit.get('id')}: нет value_mm")
            count = len(hit_items)
            candidates += count
            source["candidates"] += count
            if raw.get("locked"):
                locked_sheets += 1
                source["locked"] += 1

    report = {
        "jobs": jobs,
        "sheets": sheets,
        "candidates": candidates,
        "locked_sheets": locked_sheets,
        "classes": dict(classes),
        "sources": sources,
        "problems_count": len(problems),
        "problems": problems,
        "warning": (
            "Изометрии_без_API — псевдоразметка программы. "
            "Locked-листы проверены человеком частично, это не независимый эталон."
        ),
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "label_audit.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value['sheets']}</td>"
        f"<td>{value['candidates']}</td><td>{value['locked']}</td></tr>"
        for name, value in sources.items()
    )
    problems_html = "".join(
        f"<li>{html.escape(problem)}</li>" for problem in problems[:200]
    ) or "<li>Структурных ошибок не найдено</li>"
    (MODELS / "label_audit.html").write_text(
        f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Проверка разметки</title><style>
body{{font-family:Arial;margin:28px;color:#263746}} h1{{color:#004c97}}
.cards{{display:flex;gap:12px}} .card{{padding:14px 18px;background:#eef5fb;border-radius:6px}}
.card b{{display:block;font-size:24px}} table{{border-collapse:collapse;margin-top:18px}}
td,th{{border:1px solid #dce4eb;padding:8px 12px;text-align:left}} .warn{{color:#9a5d00}}
</style></head><body><h1>Проверка структуры разметки</h1>
<div class="cards"><div class="card"><b>{sheets}</b>листов</div>
<div class="card"><b>{candidates}</b>размеров</div>
<div class="card"><b>{len(problems)}</b>проблем</div></div>
<p class="warn">{html.escape(report['warning'])}</p>
<table><tr><th>Набор</th><th>Листов</th><th>Размеров</th><th>locked</th></tr>{rows}</table>
<h2>Проблемы</h2><ul>{problems_html}</ul></body></html>""",
        encoding="utf-8",
    )
    return report


def select_rows(out_root: Path = OUT) -> tuple[list[dict], int]:
    export_dataset(out_root)
    rows = [
        r
        for r in iter_examples(out_root)
        if r.get("id") and r.get("value_mm") is not None
    ]
    big = [r for r in rows if r.get("folder") == "Изометрии_без_API"]
    gold = [r for r in rows if r.get("locked")]
    if big:
        rows = big + [r for r in gold if r.get("folder") != "Изометрии_без_API"]
    elif len(gold) >= 30:
        rows = gold
    skipped = sum(1 for r in rows if r.get("decision") == "ambiguous")
    rows = [r for r in rows if r.get("decision") in RESOLVED]
    return rows, skipped


def group(row: dict) -> str:
    return f"{row.get('folder')}::{int(row.get('sheet_no') or 0)}"


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    groups = np.array([group(r) for r in rows])
    indices = np.arange(len(rows))
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    dev_idx, test_idx = next(first.split(indices, groups=groups))
    dev_groups = groups[dev_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED + 1)
    train_rel, val_rel = next(second.split(dev_idx, groups=dev_groups))
    return (
        [rows[int(dev_idx[i])] for i in train_rel],
        [rows[int(dev_idx[i])] for i in val_rel],
        [rows[int(i)] for i in test_idx],
    )


def xy(rows: list[dict]):
    X = np.asarray([vector(r) for r in rows], dtype=float)
    y = np.asarray([str(r["decision"]) for r in rows])
    groups = np.asarray([group(r) for r in rows])
    weights = np.asarray([3.0 if r.get("locked") else 1.0 for r in rows])
    return X, y, groups, weights


def linear_model(penalty: str, c: float) -> Pipeline:
    l1_ratio = 1.0 if penalty == "l1" else 0.0
    solver = "saga" if penalty == "l1" else "lbfgs"
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=l1_ratio,
                    C=c,
                    solver=solver,
                    class_weight="balanced",
                    max_iter=3000,
                    tol=1e-3,
                    random_state=SEED,
                ),
            ),
        ]
    )


def tree_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def fit_linear(model: Pipeline, X, y, weights):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y, model__sample_weight=weights)
    return model


def choose_linear(train_rows: list[dict]) -> tuple[str, float, list[dict]]:
    X, y, groups, weights = xy(train_rows)
    folds = min(5, len(set(groups)))
    splitter = GroupKFold(n_splits=folds)
    configs = [("l1", 0.3), ("l1", 1.0), ("l2", 0.3), ("l2", 1.0)]
    scores = []
    for penalty, c in configs:
        fold_scores = []
        converged = []
        for tr, va in splitter.split(X, y, groups):
            model = fit_linear(linear_model(penalty, c), X[tr], y[tr], weights[tr])
            pred = model.predict(X[va])
            fold_scores.append(float(f1_score(y[va], pred, average="macro", zero_division=0)))
            estimator = model.named_steps["model"]
            converged.append(bool(np.max(estimator.n_iter_) < estimator.max_iter))
        scores.append(
            {
                "penalty": penalty,
                "C": c,
                "macro_f1_mean": float(np.mean(fold_scores)),
                "macro_f1_std": float(np.std(fold_scores)),
                "folds": fold_scores,
                "all_folds_converged": all(converged),
            }
        )
    stable = [x for x in scores if x["all_folds_converged"]]
    best = max(stable or scores, key=lambda x: x["macro_f1_mean"])
    return str(best["penalty"]), float(best["C"]), scores


def cascade_predict(linear, trees, X, threshold: float) -> np.ndarray:
    lp = linear.predict_proba(X)
    tp = trees.predict_proba(X)
    lc = np.asarray(linear.classes_)
    tc = np.asarray(trees.classes_)
    out = []
    for i in range(len(X)):
        li = int(np.argmax(lp[i]))
        ti = int(np.argmax(tp[i]))
        ll, tl = str(lc[li]), str(tc[ti])
        confidence = min(float(lp[i][li]), float(tp[i][ti]))
        out.append(ll if ll == tl and confidence >= threshold else "ambiguous")
    return np.asarray(out)


def threshold_on_validation(linear, trees, X, y) -> tuple[float, list[dict]]:
    trials = []
    for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        pred = cascade_predict(linear, trees, X, threshold)
        accepted = pred != "ambiguous"
        coverage = float(np.mean(accepted))
        accepted_accuracy = (
            float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else 0.0
        )
        macro = float(f1_score(y, pred, labels=RESOLVED, average="macro", zero_division=0))
        trials.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "macro_f1_with_abstain": macro,
            }
        )
    eligible = [x for x in trials if x["accepted_accuracy"] >= 0.92]
    best = max(eligible or trials, key=lambda x: (x["coverage"], x["accepted_accuracy"]))
    return float(best["threshold"]), trials


def metrics(rows: list[dict], pred: np.ndarray) -> dict:
    _, y, groups, _ = xy(rows)
    accepted = pred != "ambiguous"
    sheet_errors = []
    values = np.asarray([int(r["value_mm"]) for r in rows])
    for name in sorted(set(groups)):
        at = groups == name
        true_sum = int(values[at & (y == "include")].sum())
        pred_sum = int(values[at & (pred == "include")].sum())
        sheet_errors.append(abs(pred_sum - true_sum))
    return {
        "rows": len(rows),
        "sheets": len(set(groups)),
        "coverage": float(np.mean(accepted)),
        "accuracy_all": float(accuracy_score(y, pred)),
        "accuracy_accepted": (
            float(accuracy_score(y[accepted], pred[accepted])) if accepted.any() else 0.0
        ),
        "macro_f1_with_abstain": float(
            f1_score(y, pred, labels=RESOLVED, average="macro", zero_division=0)
        ),
        "weighted_f1_with_abstain": float(
            f1_score(y, pred, labels=RESOLVED, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y, pred, labels=RESOLVED, output_dict=True, zero_division=0
        ),
        "confusion_labels": [*RESOLVED, "ambiguous"],
        "confusion_matrix": confusion_matrix(
            y, pred, labels=[*RESOLVED, "ambiguous"]
        ).tolist(),
        "sheet_sum_mae_mm": float(np.mean(sheet_errors)),
        "sheet_sum_median_ae_mm": float(np.median(sheet_errors)),
        "sheet_sum_max_ae_mm": int(max(sheet_errors, default=0)),
    }


def train() -> dict:
    audit = audit_labels()
    if audit["problems_count"]:
        print(f"ВНИМАНИЕ: проблем разметки {audit['problems_count']}; см. models/label_audit.json")
    rows, skipped = select_rows()
    train_rows, val_rows, test_rows = split_rows(rows)
    penalty, c, cv = choose_linear(train_rows)

    X_train, y_train, _, w_train = xy(train_rows)
    X_val, y_val, _, _ = xy(val_rows)
    linear = fit_linear(linear_model(penalty, c), X_train, y_train, w_train)
    trees = tree_model().fit(X_train, y_train, sample_weight=w_train)
    threshold, trials = threshold_on_validation(linear, trees, X_val, y_val)

    val_pred = cascade_predict(linear, trees, X_val, threshold)
    X_test, _, _, _ = xy(test_rows)
    test_pred = cascade_predict(linear, trees, X_test, threshold)

    deploy_rows = train_rows + val_rows
    X_deploy, y_deploy, _, w_deploy = xy(deploy_rows)
    linear = fit_linear(linear_model(penalty, c), X_deploy, y_deploy, w_deploy)
    trees = tree_model().fit(X_deploy, y_deploy, sample_weight=w_deploy)

    report = {
        "data_warning": audit["warning"],
        "resolved_rows": len(rows),
        "ambiguous_skipped": skipped,
        "split": {
            "train": {"rows": len(train_rows), "sheets": len({group(r) for r in train_rows})},
            "validation": {"rows": len(val_rows), "sheets": len({group(r) for r in val_rows})},
            "test": {"rows": len(test_rows), "sheets": len({group(r) for r in test_rows})},
        },
        "class_counts": dict(Counter(r["decision"] for r in rows)),
        "linear_group_kfold": cv,
        "selected_linear": {"penalty": penalty, "C": c},
        "threshold_trials": trials,
        "selected_threshold": threshold,
        "validation": metrics(val_rows, val_pred),
        "test": metrics(test_rows, test_pred),
        "deploy_fit": "train + validation; test не использован",
        "cascade": "гарды -> LogisticRegression -> ExtraTrees; разногласие/низкая уверенность -> ambiguous",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(
            {
                "kind": "cascade_v2",
                "linear": linear,
                "trees": trees,
                "threshold": threshold,
                "features": FEATURE_NAMES,
                "report": report,
            },
            f,
        )
    report_path = MODEL_PATH.parent / "training_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка разметки и обучение каскада без API")
    parser.add_argument("--audit-only", action="store_true", help="только проверить JSON")
    args = parser.parse_args()
    if args.audit_only:
        report = audit_labels()
        print(
            f"jobs={report['jobs']}, sheets={report['sheets']}, "
            f"candidates={report['candidates']}, problems={report['problems_count']}"
        )
        print("отчёт: models/label_audit.json")
        return
    report = train()
    print("разбиение:", report["split"])
    print("выбрано:", report["selected_linear"], "threshold", report["selected_threshold"])
    print("validation:", json.dumps(report["validation"], ensure_ascii=False))
    print("test:", json.dumps(report["test"], ensure_ascii=False))
    print("модель:", MODEL_PATH)
    print("отчёт:", MODEL_PATH.parent / "training_report.json")


if __name__ == "__main__":
    main()
