from __future__ import annotations

import math
import pickle
import re
from pathlib import Path

import numpy as np

from dataset import export_dataset, iter_examples

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iso_clf.pkl"
LABELS = ("include", "exclude_nested", "not_length", "ambiguous")
FLAG_NAMES = (
    "overall_mark",
    "overall_pair",
    "handwheel",
    "insulation",
    "support_repeat",
    "near_plant_coord",
    "near_endpoint",
    "nested_offset",
)
FEATURE_NAMES = (
    "value",
    "log_value",
    "x",
    "y",
    "big",
    "small",
    "has_z",
    "has_sm",
    "has_dn",
    "has_lt",
    "n_near",
    "has_larger_near",
    "nested_ratio",
    "support_mark",
    *("f_" + n for n in FLAG_NAMES),
    "has_parent",
    "geo_confidence",
    "parent_ratio",
    "parent_dist_norm",
    "dist_to_line",
    "parallel_score",
)

_Z = re.compile(r"\bz\+?\b", re.I)
_NEAR_NUM = re.compile(r"\d+")
_SUPPORT_MARK = re.compile(r"[оoо]\d+", re.I)


def _flags(row: dict) -> set[str]:
    raw = row.get("flags")
    if isinstance(raw, (list, set, tuple)):
        return {str(x) for x in raw if x}
    if isinstance(raw, str) and raw.strip():
        return {p for p in raw.replace(",", "|").split("|") if p}
    return set()


def row_features(row: dict) -> dict[str, float]:
    nearby = str(row.get("nearby") or "").lower()
    flags = _flags(row)
    value = int(row.get("value_mm") or 0)
    feats = {
        "value": float(value),
        "log_value": math.log1p(max(value, 0)),
        "x": float(row.get("x") or 0),
        "y": float(row.get("y") or 0),
        "big": float(value >= 8000),
        "small": float(value <= 200),
        "has_z": float(bool(_Z.search(nearby))),
        "has_sm": float("см" in nearby),
        "has_dn": float("dn" in nearby),
        "has_lt": float("lt" in nearby),
        "n_near": float(nearby.count("|") + (1 if nearby.strip() else 0)),
    }
    near_vals = [int(n) for n in _NEAR_NUM.findall(nearby) if int(n) != value]
    larger = [n for n in near_vals if n > value]
    feats["has_larger_near"] = float(bool(larger))
    feats["nested_ratio"] = float(value / max(larger)) if larger else 0.0
    feats["support_mark"] = float(bool(_SUPPORT_MARK.search(nearby)))
    parent_id = row.get("parent_id")
    conf = float(row.get("geometry_confidence") or 0.0)
    parent_value = int(row.get("parent_value_mm") or 0)
    feats["has_parent"] = float(bool(parent_id) and conf >= 0.75)
    feats["geo_confidence"] = conf
    feats["parent_ratio"] = float(value / parent_value) if parent_value > 0 else 0.0
    feats["parent_dist_norm"] = float(min(1.0, float(row.get("parent_dist") or 0.0) / 190.0))
    feats["dist_to_line"] = float(row.get("dist_to_line") or 0.0)
    feats["parallel_score"] = float(row.get("parallel_score") or 0.0)
    for name in FLAG_NAMES:
        feats["f_" + name] = float(name in flags)
    return feats


def vector(row: dict, names: tuple[str, ...] = FEATURE_NAMES) -> list[float]:
    f = row_features(row)
    return [float(f.get(n, 0.0)) for n in names]


def seed_include(sheet_no: int, line_id: str, candidates: list[dict]) -> dict:
    return {
        "line_id": line_id,
        "items": [
            {
                "id": c["id"],
                "decision": "include",
                "role": "main",
                "reason": "черновик без API",
            }
            for c in candidates
        ],
        "notes": [f"лист {sheet_no}: без API, правь кликом"],
    }


def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def classify(sheet_no: int, line_id: str, candidates: list[dict]) -> dict:
    bundle = load_model()
    if not bundle:
        return seed_include(sheet_no, line_id, candidates)
    names = tuple(bundle.get("features") or FEATURE_NAMES)
    filled = list(candidates)
    if filled and not any(c.get("parent_id") or c.get("geometry_confidence") for c in filled):
        from relations import compute_relations

        class _Row:
            def __init__(self, raw: dict):
                self.cid = raw.get("id")
                self.value_mm = int(raw.get("value_mm") or 0)
                self.x = float(raw.get("x") or 0)
                self.y = float(raw.get("y") or 0)
                self.nearby = raw.get("nearby") or ""
                self.local_hint = raw.get("local_hint") or ""
                self.flags = set(raw.get("flags") or [])
                self.direction = tuple(raw.get("direction") or (1.0, 0.0))
                self.dim_line_id = raw.get("dim_line_id") or ""
                self.dist_to_line = float(raw.get("dist_to_line") or 0.0)
                self.parallel_score = float(raw.get("parallel_score") or 0.0)
                ends = raw.get("dim_endpoints") or [[0.0, 0.0], [0.0, 0.0]]
                self.dim_endpoints = ((float(ends[0][0]), float(ends[0][1])), (float(ends[1][0]), float(ends[1][1])))
                self.dim_axis = tuple(raw.get("dim_axis") or self.direction)
                self.dim_offset = float(raw.get("dim_offset") or 0.0)
                self.parent_id = None
                self.relation_kind = ""
                self.geometry_confidence = 0.0
                self.parent_value_mm = None
                self.parent_dist = 0.0

        rows = [_Row(c) for c in filled]
        compute_relations(rows)
        by = {r.cid: r for r in rows}
        for c in filled:
            r = by.get(c.get("id"))
            if not r:
                continue
            c["parent_id"] = r.parent_id
            c["relation_kind"] = r.relation_kind
            c["geometry_confidence"] = r.geometry_confidence
            c["parent_value_mm"] = r.parent_value_mm
            c["parent_dist"] = r.parent_dist
    items = []
    for c in candidates:
        x = [vector(c, names)]
        if bundle.get("kind") == "catboost_gpu_ensemble_v1":
            models = bundle["models"]
            proba = np.mean([model.predict_proba(x)[0] for model in models], axis=0)
            i = int(proba.argmax())
            p = float(proba[i])
            lab = str(models[0].classes_[i])
            reason = f"CatBoost×{len(models)} {p:.2f}"
        elif bundle.get("kind") == "catboost_gpu_v1":
            model = bundle["model"]
            proba = model.predict_proba(x)[0]
            i = int(proba.argmax())
            p = float(proba[i])
            lab = str(model.classes_[i])
            reason = f"CatBoost {p:.2f}"
        elif bundle.get("kind") == "cascade_v2":
            linear = bundle["linear"]
            trees = bundle["trees"]
            lp = linear.predict_proba(x)[0]
            tp = trees.predict_proba(x)[0]
            li = int(lp.argmax())
            ti = int(tp.argmax())
            linear_lab = str(linear.classes_[li])
            tree_lab = str(trees.classes_[ti])
            p = min(float(lp[li]), float(tp[ti]))
            threshold = float(bundle.get("threshold", 0.55))
            agreed = linear_lab == tree_lab
            lab = linear_lab if agreed and p >= threshold else "ambiguous"
            reason = (
                f"каскад {p:.2f}"
                if lab != "ambiguous"
                else f"каскад не уверен {linear_lab}/{tree_lab}"
            )
        else:
            clf = bundle["clf"]
            proba = clf.predict_proba(x)[0]
            i = int(proba.argmax())
            p = float(proba[i])
            lab = str(clf.classes_[i])
            if p < 0.55:
                lab = "ambiguous"
            reason = f"локальная модель {p:.2f}"
        if lab not in LABELS:
            lab = "ambiguous"
        items.append(
            {
                "id": c["id"],
                "decision": lab,
                "role": "main" if lab == "include" else "other",
                "reason": reason,
            }
        )
    return {
        "line_id": line_id,
        "items": items,
        "notes": ["локальная модель, без API"],
    }


def _is_holdout(row: dict) -> bool:
    folder = str(row.get("folder") or "")
    n = int(row.get("sheet_no") or 0)
    if "02_Изометрии_10_листов" in folder and n == 7:
        return True
    if folder in ("Изометрии", "Изометрии_без_API") and n > 0 and n % 10 == 0:
        return True
    return False


def train(out_root: Path | None = None, folders: list[str] | None = None) -> Path:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise SystemExit("нужен scikit-learn: pip install scikit-learn") from e

    export_dataset(out_root)
    rows = [r for r in iter_examples(out_root) if r.get("id") and r.get("value_mm") is not None]
    if folders:
        want = set(folders)
        rows = [r for r in rows if r.get("folder") in want]
    # Большой прогон без API — основные примеры. Закрытые 10 листов — эталон сверху.
    big = [r for r in rows if r.get("folder") in ("Изометрии", "Изометрии_без_API")]
    gold = [r for r in rows if r.get("locked")]
    if len(big) >= 200:
        seen = {(r.get("folder"), r.get("sheet_no"), r.get("id")) for r in big}
        extra = [r for r in gold if (r.get("folder"), r.get("sheet_no"), r.get("id")) not in seen]
        rows = big + extra
        print(f"учу большой прогон без API: {len(big)} + закрытые {len(extra)}")
    elif len(gold) >= 30:
        rows = gold
        print(f"учу только закрытые листы: {len(rows)}")
    if len(rows) < 30:
        raise SystemExit(f"мало примеров: {len(rows)}. Сначала разметь листы кликом и python dataset.py")

    train_rows = [r for r in rows if not _is_holdout(r)]
    hold = [r for r in rows if _is_holdout(r)]
    if len(train_rows) < 20:
        train_rows = rows
        hold = []

    X = [vector(r) for r in train_rows]
    y = [str(r.get("decision") or "ambiguous") for r in train_rows]
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump({"clf": pipe, "features": FEATURE_NAMES}, f)

    def acc(part: list[dict]) -> str:
        if not part:
            return "нет"
        ok = 0
        for r in part:
            pred = pipe.predict([vector(r)])[0]
            if pred == (r.get("decision") or "ambiguous"):
                ok += 1
        return f"{ok}/{len(part)} = {ok / len(part):.2%}"

    print(f"примеров всего {len(rows)}, обучение {len(train_rows)}")
    print("на обучении:", acc(train_rows))
    print("контроль (каждый 10-й лист большого pdf + лист 7):", acc(hold))
    from collections import Counter

    print("классы:", dict(Counter(y)))
    print(f"модель: {MODEL_PATH}")
    return MODEL_PATH


def main() -> None:
    # Старый вход оставлен: теперь полное обучение с аудитом и тестом в train_local.py.
    from train_local import train as train_cascade

    train_cascade()


if __name__ == "__main__":
    main()
