from __future__ import annotations

import json
import os
import threading
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

from jobs import list_jobs, migrate_legacy
from review import apply_decision, confirm_sheet, sheet_view
from run import ROOT, process_pdf

load_dotenv(ROOT / ".env")

PRESETS = {
    "Локальная модель": {"provider": "none", "model": "local", "key_env": ""},
    "Groq": {"provider": "groq", "model": "qwen/qwen3.8-27b", "key_env": "GROQ_API_KEY"},
    "OpenAI": {"provider": "openai", "model": "gpt-4o", "key_env": "OPENAI_API_KEY"},
    "OpenAI Codex": {"provider": "codex", "model": "gpt-5-codex", "key_env": "OPENAI_API_KEY"},
}

WEB = ROOT / "web"
OUT = ROOT / "output"
MODELS = ROOT / "models"
app = Flask(
    __name__,
    template_folder=str(WEB / "templates"),
    static_folder=str(WEB / "static"),
)

_lock = threading.Lock()
_cancel = threading.Event()
_job = {
    "running": False,
    "mode": "",
    "log": [],
    "metrics": None,
    "error": None,
    "thumbs": [],
    "jobs": [],
    "train": None,
}


def _train_snapshot() -> dict:
    report = MODELS / "training_report.json"
    model = MODELS / "iso_clf.pkl"
    audit = MODELS / "label_audit.json"
    out: dict = {
        "model_exists": model.exists(),
        "model_path": str(model) if model.exists() else "",
        "report_exists": report.exists(),
        "audit_exists": audit.exists(),
    }
    try:
        from project_state import load_manifest

        m = load_manifest()
        out["training_sheets"] = len(m.get("training_sheets") or [])
        out["holdout_test_pages"] = m.get("holdout_test_pages")
    except Exception as exc:
        out["project_error"] = str(exc)
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            out["model_kind"] = data.get("model_kind")
            out["device"] = data.get("device")
            out["threshold"] = data.get("threshold")
            out["train"] = data.get("train")
            out["validation"] = data.get("validation")
            out["holdout_check"] = data.get("holdout_check")
            out["overfit_ok"] = data.get("overfit_ok")
            out["overfit_gap_macro_f1"] = data.get("overfit_gap_macro_f1")
            out["excluded_test_pages"] = data.get("excluded_test_pages")
        except Exception:
            pass
    if audit.exists():
        try:
            a = json.loads(audit.read_text(encoding="utf-8"))
            out["audit"] = {
                "sheets": a.get("sheets"),
                "candidates": a.get("candidates"),
                "problems_count": a.get("problems_count"),
            }
        except Exception:
            pass
    return out


def _safe_job(name: str) -> Path | None:
    if not name or ".." in name or "/" in name or "\\" in name:
        return None
    folder = (OUT / name).resolve()
    try:
        folder.relative_to(OUT.resolve())
    except ValueError:
        return None
    if not folder.is_dir():
        return None
    return folder


def _thumbs_from_metrics(metrics: dict | None) -> list[str]:
    if not metrics:
        return []
    thumbs = metrics.get("thumbs") or []
    if thumbs:
        return ["/media/" + t.replace("\\", "/") for t in thumbs]
    job = metrics.get("job")
    if not job:
        return []
    markup = OUT / job / "markup"
    if not markup.exists():
        return []
    return ["/media/" + job + "/markup/" + p.name for p in sorted(markup.glob("*.png"))]


def _load_latest() -> None:
    jobs = list_jobs(OUT)
    _job["jobs"] = jobs
    _job["train"] = _train_snapshot()
    if not jobs:
        return
    folder = OUT / jobs[0]["id"]
    path = folder / "metrics.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        _job["metrics"] = data
        _job["thumbs"] = _thumbs_from_metrics(data)
    except Exception:
        pass


_load_latest()


def _log(msg: str) -> None:
    with _lock:
        _job["log"].append(str(msg).rstrip())


def _run_job(payload: dict) -> None:
    try:
        preset = PRESETS[payload["api"]]
        os.environ["ISO_LLM_PROVIDER"] = preset["provider"]
        os.environ["ISO_LLM_MODEL"] = payload["model"] or preset["model"]
        key_env = preset.get("key_env") or ""
        if payload["key"] and key_env:
            os.environ[key_env] = payload["key"]
        pdf = payload["pdf"]
        job = migrate_legacy(OUT, pdf)
        if payload["replay"]:
            os.environ["ISO_LLM_PROVIDER"] = "replay"
            os.environ["ISO_REPLAY_DIR"] = str(job / "llm_raw")
        pages = payload["pages"]
        page_filter = None
        if pages:
            page_filter = {int(x.strip()) for x in pages.split(",") if x.strip()}
        _log(f"старт: {payload['api']} / {os.environ.get('ISO_LLM_MODEL')}")
        if preset["provider"] == "none":
            model_path = MODELS / "iso_clf.pkl"
            if model_path.exists():
                _log(f"локальная модель: {model_path.name}")
            else:
                _log("модели нет — черновик include + гарды; сначала вкладка Обучение")
        _log(f"PDF: {Path(pdf).name}")
        metrics = process_pdf(
            pdf,
            OUT,
            page_filter,
            log=_log,
            use_cache=not payload["fresh"],
            cancel=_cancel,
        )
        with _lock:
            _job["metrics"] = metrics
            _job["thumbs"] = _thumbs_from_metrics(metrics)
            _job["jobs"] = list_jobs(OUT)
            _job["train"] = _train_snapshot()
            _job["error"] = None
    except Exception as e:
        _log(f"ошибка: {e}")
        with _lock:
            _job["error"] = str(e)
    finally:
        with _lock:
            _job["running"] = False
            _job["mode"] = ""


def _run_train(kind: str) -> None:
    try:
        from train_local import audit_labels
        from train_gpu import train as train_models

        if kind == "audit":
            _log("проверка llm_raw и hits.json...")
            report = audit_labels(OUT)
            _log(
                f"готово: {report['sheets']} листов, {report['candidates']} размеров, "
                f"проблем {report['problems_count']}"
            )
            for problem in (report.get("problems") or [])[:20]:
                _log(" - " + problem)
            if report.get("warning"):
                _log(report["warning"])
            _log("отчёт: models\\label_audit.json")
        else:
            _log("GPU: CatBoost на NVIDIA CUDA (device 0)")
            _log("holdout: только листы 278-307 вне train/val (~277 листов в обучении)")
            report = train_models()
            train = report.get("train") or {}
            val = report.get("validation") or {}
            _log(
                "train: macro-F1={:.3f} · val: macro-F1={:.3f}, accuracy={:.3f}".format(
                    float(train.get("macro_f1") or 0),
                    float(val.get("macro_f1") or 0),
                    float(val.get("accuracy") or 0),
                )
            )
            gap = float(report.get("overfit_gap_macro_f1") or 0)
            _log(
                "переобучение: зазор={:.3f} ({})".format(
                    gap, "ok" if report.get("overfit_ok") else "высокий"
                )
            )
            hold = report.get("holdout_check") or {}
            _log("holdout_check: " + ("ok, тесты не в train" if hold.get("ok") else "ошибка"))
            _log("модель: models\\iso_clf.pkl")
            _log("отчёт: models\\training_report.html")
            _log("дальше: Расчёт → загрузи свои тест PDF — модель разметит сама")
        with _lock:
            _job["error"] = None
            _job["train"] = _train_snapshot()
    except Exception as e:
        _log(f"ошибка: {e}")
        with _lock:
            _job["error"] = str(e)
    finally:
        with _lock:
            _job["running"] = False
            _job["mode"] = ""


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/start")
def api_start():
    with _lock:
        if _job["running"]:
            return jsonify({"error": "уже считается"}), 409
        _job["running"] = True
        _job["mode"] = "calc"
        _job["log"] = []
        _job["error"] = None
        _cancel.clear()

    use_default = request.form.get("use_default") == "1"
    upload = request.files.get("pdf")
    pdf = ROOT / "data" / "02_Изометрии_10_листов.pdf"
    if upload and upload.filename:
        dest = ROOT / "web" / "uploads"
        dest.mkdir(parents=True, exist_ok=True)
        pdf = dest / Path(upload.filename).name
        upload.save(pdf)
    elif not use_default or not pdf.exists():
        with _lock:
            _job["running"] = False
            _job["mode"] = ""
        return jsonify({"error": "нет PDF"}), 400

    api = request.form.get("api") or "Локальная модель"
    if api not in PRESETS:
        with _lock:
            _job["running"] = False
            _job["mode"] = ""
        return jsonify({"error": "неизвестный режим"}), 400

    payload = {
        "api": api,
        "model": (request.form.get("model") or "").strip(),
        "key": (request.form.get("key") or "").strip(),
        "pages": (request.form.get("pages") or "").strip(),
        "replay": request.form.get("replay") == "1",
        "fresh": request.form.get("fresh") == "1",
        "pdf": pdf,
    }
    threading.Thread(target=_run_job, args=(payload,), daemon=True).start()
    return jsonify({"ok": True})


@app.post("/api/stop")
def api_stop():
    with _lock:
        running = _job["running"]
    if not running:
        return jsonify({"ok": True, "running": False})
    _cancel.set()
    _log("остановка: сохраню уже посчитанное")
    return jsonify({"ok": True, "running": True})


@app.post("/api/train")
def api_train():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "train").strip().lower()
    if kind not in ("train", "audit"):
        return jsonify({"error": "kind = train или audit"}), 400
    with _lock:
        if _job["running"]:
            return jsonify({"error": "уже идёт работа"}), 409
        _job["running"] = True
        _job["mode"] = kind
        _job["log"] = []
        _job["error"] = None
    threading.Thread(target=_run_train, args=(kind,), daemon=True).start()
    return jsonify({"ok": True})


@app.get("/api/train/status")
def api_train_status():
    with _lock:
        return jsonify(
            {
                "running": _job["running"],
                "mode": _job["mode"],
                "log": list(_job["log"]),
                "error": _job["error"],
                "train": _job["train"] or _train_snapshot(),
            }
        )


@app.post("/api/open-path")
def api_open_path():
    data = request.get_json(silent=True) or {}
    kind = str(data.get("kind") or "output").strip().lower()
    mapping = {
        "output": OUT,
        "models": MODELS,
        "report": MODELS / "training_report.html",
        "audit": MODELS / "label_audit.html",
        "guide": ROOT / "РАЗМЕТКА.md",
    }
    target = mapping.get(kind, OUT)
    if target.is_file():
        os.startfile(target)
    else:
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(target)
    return jsonify({"ok": True, "path": str(target)})


@app.get("/api/project/status")
def api_project_status():
    return jsonify(_train_snapshot())


@app.get("/api/sheet/<job_id>/<int:sheet_no>")
def api_sheet(job_id: str, sheet_no: int):
    folder = _safe_job(job_id)
    if not folder:
        return jsonify({"error": "нет такой папки"}), 404
    try:
        return jsonify(sheet_view(folder, sheet_no))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/decide")
def api_decide():
    with _lock:
        if _job["running"]:
            return jsonify({"error": "сейчас идёт работа — останови или подожди"}), 409
    data = request.get_json(silent=True) or {}
    folder = _safe_job(str(data.get("job") or ""))
    if not folder:
        return jsonify({"error": "нет такой папки"}), 404
    cid = str(data.get("id") or "").strip()
    if not cid:
        return jsonify({"error": "нет id размера"}), 400
    try:
        sheet_no = int(data.get("sheet_no"))
        metrics = apply_decision(folder, sheet_no, cid, data.get("decision"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    last = metrics.pop("_last_edit", None)
    thumbs = _thumbs_from_metrics(metrics)
    with _lock:
        _job["metrics"] = metrics
        _job["thumbs"] = thumbs
        _job["jobs"] = list_jobs(OUT)
    return jsonify(
        {
            "ok": True,
            "edit": last,
            "metrics": metrics,
            "thumbs": thumbs,
            "jobs": _job["jobs"],
        }
    )


@app.post("/api/confirm-sheet")
def api_confirm_sheet():
    data = request.get_json(silent=True) or {}
    folder = _safe_job(str(data.get("job") or ""))
    if not folder:
        return jsonify({"error": "нет такой папки"}), 404
    try:
        sheet_no = int(data.get("sheet_no"))
        metrics = confirm_sheet(
            folder, sheet_no, bool(data.get("confirmed", True))
        )
        with _lock:
            _job["train"] = _train_snapshot()
        return jsonify(
            {
                "ok": True,
                "metrics": metrics,
                "train": _job["train"],
                "view": sheet_view(folder, sheet_no),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/status")
def api_status():
    with _lock:
        return jsonify(
            {
                "running": _job["running"],
                "mode": _job["mode"],
                "log": list(_job["log"]),
                "metrics": _job["metrics"],
                "error": _job["error"],
                "thumbs": list(_job["thumbs"] or _thumbs_from_metrics(_job["metrics"])),
                "jobs": list(_job["jobs"] or list_jobs(OUT)),
                "train": _job["train"] or _train_snapshot(),
            }
        )


@app.get("/api/jobs")
def api_jobs():
    return jsonify({"jobs": list_jobs(OUT)})


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    folder = _safe_job(job_id)
    if not folder:
        return jsonify({"error": "нет такой папки"}), 404
    path = folder / "metrics.json"
    if not path.exists():
        return jsonify({"error": "нет metrics.json"}), 404
    data = json.loads(path.read_text(encoding="utf-8"))
    locked_sheets = []
    for raw_path in (folder / "llm_raw").glob("sheet_*.json"):
        try:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if raw.get("locked") and raw.get("reviewed_by_human"):
                locked_sheets.append(int(raw_path.stem.split("_")[-1]))
        except Exception:
            continue
    data["human_checked_sheets"] = sorted(locked_sheets)
    thumbs = _thumbs_from_metrics(data)
    with _lock:
        _job["metrics"] = data
        _job["thumbs"] = thumbs
    return jsonify({"metrics": data, "thumbs": thumbs})


@app.post("/api/open-output")
def api_open_output():
    name = request.form.get("job") or ""
    if request.is_json and request.json:
        name = name or (request.json.get("job") or "")
    folder = _safe_job(name) if name else None
    if folder is None:
        with _lock:
            job = (_job.get("metrics") or {}).get("job")
        folder = _safe_job(job) if job else OUT
    folder = folder or OUT
    folder.mkdir(parents=True, exist_ok=True)
    os.startfile(folder)
    return jsonify({"ok": True, "folder": str(folder)})


@app.get("/media/<path:sub>")
def media(sub: str):
    resp = send_from_directory(OUT, sub)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


def main() -> None:
    url = "http://127.0.0.1:8765"
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=8765, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
