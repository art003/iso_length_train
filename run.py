from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from annotate import annotate_final, overlay_candidate_ids
from extract import extract_pdf
from guards import apply_guards
from jobs import migrate_legacy, now_stamp, save_index
from llm_client import LlmClient
from pricing import estimate_cost_usd, PRICES_USD_PER_M
from report import (
    SheetResult,
    build_formula,
    decide_status,
    write_csv,
    write_html_report,
    write_json,
    write_readme,
    write_xlsx,
    metrics_text,
)
from review import write_hits


def _provider_pause(client: LlmClient, log, why: str, cancel=None) -> bool:
    if client.provider == "groq":
        pause = int(os.getenv("GROQ_PAUSE_S", "70"))
        log(f"пауза {pause} с ({why})")
        for _ in range(max(pause, 1)):
            if cancel is not None and cancel.is_set():
                log("остановлено во время паузы")
                return True
            time.sleep(1)
    elif client.provider == "openrouter":
        time.sleep(8)
        if cancel is not None and cancel.is_set():
            return True
    return False


def _cache_matches(cached: dict, cand_dicts: list[dict]) -> bool:
    """Кэш годится только если те же id и те же мм — иначе после отсечения таблиц D1 это другое число."""
    items = {it.get("id"): it for it in (cached.get("items") or []) if it.get("id")}
    wanted = {c["id"] for c in cand_dicts}
    if wanted - items.keys():
        return False
    for c in cand_dicts:
        it = items[c["id"]]
        try:
            if int(it.get("value_mm") or -1) != int(c["value_mm"]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    page_filter: set[int] | None = None,
    log=print,
    use_cache: bool = True,
    cancel=None,
) -> dict:
    t0 = time.perf_counter()
    out_root = Path(out_dir)
    job = migrate_legacy(out_root, pdf_path)
    log(f"PDF: {pdf_path.name}")
    log(f"папка: {job}")
    dest_pdf = job / pdf_path.name
    if pdf_path.resolve() != dest_pdf.resolve() and pdf_path.exists():
        try:
            shutil.copy2(pdf_path, dest_pdf)
        except OSError:
            pass
    sheets = extract_pdf(pdf_path, page_filter)
    if not sheets:
        raise RuntimeError("Не удалось извлечь листы из PDF (проверьте файл и --pages).")
    if os.getenv("ISO_LLM_PROVIDER", "").strip().lower() == "replay":
        replay_dir = os.getenv("ISO_REPLAY_DIR")
        replay_path = Path(replay_dir) if replay_dir else None
        if replay_path is None or not replay_path.exists():
            os.environ["ISO_REPLAY_DIR"] = str(job / "llm_raw")
    client = LlmClient()
    results: list[SheetResult] = []
    markup_dir = job / "markup"
    overlay_dir = job / "overlays"
    raw_dir = job / "llm_raw"
    markup_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    skip_api = False
    for sheet in sheets:
        if cancel is not None and cancel.is_set() and not skip_api:
            skip_api = True
            log("остановка: дальше без API")
        cand_dicts = [
            {
                "id": c.cid,
                "value_mm": c.value_mm,
                "local_hint": c.local_hint,
                "nearby": c.nearby,
            }
            for c in sheet.candidates
        ]
        overlay = overlay_candidate_ids(sheet.pixmap_png, sheet.candidates, sheet.render_scale)
        (overlay_dir / f"sheet_{sheet.sheet_no:02d}_{sheet.line_id}_ids.png").write_bytes(overlay)
        write_hits(sheet, overlay_dir)

        calls_before = client.usage.calls
        raw_path = raw_dir / f"sheet_{sheet.sheet_no:02d}.json"
        wanted = {c["id"] for c in cand_dicts}
        llm = None
        locked = False
        if raw_path.exists():
            try:
                cached = json.loads(raw_path.read_text(encoding="utf-8"))
                got = {it.get("id") for it in (cached.get("items") or [])}
                locked = bool(cached.get("locked"))
                fits = _cache_matches(cached, cand_dicts)
                if locked and fits:
                    llm = cached
                    log(f"Лист {sheet.sheet_no:02d} закрыт, json не трогаю")
                elif use_cache and fits:
                    llm = cached
                    log(f"Лист {sheet.sheet_no:02d} уже есть, пропускаю")
            except json.JSONDecodeError:
                llm = None
                locked = False
        from_api = False
        generated = False
        if llm is None:
            if skip_api:
                log(f"Лист {sheet.sheet_no:02d} нет json, пропуск")
                continue
            if client.provider == "none":
                from local_model import classify as local_classify

                llm = local_classify(
                    sheet.sheet_no,
                    sheet.line_id,
                    [
                        {
                            "id": c.cid,
                            "value_mm": c.value_mm,
                            "local_hint": c.local_hint,
                            "nearby": c.nearby,
                            "x": c.x,
                            "y": c.y,
                            "flags": sorted(c.flags),
                            "direction": list(c.direction),
                            "dim_line_id": getattr(c, "dim_line_id", "") or "",
                            "dist_to_line": getattr(c, "dist_to_line", 0.0),
                            "parallel_score": getattr(c, "parallel_score", 0.0),
                            "dim_endpoints": list(getattr(c, "dim_endpoints", ((0.0, 0.0), (0.0, 0.0)))),
                            "dim_axis": list(getattr(c, "dim_axis", (1.0, 0.0))),
                            "dim_offset": getattr(c, "dim_offset", 0.0),
                            "parent_id": getattr(c, "parent_id", None),
                            "relation_kind": getattr(c, "relation_kind", "") or "",
                            "geometry_confidence": getattr(c, "geometry_confidence", 0.0),
                            "parent_value_mm": getattr(c, "parent_value_mm", None),
                            "parent_dist": getattr(c, "parent_dist", 0.0),
                        }
                        for c in sheet.candidates
                    ],
                )
                log(f"Лист {sheet.sheet_no:02d} без API")
            else:
                llm = client.classify_sheet(sheet.sheet_no, sheet.line_id, cand_dicts, overlay)
                from_api = True
            generated = True
            two_passes = (
                client.provider not in ("replay", "none")
                and int(os.getenv("ISO_LLM_PASSES", "2")) >= 2
            )
            if two_passes:
                stopped = _provider_pause(client, log, "перед вторым проходом", cancel)
                draft, hint_notes = apply_guards(sheet.candidates, llm)
                hint = ", ".join(hint_notes)
                if stopped:
                    skip_api = True
                    llm = draft
                else:
                    log(f"Лист {sheet.sheet_no:02d} второй проход...")
                    try:
                        llm = client.classify_sheet(
                            sheet.sheet_no,
                            sheet.line_id,
                            cand_dicts,
                            overlay,
                            first=draft,
                            hint=hint,
                        )
                    except Exception as e:
                        log(f"Лист {sheet.sheet_no:02d} второй проход не вышел, оставляю первый: {e}")
                        llm = draft
            last_sheet = sheet is sheets[-1]
            if not last_sheet and not skip_api and client.provider not in ("replay", "none"):
                if _provider_pause(client, log, "следующий лист", cancel):
                    skip_api = True
        if not locked:
            llm, guard_notes = apply_guards(sheet.candidates, llm)
            if guard_notes:
                log(f"Лист {sheet.sheet_no:02d} авто: " + ", ".join(guard_notes))
            by_tmp = {c.cid: c for c in sheet.candidates}
            for it in llm.get("items") or []:
                c = by_tmp.get(it.get("id"))
                if c is not None:
                    it["value_mm"] = c.value_mm
            raw_path.write_text(json.dumps(llm, ensure_ascii=False, indent=2), encoding="utf-8")

        by_id = {c.cid: c for c in sheet.candidates}
        decisions: dict[str, str] = {}
        included, excluded, ambiguous = [], [], []
        notes = list(llm.get("notes") or [])
        for item in llm.get("items") or []:
            cid = item.get("id")
            if cid not in by_id:
                continue
            dec = item.get("decision") or "ambiguous"
            if dec not in ("include", "exclude_nested", "not_length", "ambiguous"):
                dec = "ambiguous"
            decisions[cid] = dec
            rec = {
                "id": cid,
                "value_mm": by_id[cid].value_mm,
                "role": item.get("role"),
                "reason": item.get("reason"),
            }
            if dec == "include":
                included.append(rec)
            elif dec == "ambiguous":
                ambiguous.append(rec)
            else:
                excluded.append(rec)
        for c in sheet.candidates:
            if c.cid not in decisions:
                decisions[c.cid] = "ambiguous"
                ambiguous.append({"id": c.cid, "value_mm": c.value_mm, "role": "other", "reason": "модель этот id не вернула"})

        # суммирование только программно
        length_mm = sum(int(x["value_mm"]) for x in included)
        formula = build_formula(included, length_mm)
        if ambiguous:
            notes.append("глянуть: " + ", ".join(f"{a['id']}={a['value_mm']}" for a in ambiguous))
        status = decide_status(ambiguous, notes, included)
        remarks = "; ".join(notes)

        title = f"Лист {sheet.sheet_no}  {sheet.line_id}"
        png_name = f"sheet_{sheet.sheet_no:02d}_{sheet.line_id}.png"
        annotate_final(
            sheet.pixmap_png,
            sheet.candidates,
            decisions,
            sheet.render_scale,
            title,
            formula,
            length_mm,
            status,
            markup_dir / png_name,
        )

        line_id = (llm.get("line_id") or sheet.line_id or "").strip() or sheet.line_id
        results.append(
            SheetResult(
                sheet_no=int(sheet.sheet_no),
                line_id=line_id,
                length_mm=length_mm,
                length_m=round(length_mm / 1000.0, 3),
                formula=formula,
                status=status,
                remarks=remarks,
                included=included,
                excluded=excluded,
                ambiguous=ambiguous,
                llm_raw=llm,
                ai_calls=client.usage.calls - calls_before,
                markup_name=png_name,
            )
        )
        log(
            f"Лист {sheet.sheet_no:02d} {line_id}: {length_mm} мм ({length_mm/1000:.3f} м) [{status}]"
        )

    if cancel is not None and cancel.is_set():
        log("остановлено")
        prev_path = job / "metrics.json"
        if not results and prev_path.exists():
            return json.loads(prev_path.read_text(encoding="utf-8"))
        if results and prev_path.exists():
            from review import _row_to_result

            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            have = {r.sheet_no for r in results}
            for s in prev.get("sheets") or []:
                if int(s["sheet_no"]) not in have:
                    results.append(_row_to_result(job, s))
        if not results:
            raise RuntimeError("остановлено до первого листа")
        results.sort(key=lambda r: r.sheet_no)

    elapsed = time.perf_counter() - t0
    usage = client.usage
    cost = estimate_cost_usd(usage)
    pages = max(len(results), 1)
    cost_per_page = cost / pages
    calls_per_page = usage.calls / pages
    total_mm = sum(r.length_mm for r in results)
    ok_n = sum(1 for r in results if r.status == "ok")
    review_n = sum(1 for r in results if r.status == "review")
    inn_p, out_p = PRICES_USD_PER_M.get(usage.model.lower(), (0.30, 2.50))
    replay = usage.provider in ("replay", "none")

    live_api = None
    prev_path = job / "metrics.json"
    if replay and prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            live_api = prev.get("live_api")
            if not live_api and prev.get("ai_calls_total"):
                live_api = {
                    "provider": prev.get("provider"),
                    "model": prev.get("model"),
                    "ai_calls_total": prev.get("ai_calls_total"),
                    "total_time_s": prev.get("total_time_s"),
                    "input_tokens": prev.get("input_tokens"),
                    "output_tokens": prev.get("output_tokens"),
                    "cost_total_usd": prev.get("cost_total_usd"),
                }
        except json.JSONDecodeError:
            live_api = None
    if not replay:
        live_api = {
            "provider": usage.provider,
            "model": usage.model,
            "ai_calls_total": usage.calls,
            "total_time_s": round(elapsed, 2),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_total_usd": round(cost, 6),
        }

    metrics = {
        "pdf": pdf_path.name,
        "job": job.name,
        "folder": job.name,
        "updated": now_stamp(),
        "provider": "без API" if replay else usage.provider,
        "model": usage.model,
        "ocr": "нет, текст из pdf (pymupdf)",
        "pages": len(results),
        "total_mm": total_mm,
        "total_m": round(total_mm / 1000.0, 3),
        "ok": ok_n,
        "review": review_n,
        "total_time_s": round(elapsed, 2),
        "ai_calls_total": 0 if replay else usage.calls,
        "ai_retries": 0 if replay else usage.retries,
        "ai_calls_avg_per_page": 0 if replay else round(calls_per_page, 2),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "tariff_date": "2026-09-18",
        "tariff_usd_per_1m_tokens": {"input": inn_p, "output": out_p},
        "cost_total_usd": round(cost, 6),
        "cost_per_page_usd": round(cost_per_page, 6),
        "cost_formula": (
            f"({usage.input_tokens} / 1e6) * {inn_p} + ({usage.output_tokens} / 1e6) * {out_p}"
        ),
        "live_api": live_api,
        "note": (
            "Суммы из json, API не вызывал."
            if replay
            else "Считал по токенам из ответа API."
        ),
        "thumbs": [f"{job.name}/markup/{r.markup_name}" for r in results if r.markup_name],
        "sheets": [
            {
                "sheet_no": r.sheet_no,
                "line_id": r.line_id,
                "length_mm": r.length_mm,
                "length_m": r.length_m,
                "formula": r.formula,
                "status": r.status,
                "remarks": r.remarks,
                "included": r.included,
                "excluded": r.excluded,
                "ambiguous": r.ambiguous,
                "ai_calls": r.ai_calls,
                "markup": f"{job.name}/markup/{r.markup_name}" if r.markup_name else "",
            }
            for r in results
        ],
    }
    write_csv(job / "lengths.csv", results)
    write_xlsx(job / "lengths.xlsx", results, metrics)
    write_json(job / "metrics.json", metrics)
    (job / "metrics.txt").write_text(metrics_text(metrics), encoding="utf-8")
    write_readme(job / "что_это.txt", metrics, results)
    write_html_report(job / "report.html", metrics, results)
    save_index(out_root, {
        "id": job.name,
        "pdf": pdf_path.name,
        "folder": job.name,
        "updated": metrics["updated"],
        "pages": metrics["pages"],
        "total_mm": total_mm,
        "total_m": metrics["total_m"],
        "ok": ok_n,
        "review": review_n,
    })
    log(metrics_text(metrics))
    log(f"готово: {job / 'lengths.xlsx'}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Расчёт длины трубопроводов по изометриям")
    p.add_argument("pdf", nargs="?", default="", help="PDF с листами изометрий")
    p.add_argument("-o", "--out", default="", help="каталог результатов")
    p.add_argument("--pages", default="", help="номера листов через запятую, например 7 или 1,7")
    p.add_argument("--from-json", default="", help="папка с уже сохранёнными json, без API")
    p.add_argument("--no-api", action="store_true", help="без Groq: локальная модель + гарды")
    args = p.parse_args()
    default_pdf = ROOT / "data" / "02_Изометрии_10_листов.pdf"
    pdf = Path(args.pdf) if args.pdf else default_pdf
    if not pdf.exists():
        raise SystemExit(f"PDF не найден: {pdf}")
    out = Path(args.out) if args.out else ROOT / "output"
    page_filter = None
    if args.pages.strip():
        page_filter = {int(x.strip()) for x in args.pages.split(",") if x.strip()}
    if args.from_json.strip():
        os.environ["ISO_LLM_PROVIDER"] = "replay"
        os.environ["ISO_REPLAY_DIR"] = str(Path(args.from_json).resolve())
    else:
        os.environ["ISO_LLM_PROVIDER"] = "none"
        os.environ["ISO_LLM_MODEL"] = "local"
    process_pdf(pdf, out, page_filter)


if __name__ == "__main__":
    main()
