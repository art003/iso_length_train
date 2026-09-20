from __future__ import annotations

import unittest
from pathlib import Path

from extract import extract_pdf
from guards import apply_guards

ROOT = Path(__file__).resolve().parent
PDF = ROOT / "data" / "02_Изометрии_10_листов.pdf"


def _decisions(sheet, mapping: dict[str, str]) -> dict:
    items = [{"id": c.cid, "decision": mapping.get(c.cid, "include"), "role": "main", "reason": mapping.get(c.cid + "_r", "")} for c in sheet.candidates]
    return {"line_id": sheet.line_id, "items": items, "notes": []}


def _sum_include(sheet, llm) -> int:
    by = {c.cid: c.value_mm for c in sheet.candidates}
    return sum(by[it["id"]] for it in llm["items"] if it["decision"] == "include" and it["id"] in by)


class GuardRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PDF.exists():
            raise unittest.SkipTest("нет учебного pdf")
        cls.by_no = {s.sheet_no: s for s in extract_pdf(PDF, set(range(1, 11)))}

    def _guarded_all_include(self, sheet):
        mapping = {c.cid: "include" for c in sheet.candidates}
        out, _ = apply_guards(sheet.candidates, _decisions(sheet, mapping))
        return out

    def test_sheet1_z_plus_is_length(self):
        s = self.by_no[1]
        d1 = next(c for c in s.candidates if c.value_mm == 1383)
        self.assertIn("near_plant_coord", d1.flags)
        llm = _decisions(s, {"D1": "not_length", "D1_r": "вертикальная координата Z+"})
        out, notes = apply_guards(s.candidates, llm)
        by = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(by[d1.cid], "include")
        self.assertEqual(_sum_include(s, out), 5563)

    def test_sheet6_two_runs_and_overlap_overall(self):
        s = self.by_no[6]
        d6 = next(c for c in s.candidates if c.value_mm == 2950)
        d8 = next(c for c in s.candidates if c.value_mm == 2400)
        d11 = next(c for c in s.candidates if c.value_mm == 3505)
        self.assertIn("overall_pair", d6.flags)
        self.assertNotIn("overall_pair", d8.flags)
        mapping = {c.cid: "include" for c in s.candidates}
        mapping[d6.cid] = "include"
        mapping[d8.cid] = "exclude_nested"
        mapping[d8.cid + "_r"] = "габарит поверх D7"
        mapping[d11.cid] = "exclude_nested"
        mapping[d11.cid + "_r"] = "габарит поверх D10"
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(by[d6.cid], "exclude_nested")
        self.assertEqual(by[d8.cid], "include")
        self.assertEqual(by[d11.cid], "exclude_nested")
        self.assertTrue(any("два прогона" in n for n in notes))
        gold, _ = apply_guards(s.candidates, _decisions(s, {c.cid: "include" for c in s.candidates}))
        self.assertEqual(_sum_include(s, gold), 10521)

    def test_sheet5_unmarked_overall(self):
        s = self.by_no[5]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm == 3505:
                self.assertEqual(by[c.cid], "exclude_nested")
            if c.value_mm == 1600:
                self.assertEqual(by[c.cid], "include")
        self.assertEqual(_sum_include(s, out), 8508)

    def test_sheet7_nested_stays_out(self):
        s = self.by_no[7]
        nested = {c.cid: "exclude_nested" for c in s.candidates if c.value_mm in (916, 300)}
        nested.update({c.cid + "_r": f"внутри {1950 if c.value_mm == 916 else 840}" for c in s.candidates if c.value_mm in (916, 300)})
        mapping = {c.cid: "include" for c in s.candidates}
        out, _ = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it for it in out["items"]}
        for c in s.candidates:
            if c.value_mm in (916, 300):
                self.assertEqual(by[c.cid]["decision"], "exclude_nested")
                self.assertNotIn("pipe_opening", c.flags)
        length = _sum_include(s, out)
        self.assertEqual(length, 4306)

    def test_sheet7_nested_from_include(self):
        s = self.by_no[7]
        mapping = {c.cid: "include" for c in s.candidates}
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm in (916, 300):
                self.assertEqual(by[c.cid], "exclude_nested")
        self.assertEqual(_sum_include(s, out), 4306)
        self.assertTrue(any("внутри" in n or "вложенный" in n for n in notes))

    def test_sheet7_restores_main_run(self):
        s = self.by_no[7]
        mapping = {c.cid: "exclude_nested" for c in s.candidates}
        out, _ = apply_guards(s.candidates, _decisions(s, mapping))
        self.assertEqual(_sum_include(s, out), 4306)

    def test_sheet9_dogleg_is_pipe(self):
        s = self.by_no[9]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm in (195, 123, 244, 175):
                self.assertEqual(by[c.cid], "include", c.cid)
            self.assertNotIn("insulation_thickness", c.flags)
        self.assertTrue(any("pipe_opening" in c.flags for c in s.candidates))
        self.assertEqual(_sum_include(s, out), 860)

    def test_sheet9_model_drops_dogleg_restored(self):
        s = self.by_no[9]
        mapping = {c.cid: "include" for c in s.candidates}
        for c in s.candidates:
            if c.value_mm in (123, 244):
                mapping[c.cid] = "not_length"
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm in (195, 123, 244, 175):
                self.assertEqual(by[c.cid], "include", c.cid)
        self.assertEqual(_sum_include(s, out), 860)
        self.assertTrue(any("вернул" in n or "отверстие" in n for n in notes))

    def test_spec_or_title_skips_stamp_and_bom(self):
        from extract import spec_or_title_reason

        w, h = 2150.0, 1521.0
        self.assertEqual(spec_or_title_reason(2005, 90, "ItemCode | Кол-во | DN", w, h), "materials_spec")
        self.assertEqual(spec_or_title_reason(1570, 1156, "Поз. № | Длина, мм | 225", w, h), "materials_spec")
        self.assertEqual(spec_or_title_reason(370, 1389, "°C | Гидр. | Пр.", w, h), "title_block")
        self.assertIsNone(spec_or_title_reason(911, 461, "<1> | DN50", w, h))
        self.assertIsNone(spec_or_title_reason(631, 359, "ИЗОЛЯЦИЯ: H | 123 | 244", w, h))

    def test_sheet10_real_segments_kept(self):
        s = self.by_no[10]
        mapping = {c.cid: "include" for c in s.candidates}
        out, _ = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        opened = {c.cid for c in s.candidates if "pipe_opening" in c.flags}
        self.assertTrue(opened)
        for c in s.candidates:
            if c.value_mm == 4116:
                self.assertEqual(by[c.cid], "exclude_nested")
            if "handwheel" in c.flags:
                self.assertEqual(by[c.cid], "not_length")
            if "pipe_opening" in c.flags and "handwheel" not in c.flags:
                self.assertEqual(by[c.cid], "include", c.cid)
            if c.value_mm in (188, 134, 550):
                self.assertEqual(by[c.cid], "include", c.cid)
        self.assertEqual(_sum_include(s, out), 5460)

    def test_sheet10_vrezka_not_same_as_handwheel(self):
        s = self.by_no[10]
        mapping = {c.cid: "not_length" for c in s.candidates if c.value_mm == 244}
        mapping.update({c.cid: "include" for c in s.candidates if c.value_mm != 244})
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if "handwheel" in c.flags:
                self.assertEqual(by[c.cid], "not_length")
            if "pipe_opening" in c.flags and "handwheel" not in c.flags:
                self.assertEqual(by[c.cid], "include", c.cid)
        self.assertEqual(_sum_include(s, out), 5460)
        self.assertTrue(any("отверстие" in n or "вернул" in n for n in notes))

    def test_sheet4_handwheels_out(self):
        s = self.by_no[4]
        mapping = {c.cid: "include" for c in s.candidates}
        out, _ = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm == 159:
                self.assertEqual(by[c.cid], "not_length")
        self.assertEqual(_sum_include(s, out), 2772)

    def test_sheet2_stacked_overall(self):
        s = self.by_no[2]
        d9 = next(c for c in s.candidates if c.value_mm == 10200)
        self.assertIn("overall_pair", d9.flags)
        d8 = next(c for c in s.candidates if c.value_mm == 4072)
        self.assertNotIn("overall_pair", d8.flags)

    def test_sheet8_overall_if_chain_incomplete(self):
        sheets = extract_pdf(PDF, {8})
        if not sheets:
            self.skipTest("нет листа 8")
        s = sheets[0]
        mapping = {c.cid: "include" for c in s.candidates}
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        d6 = next(c for c in s.candidates if c.value_mm == 23525)
        d12 = next(c for c in s.candidates if c.value_mm == 24100)
        d7 = next(c for c in s.candidates if c.value_mm == 1000)
        self.assertEqual(by[d6.cid], "include")
        self.assertEqual(by[d12.cid], "include")
        self.assertEqual(by[d7.cid], "include")
        self.assertEqual(_sum_include(s, out), 48625)
        self.assertTrue(any("полный размер" in n for n in notes))

    def test_gold_lengths_all_ten(self):
        gold = {
            1: 5563,
            2: 21956,
            3: 6880,
            4: 2772,
            5: 8508,
            6: 10521,
            7: 4306,
            8: 48625,
            9: 860,
            10: 5460,
        }
        for no, length in gold.items():
            s = self.by_no[no]
            out = self._guarded_all_include(s)
            self.assertEqual(_sum_include(s, out), length, f"лист {no}")

    def test_sheet3_bom_out_branch_in(self):
        s = self.by_no[3]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm == 13:
                self.assertEqual(by[c.cid], "not_length")
            if c.value_mm == 500:
                self.assertEqual(by[c.cid], "include")

    def test_sheet2_overall_not_the_run(self):
        s = self.by_no[2]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        d9 = next(c for c in s.candidates if c.value_mm == 10200)
        d8 = next(c for c in s.candidates if c.value_mm == 4072)
        self.assertEqual(by[d9.cid], "exclude_nested")
        self.assertEqual(by[d8.cid], "include")

    def test_stamp_nearby_stays_out(self):
        from extract import Candidate

        c = Candidate(
            cid="D9",
            value_mm=120,
            x=1570,
            y=1130,
            bbox=(0, 0, 0, 0),
            nearby="Поз. № | DN | Длина, мм | 50",
            local_hint="dimension_candidate",
            flags=set(),
        )
        llm = {"line_id": "X", "items": [{"id": "D9", "decision": "include", "role": "main", "reason": ""}], "notes": []}
        out, notes = apply_guards([c], llm)
        self.assertEqual(out["items"][0]["decision"], "not_length")
        self.assertTrue(any("штамп" in n for n in notes))

    def test_locked_untouched(self):
        s = self.by_no[1]
        llm = _decisions(s, {"D1": "not_length", "D1_r": "координата Z+"})
        llm["locked"] = True
        out, notes = apply_guards(s.candidates, llm)
        self.assertEqual(notes, [])
        by = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(by[next(c.cid for c in s.candidates if c.value_mm == 1383)], "not_length")


if __name__ == "__main__":
    unittest.main()
