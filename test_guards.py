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

    def test_sheet6_full_runs_replace_support_offsets(self):
        s = self.by_no[6]
        d6 = next(c for c in s.candidates if c.value_mm == 2950)
        d8 = next(c for c in s.candidates if c.value_mm == 2400)
        d11 = next(c for c in s.candidates if c.value_mm == 3505)
        self.assertNotIn("overall_pair", d6.flags)
        self.assertNotIn("overall_pair", d8.flags)
        mapping = {c.cid: "include" for c in s.candidates}
        mapping[d6.cid] = "include"
        mapping[d8.cid] = "exclude_nested"
        mapping[d8.cid + "_r"] = "габарит поверх D7"
        mapping[d11.cid] = "exclude_nested"
        mapping[d11.cid + "_r"] = "габарит поверх D10"
        out, notes = apply_guards(s.candidates, _decisions(s, mapping))
        by = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(by[d6.cid], "include")
        self.assertEqual(by[d8.cid], "include")
        self.assertEqual(by[d11.cid], "include")
        gold, _ = apply_guards(s.candidates, _decisions(s, {c.cid: "include" for c in s.candidates}))
        for value in (2463, 2000, 1600):
            cid = next(c.cid for c in s.candidates if c.value_mm == value)
            self.assertEqual({it["id"]: it["decision"] for it in gold["items"]}[cid], "exclude_nested")
        self.assertEqual(_sum_include(s, gold), 10913)

    def test_sheet5_full_runs_replace_support_offsets(self):
        s = self.by_no[5]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        for c in s.candidates:
            if c.value_mm in (238, 2000, 1600):
                self.assertEqual(by[c.cid], "exclude_nested")
            if c.value_mm in (450, 2400, 3505):
                self.assertEqual(by[c.cid], "include")
        self.assertEqual(_sum_include(s, out), 8175)

    def test_test1_sheet8_support_offsets_are_nested(self):
        """Все четыре размера до опор не добавляются поверх полного участка."""
        pdf = ROOT / "data" / "контрольные_тесты" / "тест_1.pdf"
        if not pdf.exists():
            self.skipTest("нет тест_1.pdf")
        s = extract_pdf(pdf, {8})[0]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        for val in (2600, 1750, 2750, 1600, 1349, 665):
            cid = next(c.cid for c in s.candidates if c.value_mm == val)
            self.assertEqual(by[cid], "exclude_nested", val)
        for val in (7400, 2200, 3316, 890):
            cid = next(c.cid for c in s.candidates if c.value_mm == val)
            self.assertEqual(by[cid], "include", val)
        self.assertEqual(_sum_include(s, out), 26960)

    def test_close_pair_nests_smaller_keeps_run(self):
        """665 внутри 890: полный участок в сумме, размер до опоры нет."""
        pdf = ROOT / "data" / "контрольные_тесты" / "тест_1.pdf"
        if not pdf.exists():
            self.skipTest("нет тест_1.pdf")
        s = extract_pdf(pdf, {7})[0]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        d665 = next(c for c in s.candidates if c.value_mm == 665)
        d890 = next(c for c in s.candidates if c.value_mm == 890)
        self.assertNotIn("overall_pair", d890.flags)
        self.assertEqual(by[d890.cid], "include")
        self.assertEqual(by[d665.cid], "exclude_nested")

    def test_run_parent_not_dropped_as_overall(self):
        pdf = ROOT / "data" / "контрольные_тесты" / "тест_2.pdf"
        if not pdf.exists():
            self.skipTest("нет тест_2.pdf")
        s = extract_pdf(pdf, {10})[0]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        d9000 = next(c for c in s.candidates if c.value_mm == 9000)
        self.assertEqual(by[d9000.cid], "include")
        nested_2100 = [c for c in s.candidates if c.value_mm == 2100]
        self.assertTrue(any(by[c.cid] == "exclude_nested" for c in nested_2100))

    def test_test1_sheet6_main_run_not_overall(self):
        """22250 — основная линия диагонали, не габарит; 3000 внутри неё."""
        pdf = ROOT / "data" / "контрольные_тесты" / "тест_1.pdf"
        if not pdf.exists():
            self.skipTest("нет тест_1.pdf")
        s = extract_pdf(pdf, {6})[0]
        out = self._guarded_all_include(s)
        by = {it["id"]: it["decision"] for it in out["items"]}
        d22250 = next(c for c in s.candidates if c.value_mm == 22250)
        d3000 = next(c for c in s.candidates if c.value_mm == 3000)
        self.assertEqual(by[d22250.cid], "include")
        self.assertEqual(by[d3000.cid], "exclude_nested")

    def _holdout_sheet(self, test_name: str, page: int):
        pdf = ROOT / "data" / "контрольные_тесты" / f"{test_name}.pdf"
        if not pdf.exists():
            self.skipTest(f"нет {test_name}.pdf")
        return extract_pdf(pdf, {page})[0]

    def test_long_dim_line_is_main_run_across_holdout(self):
        """Длинная размерная линия — основной участок; куски до опор внутри неё."""
        cases = [
            ("тест_1", 7, 21411, (6150, 6350, 5000)),
            ("тест_2", 4, 14550, (5500, 6000)),
            ("тест_2", 5, 8750, (2700, 5500)),
            ("тест_2", 6, 8530, (3530, 3000)),
            ("тест_2", 8, 10500, (3000,)),
            ("тест_2", 9, 41150, (6000, 5500, 3000)),
            ("тест_3", 1, 11000, (2000, 2500, 2265)),
            ("тест_3", 2, 26000, (6350, 5650, 6000)),
            ("тест_3", 4, 15050, (6000, 550)),
        ]
        for name, page, main, inners in cases:
            with self.subTest(name=name, page=page, main=main):
                s = self._holdout_sheet(name, page)
                out = self._guarded_all_include(s)
                by = {it["id"]: it["decision"] for it in out["items"]}
                mains = [c for c in s.candidates if c.value_mm == main]
                self.assertTrue(mains, f"нет {main}")
                self.assertTrue(any(by[c.cid] == "include" for c in mains), f"{main} должен быть в сумме")
                for val in inners:
                    kids = [c for c in s.candidates if c.value_mm == val]
                    self.assertTrue(
                        any(by[c.cid] == "exclude_nested" for c in kids),
                        f"{val} должен быть вложен в {main}",
                    )

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
            5: 8175,
            6: 10913,
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

    def test_overall_mark_accepts_trailing_comma(self):
        from extract import Candidate, Span, _tag_specials

        c = Candidate("D12", 13950, 10, 10, (0, 0, 0, 0), "Н.О, | 3000 | 6000", "dim")
        _tag_specials([c], [Span("Н.О,", 12, 12, (0, 0, 0, 0), 8)])
        self.assertIn("overall_mark", c.flags)

    def test_large_nearby_ignores_tiny_and_slope(self):
        from extract import Candidate
        from guards import _large_nearby_is_overall

        big = Candidate("D20", 14054, 0, 0, (0, 0, 0, 0), "305 | УКЛОН | 1000 | 8502", "dim")
        self.assertFalse(_large_nearby_is_overall(big, Candidate("D16", 305, 50, 0, (0, 0, 0, 0), "", "dim")))
        self.assertFalse(_large_nearby_is_overall(big, Candidate("D18", 1000, 40, 20, (0, 0, 0, 0), "", "dim")))
        self.assertFalse(_large_nearby_is_overall(big, Candidate("D19", 8502, 30, 10, (0, 0, 0, 0), "", "dim")))
        env = Candidate("D1", 22250, 0, 0, (0, 0, 0, 0), "3000", "dim")
        self.assertTrue(_large_nearby_is_overall(env, Candidate("D2", 3000, 15, 0, (0, 0, 0, 0), "", "dim")))
        self.assertTrue(_large_nearby_is_overall(env, Candidate("D2", 3000, 45, 0, (0, 0, 0, 0), "", "dim")))
        self.assertFalse(
            _large_nearby_is_overall(
                Candidate("D6", 17228, 0, 0, (0, 0, 0, 0), "90", "dim"),
                Candidate("D4", 90, 20, 0, (0, 0, 0, 0), "", "dim"),
            )
        )

    def test_sequential_neighbor_restored_to_include(self):
        from extract import Candidate

        a = Candidate("D7", 9256, 0, 0, (0, 0, 0, 0), "2006 | 5300", "dim")
        b = Candidate("D6", 5300, 18, 0, (0, 0, 0, 0), "9256 | 2006", "dim")
        c = Candidate("D5", 2006, 45, 0, (0, 0, 0, 0), "9256 | 5300", "dim")
        llm = {
            "line_id": "X",
            "items": [
                {"id": "D7", "decision": "exclude_nested", "role": "other", "reason": "CatBoost 0.90"},
                {"id": "D6", "decision": "include", "role": "main", "reason": ""},
                {"id": "D5", "decision": "include", "role": "main", "reason": ""},
            ],
            "notes": [],
        }
        out, _ = apply_guards([a, b, c], llm)
        by = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(by["D7"], "include")
        self.assertEqual(by["D6"], "include")
        self.assertEqual(by["D5"], "include")

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
