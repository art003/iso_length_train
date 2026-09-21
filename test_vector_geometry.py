from __future__ import annotations

import math
import unittest
from pathlib import Path

from extract import Candidate, extract_pdf
from guards import apply_guards
from relations import NEST_AUTO_THRESHOLD, compute_relations, find_nested_parent
from vector_geometry import (
    Segment,
    classify_segments,
    cluster_dimension_lines,
    flatten_drawings,
    link_candidate_to_line,
)


ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "data" / "02_Изометрии_10_листов.pdf"
TEST1 = ROOT / "data" / "контрольные_тесты" / "тест_1.pdf"


class _P:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


class _Page:
    def __init__(self, drawings):
        self._drawings = drawings

    def get_drawings(self):
        return self._drawings


def _seg(p1, p2, width=0.48, seqno=1) -> Segment:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    n = length or 1.0
    axis = (dx / n, dy / n)
    return Segment(
        p1=p1,
        p2=p2,
        length=length,
        angle=math.atan2(axis[1], axis[0]),
        axis=axis,
        width=width,
        drawing_type="s",
        seqno=seqno,
    )


def _cand(cid, value, x, y, nearby="", **kwargs) -> Candidate:
    return Candidate(
        cid=cid,
        value_mm=value,
        x=x,
        y=y,
        bbox=(x - 4, y - 4, x + 4, y + 4),
        nearby=nearby,
        local_hint="dim",
        flags=set(),
        **kwargs,
    )


class VectorGeometryUnit(unittest.TestCase):
    def test_flatten_lines_and_rects(self):
        page = _Page(
            [
                {
                    "width": 0.48,
                    "type": "s",
                    "seqno": 1,
                    "items": [("l", _P(0, 10), _P(80, 10))],
                },
                {
                    "width": 0.4,
                    "type": "s",
                    "seqno": 2,
                    "items": [("re", _Rect(0, 0, 40, 8))],
                },
            ]
        )
        segs = flatten_drawings(page)
        self.assertGreaterEqual(len(segs), 5)
        self.assertTrue(any(abs(s.length - 80) < 1e-6 for s in segs))

    def test_classify_body_and_ticks(self):
        bodies, ticks = classify_segments(
            [
                _seg((0, 0), (60, 0), width=0.48),
                _seg((0, 0), (8, 0), width=0.4),
                _seg((0, 0), (2, 0), width=0.4),
                _seg((0, 0), (80, 0), width=2.5),
            ]
        )
        self.assertEqual(len(bodies), 1)
        self.assertEqual(len(ticks), 1)

    def test_cluster_merges_collinear_gap(self):
        bodies = [_seg((0, 20), (40, 20)), _seg((50, 20), (90, 20))]
        ticks = [_seg((0, 16), (0, 24), width=0.4), _seg((90, 16), (90, 24), width=0.4)]
        lines = cluster_dimension_lines(bodies, ticks)
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0].length, 90.0, places=1)
        self.assertGreaterEqual(len(lines[0].extension_ticks), 2)

    def test_link_nearest_parallel_line(self):
        lines = cluster_dimension_lines([_seg((0, 20), (100, 20))], [])
        c = _cand("D1", 1000, 50, 25, direction=(1.0, 0.0))
        link = link_candidate_to_line(c, lines)
        self.assertIsNotNone(link)
        self.assertLess(link.dist_text_to_line, 6.0)
        self.assertGreaterEqual(link.parallel_score, 0.97)


class RelationGraphUnit(unittest.TestCase):
    def test_stacked_nested_with_shared_endpoint(self):
        child = _cand(
            "D1",
            300,
            40,
            30,
            nearby="О12 | 840",
            direction=(1.0, 0.0),
            dim_line_id="L1",
            dim_axis=(1.0, 0.0),
            dim_offset=30.0,
            dim_endpoints=((0.0, 30.0), (80.0, 30.0)),
        )
        parent = _cand(
            "D2",
            840,
            40,
            50,
            nearby="300 | О12",
            direction=(1.0, 0.0),
            dim_line_id="L2",
            dim_axis=(1.0, 0.0),
            dim_offset=50.0,
            dim_endpoints=((0.0, 50.0), (120.0, 50.0)),
        )
        rel = compute_relations([child, parent])
        self.assertEqual(rel["D1"].parent_id, "D2")
        self.assertGreaterEqual(rel["D1"].geometry_confidence, NEST_AUTO_THRESHOLD)
        self.assertIsNone(rel["D2"].parent_id)
        self.assertEqual(find_nested_parent(child, [child, parent]).cid, "D2")

    def test_sequential_collinear_not_nested(self):
        a = _cand(
            "D7",
            2006,
            20,
            10,
            nearby="5300 | 9256",
            direction=(1.0, 0.0),
            dim_line_id="L1",
            dim_axis=(1.0, 0.0),
            dim_offset=10.0,
            dim_endpoints=((0.0, 10.0), (40.0, 10.0)),
        )
        b = _cand(
            "D6",
            5300,
            70,
            10,
            nearby="2006 | 9256",
            direction=(1.0, 0.0),
            dim_line_id="L2",
            dim_axis=(1.0, 0.0),
            dim_offset=10.0,
            dim_endpoints=((40.0, 10.0), (100.0, 10.0)),
        )
        c = _cand(
            "D5",
            9256,
            160,
            10,
            nearby="2006 | 5300",
            direction=(1.0, 0.0),
            dim_line_id="L3",
            dim_axis=(1.0, 0.0),
            dim_offset=10.0,
            dim_endpoints=((100.0, 10.0), (220.0, 10.0)),
        )
        rel = compute_relations([a, b, c])
        self.assertIsNone(rel["D7"].parent_id)
        self.assertIsNone(rel["D6"].parent_id)
        self.assertIsNone(rel["D5"].parent_id)

    def test_cascade_keeps_middle_as_parent(self):
        small = _cand(
            "D1",
            300,
            20,
            20,
            nearby="О1 | 840",
            direction=(1.0, 0.0),
            dim_line_id="L1",
            dim_axis=(1.0, 0.0),
            dim_offset=20.0,
            dim_endpoints=((0.0, 20.0), (40.0, 20.0)),
        )
        mid = _cand(
            "D2",
            840,
            20,
            40,
            nearby="300 | 1950",
            direction=(1.0, 0.0),
            dim_line_id="L2",
            dim_axis=(1.0, 0.0),
            dim_offset=40.0,
            dim_endpoints=((0.0, 40.0), (80.0, 40.0)),
        )
        big = _cand(
            "D3",
            1950,
            20,
            60,
            nearby="840",
            direction=(1.0, 0.0),
            dim_line_id="L3",
            dim_axis=(1.0, 0.0),
            dim_offset=60.0,
            dim_endpoints=((0.0, 60.0), (140.0, 60.0)),
        )
        rel = compute_relations([small, mid, big])
        self.assertEqual(rel["D1"].parent_id, "D2")
        self.assertIsNone(rel["D2"].parent_id)


class PdfRelationRegression(unittest.TestCase):
    def _guarded(self, sheet):
        mapping = {
            "line_id": sheet.line_id,
            "items": [
                {"id": c.cid, "decision": "include", "role": "main", "reason": ""}
                for c in sheet.candidates
            ],
            "notes": [],
        }
        out, _ = apply_guards(sheet.candidates, mapping)
        return out

    def test_gold_sheet7_300_in_840_and_916_in_1950(self):
        if not GOLD.exists():
            self.skipTest("нет учебного pdf")
        sheet = extract_pdf(GOLD, {7})[0]
        by_val = {}
        for c in sheet.candidates:
            by_val.setdefault(c.value_mm, []).append(c)
        child_300 = by_val[300][0]
        child_916 = by_val[916][0]
        parent_840 = find_nested_parent(child_300, sheet.candidates)
        parent_1950 = find_nested_parent(child_916, sheet.candidates)
        self.assertIsNotNone(parent_840)
        self.assertEqual(parent_840.value_mm, 840)
        self.assertIsNotNone(parent_1950)
        self.assertEqual(parent_1950.value_mm, 1950)
        out = self._guarded(sheet)
        decisions = {it["id"]: it["decision"] for it in out["items"]}
        self.assertEqual(decisions[child_300.cid], "exclude_nested")
        self.assertEqual(decisions[child_916.cid], "exclude_nested")
        self.assertEqual(decisions[parent_840.cid], "include")
        self.assertEqual(decisions[parent_1950.cid], "include")

    def test_test1_sheet8_six_nested_and_sum(self):
        if not TEST1.exists():
            self.skipTest("нет тест_1.pdf")
        sheet = extract_pdf(TEST1, {8})[0]
        expected = {
            2600: 4350,
            1750: 4350,
            2750: 7400,
            1600: 2200,
            1349: 3316,
            665: 890,
        }
        by_val = {c.value_mm: c for c in sheet.candidates if c.value_mm in expected or c.value_mm in expected.values()}
        for child_val, parent_val in expected.items():
            child = by_val[child_val]
            parent = find_nested_parent(child, sheet.candidates)
            self.assertIsNotNone(parent, child_val)
            self.assertEqual(parent.value_mm, parent_val, child_val)
            self.assertGreaterEqual(child.geometry_confidence, NEST_AUTO_THRESHOLD, child_val)
        out = self._guarded(sheet)
        decisions = {it["id"]: it["decision"] for it in out["items"]}
        by = {c.cid: c.value_mm for c in sheet.candidates}
        for val in expected:
            self.assertEqual(decisions[by_val[val].cid], "exclude_nested", val)
        for val in expected.values():
            self.assertEqual(decisions[by_val[val].cid], "include", val)
        total = sum(by[it["id"]] for it in out["items"] if it["decision"] == "include")
        self.assertEqual(total, 26960)


class FeatureLeakage(unittest.TestCase):
    def test_features_do_not_include_labels(self):
        from local_model import FEATURE_NAMES, row_features

        forbidden = {"decision", "role", "reason", "locked", "reviewed_by_human"}
        self.assertFalse(forbidden.intersection(FEATURE_NAMES))
        feats = row_features(
            {
                "value_mm": 300,
                "decision": "exclude_nested",
                "role": "support_offset",
                "reason": "внутри D2",
                "parent_id": "D2",
                "geometry_confidence": 0.9,
                "parent_value_mm": 840,
                "parent_dist": 40,
                "dist_to_line": 5.5,
                "parallel_score": 1.0,
            }
        )
        self.assertFalse(forbidden.intersection(feats))
        self.assertEqual(feats["has_parent"], 1.0)
        self.assertGreater(feats["geo_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
