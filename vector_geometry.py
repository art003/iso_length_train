from __future__ import annotations

import math
from dataclasses import dataclass, field


BODY_WIDTH_MIN = 0.18
BODY_WIDTH_MAX = 1.20
BODY_LEN_MIN = 28.0
BODY_LEN_MAX = 900.0
TICK_LEN_MIN = 3.5
TICK_LEN_MAX = 16.0
LINK_DIST_MAX = 22.0
PARALLEL_MIN = 0.97


Point = tuple[float, float]


@dataclass
class Segment:
    p1: Point
    p2: Point
    length: float
    angle: float
    axis: Point
    width: float
    drawing_type: str
    seqno: int


@dataclass
class DimensionLine:
    line_id: str
    angle: float
    offset: float
    axis: Point
    a: Point
    b: Point
    body_segments: list[Segment] = field(default_factory=list)
    extension_ticks: list[Segment] = field(default_factory=list)

    @property
    def length(self) -> float:
        return _dist(self.a, self.b)


@dataclass
class SpanDimLink:
    dim_line_id: str
    dist_text_to_line: float
    parallel_score: float
    t_center: float
    endpoints: tuple[Point, Point]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(dx: float, dy: float) -> Point:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (dx / n, dy / n)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def _normal(axis: Point) -> Point:
    return (-axis[1], axis[0])


def _point_seg_dist(p: Point, a: Point, b: Point) -> float:
    ab = _sub(b, a)
    den = _dot(ab, ab)
    if den < 1e-9:
        return _dist(p, a)
    t = max(0.0, min(1.0, _dot(_sub(p, a), ab) / den))
    proj = _add(a, _scale(ab, t))
    return _dist(p, proj)


def _point_line_dist(p: Point, a: Point, axis: Point) -> float:
    return abs(_dot(_sub(p, a), _normal(axis)))


def _project_t(p: Point, a: Point, b: Point) -> float:
    ab = _sub(b, a)
    den = _dot(ab, ab)
    if den < 1e-9:
        return 0.0
    return _dot(_sub(p, a), ab) / den


def _rect_to_lines(rect) -> list[tuple[Point, Point]]:
    x0, y0, x1, y1 = float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def flatten_drawings(page) -> list[Segment]:
    segs: list[Segment] = []
    for drawing in page.get_drawings():
        width = float(drawing.get("width") or 0.0)
        dtype = str(drawing.get("type") or "")
        seqno = int(drawing.get("seqno") or 0)
        items = drawing.get("items") or []
        pairs: list[tuple[Point, Point]] = []
        for item in items:
            kind = item[0]
            if kind == "l" and len(item) >= 3:
                p1, p2 = item[1], item[2]
                pairs.append(((float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))))
            elif kind == "re" and len(item) >= 2:
                pairs.extend(_rect_to_lines(item[1]))
        for p1, p2 in pairs:
            length = _dist(p1, p2)
            if length < 1.0:
                continue
            axis = _unit(p2[0] - p1[0], p2[1] - p1[1])
            angle = math.atan2(axis[1], axis[0])
            segs.append(
                Segment(
                    p1=p1,
                    p2=p2,
                    length=length,
                    angle=angle,
                    axis=axis,
                    width=width,
                    drawing_type=dtype,
                    seqno=seqno,
                )
            )
    return segs


def classify_segments(segs: list[Segment]) -> tuple[list[Segment], list[Segment]]:
    bodies: list[Segment] = []
    ticks: list[Segment] = []
    for s in segs:
        if BODY_LEN_MIN <= s.length <= BODY_LEN_MAX and BODY_WIDTH_MIN <= s.width <= BODY_WIDTH_MAX:
            bodies.append(s)
        elif TICK_LEN_MIN <= s.length <= TICK_LEN_MAX and s.width <= BODY_WIDTH_MAX:
            ticks.append(s)
    return bodies, ticks


def _canon_axis(axis: Point) -> Point:
    if axis[0] < -1e-9 or (abs(axis[0]) <= 1e-9 and axis[1] < 0):
        return (-axis[0], -axis[1])
    return axis


def cluster_dimension_lines(bodies: list[Segment], ticks: list[Segment]) -> list[DimensionLine]:
    if not bodies:
        return []
    buckets: dict[tuple[int, int], list[Segment]] = {}
    for seed in bodies:
        axis = _canon_axis(seed.axis)
        offset = _dot(seed.p1, _normal(axis))
        angle_bin = int(round(math.atan2(axis[1], axis[0]) * 36 / math.pi))
        off_bin = int(round(offset / 2.0))
        buckets.setdefault((angle_bin, off_bin), []).append(seed)

    lines: list[DimensionLine] = []
    idx = 0
    for group in buckets.values():
        axis = _canon_axis(group[0].axis)
        origin = group[0].p1
        items = []
        for seg in group:
            t1 = _dot(_sub(seg.p1, origin), axis)
            t2 = _dot(_sub(seg.p2, origin), axis)
            items.append((min(t1, t2), max(t1, t2), seg))
        items.sort(key=lambda x: x[0])
        cur = [items[0]]
        clusters = []
        for item in items[1:]:
            prev_end = max(x[1] for x in cur)
            if item[0] <= prev_end + 14.0:
                cur.append(item)
            else:
                clusters.append(cur)
                cur = [item]
        clusters.append(cur)
        for cl in clusters:
            segs = [x[2] for x in cl]
            t0 = min(x[0] for x in cl)
            t1 = max(x[1] for x in cl)
            a = _add(origin, _scale(axis, t0))
            b = _add(origin, _scale(axis, t1))
            offset = _dot(a, _normal(axis))
            idx += 1
            line = DimensionLine(
                line_id=f"L{idx}",
                angle=math.atan2(axis[1], axis[0]),
                offset=offset,
                axis=axis,
                a=a,
                b=b,
                body_segments=segs,
            )
            for tick in ticks:
                if abs(_dot(tick.axis, axis)) > 0.35:
                    continue
                near_end = min(_point_seg_dist(tick.p1, a, b), _point_seg_dist(tick.p2, a, b))
                if near_end <= 4.0:
                    line.extension_ticks.append(tick)
            lines.append(line)
    return lines


def link_candidate_to_line(c, lines: list[DimensionLine]) -> SpanDimLink | None:
    text_dir = getattr(c, "direction", (1.0, 0.0))
    best: SpanDimLink | None = None
    best_key = None
    p = (float(c.x), float(c.y))
    for line in lines:
        parallel = abs(_dot(text_dir, line.axis))
        if parallel < PARALLEL_MIN:
            continue
        dist = _point_seg_dist(p, line.a, line.b)
        if dist > LINK_DIST_MAX:
            inf = _point_line_dist(p, line.a, line.axis)
            t = _project_t(p, line.a, line.b)
            if inf > LINK_DIST_MAX or t < -0.15 or t > 1.15:
                continue
            dist = inf
        t_center = max(0.0, min(1.0, _project_t(p, line.a, line.b)))
        key = (dist, -parallel)
        if best_key is None or key < best_key:
            best_key = key
            best = SpanDimLink(
                dim_line_id=line.line_id,
                dist_text_to_line=dist,
                parallel_score=parallel,
                t_center=t_center,
                endpoints=(line.a, line.b),
            )
    return best


def attach_geometry(page, candidates) -> list[DimensionLine]:
    segs = flatten_drawings(page)
    bodies, ticks = classify_segments(segs)
    lines = cluster_dimension_lines(bodies, ticks)
    by_id = {line.line_id: line for line in lines}
    for c in candidates:
        link = link_candidate_to_line(c, lines)
        if not link:
            c.dim_line_id = ""
            c.dist_to_line = 0.0
            c.parallel_score = 0.0
            c.dim_endpoints = ((0.0, 0.0), (0.0, 0.0))
            continue
        c.dim_line_id = link.dim_line_id
        c.dist_to_line = float(link.dist_text_to_line)
        c.parallel_score = float(link.parallel_score)
        c.dim_endpoints = link.endpoints
        line = by_id.get(link.dim_line_id)
        if line is not None:
            c.dim_axis = line.axis
            c.dim_offset = float(line.offset)
        else:
            c.dim_axis = getattr(c, "direction", (1.0, 0.0))
            c.dim_offset = 0.0
    return lines


def extract_page_geometry(page, candidates) -> list[DimensionLine]:
    return attach_geometry(page, candidates)
