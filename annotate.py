from __future__ import annotations
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from extract import Candidate


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_candidate_ids(png_bytes: bytes, candidates: list[Candidate], scale: float) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(14)
    for c in candidates:
        x = c.x * scale
        y = c.y * scale
        label = c.cid
        bb = draw.textbbox((0, 0), label, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        pad = 2
        box = [x - tw / 2 - pad, y - th - 10, x + tw / 2 + pad, y - 8]
        draw.rectangle(box, fill=(255, 230, 0), outline=(0, 0, 0))
        draw.text((box[0] + pad, box[1]), label, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


DECISION_COLORS = {
    "include": (0, 140, 70),
    "exclude_nested": (120, 50, 160),
    "not_length": (140, 140, 140),
    "ambiguous": (210, 120, 0),
}


def annotate_final(
    png_bytes: bytes,
    candidates: list[Candidate],
    decisions: dict[str, str],
    scale: float,
    title: str,
    formula: str,
    length_mm: int,
    status: str,
    out_path: Path,
) -> None:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(13)
    font_sm = _font(11)
    font_lg = _font(18)

    for c in candidates:
        dec = decisions.get(c.cid, "ambiguous")
        color = DECISION_COLORS.get(dec, (210, 120, 0))
        x0, y0, x1, y1 = [v * scale for v in c.bbox]
        pad = 4
        box = [x0 - pad, y0 - pad, x1 + pad, y1 + pad]
        draw.rectangle(box, outline=color, width=3)
        tag = f"{c.cid}:{c.value_mm}"
        tx, ty = x0 - pad, max(0, y0 - 16)
        tw = draw.textbbox((0, 0), tag, font=font_sm)[2]
        draw.rectangle([tx, ty, tx + tw + 4, ty + 14], fill=(255, 255, 255), outline=color)
        draw.text((tx + 2, ty), tag, fill=color, font=font_sm)

    legend = [
        ((0, 140, 70), "зелёный - в сумме"),
        ((120, 50, 160), "фиол. - вложенный"),
        ((210, 120, 0), "оранж. - проверить"),
        ((140, 140, 140), "серый - не длина"),
    ]
    lx, ly = 16, 16
    draw.rectangle([10, 10, 420, 118], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((lx, ly), title, fill=(0, 0, 0), font=font_lg)
    draw.text((lx, ly + 24), f"{length_mm} мм = {length_mm/1000:.3f} м  [{status}]", fill=(0, 90, 60), font=font)
    y = ly + 44
    for color, text in legend:
        draw.rectangle([lx, y, lx + 12, y + 12], fill=color)
        draw.text((lx + 18, y - 2), text, fill=(0, 0, 0), font=font_sm)
        y += 16

    # формула снизу
    w, h = img.size
    draw.rectangle([0, h - 36, w, h], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((12, h - 28), formula[:180], fill=(0, 0, 0), font=font_sm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
