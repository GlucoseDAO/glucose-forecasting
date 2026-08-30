"""Build docs/manuscript2/sugar_jepa.png from the 128-step original.

The original (jepa_paper/sugar_jepa.png) is copied pixel-for-pixel; only the
288-window / 36-patch labels are rewritten.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SRC = HERE / "jepa_paper" / "sugar_jepa.png"
OUT = HERE / "sugar_jepa.png"

FONTS = Path(r"C:\Windows\Fonts")
TIMES = FONTS / "times.ttf"
TIMES_I = FONTS / "timesi.ttf"

WHITE = (255, 255, 255)
CAPTION = (40, 40, 40)
SHAPE = (70, 70, 70)
GREEN_NOTE = (90, 140, 105)
KV_GREEN = (80, 130, 95)


def _font(path: Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _cover(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    draw.rectangle(box, fill=color)


def _clone_rows(im: Image.Image, box: tuple[int, int, int, int], src_y: int) -> None:
    """Copy one fill row across a rectangle so box shading is preserved."""
    x0, y0, x1, y1 = box
    row = im.crop((x0, src_y, x1, src_y + 1))
    for y in range(y0, y1):
        im.paste(row, (x0, y))


def _center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    x = x0 + (x1 - x0 - tw) / 2 - l
    y = y0 + (y1 - y0 - th) / 2 - t
    draw.text((x, y), text, font=font, fill=fill)


def main() -> None:
    im = Image.open(SRC).convert("RGB")
    draw = ImageDraw.Draw(im)

    times13 = _font(TIMES, 13)
    times11 = _font(TIMES, 11)
    timesi11 = _font(TIMES_I, 11)
    timesi10 = _font(TIMES_I, 10.5)
    timesi12 = _font(TIMES_I, 12)

    # Input caption (do not touch "128 steps" under the glucose box).
    cap = (460, 1434, 922, 1450)
    _cover(draw, cap, WHITE)
    _center(
        draw,
        cap,
        "Input window x : (B, 288, 4) — 288 × 5 min = 24 h, MinMax-scaled",
        times13,
        CAPTION,
    )

    # Note that SugarOne still uses the last 128 of this 288 window.
    note = (460, 1484, 900, 1502)
    _cover(draw, note, WHITE)
    _center(
        draw,
        note,
        "SugarOne trunk uses last 128 of 288",
        timesi12,
        GREEN_NOTE,
    )

    # Green italic JEPA feed line.
    gnote = (460, 1543, 880, 1559)
    _cover(draw, gnote, WHITE)
    _center(
        draw,
        gnote,
        "glucose channel x[..., 0] — full 288-step window feeds the JEPA encoder",
        timesi12,
        GREEN_NOTE,
    )

    # JEPA glucose box: (B, 128) → (B, 288)
    glu = (1206, 1307, 1254, 1321)
    _clone_rows(im, glu, 1286)
    _center(draw, glu, "(B, 288)", times11, CAPTION)

    # Arrow label after instance z-score (leave the vertical arrow).
    zlab = (1230, 1176, 1292, 1192)
    _cover(draw, zlab, WHITE)
    _center(draw, zlab, "(B, 288)", timesi11, SHAPE)

    # PE subtitle inside the green box.
    pe = (1182, 1065, 1283, 1079)
    _clone_rows(im, pe, 1062)
    _center(draw, pe, "own buffer (1, 36, 96)", times11, CAPTION)

    # Arrow label after LayerNorm: (B, 16, 96) → (B, 36, 96)
    lnlab = (1236, 756, 1308, 774)
    _cover(draw, lnlab, WHITE)
    _center(draw, lnlab, "(B, 36, 96)", timesi11, SHAPE)

    # jepa K/V shape only; keep the "jepa K / V" line.
    kv = (1018, 996, 1100, 1014)
    _clone_rows(im, kv, 994)
    _center(draw, kv, "(B, 36, 32)", timesi10, KV_GREEN)

    im.save(OUT, "PNG")
    print(OUT)


if __name__ == "__main__":
    main()
