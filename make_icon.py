#!/usr/bin/env python3
"""
make_icon.py — generate icon.ico (Windows) and icon.icns (Mac) for PQ Analyzer.
Run once: python3 make_icon.py
"""

import math
import os
import shutil
import struct
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).parent

# ── Design constants ──────────────────────────────────────────────────────────
_BG_TOP    = (15,  45,  95)   # deep navy
_BG_BOT    = (26, 111, 191)   # Xcel blue
_WAVE_CLR  = (255, 255, 255, 230)
_HARM_CLR  = (255, 210,  80, 160)  # amber — harmonic overlay
_TEXT_CLR  = (255, 255, 255, 255)


def _draw_icon(size: int) -> Image.Image:
    img  = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Background: vertical gradient via scanlines ───────────────────────
    r0, g0, b0 = _BG_TOP
    r1, g1, b1 = _BG_BOT
    for y in range(size):
        t = y / (size - 1)
        r = int(r0 + (r1 - r0) * t)
        g = int(g0 + (g1 - g0) * t)
        b = int(b0 + (b1 - b0) * t)
        draw.line([(0, y), (size - 1, y)], fill=(r, g, b, 255))

    # Rounded corners — mask out the corners with transparency
    radius = max(4, size // 8)
    mask   = Image.new("L", (size, size), 0)
    md     = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    img.putalpha(mask)
    draw = ImageDraw.Draw(img)

    # ── Waveform ──────────────────────────────────────────────────────────
    n      = max(300, size * 2)
    cx     = size / 2
    cy     = size * 0.48
    amp    = size * 0.22          # fundamental amplitude
    x0     = size * 0.08
    x1     = size * 0.92
    cycles = 1.6                  # full cycles across the icon

    def wave_y(t, harmonics=True):
        phase = t * cycles * 2 * math.pi
        y = math.sin(phase) * amp
        if harmonics:
            # 5th harmonic (VFD signature): ~20% of fundamental
            y += math.sin(5 * phase) * amp * 0.22
            # 7th harmonic: ~12%
            y += math.sin(7 * phase) * amp * 0.12
        return cy - y

    # Harmonic overlay (amber, thinner)
    harm_pts = [(x0 + (x1 - x0) * i / n, wave_y(i / n, harmonics=True))
                for i in range(n + 1)]
    lw = max(1, size // 80)
    for i in range(len(harm_pts) - 1):
        draw.line([harm_pts[i], harm_pts[i + 1]], fill=_HARM_CLR, width=lw)

    # Clean fundamental (white, thicker, on top)
    fund_pts = [(x0 + (x1 - x0) * i / n, wave_y(i / n, harmonics=False))
                for i in range(n + 1)]
    lw2 = max(2, size // 44)
    for i in range(len(fund_pts) - 1):
        draw.line([fund_pts[i], fund_pts[i + 1]], fill=_WAVE_CLR, width=lw2)

    # ── "PQ" label ────────────────────────────────────────────────────────
    font_size = max(6, size // 4)
    font      = None
    # Try a few system fonts; fall back to default
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        if Path(candidate).exists():
            try:
                from PIL import ImageFont as _IFont
                font = _IFont.truetype(candidate, font_size)
                break
            except Exception:
                pass

    text     = "PQ"
    bbox     = draw.textbbox((0, 0), text, font=font)
    tw, th   = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx       = (size - tw) / 2 - bbox[0]
    ty       = size * 0.70 - bbox[1]

    # Subtle shadow
    shadow_offset = max(1, size // 128)
    draw.text((tx + shadow_offset, ty + shadow_offset), text,
              font=font, fill=(0, 0, 0, 90))
    draw.text((tx, ty), text, font=font, fill=_TEXT_CLR)

    # ── Thin bottom bar (accent) ──────────────────────────────────────────
    bar_h = max(2, size // 48)
    bar_y = size - bar_h - max(2, size // 32)
    draw.rectangle([size * 0.1, bar_y, size * 0.9, bar_y + bar_h],
                   fill=(255, 210, 80, 200))

    return img


def make_ico(out: Path):
    """Windows multi-resolution .ico"""
    sizes  = [16, 24, 32, 48, 64, 128, 256]
    frames = [_draw_icon(s).convert("RGBA") for s in sizes]
    frames[0].save(
        out, format="ICO",
        sizes=[(s, s) for s in sizes],
        append_images=frames[1:],
    )
    print(f"  Created {out}")


def make_icns(out: Path):
    """macOS .icns via iconutil"""
    iconset = HERE / "icon.iconset"
    iconset.mkdir(exist_ok=True)

    # iconutil expects specific filenames
    spec = [
        ("icon_16x16.png",       16),
        ("icon_16x16@2x.png",    32),
        ("icon_32x32.png",       32),
        ("icon_32x32@2x.png",    64),
        ("icon_128x128.png",    128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png",    256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png",    512),
        ("icon_512x512@2x.png",1024),
    ]
    for fname, sz in spec:
        img = _draw_icon(sz).convert("RGBA")
        img.save(iconset / fname)

    import subprocess
    result = subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
        capture_output=True, text=True,
    )
    shutil.rmtree(iconset)
    if result.returncode == 0:
        print(f"  Created {out}")
    else:
        print(f"  iconutil failed: {result.stderr.strip()}")
        sys.exit(1)


def make_png(out: Path, size: int = 512):
    img = _draw_icon(size).convert("RGBA")
    img.save(out)
    print(f"  Created {out}")


if __name__ == "__main__":
    print("Generating PQ Analyzer icons…")
    make_png(HERE / "icon.png")
    make_ico(HERE / "icon.ico")
    if sys.platform == "darwin":
        make_icns(HERE / "icon.icns")
    print("Done.")
