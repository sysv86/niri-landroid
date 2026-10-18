#!/usr/bin/env python3
"""Extract the Landroid (Android 17 Easter Egg) vector assets into Rust constants.

Source of truth:
  AOSP packages/apps/EasterEgg, src/com/android/egg/landroid/Assets.kt
  AOSP packages/apps/EasterEgg, src/com/android/egg/landroid/PathTools.kt
  AOSP packages/apps/EasterEgg, src/com/android/egg/landroid/Colors.kt

The Kotlin code stores every asset as an android.graphics.Path built from a
SVG path string via Path.parseSvgPathData() (see PathTools.kt).  This script
re-implements that parser with identical semantics and flattens the paths into
polylines that a Rust renderer can consume directly:

  * cubic beziers (C) are adaptively subdivided into line segments
  * paths are split into subpaths on M / Z
  * a recognised Z closes the current subpath (the start vertex is appended,
    exactly like android.graphics.Path.close() emits the closing segment)

IMPORTANT -- faithful replication of an AOSP tokenizer quirk:
PathTools.kt tokenises with Regex("([A-Za-z])\\s*([-.,0-9e ]+)"), whose second
group does NOT include the newline character.  A "Z" immediately followed by a
newline therefore never matches, and the corresponding close() is silently
dropped by the real app.  Only "Z" followed by a space is recognised (60 of
the 127 Z's in Assets.kt).  This script reproduces that behaviour exactly, so
the flattened geometry matches what the Easter Egg actually draws (e.g.
spaceshipLegs and texture_bigradient genuinely come out as open subpaths).

Output: src/assets.rs  (do not edit by hand)

Usage:
    python3 tools/extract_landroid_assets.py --src PATH [--out PATH]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "assets.rs",
)

# The 10 planet textures, in the exact order of Assets.kt's planetTextures array.
TEXTURE_VALS = [
    "texture_gradient1",
    "texture_contour1",
    "texture_contour2",
    "texture_contour3",
    "texture_planet17",
    "texture_planet17_nostorm",
    "texture_bigradient",
    "texture_bands1",
    "texture_bands2",
    "texture_offset_circles",
]

# Kotlin val name -> Rust const name
VAL_TO_CONST = {
    "spaceshipPath": "SPACESHIP_PATH",
    "spaceshipLegs": "SPACESHIP_LEGS",
    "thrustPath": "THRUST_PATH",
    "texture_gradient1": "TEXTURE_GRADIENT1",
    "texture_contour1": "TEXTURE_CONTOUR1",
    "texture_contour2": "TEXTURE_CONTOUR2",
    "texture_contour3": "TEXTURE_CONTOUR3",
    "texture_planet17": "TEXTURE_PLANET17",
    "texture_planet17_nostorm": "TEXTURE_PLANET17_NOSTORM",
    "texture_bigradient": "TEXTURE_BIGRADIENT",
    "texture_bands1": "TEXTURE_BANDS1",
    "texture_bands2": "TEXTURE_BANDS2",
    "texture_offset_circles": "TEXTURE_OFFSET_CIRCLES",
}

PLANET_TEXTURE_SIZE = 128.0

# Colours from Colors.kt (ARGB hex) -> Rust const name.
COLORS = [
    ("EIGENGRAU", 0xFF16161D),
    ("EIGENGRAU2", 0xFF292936),
    ("EIGENGRAU3", 0xFF3C3C4F),
    ("EIGENGRAU4", 0xFFA7A7CA),
    ("CONSOLE", 0xFFB7B7FF),
    ("AUTOPILOT", 0xFF4285F4),
    ("TRACK", 0xFF34A853),
    ("FLAG", 0xFFC6FF00),
]

# Adaptive bezier flattening: recurse until the control points are within
# FLATTEN_TOL of the chord, or MAX_DEPTH is hit (2**MAX_DEPTH max segments).
FLATTEN_TOL = 0.01
MAX_DEPTH = 10

# --------------------------------------------------------------------------- #
# SVG path parser (mirrors PathTools.kt::parseSvgPathData)
# --------------------------------------------------------------------------- #

# PathTools.kt tokenises with Regex("([A-Za-z])\\s*([-.,0-9e ]+)").
_TOKEN_RE = re.compile(r"([A-Za-z])\s*([-.,0-9e ]+)")


class SvgParseError(Exception):
    pass


def _numbers(raw: str) -> list[float]:
    # Kotlin splits on \s+ only; commas are also accepted here for robustness.
    parts = [p for p in re.split(r"[\s,]+", raw) if p]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError as exc:  # pragma: no cover - malformed input
            raise SvgParseError(f"cannot parse number {p!r} in {raw!r}") from exc
    return out


def _flatness(p0, c1, c2, p3) -> float:
    """Max distance of the control points from the chord p0->p3."""
    dx = p3[0] - p0[0]
    dy = p3[1] - p0[1]
    seg = math.hypot(dx, dy)
    if seg < 1e-12:
        return max(
            math.hypot(c1[0] - p0[0], c1[1] - p0[1]),
            math.hypot(c2[0] - p0[0], c2[1] - p0[1]),
        )
    # |cross| / |seg| == perpendicular distance
    d1 = abs((c1[0] - p0[0]) * dy - (c1[1] - p0[1]) * dx) / seg
    d2 = abs((c2[0] - p0[0]) * dy - (c2[1] - p0[1]) * dx) / seg
    return max(d1, d2)


def _subdivide(p0, c1, c2, p3, out, depth=0):
    """Adaptively flatten a cubic bezier, appending points after p0."""
    if depth >= MAX_DEPTH or _flatness(p0, c1, c2, p3) <= FLATTEN_TOL:
        out.append(p3)
        return

    # de Casteljau split at t = 0.5
    def mid(a, b):
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    a = mid(p0, c1)
    b = mid(c1, c2)
    c = mid(c2, p3)
    d = mid(a, b)
    e = mid(b, c)
    f = mid(d, e)
    _subdivide(p0, a, d, f, out, depth + 1)
    _subdivide(f, e, c, p3, out, depth + 1)


def parse_svg_path_data(d: str) -> list[list[tuple[float, float]]]:
    """Parse an SVG path string into subpaths of points.

    Returns a list of closed/open polylines.  Z closes a subpath by repeating
    its first vertex, matching android.graphics.Path.close().
    """
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    lastx = 0.0
    lasty = 0.0

    def flush():
        if current:
            subpaths.append(current)

    for m in _TOKEN_RE.finditer(d):
        cmd = m.group(1)
        args = _numbers(m.group(2) or "")
        if cmd == "M":
            flush()
            current = [(args[0], args[1])]
            lastx, lasty = args[0], args[1]
        elif cmd == "C":
            x0, y0 = current[-1]
            p3 = (args[4], args[5])
            _subdivide((x0, y0), (args[0], args[1]), (args[2], args[3]), p3, current)
            lastx, lasty = args[4], args[5]
        elif cmd == "L":
            current.append((args[0], args[1]))
            lastx, lasty = args[0], args[1]
        elif cmd == "l":
            nx, ny = lastx + args[0], lasty + args[1]
            current.append((nx, ny))
            lastx, lasty = nx, ny
        elif cmd == "H":
            current.append((args[0], lasty))
            lastx = args[0]
        elif cmd == "h":
            nx = lastx + args[0]
            current.append((nx, lasty))
            lastx = nx
        elif cmd == "V":
            current.append((lastx, args[0]))
            lasty = args[0]
        elif cmd == "v":
            ny = lasty + args[0]
            current.append((lastx, ny))
            lasty = ny
        elif cmd == "Z":
            # Recognised Z only (see module docstring: a "Z" followed by a
            # newline is dropped by PathTools.kt's tokenizer and never reaches
            # close()).  close() returns the pen to the contour start, but
            # PathTools.kt does *not* update lastx/lasty on Z, so relative
            # commands and H/V after a closed contour stay relative to the
            # previous vertex.  Mirror that exactly.
            if current and current[0] != current[-1]:
                current.append(current[0])
            flush()
            current = []
        else:
            raise SvgParseError(f"unsupported SVG command: {cmd}")

    flush()
    return subpaths


# --------------------------------------------------------------------------- #
# createPolygon (mirrors PathTools.kt::createPolygon)
# --------------------------------------------------------------------------- #

PI2 = 2.0 * math.pi


def create_polygon(radius: float, sides: int) -> list[list[tuple[float, float]]]:
    pts = [(radius, 0.0)]
    angle_step = PI2 / sides
    for i in range(1, sides):
        pts.append((radius * math.cos(angle_step * i), radius * math.sin(angle_step * i)))
    pts.append(pts[0])  # close()
    return [pts]


def translate(subpaths, dx: float, dy: float):
    return [[(x + dx, y + dy) for (x, y) in sp] for sp in subpaths]


def thrust_path() -> list[list[tuple[float, float]]]:
    # val thrustPath = createPolygon(-3f, 3).also { it.translate(Vec2(-5f, 0f)) }
    return translate(create_polygon(-3.0, 3), -5.0, 0.0)


# --------------------------------------------------------------------------- #
# Kotlin source extraction
# --------------------------------------------------------------------------- #

# val <name> ... """ <data> """
_VAL_RE_TMPL = r"val\s+{name}\s*=.*?\"\"\"(.*?)\"\"\""


def extract_svg(kt_source: str, val_name: str) -> str:
    m = re.search(_VAL_RE_TMPL.format(name=re.escape(val_name)), kt_source, re.DOTALL)
    if not m:
        raise SvgParseError(f"could not find SVG literal for val {val_name}")
    return m.group(1)


# --------------------------------------------------------------------------- #
# Rust emission
# --------------------------------------------------------------------------- #


def fmt(v: float) -> str:
    if abs(v) < 1e-6:
        v = 0.0
    s = f"{v:.6f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


def emit_path(name: str, subpaths, indent: str = "") -> str:
    """Emit `pub const NAME: &[&[(f32, f32)]] = &[ ... ];`"""
    pad = indent + "    "
    lines = [f"{indent}pub const {name}: &[&[(f32, f32)]] = &["]
    for sp in subpaths:
        lines.append(f"{pad}&[")
        row = pad + "    "
        for (x, y) in sp:
            item = f"({fmt(x)}, {fmt(y)}), "
            if len(row) + len(item) > 98 and row != pad + "    ":
                lines.append(row.rstrip())
                row = pad + "    "
            row += item
        if row.strip():
            lines.append(row.rstrip())
        lines.append(f"{pad}],")
    lines.append(f"{indent}];")
    return "\n".join(lines)


def srgb_component(byte: int) -> str:
    return fmt(round(byte / 255.0, 3))


def rust_const_name(name: str) -> str:
    return VAL_TO_CONST.get(name, name.upper())


def generate(src_dir: str) -> tuple[str, dict]:
    assets_kt = os.path.join(src_dir, "Assets.kt")
    with open(assets_kt, "r", encoding="utf-8") as fh:
        kt = fh.read()

    # 1. ship + textures
    paths: dict[str, list[list[tuple[float, float]]]] = {}
    for val in ["spaceshipPath", "spaceshipLegs"] + TEXTURE_VALS:
        paths[val] = parse_svg_path_data(extract_svg(kt, val))
    paths["thrustPath"] = thrust_path()

    stats = {
        val: {
            "subpaths": len(sp),
            "points": sum(len(s) for s in sp),
            "closed": sum(1 for s in sp if len(s) > 1 and s[0] == s[-1]),
        }
        for val, sp in paths.items()
    }

    out = []
    out.append("//! Generated by tools/extract_landroid_assets.py from AOSP packages/apps/EasterEgg (android-17.0.0_r1).")
    out.append("//! DO NOT EDIT BY HAND.")
    out.append("")
    out.append("/// Silhouette of the Landroid ship.")
    out.append(emit_path("SPACESHIP_PATH", paths["spaceshipPath"]))
    out.append("")
    out.append("/// Landing legs of the ship.")
    out.append(emit_path("SPACESHIP_LEGS", paths["spaceshipLegs"]))
    out.append("")
    out.append("/// Closed exhaust triangle (createPolygon(-3, 3), shifted by -5 along X).")
    out.append(emit_path("THRUST_PATH", paths["thrustPath"]))
    out.append("")

    for val in TEXTURE_VALS:
        out.append(emit_path(rust_const_name(val), paths[val]))
        out.append("")

    out.append("/// All planet textures, in the order of planetTextures[] from Assets.kt.")
    out.append("pub const PLANET_TEXTURES: &[&[&[(f32, f32)]]] = &[")
    for val in TEXTURE_VALS:
        out.append(f"    {rust_const_name(val)},")
    out.append("];")
    out.append("")
    out.append("/// Planet textures are specified in the coordinate system 0..PLANET_TEXTURE_SIZE.")
    out.append("pub const PLANET_TEXTURE_SIZE: f32 = 128.0;")
    out.append("")
    out.append("/// Color in sRGB, components 0..1. Conversion to linear space is done by the renderer.")
    out.append("pub struct LinearColor(pub f32, pub f32, pub f32, pub f32);")
    out.append("")
    for name, argb in COLORS:
        r = (argb >> 16) & 0xFF
        g = (argb >> 8) & 0xFF
        b = argb & 0xFF
        hexval = f"0x{argb:08X}"
        out.append(
            f"pub const {name}: LinearColor = "
            f"LinearColor({srgb_component(r)}, {srgb_component(g)}, "
            f"{srgb_component(b)}, 1.0); // {hexval}"
        )
    out.append("")

    return "\n".join(out), stats


def validate(paths_stats: dict, subpaths: dict) -> None:
    problems = []
    for val, st in paths_stats.items():
        if st["subpaths"] < 1:
            problems.append(f"{val}: no subpaths")
        for sp in subpaths[val]:
            for (x, y) in sp:
                if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                    problems.append(f"{val}: NaN/Inf coordinate")
    if problems:
        raise SystemExit("VALIDATION FAILED:\n  " + "\n  ".join(problems))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="directory holding Assets.kt")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output assets.rs path")
    ap.add_argument("--report", action="store_true", help="print per-asset statistics")
    args = ap.parse_args()

    rust, stats = generate(args.src)

    # re-parse for validation (cheap) so validate() sees the real polylines
    paths = {}
    with open(os.path.join(args.src, "Assets.kt"), encoding="utf-8") as fh:
        kt = fh.read()
    for val in ["spaceshipPath", "spaceshipLegs"] + TEXTURE_VALS:
        paths[val] = parse_svg_path_data(extract_svg(kt, val))
    paths["thrustPath"] = thrust_path()
    validate(stats, paths)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(rust)

    print(f"wrote {args.out} ({len(rust)} bytes)")
    print(f"{'asset':<26} {'subpaths':>8} {'points':>7} {'closed':>6}  first point")
    for val, st in stats.items():
        first = paths[val][0][0] if paths[val] else None
        first_s = f"({fmt(first[0])}, {fmt(first[1])})" if first else "-"
        print(
            f"{val:<26} {st['subpaths']:>8} {st['points']:>7} {st['closed']:>6}  {first_s}"
        )

    if args.report:
        print()
        for val, sp in paths.items():
            pts = [p for s in sp for p in s]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            print(
                f"{val}: x=[{fmt(min(xs))}, {fmt(max(xs))}] "
                f"y=[{fmt(min(ys))}, {fmt(max(ys))}]"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
