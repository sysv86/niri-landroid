#!/usr/bin/env python3
"""Numerical check of a niri-landroid wallpaper frame (PPM from RWALLPPRS_DUMP_PPM).

Checks for the presence of the canonical Android 17 Easter Egg scene elements:
- the white outline of the Landroid ship,
- the green track (Colors.Track = 0xFF34A853),
- the blue autopilot (Colors.Autopilot = 0xFF4285F4; informational, it may be absent during
  the SIGHTSEEING phase),
- the chartreuse landing flag (Colors.Flag = 0xFFC6FF00),
- the Container dashes (canon color 0xFF800000, an opaque dark red over the Eigengrau background),
- a non-empty scene (number of unique colors).

Usage: verify_landroid_shot.py <frame.ppm> [--json]
"""

import sys
import json
from collections import Counter


def load_ppm(path: str):
    data = open(path, "rb").read()
    if not data.startswith(b"P6"):
        raise SystemExit(f"{path}: not a P6 PPM")
    pos = 2
    fields = []
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos : pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1
    w, h, maxval = fields
    assert maxval == 255, f"expected maxval=255, got {maxval}"
    px = data[pos:]
    if len(px) < w * h * 3:
        raise SystemExit(f"{path}: truncated pixels ({len(px)} < {w * h * 3})")
    return w, h, px


def near(rgb, target, tol=28):
    return all(abs(a - b) <= tol for a, b in zip(rgb, target))


EIGENGRAU = (0x16, 0x16, 0x1D)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    w, h, px = load_ppm(path)

    full = Counter()
    white = track = auto = flag = 0
    dashed = 0
    for i in range(0, len(px) - 2, 3):
        r, g, b = px[i], px[i + 1], px[i + 2]
        full[(r, g, b)] += 1
        if r > 220 and g > 220 and b > 220:
            white += 1
        elif near((r, g, b), (0x34, 0xA8, 0x53), 24):
            track += 1
        elif near((r, g, b), (0x42, 0x85, 0xF4), 24):
            auto += 1
        elif near((r, g, b), (0xC6, 0xFF, 0x00), 30):
            flag += 1
        # Container dashes: canon Color(0xFF800000) is an opaque dark red, so the dash pixels
        # land near (0x80, 0x00, 0x00) with antialiased edges toward the Eigengrau background.
        elif 0x4B <= r <= 0x9B and g <= 0x20 and b <= 0x24:
            dashed += 1

    report = {
        "file": path,
        "resolution": [w, h],
        "unique_colors": len(full),
        "ship_outline_white_px": white,
        "track_green_px": track,
        "autopilot_blue_px": auto,
        "flag_chartreuse_px": flag,
        "dashed_container_px": dashed,
        "checks": {
            "scene_not_empty": len(full) > 100,
            "ship_visible": white > 10,
            "track_visible": track > 0,
        },
    }
    if "--json" in sys.argv:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for k, v in report.items():
            if k == "checks":
                print("checks:")
                for ck, cv in v.items():
                    print(f"  {'OK ' if cv else 'FAIL'} {ck}")
            else:
                print(f"{k}: {v}")
    if not all(report["checks"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
