# niri-landroid — Android 17 "Landroid" Easter Egg wallpaper for Niri

A full port of the Android 17 Easter Egg screensaver (`DreamService`) to the scrollable-tiling Wayland
compositor **Niri**, rendered on the GPU through the [RWallpprs](https://github.com/sysv86/rwallpprs)
`wgpu` scene API.

There is no UI and no gameplay: you watch the starship **Landroid** fly a procedural universe on
autopilot, with the camera following it on a dynamic zoom, exactly like `DreamUniverse` /
`MainActivity.Spaaaaace` in the original.

Ported from the Android sources:

- **Simulation** — `Physics.kt`, `Universe.kt`, `Autopilot.kt`: gravity, landings and landing
  constraints, the flight track, exhaust sparks, the `Container` ringfence.
- **Rendering** — `VisibleUniverse.kt`: gravitational fields, star coronas, planet discs, strokes and
  textures, orbits, the dashed `Container`, the landing flag, the autopilot overlay and the track.

All geometry is drawn by a single instanced SDF pipeline in `wgpu`, plus a CPU-triangulated polygon
for the ship silhouette.

## Requirements

- **Niri** on Wayland, with `NIRI_SOCKET` available in the session.
- **Arch Linux** or an Arch-based distribution (CachyOS, EndeavourOS, Manjaro, Garuda, …).
- The **RWallpprs** engine (`rwallpprs` and `rwallpprs-daemon`) — the installer below takes care of it.
- A GPU with working **Vulkan** (or GL) support for `wgpu`.

## Install

One command: installs the dependencies, the RWallpprs engine if it is missing, builds the wallpaper
against the SDK, installs it, and starts it in a running Niri session.

```bash
curl -fsSL https://raw.githubusercontent.com/sysv86/niri-landroid/main/install.sh | bash
```

Options: `--prefix DIR`, `--no-deps`, `--no-start`, `--uninstall`. Run `./install.sh --help`.

### Manual

```bash
git clone https://github.com/sysv86/niri-landroid
cd niri-landroid
cargo build --release
rwallpprs set-bin ./target/release/niri-landroid --mode animated --fps 60
```

## Usage

```bash
# Show the wallpaper on the Niri desktop (the daemon must be running)
rwallpprs set-bin ./target/release/niri-landroid --mode animated --fps 60

# Fix the universe instead of using the daily seed
rwallpprs set-bin ./target/release/niri-landroid --mode animated --fps 60 -- --seed 12345
```

### Universe seed

Resolution order, highest priority first:

1. `--seed N` (or `--seed=N`) on the command line.
2. The `RWALLPPRS_SEED=N` environment variable.
3. `dailySeed` — `year*10000 + month*100 + day` in UTC, matching `randomSeed()` in the original.

The same seed produces a bit-identical universe, which is what makes the checks below reproducible.

### Standalone mode and self-test

Without `RWALLPPRS_SHM_FD` the binary runs standalone instead of talking to the daemon:

```bash
./target/release/niri-landroid --seed 12345                    # 900 frames, then exit
RWALLPPRS_SELFTEST=300 RWALLPPRS_DUMP_PPM=/tmp/f.ppm \
    ./target/release/niri-landroid --seed 12345                # N frames + PPM dump
RWALLPPRS_STATS=1 ./target/release/niri-landroid --seed 12345  # per-frame telemetry on stderr
RWALLPPRS_DEBUG_SHIP_R=190000 ./target/release/niri-landroid --seed 12345  # ship at the ringfence
```

`RWALLPPRS_DEBUG_SHIP_R` parks the ship at a given distance from the universe centre. It exists
because in normal flight the ringfence is almost always outside the frame — the autopilot stays inside
planetary orbits while the camera follows the ship — so this is the only practical way to inspect the
`Container` rendering pixel by pixel.

### Verifying a frame

`tools/verify_landroid_shot.py` checks a dumped frame for the canonical elements of the scene: the
white ship outline, the green track (`Colors.Track`), the blue autopilot overlay (`Colors.Autopilot`),
the chartreuse landing flag (`Colors.Flag`), the dashed `Container`, and the overall non-emptiness of
the scene.

```bash
RWALLPPRS_SELFTEST=900 RWALLPPRS_DUMP_PPM=/tmp/frame.ppm \
    ./target/release/niri-landroid --seed 12345
python3 tools/verify_landroid_shot.py /tmp/frame.ppm --json
```

The autopilot overlay is informational: it can legitimately be absent during the `SIGHTSEEING` phase.

## Fidelity and performance

The renderer was checked against the originals frame by frame: draw order
(grid → constraints → star → planets/sparks → autopilot → ship → track), line hatching measured in
screen pixels (`1/zoom`, `2/zoom`, `3/zoom`), the ringfence dashed at 8/8 screen pixels regardless of
zoom, the gravitational field formulas, star coronas, planet textures and the autopilot overlay.

Invisible objects, sparks and track segments are culled against a frame expanded by 10%. The culling
is conservative — thin circles and very large rings are tested for intersection with the frame rather
than by their centre — so the image is unchanged (verified by pixel diff with and without culling:
0 differing pixels across scenes) while frame time drops by roughly half: at 1920×1080 the self-test
measures about **3.3 ms/frame with culling** against **~6.5 ms/frame without** (median of 3 paired
runs of 1200 frames on Vulkan; absolute numbers depend on GPU load).

### Known deviations from the original

- `PathEffect.cornerPathEffect(radius = 200f)` on the star coronas is not reproduced: the coronas are
  drawn as polylines without corner rounding. At a 37-ray star the corners are obtuse and the
  rounding radius is only about 17 world units per ray.
- Grid step: the original uses `32.dp.toPx()`; here it is 32 screen pixels (on desktop, dp ≈ px).
- The ship silhouette fill is CPU-triangulated (ear clipping) without edge antialiasing, while
  Compose draws it with AA.
- `Container` is drawn in the canonical `Colors` value `Color(0xFF800000)`. That is ARGB with
  alpha = `0xFF`, i.e. an **opaque** dark red `#800000`, not "alpha = 0.5" — `0x80` there is the red
  channel, not the alpha channel.
- `STAR_POINTS`: the canonical expression
  `Build.VERSION.SDK_INT.takeIf { it in 1..99 } ?: 31` yields 37 for Android 17 (API 37), so the
  coronas have 37 and 38 rays.

## Regenerating the vector assets

`src/assets.rs` is generated, do not edit it by hand. It is flattened from the AOSP vector paths by
`tools/extract_landroid_assets.py`, which reimplements the `PathTools.kt` SVG parser with identical
semantics (including its tokenizer quirk, where a `Z` followed by a newline is silently dropped, so
the flattened geometry matches what the Easter Egg actually draws).

```bash
python3 tools/extract_landroid_assets.py --src /path/to/ee/src/com/android/egg/landroid \
    --out src/assets.rs --report
```

## Attribution

The vector geometry and palette in `src/assets.rs` are derived from the AOSP Easter Egg application
(`packages/apps/EasterEgg`, `android-17.0.0_r1`), which is distributed under the Apache License 2.0.
They are included here as a faithful port. The port itself — simulation, renderer and Rust code — is
MIT-licensed, see `LICENSE`.

## License

MIT. See `LICENSE`.
