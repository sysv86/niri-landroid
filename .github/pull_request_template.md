## What changed

<!-- One paragraph. What is different after this PR? -->

## Why

<!-- The problem this solves, or the behaviour that was wrong. Fixes #123 -->

## Fidelity checklist

If this PR changes rendering, state what you compared against and how:

- [ ] The change does not alter the picture for existing seeds (a self-test PPM diff is byte-identical), **or**
- [ ] The change intentionally alters the picture and is closer to the Android original; the evidence is below.

```
RWALLPPRS_SELFTEST=900 RWALLPPRS_DUMP_PPM=/tmp/after.ppm ./target/release/niri-landroid --seed 12345
python3 tools/verify_landroid_shot.py /tmp/after.ppm
```

## Affected areas

- [ ] `src/sim.rs` (simulation)
- [ ] `src/render.rs` (renderer)
- [ ] `src/main.rs` (entry point, seed handling, self-test)
- [ ] `src/assets.rs` (generated — must be regenerated, not hand-edited)
- [ ] `tools/` (asset extraction, frame verification)
- [ ] Packaging, install script
- [ ] Documentation

## Checklist

- [ ] `cargo build --release` succeeds.
- [ ] A `RWALLPPRS_SELFTEST` run with a fixed seed still passes `tools/verify_landroid_shot.py`.
- [ ] Determinism is preserved: the same seed produces the same output.
- [ ] `src/assets.rs` was regenerated with `tools/extract_landroid_assets.py` if geometry changed.
- [ ] README "Known deviations" is updated if a deviation was introduced or removed.

## Screenshots

<!-- Before/after frames for anything visual. -->
