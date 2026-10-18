//! The `niri-landroid` wallpaper — a port of the Android 17 Easter Egg screensaver
//! (DreamService) "Landroid" to a full wgpu renderer SDK.
//!
//! The UI and gameplay are NOT ported: an endless observation of the Landroid ship,
//! which flies around the universe on autopilot.
//!
//! ## Standalone / self-test
//! If there is no `RWALLPPRS_SHM_FD` (or `RWALLPPRS_SELFTEST=N` is set), `main` runs
//! N simulation frames with `dt = 1/60`, calling `render_scene` through the SDK adapter,
//! and (with `RWALLPPRS_DUMP_PPM`) saves the last frame to a PPM for checking.
//!
//! Environment variables / arguments:
//! * `--seed N` (or `--seed=N`) — universe seed; takes priority over `RWALLPPRS_SEED`,
//!   which in turn takes priority over dailySeed (year*10000+month*100+day in UTC);
//! * `RWALLPPRS_SELFTEST=N` — run N frames without SHM and exit;
//! * `RWALLPPRS_DUMP_PPM=<path>` — save the last self-test frame to a PPM (P6);
//! * `RWALLPPRS_STATS=1` — print telemetry (position, zoom, instance count) to stderr;
//! * `RWALLPPRS_DEBUG_SHIP_R=<r>` — place the ship at distance `r` from the center
//!   of the universe (diagnostics: the ringfence is almost always outside the frame in normal flight).
//!
//! Note: instead of the `run_scene_wallpaper!` macro, a custom `main` is written here,
//! because the SDK's regular standalone path draws exactly one frame, while we need a
//! multi-frame self-test (the same `SceneWallpaperAdapter` and `run_wallpaper_loop`
//! are used for the daemon's production mode).

#[allow(dead_code)] // assets.rs is generated and contains constants, some of which are unused
mod assets;
mod render;
mod sim;

use rwallpprs_sdk::{
    Canvas, Color, FrameContext, GpuScene, GpuSceneWallpaper, NiriViewportState,
    SceneWallpaperAdapter, Wallpaper,
};

use render::Renderer;
use sim::{Simulator, Vec2};

/// From MainActivity.kt.
const DEFAULT_CAMERA_ZOOM: f32 = 1.0;
const MIN_CAMERA_ZOOM: f32 = 250.0 / sim::UNIVERSE_RANGE;
const MAX_CAMERA_ZOOM: f32 = 5.0;

pub struct LandroidWallpaper {
    universe: Simulator,
    renderer: Renderer,
    camera_zoom: f32,
    /// `RWALLPPRS_STATS=1` — print per-frame telemetry to stderr (diagnostics).
    stats: bool,
}

impl LandroidWallpaper {
    /// Seed priority: the `--seed N` argument (or `--seed=N`) -> the environment variable
    /// `RWALLPPRS_SEED` -> dailySeed = year*10000 + month*100 + day (today in UTC).
    fn resolve_seed(args: &[String]) -> i64 {
        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_str();
            if let Some(v) = a.strip_prefix("--seed=") {
                if let Ok(v) = v.trim().parse::<i64>() {
                    return v;
                }
            } else if a == "--seed" {
                if let Some(v) = args.get(i + 1) {
                    if let Ok(v) = v.trim().parse::<i64>() {
                        return v;
                    }
                }
            }
            i += 1;
        }
        if let Ok(s) = std::env::var("RWALLPPRS_SEED") {
            if let Ok(v) = s.trim().parse::<i64>() {
                return v;
            }
        }
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let days = (secs / 86_400) as i64;
        let (y, m, d) = civil_from_days(days);
        y * 10_000 + m * 100 + d
    }

    /// One simulation step per frame.
    ///
    /// Normal frame: a single step with the real `dt` (as in the original, which skips
    /// `dt > MAX_VALID_DT`). An abnormally large `dt` (the machine slept, the tab was
    /// frozen) is split into fixed 1/60 steps with an overall cap of
    /// `MAX_VALID_DT` — with no catch-up to real time.
    fn advance(&mut self, dt_raw: f32) {
        let dt = dt_raw.clamp(0.0, sim::MAX_VALID_DT);
        if dt_raw > sim::MAX_VALID_DT {
            const FIXED_DT: f32 = 1.0 / 60.0;
            let mut remaining = sim::MAX_VALID_DT;
            while remaining > 1e-4 {
                let d = FIXED_DT.min(remaining);
                self.universe.step(d);
                remaining -= d;
            }
        } else {
            self.universe.step(dt);
        }
        self.update_camera(dt);
    }

    /// Camera zoom from MainActivity.kt (DYNAMIC_ZOOM = true, as in DreamUniverse).
    fn update_camera(&mut self, dt: f32) {
        let ci = self.universe.closest_body_index();
        let (pos, radius) = self.universe.body_pos_radius(ci);
        let dist_to_surface =
            ((self.universe.ship.body.pos - pos).mag() - radius * 1.2).max(0.0);
        let target_zoom = (500.0 / dist_to_surface).clamp(MIN_CAMERA_ZOOM, MAX_CAMERA_ZOOM);
        self.camera_zoom = sim::exp_smooth(self.camera_zoom, target_zoom, dt, 1.5);
    }

    fn print_stats(&self, frame_index: u64) {
        let u = &self.universe;
        let ship = u.ship.body.pos;
        let ci = u.closest_body_index();
        let (bp, br) = u.body_pos_radius(ci);
        eprintln!(
            "[stats] f={} now={:.1} ship=({:.2},{:.2}) |ship|={:.0} closest[{}] d={:.0} r={:.0} \
             landing={} thrust={:.2} strategy={:?} zoom={:.7} inst={} sparks={} star_r={:.0} star={:?}",
            frame_index,
            u.now,
            ship.x,
            ship.y,
            ship.mag(),
            ci,
            (ship - bp).mag(),
            br,
            u.ship.landing.is_some(),
            u.ship.thrust.mag(),
            u.autopilot.strategy,
            self.camera_zoom,
            self.renderer.last_instance_count(),
            u.sparks.len(),
            u.star.body.radius,
            u.star.cls,
        );
    }
}

impl GpuSceneWallpaper for LandroidWallpaper {
    fn init(args: &[String]) -> Self {
        let seed = Self::resolve_seed(args);
        let mut universe = Simulator::new(seed);
        universe.init_random();
        // DreamUniverse.onAttachedToWindow: the autopilot is created and enabled in screensaver mode.
        universe.autopilot.enabled = true;

        // Self-test diagnostics: `RWALLPPRS_DEBUG_SHIP_R=r` places the ship at distance
        // `r` from the center of the universe. Needed to pixel-check the rendering of the
        // ringfence (Container): in normal flight it is almost always outside the frame,
        // because the autopilot stays inside the planetary orbits and the camera follows the ship.
        if let Ok(v) = std::env::var("RWALLPPRS_DEBUG_SHIP_R") {
            if let Ok(r) = v.trim().parse::<f32>() {
                let a = 0.9f32;
                universe.ship.body.pos = Vec2::make_with_angle_mag(a, r);
                universe.ship.body.angle = a + sim::PIf / 2.0;
                universe.ship.body.velocity = Vec2::ZERO;
            }
        }

        Self {
            universe,
            renderer: Renderer::new(),
            camera_zoom: DEFAULT_CAMERA_ZOOM,
            stats: std::env::var("RWALLPPRS_STATS").is_ok(),
        }
    }

    fn render_scene(&mut self, ctx: &FrameContext, scene: &mut GpuScene<'_>) -> bool {
        self.advance(ctx.dt_secs);

        self.renderer.draw(scene, &self.universe, self.camera_zoom);

        if self.stats && ctx.frame_index % 60 == 0 {
            self.print_stats(ctx.frame_index);
        }
        true
    }

    fn fallback_cpu_render(&mut self, ctx: &FrameContext, canvas: &mut Canvas<'_>) {
        self.advance(ctx.dt_secs);

        let zoom = self.camera_zoom.max(1e-6);
        let ship = self.universe.ship.body.pos;
        let w = ctx.width as f32;
        let h = ctx.height as f32;
        let project = |p: Vec2| -> (f32, f32) {
            ((p.x - ship.x) * zoom + w * 0.5, (p.y - ship.y) * zoom + h * 0.5)
        };

        canvas.clear(Color::from_hex(0x16161D));

        let sp = self.universe.star.body.pos;
        let (sx, sy) = project(sp);
        let sr = (self.universe.star.body.radius * zoom).max(1.5);
        canvas.fill_circle_sdf(sx, sy, sr, to_color(self.universe.star.color));

        for p in &self.universe.planets {
            let (px, py) = project(p.body.pos);
            let pr = (p.body.radius * zoom).max(1.0);
            let c = if p.explored { to_color(p.color) } else { Color::from_hex(0xA7A7CA) };
            canvas.fill_circle_sdf(px, py, pr, c);
        }

        let (shx, shy) = project(ship);
        canvas.fill_circle_sdf(shx, shy, 2.5, Color::rgb(0xFF, 0xFF, 0xFF));
        let pos = &self.universe.ship.track.positions;
        let n = pos.len();
        if n >= 2 {
            let start = n.saturating_sub(64);
            let track = to_color([0.204, 0.659, 0.325, 1.0]);
            for i in start..n - 1 {
                let a = project(pos[i]);
                let b = project(pos[i + 1]);
                canvas.draw_line(a.0 as i32, a.1 as i32, b.0 as i32, b.1 as i32, track);
            }
        }
    }
}

#[inline]
fn to_color(c: [f32; 4]) -> Color {
    let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    Color::rgb(b(c[0]), b(c[1]), b(c[2]))
}

/// Days from the epoch -> (year, month, day) in the proleptic Gregorian calendar.
fn civil_from_days(z0: i64) -> (i64, i64, i64) {
    let z = z0 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn write_ppm(path: &str, width: u32, height: u32, pixels: &[u32]) {
    let count = (width as usize) * (height as usize);
    let mut buf = Vec::with_capacity(32 + count * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
    for &px in pixels.iter().take(count) {
        buf.push(((px >> 16) & 0xFF) as u8); // R
        buf.push(((px >> 8) & 0xFF) as u8); // G
        buf.push((px & 0xFF) as u8); // B
    }
    let _ = std::fs::write(path, buf);
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has_shm = std::env::var("RWALLPPRS_SHM_FD").is_ok();
    let selftest: Option<u32> = std::env::var("RWALLPPRS_SELFTEST")
        .ok()
        .and_then(|v| v.trim().parse().ok());

    if has_shm && selftest.is_none() {
        // Production mode: the daemon passes the SHM fd and events over stdin.
        return rwallpprs_sdk::run_wallpaper_loop::<SceneWallpaperAdapter<LandroidWallpaper>>();
    }

    // Standalone (or an explicit self-test): run N frames so the scene has time to come alive.
    let frames = selftest.unwrap_or(900);
    let dump_path = std::env::var("RWALLPPRS_DUMP_PPM").ok();
    // Do not make the adapter write a PPM on every frame — we only save the last one.
    std::env::remove_var("RWALLPPRS_DUMP_PPM");

    let width = 1920u32;
    let height = 1080u32;
    let mut buffer = vec![0u32; (width * height) as usize];
    let mut canvas = Canvas::from_raw_slice(width, height, &mut buffer);

    let mut wallpaper =
        <SceneWallpaperAdapter<LandroidWallpaper> as Wallpaper>::init(&args);

    for i in 0..frames {
        let ctx = FrameContext {
            frame_index: i as u64,
            time_secs: i as f64 / 60.0,
            dt_secs: 1.0 / 60.0,
            width,
            height,
            niri: NiriViewportState::default(),
        };
        let _ = wallpaper.render(&ctx, &mut canvas);
    }

    if let Some(path) = dump_path {
        write_ppm(&path, width, height, canvas.pixels_mut());
        eprintln!(
            "[niri-landroid] standalone self-test: {} frames, PPM saved to '{}'",
            frames, path
        );
    } else {
        eprintln!("[niri-landroid] standalone self-test: {} frames", frames);
    }

    Ok(())
}
