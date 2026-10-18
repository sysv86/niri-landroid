//! Exact port of the physics simulation of the Android Easter Egg "Landroid" screensaver
//! (packages/apps/EasterEgg, com.android.egg.landroid: Physics.kt, Universe.kt, Autopilot.kt,
//! Vec2.kt, Maths.kt, Randomness.kt).
//!
//! This is the "engine" without UI: all entities (star, planets, ship, sparks, autopilot)
//! are integrated in the same order as in the original:
//!   now += dt -> update_all -> solve_all -> post_update_all -> remove expired (Fuse < 0).
//!
//! The original uses `ArraySet<Entity>` (which preserves insertion order) and takes a
//! "local copy" of the lists at the start of a step, so newly created entities (exhaust
//! sparks, landing) are not processed during the same step. Here that is reproduced with
//! length snapshots.

#![allow(dead_code)]

use std::collections::VecDeque;

// Constant names kept as in the original (Vec2.kt: PIf / PI2f).
#[allow(non_upper_case_globals)]
pub const PIf: f32 = std::f32::consts::PI;
#[allow(non_upper_case_globals)]
pub const PI2f: f32 = std::f32::consts::TAU;

// ---- Universe.kt ----
pub const UNIVERSE_RANGE: f32 = 200_000.0;
pub const NUM_PLANETS_MIN: i32 = 1;
pub const NUM_PLANETS_MAX: i32 = 10; // inclusive
pub const STAR_RADIUS_MIN: f32 = 1_000.0;
pub const STAR_RADIUS_MAX: f32 = 8_000.0;
pub const PLANET_RADIUS_MIN: f32 = 50.0;
pub const PLANET_RADIUS_MAX: f32 = 2_000.0;
/// PLANET_ORBIT_RANGE = (STAR_RADIUS_RANGE.endInclusive * 2f)..(UNIVERSE_RANGE * 0.75f)
pub const PLANET_ORBIT_MIN: f32 = STAR_RADIUS_MAX * 2.0;
pub const PLANET_ORBIT_MAX: f32 = UNIVERSE_RANGE * 0.75;

pub const GRAVITATION: f32 = 1e-2;
pub const KEPLER_CONSTANT: f32 = 50.0;

pub const PLANETARY_DENSITY: f32 = 2.5;
pub const STELLAR_DENSITY: f32 = 0.5;

pub const SPACECRAFT_MASS: f32 = 10.0;

pub const CRAFT_SPEED_LIMIT: f32 = 5_000.0;
pub const MAIN_ENGINE_ACCEL: f32 = 1_000.0;
pub const LAUNCH_MECO: f32 = 2.0;

pub const LANDING_REMOVAL_TIME: f32 = 60.0 * 15.0;
pub const SCALED_THRUST: bool = true;

// ---- Physics.kt ----
pub const MAX_VALID_DT: f32 = 1.0;
pub const TRACK_LENGTH: usize = 10_000;

// ---- Autopilot.kt ----
pub const BRAKING_TIME: f32 = 5.0;
pub const SIGHTSEEING_TIME: f32 = 15.0;
pub const LAUNCH_THRUST_TIME: f32 = 5.0;
pub const STRATEGY_MIN_TIME: f32 = 0.5;

/// `lerp` from Compose (`androidx.compose.ui.util.lerp`) — without clamping.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// `expSmooth` from Maths.kt.
#[inline]
pub fn exp_smooth(current: f32, target: f32, dt: f32, speed: f32) -> f32 {
    current + (target - current) * (1.0 - (-dt * speed).exp())
}

/// sRGB 0xRRGGBB -> [r,g,b,1] in 0..1 (like Compose Color(0xFF...)).
#[inline]
pub fn srgb(hex: u32) -> [f32; 4] {
    [
        ((hex >> 16) & 0xFF) as f32 / 255.0,
        ((hex >> 8) & 0xFF) as f32 / 255.0,
        (hex & 0xFF) as f32 / 255.0,
        1.0,
    ]
}

/// `Color.hsv(h, s, v)` (Compose) — h in degrees.
pub fn hsv(h: f32, s: f32, v: f32) -> [f32; 4] {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let hh = h / 60.0;
    let x = c * (1.0 - ((hh % 2.0) - 1.0).abs());
    let (r, g, b) = match hh as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m, 1.0]
}

// ---------------------------------------------------------------------------
// RNG: a custom deterministic xorshift64* (it does not need to match kotlin.random.Random
// byte for byte — only reproducibility from the seed).
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Rng {
    s: u64,
}

impl Rng {
    pub fn new(seed: i64) -> Self {
        let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
        if s == 0 {
            s = 0x1234_5678_9ABC_DEF0;
        }
        Self { s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// [0, 1)
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    /// [from, until)
    #[inline]
    pub fn next_f32_in_range(&mut self, from: f32, until: f32) -> f32 {
        from + (until - from) * self.next_f32()
    }

    /// [from, until)
    #[inline]
    pub fn next_i32_in_range(&mut self, from: i32, until: i32) -> i32 {
        if until <= from {
            return from;
        }
        from + (self.next_u32() % ((until - from) as u32)) as i32
    }

    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        let i = self.next_i32_in_range(0, items.len() as i32) as usize;
        &items[i.min(items.len() - 1)]
    }
}

// ---------------------------------------------------------------------------
// Vec2 (Vec2.kt)
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn mag(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn distance(self, other: Vec2) -> f32 {
        (self - other).mag()
    }

    #[inline]
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    #[inline]
    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    #[inline]
    pub fn make_with_angle_mag(a: f32, m: f32) -> Vec2 {
        Vec2::new(m * a.cos(), m * a.sin())
    }

    /// Rotation about the origin (Vec2.rotate with origin = Zero).
    #[inline]
    pub fn rotate(self, angle: f32) -> Vec2 {
        let (s, c) = angle.sin_cos();
        Vec2::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.x == 0.0 && self.y == 0.0
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;
    #[inline]
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}
impl std::ops::Sub for Vec2 {
    type Output = Vec2;
    #[inline]
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}
impl std::ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, f: f32) -> Vec2 {
        Vec2::new(self.x * f, self.y * f)
    }
}
impl std::ops::Neg for Vec2 {
    type Output = Vec2;
    #[inline]
    fn neg(self) -> Vec2 {
        Vec2::new(-self.x, -self.y)
    }
}

// ---------------------------------------------------------------------------
// Physics.kt
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Body {
    pub pos: Vec2,
    pub opos: Vec2,
    pub velocity: Vec2,
    pub mass: f32,
    pub angle: f32,
    pub radius: f32,
    pub collides: bool,
}

impl Default for Body {
    fn default() -> Self {
        Self {
            pos: Vec2::ZERO,
            opos: Vec2::ZERO,
            velocity: Vec2::ZERO,
            mass: 0.0,
            angle: 0.0,
            radius: 0.0,
            collides: true,
        }
    }
}

impl Body {
    #[inline]
    pub fn update(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let vscaled = self.velocity * dt;
        self.opos = self.pos;
        self.pos = self.pos + vscaled;
    }

    #[inline]
    pub fn post_update(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.velocity = (self.pos - self.opos) * (1.0 / dt);
    }
}

#[derive(Clone, Copy)]
pub struct Fuse {
    pub lifetime: f32,
}

impl Fuse {
    #[inline]
    pub fn new(lifetime: f32) -> Self {
        Self { lifetime }
    }
    #[inline]
    pub fn update(&mut self, dt: f32) {
        self.lifetime -= dt;
    }
    #[inline]
    pub fn can_be_removed(&self) -> bool {
        self.lifetime < 0.0
    }
}

// ---------------------------------------------------------------------------
// Universe.kt: Planet / StarClass / Star
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Planet {
    pub body: Body,
    pub orbit_center: Vec2,
    pub orbit_radius: f32,
    pub speed: f32,
    pub color: [f32; 4],
    pub explored: bool,
    /// Creation order (the insertion order in the original's `entities`) — sets the planet
    /// draw order (ArraySet preserves insertion order, not sorting).
    pub spawn_order: u32,
}

impl Planet {
    pub fn new(orbit_center: Vec2, radius: f32, pos: Vec2, speed: f32, color: [f32; 4]) -> Self {
        let mut body = Body::default();
        body.radius = radius;
        body.pos = pos;
        body.mass = 4.0 / 3.0 * PIf * radius.powi(3) * PLANETARY_DENSITY;
        let orbit_radius = pos.distance(orbit_center);
        Self {
            body,
            orbit_center,
            orbit_radius,
            speed,
            color,
            explored: false,
            spawn_order: 0,
        }
    }

    pub fn update(&mut self, dt: f32) {
        let orbit_angle = (self.body.pos - self.orbit_center).angle();
        self.body.velocity = Vec2::make_with_angle_mag(orbit_angle + PIf / 2.0, self.speed);
        self.body.update(dt);
    }

    pub fn post_update(&mut self, dt: f32) {
        let orbit_angle = (self.body.pos - self.orbit_center).angle();
        self.body.pos = self.orbit_center + Vec2::make_with_angle_mag(orbit_angle, self.orbit_radius);
        self.body.post_update(dt);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarClass {
    O,
    B,
    A,
    F,
    G,
    K,
    M,
}

impl StarClass {
    pub const ALL: [StarClass; 7] = [
        StarClass::O,
        StarClass::B,
        StarClass::A,
        StarClass::F,
        StarClass::G,
        StarClass::K,
        StarClass::M,
    ];

    pub fn color(self) -> [f32; 4] {
        match self {
            StarClass::O => srgb(0xFF6666FF),
            StarClass::B => srgb(0xFFCCCCFF),
            StarClass::A => srgb(0xFFEEEEFF),
            StarClass::F => srgb(0xFFFFFFFF),
            StarClass::G => srgb(0xFFFFFF66),
            StarClass::K => srgb(0xFFFFCC33),
            StarClass::M => srgb(0xFFFF8800),
        }
    }
}

#[derive(Clone)]
pub struct Star {
    pub body: Body,
    pub cls: StarClass,
    pub color: [f32; 4],
    pub anim: f32,
}

impl Star {
    pub fn new(cls: StarClass, radius: f32) -> Self {
        let mut body = Body::default();
        body.pos = Vec2::ZERO;
        body.radius = radius;
        body.mass = 4.0 / 3.0 * PIf * radius.powi(3) * STELLAR_DENSITY;
        body.collides = false;
        Self {
            body,
            cls,
            color: cls.color(),
            anim: 0.0,
        }
    }

    /// Star.update: only advances the animation (no integration).
    pub fn update(&mut self, dt: f32) {
        self.anim += dt;
    }

    /// Inherited from Planet.postUpdate, but orbitRadius = 0 -> the position stays at Zero.
    pub fn post_update(&mut self, dt: f32) {
        self.body.pos = Vec2::ZERO;
        self.body.post_update(dt);
    }
}

// ---------------------------------------------------------------------------
// Universe.kt: Track / Spacecraft
// ---------------------------------------------------------------------------
#[derive(Clone)]
pub struct Track {
    pub positions: VecDeque<Vec2>,
    pub angles: VecDeque<f32>,
}

impl Default for Track {
    fn default() -> Self {
        Self {
            positions: VecDeque::with_capacity(TRACK_LENGTH),
            angles: VecDeque::with_capacity(TRACK_LENGTH),
        }
    }
}

impl Track {
    /// Literal port: on overflow the first TWO elements are removed.
    pub fn add(&mut self, x: f32, y: f32, a: f32) {
        if self.positions.len() >= (TRACK_LENGTH - 1) {
            self.positions.pop_front();
            self.angles.pop_front();
            self.positions.pop_front();
            self.angles.pop_front();
        }
        self.positions.push_back(Vec2::new(x, y));
        self.angles.push_back(a);
    }
}

#[derive(Clone)]
pub struct Spacecraft {
    pub body: Body,
    pub thrust: Vec2,
    pub launch_clock: f32,
    pub transit: bool,
    pub track: Track,
    /// Index into `Simulator::landings` if the ship is docked.
    pub landing: Option<usize>,
}

impl Default for Spacecraft {
    fn default() -> Self {
        let mut body = Body::default();
        body.mass = SPACECRAFT_MASS;
        body.radius = 12.0;
        Self {
            body,
            thrust: Vec2::ZERO,
            launch_clock: 0.0,
            transit: false,
            track: Track::default(),
            landing: None,
        }
    }
}

impl Spacecraft {
    pub fn update(&mut self, now: f32, dt: f32, landings: &mut [Landing]) {
        let thrust_mag = self.thrust.mag();
        if thrust_mag > 0.0 {
            let mut delta_v = MAIN_ENGINE_ACCEL * dt;
            if SCALED_THRUST {
                delta_v *= thrust_mag.clamp(0.0, 1.0);
            }

            if let Some(li) = self.landing {
                if self.launch_clock == 0.0 {
                    self.launch_clock = now + 1.0; // @@@ TODO extract (as in the original)
                }
                if now > self.launch_clock {
                    // detach from the landing pad
                    if let Some(l) = landings.get_mut(li) {
                        l.ship_attached = false;
                    }
                    self.landing = None;
                } else {
                    delta_v = 0.0;
                }
            }

            // always thrust forward
            self.body.velocity = self.body.velocity + Vec2::make_with_angle_mag(self.body.angle, delta_v);
        } else if self.launch_clock != 0.0 {
            self.launch_clock = 0.0;
        }

        if self.body.velocity.mag() > CRAFT_SPEED_LIMIT {
            self.body.velocity = Vec2::make_with_angle_mag(self.body.velocity.angle(), CRAFT_SPEED_LIMIT);
        }

        self.body.update(dt);
    }

    pub fn post_update(&mut self, dt: f32, rng: &mut Rng, sparks: &mut Vec<Spark>) {
        self.body.post_update(dt);

        self.track.add(self.body.pos.x, self.body.pos.y, self.body.angle);

        let mag = self.thrust.mag();
        if rng.next_f32() < mag {
            let mut spark = Spark::new(
                rng.next_f32_in_range(0.5, 1.0),
                true,
                1.0,
                SparkStyle::Ring,
                [1.0, 1.0, 1.0, 0x40 as f32 / 255.0],
                1.0,
            );
            spark.body.pos = self.body.pos;
            spark.body.opos = self.body.pos;
            spark.body.velocity = self.body.velocity
                + Vec2::make_with_angle_mag(
                    self.body.angle + rng.next_f32_in_range(-0.2, 0.2),
                    -MAIN_ENGINE_ACCEL * mag * 10.0 * dt,
                );
            sparks.push(spark);
        }
    }
}

// ---------------------------------------------------------------------------
// Universe.kt: Landing / Spark
// ---------------------------------------------------------------------------
#[derive(Clone, Copy)]
pub struct Landing {
    pub planet: usize,
    pub angle: f32,
    pub fuse: Fuse,
    pub ship_attached: bool,
}

impl Landing {
    pub fn new(planet: usize, angle: f32) -> Self {
        Self {
            planet,
            angle,
            fuse: Fuse::new(LANDING_REMOVAL_TIME),
            ship_attached: true,
        }
    }

    /// Landing.solve: ship.pos = ship.pos*0.5 + desired*0.5; ship.angle = angle.
    pub fn solve(&mut self, ship: &mut Spacecraft, planet_pos: Vec2, planet_radius: f32, dt: f32) {
        let desired = planet_pos + Vec2::make_with_angle_mag(self.angle, ship.body.radius + planet_radius);
        ship.body.pos = ship.body.pos * 0.5 + desired * 0.5;
        ship.body.angle = self.angle;
        self.fuse.update(dt);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparkStyle {
    Line,
    LineAbsolute,
    Dot,
    DotAbsolute,
    Ring,
}

#[derive(Clone)]
pub struct Spark {
    pub body: Body,
    pub ttl: f32,
    pub fuse: Fuse,
    pub style: SparkStyle,
    pub color: [f32; 4],
    pub size: f32,
}

impl Spark {
    pub fn new(ttl: f32, collides: bool, mass: f32, style: SparkStyle, color: [f32; 4], size: f32) -> Self {
        let mut body = Body::default();
        body.collides = collides;
        body.mass = mass;
        Self {
            body,
            ttl,
            fuse: Fuse::new(ttl),
            style,
            color,
            size,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.body.update(dt);
        self.fuse.update(dt);
    }

    pub fn post_update(&mut self, dt: f32) {
        self.body.post_update(dt);
    }
}

// ---------------------------------------------------------------------------
// Autopilot.kt
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    None,
    Chasing,
    Approaching,
    Landing,
    Landed,
    Launching,
}

#[derive(Clone)]
pub struct Autopilot {
    pub enabled: bool,
    /// Index of the target planet in `Simulator::planets`.
    pub target: Option<usize>,
    pub landing_altitude: f32,
    pub next_strategy_time: f32,
    pub braking_distance: f32,
    pub leading_pos: Vec2,
    pub leading_vector: Vec2,
    pub strategy: Strategy,
}

impl Default for Autopilot {
    fn default() -> Self {
        Self {
            enabled: false,
            target: None,
            landing_altitude: 0.0,
            next_strategy_time: 0.0,
            braking_distance: 0.0,
            leading_pos: Vec2::ZERO,
            leading_vector: Vec2::ZERO,
            strategy: Strategy::None,
        }
    }
}

impl Autopilot {
    pub fn update(&mut self, ship: &mut Spacecraft, planets: &[Planet], now: f32, dt: f32, rng: &mut Rng) {
        if !self.enabled {
            return;
        }
        if now < self.next_strategy_time {
            return;
        }

        let current_strategy = self.strategy;

        if ship.landing.is_some() {
            if self.target.is_some() {
                self.strategy = Strategy::Landed;
                // just arrived — look around
                self.target = None;
                self.landing_altitude = 0.0;
                self.next_strategy_time = now + SIGHTSEEING_TIME;
            } else {
                // full thrust to take off
                ship.thrust = Vec2::make_with_angle_mag(ship.body.angle, 1.0);
                self.strategy = Strategy::Launching;
                self.next_strategy_time = now + LAUNCH_THRUST_TIME;
            }
        } else {
            // pick a new target
            if self.target.is_none() {
                // nearest unexplored planet
                let mut best: Option<(f32, usize)> = None;
                for (i, p) in planets.iter().enumerate() {
                    if p.explored {
                        continue;
                    }
                    let d = (p.body.pos - ship.body.pos).mag();
                    if best.map_or(true, |(bd, _)| d < bd) {
                        best = Some((d, i));
                    }
                }
                self.target = best.map(|(_, i)| i);
                self.braking_distance = 0.0;

                // if everything has been explored — a random planet
                if self.target.is_none() && !planets.is_empty() {
                    let i = rng.next_i32_in_range(0, planets.len() as i32) as usize;
                    self.target = Some(i.min(planets.len() - 1));
                }
            }

            if let Some(ti) = self.target {
                let target = &planets[ti];
                let ship_v = ship.body.velocity;
                let target_v = target.body.velocity;
                let target_vector = target.body.pos - ship.body.pos;
                let altitude = target_vector.mag() - target.body.radius;

                self.landing_altitude = target.body.radius.min(100.0);

                // in the target's frame of reference
                let relative_v = ship_v - target_v;
                let projection = relative_v.dot(target_vector * (1.0 / target_vector.mag().max(1e-6)));
                let relative_speed = relative_v.mag() * projection.signum();
                let time_to_target = if relative_speed != 0.0 {
                    altitude / relative_speed
                } else {
                    1_000.0
                };

                let new_braking_distance =
                    BRAKING_TIME * if relative_speed > 0.0 { relative_speed } else { MAIN_ENGINE_ACCEL };
                self.braking_distance = exp_smooth(self.braking_distance, new_braking_distance, dt, 5.0);

                self.leading_pos = target.body.pos
                    + Vec2::make_with_angle_mag(
                        target.body.velocity.angle(),
                        (altitude / 2.0).min(target.body.velocity.mag()),
                    );
                self.leading_vector = self.leading_pos - ship.body.pos;

                if altitude < self.landing_altitude {
                    self.strategy = Strategy::Landing;
                    // zero thrust, turn "away from the ground"
                    ship.body.angle = (ship.body.pos - target.body.pos).angle();
                    ship.thrust = Vec2::ZERO;
                } else if relative_speed < 0.0 || altitude > self.braking_distance {
                    self.strategy = Strategy::Chasing;
                    ship.body.angle = self.leading_vector.angle();
                    ship.thrust = Vec2::make_with_angle_mag(ship.body.angle, 1.0);
                } else {
                    self.strategy = Strategy::Approaching;
                    ship.body.angle = (-ship.body.velocity).angle();
                    let decel = relative_speed / time_to_target;
                    let decel_thrust = decel / MAIN_ENGINE_ACCEL * 0.9;
                    ship.thrust = Vec2::make_with_angle_mag(ship.body.angle, decel_thrust);
                }
            }
            if self.strategy != current_strategy {
                self.next_strategy_time = now + STRATEGY_MIN_TIME;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simulator (Physics.kt) + Universe (Universe.kt)
// ---------------------------------------------------------------------------
pub struct Simulator {
    pub now: f32,
    pub dt: f32,
    pub rng: Rng,

    pub star: Star,
    pub planets: Vec<Planet>,
    pub ship: Spacecraft,
    pub sparks: Vec<Spark>,
    pub landings: Vec<Landing>,
    pub autopilot: Autopilot,

    /// Container (ringfence) — only the ship is added to it.
    pub ringfence_radius: f32,
    pub follow_ship: bool,
}

impl Simulator {
    pub fn new(seed: i64) -> Self {
        Self {
            now: 0.0,
            dt: 0.0,
            rng: Rng::new(seed),
            star: Star::new(StarClass::F, STAR_RADIUS_MAX),
            planets: Vec::new(),
            ship: Spacecraft::default(),
            sparks: Vec::new(),
            landings: Vec::new(),
            autopilot: Autopilot::default(),
            ringfence_radius: UNIVERSE_RANGE,
            follow_ship: true,
        }
    }

    /// Universe.initRandom
    pub fn init_random(&mut self) {
        let cls = *self.rng.choose(&StarClass::ALL);
        let star_radius = self.rng.next_f32_in_range(STAR_RADIUS_MIN, STAR_RADIUS_MAX);
        self.star = Star::new(cls, star_radius);

        let n = self.rng.next_i32_in_range(NUM_PLANETS_MIN, NUM_PLANETS_MAX + 1);
        for _ in 0..n {
            let radius = self.rng.next_f32_in_range(PLANET_RADIUS_MIN, PLANET_RADIUS_MAX);
            let orbit_radius = lerp(
                PLANET_ORBIT_MIN,
                PLANET_ORBIT_MAX,
                self.rng.next_f32().powi(1),
            );

            // Kepler's third law
            let period = (orbit_radius.powi(3) / self.star.body.mass).sqrt() * KEPLER_CONSTANT;
            let speed = PI2f * orbit_radius / period;

            let hue = radius % 360.0;

            let p = Planet::new(
                self.star.body.pos,
                radius,
                self.star.body.pos + Vec2::make_with_angle_mag(self.rng.next_f32() * PI2f, orbit_radius),
                speed,
                hsv(hue, 0.75, 1.0),
            );
            self.planets.push(p);
            let n = self.planets.len();
            self.planets[n - 1].spawn_order = (n - 1) as u32;
        }
        self.planets
            .sort_by(|a, b| a.body.pos.distance(self.star.body.pos).partial_cmp(&b.body.pos.distance(self.star.body.pos)).unwrap());

        self.ship = Spacecraft::default();
        self.ship.body.pos = self.star.body.pos
            + Vec2::make_with_angle_mag(
                self.rng.next_f32() * PI2f,
                self.rng.next_f32_in_range(PLANET_ORBIT_MIN, PLANET_ORBIT_MAX),
            );
        self.ship.body.angle = self.rng.next_f32() * PI2f;

        self.follow_ship = true;
    }

    /// Universe.closestPlanet — index into `planets`, or `planets.len()` for the star.
    pub fn closest_body_index(&self) -> usize {
        let mut best = usize::MAX;
        let mut best_d = f32::INFINITY;
        for (i, p) in self.planets.iter().enumerate() {
            let d = (p.body.pos - self.ship.body.pos).mag();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        let ds = (self.star.body.pos - self.ship.body.pos).mag();
        if ds < best_d {
            best = self.planets.len();
        }
        best
    }

    pub fn body_pos_radius(&self, i: usize) -> (Vec2, f32) {
        if i < self.planets.len() {
            (self.planets[i].body.pos, self.planets[i].body.radius)
        } else {
            (self.star.body.pos, self.star.body.radius)
        }
    }

    /// One simulation step (`Simulator.step`).
    pub fn step(&mut self, dt: f32) {
        self.dt = dt;
        // As in the original: an invalid / too large dt (pause) — do not simulate.
        if dt > MAX_VALID_DT {
            return;
        }
        // Protection against a "backwards clock" (absent in the canon): negative dt is
        // skipped so that `now` is not rolled back and the fuses are not corrupted.
        if dt < 0.0 {
            return;
        }
        self.now += dt;

        self.update_all(dt);
        self.solve_all(dt);
        self.post_update_all(dt);
        self.remove_expired();
    }

    fn update_all(&mut self, dt: f32) {
        let now = self.now;

        // transit in front of the star
        self.ship.transit = false;

        let apply_gravity = |ship: &mut Spacecraft, pos: Vec2, radius: f32, mass: f32, is_star: bool| {
            let vector = pos - ship.body.pos;
            let d = vector.mag();
            if d < radius {
                if is_star {
                    ship.transit = true;
                }
            } else if now > ship.launch_clock + LAUNCH_MECO {
                // f_g = G * m1 * m2 / d^2
                ship.body.velocity = ship.body.velocity
                    + Vec2::make_with_angle_mag(
                        vector.angle(),
                        GRAVITATION * (ship.body.mass * mass) / d.powi(2),
                    ) * dt;
            }
        };

        for i in 0..self.planets.len() {
            let p = &self.planets[i];
            apply_gravity(&mut self.ship, p.body.pos, p.body.radius, p.body.mass, false);
        }
        {
            let s = &self.star;
            apply_gravity(&mut self.ship, s.body.pos, s.body.radius, s.body.mass, true);
        }

        // update of all entities (insertion order: planets -> star -> ship -> autopilot -> sparks)
        for p in self.planets.iter_mut() {
            p.update(dt);
        }
        self.star.update(dt);
        self.ship.update(now, dt, &mut self.landings);
        self.autopilot
            .update(&mut self.ship, &self.planets, now, dt, &mut self.rng);
        let spark_count = self.sparks.len();
        for s in self.sparks[..spark_count].iter_mut() {
            s.update(dt);
        }
    }

    fn solve_all(&mut self, dt: f32) {
        // Snapshot of the existing landings: a newly created one is solved on the next step.
        let landing_count = self.landings.len();

        if self.ship.landing.is_none() {
            let ci = self.closest_body_index();
            let collides = if ci < self.planets.len() {
                self.planets[ci].body.collides
            } else {
                self.star.body.collides
            };

            if collides && ci < self.planets.len() {
                let planet_pos = self.planets[ci].body.pos;
                let planet_radius = self.planets[ci].body.radius;
                let planet_velocity = self.planets[ci].body.velocity;

                let d = (self.ship.body.pos - planet_pos).mag() - self.ship.body.radius - planet_radius;
                let a = (self.ship.body.pos - planet_pos).angle();

                if d < 0.0 {
                    // landing or impact?
                    let a_diff = (self.ship.body.angle - a).abs();

                    if a_diff < PIf / 4.0 {
                        // landing criterion
                        self.ship.landing = Some(self.landings.len());
                        self.ship.body.velocity = planet_velocity;
                        self.landings.push(Landing::new(ci, a));
                        self.planets[ci].explored = true;
                    } else {
                        let impact = planet_pos + Vec2::make_with_angle_mag(a, planet_radius);
                        self.ship.body.pos =
                            planet_pos + Vec2::make_with_angle_mag(a, planet_radius + self.ship.body.radius - d);

                        for _ in 0..10 {
                            let mut spark = Spark::new(
                                self.rng.next_f32_in_range(0.5, 2.0),
                                false,
                                0.0,
                                SparkStyle::Dot,
                                srgb(0xFFFFFF),
                                1.0,
                            );
                            spark.body.pos = impact
                                + Vec2::make_with_angle_mag(
                                    self.rng.next_f32_in_range(0.0, 2.0 * PIf),
                                    self.rng.next_f32_in_range(0.1, 0.5),
                                );
                            spark.body.opos = spark.body.pos;
                            spark.body.velocity = self.ship.body.velocity * 0.8
                                + Vec2::make_with_angle_mag(
                                    self.rng.next_f32_in_range(0.0, 2.0 * PIf),
                                    self.rng.next_f32_in_range(0.1, 0.5),
                                );
                            self.sparks.push(spark);
                        }
                    }
                }
            }
        }

        // constraints: ringfence (Container with softness = 0), then the existing landings
        if (self.ship.body.pos.mag() + self.ship.body.radius) > self.ringfence_radius {
            self.ship.body.pos = Vec2::make_with_angle_mag(
                self.ship.body.pos.angle(),
                self.ringfence_radius - self.ship.body.radius,
            );
        }

        for i in 0..landing_count {
            let ship_landing = self.ship.landing;
            let l = self.landings[i];
            // expired constraints are removed in the original and are no longer solved
            if ship_landing == Some(i) && l.ship_attached && l.fuse.lifetime >= 0.0 {
                let planet_pos = self.planets[l.planet].body.pos;
                let planet_radius = self.planets[l.planet].body.radius;
                self.landings[i].solve(&mut self.ship, planet_pos, planet_radius, dt);
            } else {
                self.landings[i].fuse.update(dt);
            }
        }
    }

    fn post_update_all(&mut self, dt: f32) {
        for p in self.planets.iter_mut() {
            p.post_update(dt);
        }
        self.star.post_update(dt);

        // snapshot: the exhaust sparks added in the ship's postUpdate are not processed during this step
        let spark_count = self.sparks.len();
        self.ship.post_update(dt, &mut self.rng, &mut self.sparks);

        for s in self.sparks[..spark_count].iter_mut() {
            s.post_update(dt);
        }
    }

    fn remove_expired(&mut self) {
        self.sparks.retain(|s| !s.fuse.can_be_removed());
        // In the original, the landing constraints are removed from constraints, but the object
        // stays reachable through ship.landing — so here we keep the object but mark it expired
        // (solve skips expired ones, see solve_all: the fuse keeps ticking anyway).
    }
}
