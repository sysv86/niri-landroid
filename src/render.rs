//! Full wgpu render of the "Landroid" screensaver (canon: VisibleUniverse.kt +
//! the `Spaaaaace` draw loop from MainActivity.kt).
//!
//! Two pipelines are used:
//! * `shapes` — instanced SDF pipeline (segments / filled circles / stroked circles),
//!   one instance = one primitive shape; 4×vec4 attributes per instance;
//! * `fill` — CPU-triangulated polygons (the ship silhouette), a static index
//!   buffer + a dynamic vertex buffer (world coordinates are recomputed on the CPU).
//!
//! All colors are given as sRGB bytes / 255 and written directly (no gamma conversion),
//! as the `Bgra8Unorm` target texture requires.
//!
//! ## Known deviations from the original
//! * `PathEffect.cornerPathEffect(radius = 200f)` on the star crowns is not reproduced —
//!   the crowns are drawn as a polyline without corner rounding (the corners of the
//!   37-ray star are obtuse, and the rounding radius is only ~17 "world" units per ray).
//! * Grid step: the original uses `32.dp.toPx()`, here it is 32 screen px (on desktop dp ≈ px).
//! * The ship silhouette fill is a CPU triangulation (ear clipping) without edge
//!   antialiasing; Compose draws it with AA.
//! * `Container` is drawn with the `Colors` canon color `Color(0xFF800000)` — that is ARGB
//!   with alpha = 0xFF (an opaque #800000), not "alpha = 0.5".
//! * `STAR_POINTS`: the canon `Build.VERSION.SDK_INT.takeIf { it in 1..99 } ?: 31` yields
//!   37 for Android 17 (API 37), so the crowns have 37 and 38 rays.

use bytemuck::{Pod, Zeroable};
use rwallpprs_sdk::GpuScene;

use crate::assets::{self, LinearColor};
use crate::sim::{self, Simulator, Vec2, PI2f};

const KIND_LINE: f32 = 0.0;
const KIND_CIRCLE_FILL: f32 = 1.0;
const KIND_CIRCLE_STROKE: f32 = 2.0;
/// Dashed stroked circle in SCREEN pixels (Container/ringfence):
/// `a3.w` is the pattern period in px (dash = gap = period/2), like
/// `PathEffect.dashPathEffect([8/zoom, 8/zoom])` (see VisibleUniverse.kt:drawContainer).
const KIND_DASHED_CIRCLE: f32 = 3.0;

/// VisibleUniverse.kt: `val STAR_POINTS = Build.VERSION.SDK_INT.takeIf { it in 1..99 } ?: 31`.
/// On a real device this is always the API level itself (the 31 fallback only applies to
/// dev builds with SDK_INT = 10000). Android 17 = API 37, so the star crowns have
/// 37 and 38 rays (the task spec said 31/32 — that is the fallback branch).
const STAR_POINTS: u32 = 37;

/// drawContainer: `PathEffect.dashPathEffect(floatArrayOf(8f / zoom, 8f / zoom), 0f)` —
/// dash 8 screen px, gap 8 screen px.
const CONTAINER_DASH_PX: f32 = 8.0;
/// drawContainer: color `Color(0xFF800000)`. In Compose that is ARGB with alpha = 0xFF, i.e.
/// an OPAQUE dark red #800000 (the task description calls it "alpha = 0.5", but
/// 0x80 there is the red channel, not alpha; see README: known deviations).
const CONTAINER_RGB: [f32; 4] = [0x80 as f32 / 255.0, 0.0, 0.0, 1.0];

// ---------------------------------------------------------------------------
// GPU structures
// ---------------------------------------------------------------------------
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    resolution: [f32; 2],
    zoom: f32,
    cam_x: f32,
    cam_y: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

/// One instance of a shape.
/// a0 = {p0.xy, p1.xy}
/// a1 = {width_or_radius, kind, alpha, clip_radius}
/// a2 = color rgba
/// a3 = {clip_center.xy, thickness, dash_period_px}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    a0: [f32; 4],
    a1: [f32; 4],
    a2: [f32; 4],
    a3: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillVertex {
    pos: [f32; 2],
    color: [f32; 4],
}

#[inline]
fn c4(c: LinearColor) -> [f32; 4] {
    [c.0, c.1, c.2, c.3]
}

#[inline]
fn with_alpha(c: [f32; 4], a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------
const SHAPES_WGSL: &str = r#"
struct Frame {
    resolution: vec2<f32>,
    zoom: f32,
    cam_x: f32,
    cam_y: f32,
    p0: f32,
    p1: f32,
    p2: f32,
};

@group(0) @binding(0) var<uniform> fr: Frame;

struct VsIn {
    @location(0) a0: vec4<f32>,
    @location(1) a1: vec4<f32>,
    @location(2) a2: vec4<f32>,
    @location(3) a3: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec2<f32>,
    @location(1) @interpolate(flat) p0: vec2<f32>,
    @location(2) @interpolate(flat) p1: vec2<f32>,
    @location(3) @interpolate(flat) prm: vec4<f32>,
    @location(4) @interpolate(flat) col: vec4<f32>,
    @location(5) @interpolate(flat) ext: vec4<f32>,
};

fn world_to_clip(w: vec2<f32>) -> vec4<f32> {
    let sp = (w + vec2<f32>(fr.cam_x, fr.cam_y)) * fr.zoom + fr.resolution * 0.5;
    let ndc = vec2<f32>(sp.x / fr.resolution.x * 2.0 - 1.0, 1.0 - sp.y / fr.resolution.y * 2.0);
    return vec4<f32>(ndc, 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: VsIn) -> VsOut {
    let p0 = inst.a0.xy;
    let p1 = inst.a0.zw;
    let kind = i32(inst.a1.y + 0.5);
    let px = 2.0 / max(fr.zoom, 1e-6);

    var center = p0;
    var ax = vec2<f32>(1.0, 0.0);
    var ay = vec2<f32>(0.0, 1.0);
    var hx = 1.0;
    var hy = 1.0;

    if (kind == 0) {
        let d = p1 - p0;
        let len = length(d);
        if (len > 1e-4) {
            ax = d / len;
        }
        ay = vec2<f32>(-ax.y, ax.x);
        center = (p0 + p1) * 0.5;
        hx = len * 0.5 + inst.a1.x * 0.5 + px;
        hy = inst.a1.x * 0.5 + px;
    } else {
        var r = inst.a1.x;
        if (kind == 2 || kind == 3) {
            r = r + inst.a3.z * 0.5;
        }
        hx = r + px;
        hy = r + px;
    }

    var c = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) { c = vec2<f32>(1.0, -1.0); }
    else if (vi == 2u) { c = vec2<f32>(-1.0, 1.0); }
    else if (vi == 3u) { c = vec2<f32>(1.0, -1.0); }
    else if (vi == 4u) { c = vec2<f32>(-1.0, 1.0); }
    else if (vi == 5u) { c = vec2<f32>(1.0, 1.0); }

    let world = center + ax * (c.x * hx) + ay * (c.y * hy);

    var out: VsOut;
    out.clip = world_to_clip(world);
    out.world = world;
    out.p0 = p0;
    out.p1 = p1;
    out.prm = inst.a1;
    out.col = inst.a2;
    out.ext = inst.a3;
    return out;
}

fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ba = b - a;
    let dd = max(dot(ba, ba), 1e-6);
    let h = clamp(dot(pa, ba) / dd, 0.0, 1.0);
    return length(pa - ba * h);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let kind = i32(in.prm.y + 0.5);

    // clip circle (planet textures) in world coordinates
    if (in.prm.w > 0.0) {
        if (length(in.world - in.ext.xy) > in.prm.w) {
            discard;
        }
    }

    var d: f32;
    var half: f32;
    if (kind == 0) {
        d = seg_dist(in.world, in.p0, in.p1);
        half = in.prm.x * 0.5;
    } else if (kind == 1) {
        d = length(in.world - in.p0) - in.prm.x;
        half = 0.0;
    } else {
        d = abs(length(in.world - in.p0) - in.prm.x) - in.ext.z * 0.5;
        half = 0.0;
    }

    let aa = max(fwidth(d), 1e-5);
    var cov: f32;
    if (kind == 1) {
        cov = clamp(0.5 - d / aa, 0.0, 1.0);
    } else {
        cov = clamp((half - d) / aa + 0.5, 0.0, 1.0);
    }

    // Container dashes: a fixed-length "dash/gap" pattern in screen px.
    // Arc length in px = angle * radius * zoom (phase 0 starts at angle 0,
    // as in PathEffect.dashPathEffect(phase = 0)).
    if (kind == 3) {
        let rel = in.world - in.p0;
        let per = max(in.ext.w, 1.0);
        let arc = atan2(rel.y, rel.x) * in.prm.x * fr.zoom;
        let x = fract(arc / per);
        let sd = min(x, 0.5 - x) * per; // px inside the dash (>0) / gap (<0)
        cov = cov * clamp(sd + 0.5, 0.0, 1.0);
    }

    let a = in.col.a * in.prm.z * cov;
    if (a <= 0.001) {
        discard;
    }
    return vec4<f32>(in.col.rgb, a);
}
"#;

const FILL_WGSL: &str = r#"
struct Frame {
    resolution: vec2<f32>,
    zoom: f32,
    cam_x: f32,
    cam_y: f32,
    p0: f32,
    p1: f32,
    p2: f32,
};

@group(0) @binding(0) var<uniform> fr: Frame;

struct FIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
};

struct FOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: FIn) -> FOut {
    let sp = (in.pos + vec2<f32>(fr.cam_x, fr.cam_y)) * fr.zoom + fr.resolution * 0.5;
    var o: FOut;
    o.clip = vec4<f32>(
        sp.x / fr.resolution.x * 2.0 - 1.0,
        1.0 - sp.y / fr.resolution.y * 2.0,
        0.0,
        1.0,
    );
    o.color = in.color;
    return o;
}

@fragment
fn fs_main(in: FOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb, in.color.a);
}
"#;

// ---------------------------------------------------------------------------
// Instance emission
// ---------------------------------------------------------------------------
#[inline]
fn push_line(
    out: &mut Vec<Instance>,
    p0: Vec2,
    p1: Vec2,
    width: f32,
    color: [f32; 4],
    alpha: f32,
    clip_c: Vec2,
    clip_r: f32,
) {
    if alpha <= 0.0 {
        return;
    }
    out.push(Instance {
        a0: [p0.x, p0.y, p1.x, p1.y],
        a1: [width, KIND_LINE, alpha, clip_r],
        a2: color,
        a3: [clip_c.x, clip_c.y, 0.0, 0.0],
    });
}

#[inline]
fn push_circle_fill(out: &mut Vec<Instance>, c: Vec2, r: f32, color: [f32; 4], alpha: f32) {
    if r <= 0.0 || alpha <= 0.0 {
        return;
    }
    out.push(Instance {
        a0: [c.x, c.y, 0.0, 0.0],
        a1: [r, KIND_CIRCLE_FILL, alpha, 0.0],
        a2: color,
        a3: [0.0, 0.0, 0.0, 0.0],
    });
}

#[inline]
fn push_circle_stroke(
    out: &mut Vec<Instance>,
    c: Vec2,
    r: f32,
    thickness: f32,
    color: [f32; 4],
    alpha: f32,
    clip_c: Vec2,
    clip_r: f32,
) {
    if r <= 0.0 || alpha <= 0.0 {
        return;
    }
    out.push(Instance {
        a0: [c.x, c.y, 0.0, 0.0],
        a1: [r, KIND_CIRCLE_STROKE, alpha, clip_r],
        a2: color,
        a3: [clip_c.x, clip_c.y, thickness, 0.0],
    });
}

/// Dashed circle (Container): `dash_period_px` is the period of the dash pattern
/// in screen pixels (dash = gap = period/2), independent of zoom.
#[inline]
fn push_dashed_circle(
    out: &mut Vec<Instance>,
    c: Vec2,
    r: f32,
    thickness: f32,
    dash_period_px: f32,
    color: [f32; 4],
    alpha: f32,
) {
    if r <= 0.0 || alpha <= 0.0 || dash_period_px <= 0.0 {
        return;
    }
    out.push(Instance {
        a0: [c.x, c.y, 0.0, 0.0],
        a1: [r, KIND_DASHED_CIRCLE, alpha, 0.0],
        a2: color,
        a3: [0.0, 0.0, thickness, dash_period_px],
    });
}

/// The visible (in world units) region of the frame, with a 10 % margin — used to cull
/// invisible objects, sparks and track segments.
///
/// All the tests are conservative: an object is discarded only if it definitely does not
/// intersect a single pixel of the frame, so culling never changes the picture.
#[derive(Clone, Copy)]
struct View {
    min: Vec2,
    max: Vec2,
    center: Vec2,
    half_diag: f32,
}

impl View {
    /// `cam` — cameraOffset from MainActivity (world meters "from the ship toward (0,0)").
    fn new(cam: Vec2, zoom: f32, width: u32, height: u32) -> Self {
        let z = zoom.max(1e-9);
        let hw = width as f32 * 0.5 / z;
        let hh = height as f32 * 0.5 / z;
        let c = Vec2::new(-cam.x, -cam.y);
        let m = 0.1;
        Self {
            min: Vec2::new(c.x - hw * (1.0 + m), c.y - hh * (1.0 + m)),
            max: Vec2::new(c.x + hw * (1.0 + m), c.y + hh * (1.0 + m)),
            center: c,
            half_diag: (hw * hw + hh * hh).sqrt() * 1.1,
        }
    }

    /// A filled disk of radius `r` centered at `c` intersects the frame.
    #[inline]
    fn disk_visible(&self, c: Vec2, r: f32) -> bool {
        (c - self.center).mag() <= r + self.half_diag
    }

    /// A thin circle of radius `r` centered at `c` intersects the frame.
    /// (The circle passes at distance `|d - r|` from the frame center.)
    #[inline]
    fn circle_visible(&self, c: Vec2, r: f32) -> bool {
        ((c - self.center).mag() - r).abs() <= self.half_diag
    }

    /// Point `p` inside the frame with an extra margin `pad` (in world units).
    #[inline]
    fn point_visible(&self, p: Vec2, pad: f32) -> bool {
        p.x >= self.min.x - pad
            && p.x <= self.max.x + pad
            && p.y >= self.min.y - pad
            && p.y <= self.max.y + pad
    }

    /// Conservative: the bbox of the segment `a..b`, expanded by `pad`, intersects the frame.
    #[inline]
    fn seg_visible(&self, a: Vec2, b: Vec2, pad: f32) -> bool {
        let x0 = a.x.min(b.x);
        let x1 = a.x.max(b.x);
        let y0 = a.y.min(b.y);
        let y1 = a.y.max(b.y);
        x1 >= self.min.x - pad
            && x0 <= self.max.x + pad
            && y1 >= self.min.y - pad
            && y0 <= self.max.y + pad
    }
}

fn push_polyline(
    out: &mut Vec<Instance>,
    pts: &[Vec2],
    closed: bool,
    width: f32,
    color: [f32; 4],
    alpha: f32,
    clip_c: Vec2,
    clip_r: f32,
) {
    if pts.len() < 2 {
        return;
    }
    for w in pts.windows(2) {
        push_line(out, w[0], w[1], width, color, alpha, clip_c, clip_r);
    }
    if closed {
        push_line(
            out,
            pts[pts.len() - 1],
            pts[0],
            width,
            color,
            alpha,
            clip_c,
            clip_r,
        );
    }
}

// ---------------------------------------------------------------------------
// GPU context
// ---------------------------------------------------------------------------
struct Gpu {
    shapes_pipeline: wgpu::RenderPipeline,
    fill_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    frame_buffer: wgpu::Buffer,
    inst_buffer: wgpu::Buffer,
    inst_capacity: usize,
    fill_vbuffer: wgpu::Buffer,
    fill_vcapacity: usize,
    index_buffer: wgpu::Buffer,
    index_count: u32,
}

impl Gpu {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, indices: &[u32]) -> Self {
        let shapes_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("landroid-shapes-shader"),
            source: wgpu::ShaderSource::Wgsl(SHAPES_WGSL.into()),
        });
        let fill_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("landroid-fill-shader"),
            source: wgpu::ShaderSource::Wgsl(FILL_WGSL.into()),
        });

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-frame-uniform"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("landroid-frame-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("landroid-frame-bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("landroid-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let inst_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 32,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 48,
                shader_location: 3,
            },
        ];
        let inst_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &inst_attrs,
        };

        let shapes_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("landroid-shapes-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shapes_module,
                entry_point: Some("vs_main"),
                buffers: &[inst_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shapes_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let fill_attrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 8,
                shader_location: 1,
            },
        ];
        let fill_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<FillVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &fill_attrs,
        };

        let fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("landroid-fill-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fill_module,
                entry_point: Some("vs_main"),
                buffers: &[fill_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fill_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        const INIT_INST: usize = 300_000;
        let inst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-instances"),
            size: (INIT_INST * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fill_vcapacity = 4096usize;
        let fill_vbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-fill-vertices"),
            size: (fill_vcapacity * std::mem::size_of::<FillVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-fill-indices"),
            size: (indices.len().max(1) * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shapes_pipeline,
            fill_pipeline,
            bind_group,
            frame_buffer,
            inst_buffer,
            inst_capacity: INIT_INST,
            fill_vbuffer,
            fill_vcapacity,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    fn ensure_instances(&mut self, device: &wgpu::Device, needed: usize) {
        if needed <= self.inst_capacity {
            return;
        }
        let cap = needed.next_power_of_two().max(self.inst_capacity * 2);
        self.inst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-instances"),
            size: (cap * std::mem::size_of::<Instance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.inst_capacity = cap;
    }

    fn ensure_fill_vertices(&mut self, device: &wgpu::Device, needed: usize) {
        if needed <= self.fill_vcapacity {
            return;
        }
        let cap = needed.next_power_of_two().max(self.fill_vcapacity * 2);
        self.fill_vbuffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("landroid-fill-vertices"),
            size: (cap * std::mem::size_of::<FillVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.fill_vcapacity = cap;
    }
}

// ---------------------------------------------------------------------------
// Triangulation (ear clipping)
// ---------------------------------------------------------------------------
fn cross2(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

fn point_in_tri(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

/// Simple and correct ear-clipping triangulation of a closed polygon.
fn triangulate(poly: &[(f32, f32)]) -> Vec<u32> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();

    let mut area = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area += poly[i].0 * poly[j].1 - poly[j].0 * poly[i].1;
    }
    if area < 0.0 {
        idx.reverse();
    }

    let mut tris: Vec<u32> = Vec::with_capacity((n - 2) * 3);
    let mut guard = 0usize;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut ear = false;
        for i in 0..m {
            let ip = idx[(i + m - 1) % m];
            let ic = idx[i];
            let inx = idx[(i + 1) % m];
            let a = poly[ip];
            let b = poly[ic];
            let c = poly[inx];
            if cross2(a, b, c) <= 0.0 {
                continue; // not a convex vertex
            }
            let mut contains = false;
            for &k in &idx {
                if k == ip || k == ic || k == inx {
                    continue;
                }
                if point_in_tri(poly[k], a, b, c) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }
            tris.push(ip as u32);
            tris.push(ic as u32);
            tris.push(inx as u32);
            idx.remove(i);
            ear = true;
            break;
        }
        if !ear {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push(idx[0] as u32);
        tris.push(idx[1] as u32);
        tris.push(idx[2] as u32);
    }
    tris
}

fn star_points(radius1: f32, radius2: f32, points: u32) -> Vec<Vec2> {
    let angle_step = PI2f / points as f32;
    let mut v = Vec::with_capacity((points as usize) * 2);
    v.push(Vec2::new(radius1, 0.0));
    v.push(Vec2::new(
        radius2 * (angle_step * 0.5).cos(),
        radius2 * (angle_step * 0.5).sin(),
    ));
    for i in 1..points {
        let i_f = i as f32;
        v.push(Vec2::new(
            radius1 * (angle_step * i_f).cos(),
            radius1 * (angle_step * i_f).sin(),
        ));
        v.push(Vec2::new(
            radius2 * (angle_step * (i_f + 0.5)).cos(),
            radius2 * (angle_step * (i_f + 0.5)).sin(),
        ));
    }
    v
}

fn polygon_points(radius: f32, sides: u32) -> Vec<Vec2> {
    let angle_step = PI2f / sides as f32;
    (0..sides)
        .map(|i| {
            let a = angle_step * i as f32;
            Vec2::new(radius * a.cos(), radius * a.sin())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------
pub struct Renderer {
    gpu: Option<Gpu>,
    instances: Vec<Instance>,
    fill_verts: Vec<FillVertex>,
    fill_indices: Vec<u32>,
    /// The visible region of the current frame (used for culling).
    view: View,
    /// Number of instances in the last drawn frame (for measurements/diagnostics).
    last_instances: usize,
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer {
    pub fn new() -> Self {
        let fill_indices = assets::SPACESHIP_PATH
            .first()
            .map(|p| triangulate(p))
            .unwrap_or_default();
        Self {
            gpu: None,
            instances: Vec::new(),
            fill_verts: Vec::new(),
            fill_indices,
            view: View::new(Vec2::ZERO, 1.0, 1, 1),
            last_instances: 0,
        }
    }

    /// Number of shape instances in the last frame (performance measurement/diagnostics).
    pub fn last_instance_count(&self) -> usize {
        self.last_instances
    }

    pub fn draw(&mut self, scene: &mut GpuScene<'_>, u: &Simulator, zoom: f32) {
        if self.gpu.is_none() {
            self.gpu = Some(Gpu::new(scene.device, scene.format, &self.fill_indices));
        }

        let now = u.now;
        let ship_pos = u.ship.body.pos;
        let cam = ship_pos * -1.0;
        self.view = View::new(cam, zoom, scene.width, scene.height);

        self.instances.clear();
        self.fill_verts.clear();

        // 1. Grid (drawn first, beneath everything else — like drawRect + grid
        //    before drawUniverse in MainActivity.Spaaaaace).
        self.emit_grid(zoom, cam, scene.width, scene.height);

        // 2. constraints: the landing flag (Landing), then the Container (ringfence).
        self.emit_landing_flag(u, zoom);
        self.emit_container(u, zoom);

        // 3. Star (drawStar: gravitational fields + fill + 2 crowns).
        self.emit_star(u, now, zoom);

        // 4. entities: planets (in creation order, as in ArraySet) and sparks.
        let mut order: Vec<usize> = (0..u.planets.len()).collect();
        order.sort_by_key(|&i| u.planets[i].spawn_order);
        for i in order {
            self.emit_planet(&u.planets[i], u, now, zoom);
        }
        for s in &u.sparks {
            self.emit_spark(s, zoom);
        }

        // 5. Autopilot.
        self.emit_autopilot(u, now, zoom);

        // 6. Ship: legs -> fill -> outline -> exhaust, and ONLY AFTER that
        //    the track (drawTrack is called inside drawSpacecraft after the ship).
        let fill_split = self.emit_ship(u, zoom);
        self.emit_track(u, zoom);
        let total = self.instances.len();
        self.last_instances = total;

        // ---- buffer upload ----
        let gpu = self.gpu.as_mut().unwrap();
        gpu.ensure_instances(scene.device, total.max(1));
        gpu.ensure_fill_vertices(scene.device, self.fill_verts.len().max(1));

        let frame = FrameUniform {
            resolution: [scene.width as f32, scene.height as f32],
            zoom,
            cam_x: cam.x,
            cam_y: cam.y,
            _p0: 0.0,
            _p1: 0.0,
            _p2: 0.0,
        };
        scene
            .queue
            .write_buffer(&gpu.frame_buffer, 0, bytemuck::bytes_of(&frame));
        if total > 0 {
            scene
                .queue
                .write_buffer(&gpu.inst_buffer, 0, bytemuck::cast_slice(&self.instances));
        }
        if !self.fill_verts.is_empty() {
            scene.queue.write_buffer(
                &gpu.fill_vbuffer,
                0,
                bytemuck::cast_slice(&self.fill_verts),
            );
        }

        let clear = bytes_to_wgpu_color(c4(assets::EIGENGRAU));

        let mut pass = scene.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("landroid-scene-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: scene.target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_bind_group(0, &gpu.bind_group, &[]);

        let idx_ok = gpu.index_count >= 3 && !self.fill_verts.is_empty();

        // 7a. everything drawn before the ship silhouette fill
        //     (grid, flag, Container, star, planets, sparks, autopilot, legs)
        if fill_split > 0 {
            pass.set_pipeline(&gpu.shapes_pipeline);
            pass.set_vertex_buffer(0, gpu.inst_buffer.slice(..));
            pass.draw(0..6, 0..fill_split as u32);
        }

        // 7b. ship polygon fill (Colors.Eigengrau, drawn "over" the legs)
        if idx_ok {
            pass.set_pipeline(&gpu.fill_pipeline);
            pass.set_vertex_buffer(0, gpu.fill_vbuffer.slice(..));
            pass.set_index_buffer(gpu.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..gpu.index_count, 0, 0..1);
        }

        // 7c. ship outline -> exhaust -> track
        if total > fill_split {
            pass.set_pipeline(&gpu.shapes_pipeline);
            pass.set_vertex_buffer(0, gpu.inst_buffer.slice(..));
            pass.draw(0..6, fill_split as u32..total as u32);
        }
    }

    fn emit_grid(&mut self, zoom: f32, cam: Vec2, width: u32, height: u32) {
        let half_w = width as f32 * 0.5;
        let half_h = height as f32 * 0.5;
        let z = zoom.max(1e-6);
        let left = (0.0 - half_w) / z - cam.x;
        let right = half_w / z - cam.x;
        let top = (0.0 - half_h) / z - cam.y;
        let bottom = half_h / z - cam.y;

        let mut step = 1000.0f32;
        let mut guard = 0;
        while step * z < 32.0 && guard < 32 {
            step *= 10.0;
            guard += 1;
        }

        let color = c4(assets::EIGENGRAU2);
        let major = step * 10.0;

        let mut x = (left / step).floor() * step;
        let mut it = 0;
        while x < right && it < 4096 {
            let w = if (x % major) == 0.0 { 3.0 } else { 1.5 } / z;
            push_line(
                &mut self.instances,
                Vec2::new(x, top),
                Vec2::new(x, bottom),
                w,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
            x += step;
            it += 1;
        }

        let mut y = (top / step).floor() * step;
        it = 0;
        while y < bottom && it < 4096 {
            let w = if (y % major) == 0.0 { 3.0 } else { 1.5 } / z;
            push_line(
                &mut self.instances,
                Vec2::new(left, y),
                Vec2::new(right, y),
                w,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
            y += step;
            it += 1;
        }
    }

    fn emit_grav_field(&mut self, pos: Vec2, mass: f32, now: f32, zoom: f32) {
        let view = self.view;
        let rings = 10;
        for i in 0..rings {
            let force = sim::lerp(2000.0, 0.01, (i as f32 - now % 1.0) / rings as f32);
            let r = (sim::GRAVITATION * mass * sim::SPACECRAFT_MASS / force.max(1e-6)).sqrt();
            let alpha = sim::lerp(0.75, 0.1, i as f32 / rings as f32);
            // The rings can be enormous (r up to ~3e6) — discard only those that
            // definitely do not intersect the frame.
            if !view.circle_visible(pos, r) {
                continue;
            }
            push_circle_stroke(
                &mut self.instances,
                pos,
                r,
                2.0 / zoom,
                [1.0, 0.0, 0.0, 1.0],
                alpha,
                Vec2::ZERO,
                0.0,
            );
        }
    }

    /// Container/ringfence: a dashed circle of radius UNIVERSE_RANGE around (0,0),
    /// color `Color(0xFF800000)`, thickness 1/zoom, dash/gap of 8 screen px each
    /// (VisibleUniverse.kt:drawContainer).
    fn emit_container(&mut self, u: &Simulator, zoom: f32) {
        let r = u.ringfence_radius;
        if !self.view.circle_visible(Vec2::ZERO, r) {
            return;
        }
        push_dashed_circle(
            &mut self.instances,
            Vec2::ZERO,
            r,
            1.0 / zoom,
            CONTAINER_DASH_PX * 2.0,
            CONTAINER_RGB,
            1.0,
        );
    }

    fn emit_star(&mut self, u: &Simulator, now: f32, zoom: f32) {
        let star = &u.star;
        let pos = star.body.pos;
        let r = star.body.radius;

        // the star's gravitational field (drawStar -> drawGravitationalField)
        self.emit_grav_field(pos, star.body.mass, now, zoom);

        // The star disk and both crowns lie within radius r + 250 — if this disk
        // does not intersect the frame, there is nothing to draw.
        if !self.view.disk_visible(pos, r + 250.0) {
            return;
        }

        push_circle_fill(&mut self.instances, pos, r, star.color, 1.0);

        // two crowns (createStar). cornerPathEffect(radius = 200) is not reproduced
        // in the shader renderer (see README: known deviations).
        let crown1 = star_points(r + 80.0, r + 250.0, STAR_POINTS);
        let rot1 = star.anim / 23.0 * PI2f;
        let pts1: Vec<Vec2> = crown1.iter().map(|p| pos + p.rotate(rot1)).collect();
        push_polyline(
            &mut self.instances,
            &pts1,
            true,
            3.0 / zoom,
            star.color,
            1.0,
            Vec2::ZERO,
            0.0,
        );

        let crown2 = star_points(r + 20.0, r + 200.0, STAR_POINTS + 1);
        let rot2 = star.anim / -19.0 * PI2f;
        let pts2: Vec<Vec2> = crown2.iter().map(|p| pos + p.rotate(rot2)).collect();
        push_polyline(
            &mut self.instances,
            &pts2,
            true,
            3.0 / zoom,
            star.color,
            1.0,
            Vec2::ZERO,
            0.0,
        );
    }

    fn emit_planet(&mut self, p: &sim::Planet, u: &Simulator, now: f32, zoom: f32) {
        let draw_color = if p.explored {
            p.color
        } else {
            c4(assets::EIGENGRAU4)
        };
        let view = self.view;

        let orbit_r = p.body.pos.distance(p.orbit_center);
        if view.circle_visible(p.orbit_center, orbit_r) {
            push_circle_stroke(
                &mut self.instances,
                p.orbit_center,
                orbit_r,
                1.0 / zoom,
                c4(assets::EIGENGRAU3),
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }

        self.emit_grav_field(p.body.pos, p.body.mass, now, zoom);

        if !view.disk_visible(p.body.pos, p.body.radius) {
            return;
        }

        push_circle_fill(&mut self.instances, p.body.pos, p.body.radius, c4(assets::EIGENGRAU), 1.0);

        // texture (only after exploration and when near the ship)
        if p.explored
            && zoom > 0.05
            && p.body.pos.distance(u.ship.body.pos) < 10_000.0
        {
            self.emit_planet_texture(p, draw_color, zoom);
        }

        push_circle_stroke(
            &mut self.instances,
            p.body.pos,
            p.body.radius,
            2.0 / zoom,
            draw_color,
            1.0,
            Vec2::ZERO,
            0.0,
        );
    }

    fn emit_planet_texture(&mut self, p: &sim::Planet, color: [f32; 4], zoom: f32) {
        let radius = p.body.radius;
        let center = p.body.pos;
        let n_tex = assets::PLANET_TEXTURES.len();
        let tex_idx = (((radius % 17.0) / 17.0 * n_tex as f32) as usize).min(n_tex - 1);
        let tex = assets::PLANET_TEXTURES[tex_idx];
        let scale = radius / (assets::PLANET_TEXTURE_SIZE / 2.0);
        let rot = PI2f * (radius % 100.0) / 100.0;
        // As in the original (VisibleUniverse.kt): the stroke width is 1f / zoom in the local
        // system; after scale(textureScale) this yields exactly 1 pixel on screen.
        let width = 1.0 / zoom;

        let to_world = |p: (f32, f32)| -> Vec2 {
            let q = Vec2::new(p.0 * scale - radius, p.1 * scale - radius);
            center + q.rotate(rot)
        };

        // A texture can consist of thousands of segments; everything outside the frame
        // is culled by the clip circle of the planet's radius anyway.
        let view = self.view;
        for sub in tex {
            for w in sub.windows(2) {
                let a = to_world(w[0]);
                let b = to_world(w[1]);
                if !view.seg_visible(a, b, width) {
                    continue;
                }
                push_line(
                    &mut self.instances,
                    a,
                    b,
                    width,
                    color,
                    1.0,
                    center,
                    radius,
                );
            }
        }
    }

    fn emit_spark(&mut self, s: &sim::Spark, zoom: f32) {
        if s.fuse.lifetime < 0.0 {
            return;
        }
        let view = self.view;
        let life = 1.0 - s.fuse.lifetime / s.ttl;
        let pos = s.body.pos;
        match s.style {
            sim::SparkStyle::Line => {
                if !s.body.opos.is_zero() && view.seg_visible(s.body.opos, pos, s.size * 0.5) {
                    push_line(
                        &mut self.instances,
                        s.body.opos,
                        pos,
                        s.size,
                        s.color,
                        1.0,
                        Vec2::ZERO,
                        0.0,
                    );
                }
            }
            sim::SparkStyle::LineAbsolute => {
                if !s.body.opos.is_zero()
                    && view.seg_visible(s.body.opos, pos, s.size * 0.5 / zoom)
                {
                    push_line(
                        &mut self.instances,
                        s.body.opos,
                        pos,
                        s.size / zoom,
                        s.color,
                        1.0,
                        Vec2::ZERO,
                        0.0,
                    );
                }
            }
            sim::SparkStyle::Dot => {
                if view.point_visible(pos, s.size) {
                    push_circle_fill(&mut self.instances, pos, s.size, s.color, 1.0);
                }
            }
            sim::SparkStyle::DotAbsolute => {
                // Canon: center is pos / zoom (VisibleUniverse.kt:drawSpark, DOT_ABSOLUTE).
                let c = pos * (1.0 / zoom);
                if view.point_visible(c, s.size) {
                    push_circle_fill(&mut self.instances, c, s.size, s.color, 1.0);
                }
            }
            sim::SparkStyle::Ring => {
                let alpha = s.color[3] * (1.0 - life);
                let r = sim::lerp(s.size, 3.0 * s.size, life).exp() - 1.0;
                let r_max = (3.0 * s.size).exp() - 1.0;
                if view.circle_visible(pos, r_max.max(r)) {
                    push_circle_stroke(
                        &mut self.instances,
                        pos,
                        r,
                        1.0 / zoom,
                        [s.color[0], s.color[1], s.color[2], 1.0],
                        alpha,
                        Vec2::ZERO,
                        0.0,
                    );
                }
            }
        }
    }

    fn emit_track(&mut self, u: &Simulator, zoom: f32) {
        let pos = &u.ship.track.positions;
        let n = pos.len();
        if n < 2 {
            return;
        }
        let view = self.view;
        let color = c4(assets::TRACK);
        let w = 1.0 / zoom;
        // drawPoints(PointMode.Lines) draws each pair independently, so segments can be
        // discarded one by one (with a 10 % margin) without "breaking" the neighbours.
        for i in 0..n - 1 {
            if !view.seg_visible(pos[i], pos[i + 1], w) {
                continue;
            }
            push_line(
                &mut self.instances,
                pos[i],
                pos[i + 1],
                w,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }
    }

    fn emit_autopilot(&mut self, u: &Simulator, now: f32, zoom: f32) {
        let ap = &u.autopilot;
        let ti = match ap.target {
            Some(t) if t < u.planets.len() => t,
            _ => return,
        };
        let target = &u.planets[ti];
        let blue = c4(assets::AUTOPILOT);
        let color = with_alpha(blue, 0.5);
        let view = self.view;

        if view.circle_visible(target.body.pos, target.body.radius + ap.braking_distance) {
            let poly = polygon_points(target.body.radius + ap.braking_distance, 15);
            let rot = now * PI2f / 10.0;
            let pts: Vec<Vec2> = poly.iter().map(|p| target.body.pos + p.rotate(rot)).collect();
            push_polyline(
                &mut self.instances,
                &pts,
                true,
                1.0 / zoom,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }

        if view.circle_visible(target.body.pos, target.body.radius + ap.landing_altitude / 2.0) {
            push_circle_stroke(
                &mut self.instances,
                target.body.pos,
                target.body.radius + ap.landing_altitude / 2.0,
                ap.landing_altitude,
                color,
                0.25,
                Vec2::ZERO,
                0.0,
            );
        }

        if view.seg_visible(u.ship.body.pos, ap.leading_pos, 1.0 / zoom) {
            push_line(
                &mut self.instances,
                u.ship.body.pos,
                ap.leading_pos,
                1.0 / zoom,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }
        if view.circle_visible(ap.leading_pos, 5.0 / zoom) {
            push_circle_stroke(
                &mut self.instances,
                ap.leading_pos,
                5.0 / zoom,
                1.0 / zoom,
                color,
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }
    }

    fn emit_landing_flag(&mut self, u: &Simulator, zoom: f32) {
        let li = match u.ship.landing {
            Some(i) => i,
            None => return,
        };
        let landing = match u.landings.get(li) {
            Some(l) => *l,
            None => return,
        };
        let planet = match u.planets.get(landing.planet) {
            Some(p) => p,
            None => return,
        };

        let v = planet.body.pos + Vec2::make_with_angle_mag(landing.angle, planet.body.radius);
        let height = 80.0f32;
        if !self.view.point_visible(v, height + 2.0 / zoom) {
            return;
        }
        let local = [
            Vec2::new(0.0, 0.0),
            Vec2::new(height, 0.0),
            Vec2::new(height * 0.875, height * 0.25),
            Vec2::new(height * 0.75, 0.0),
        ];
        let pts: Vec<Vec2> = local.iter().map(|p| v + p.rotate(landing.angle)).collect();
        push_polyline(
            &mut self.instances,
            &pts,
            true,
            2.0 / zoom,
            c4(assets::FLAG),
            1.0,
            Vec2::ZERO,
            0.0,
        );
    }

    /// Draws the ship and returns the instance index up to which the "pre-fill layer"
    /// (the legs) extends — the silhouette fill is drawn between `[0, split)` and `[split, len)`.
    fn emit_ship(&mut self, u: &Simulator, zoom: f32) -> usize {
        let ship = &u.ship;
        let pos = ship.body.pos;
        let ang = ship.body.angle;
        let (sin_a, cos_a) = ang.sin_cos();
        let transform = |p: (f32, f32)| -> Vec2 {
            Vec2::new(
                pos.x + p.0 * cos_a - p.1 * sin_a,
                pos.y + p.0 * sin_a + p.1 * cos_a,
            )
        };

        if ship.landing.is_some() {
            for sub in assets::SPACESHIP_LEGS {
                let pts: Vec<Vec2> = sub.iter().map(|&p| transform(p)).collect();
                push_polyline(
                    &mut self.instances,
                    &pts,
                    false,
                    2.0 / zoom,
                    [0xCC as f32 / 255.0, 0xCC as f32 / 255.0, 0xCC as f32 / 255.0, 1.0],
                    1.0,
                    Vec2::ZERO,
                    0.0,
                );
            }
        }

        let fill_split = self.instances.len();

        // silhouette fill (gather the polygon's vertices in world coordinates)
        if let Some(sub) = assets::SPACESHIP_PATH.first() {
            let eg = c4(assets::EIGENGRAU);
            for &p in sub.iter() {
                let w = transform(p);
                self.fill_verts.push(FillVertex {
                    pos: [w.x, w.y],
                    color: eg,
                });
            }
        }

        // silhouette outline (white; black while transiting in front of the star)
        let outline = if ship.transit {
            [0.0, 0.0, 0.0, 1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        };
        for sub in assets::SPACESHIP_PATH {
            let pts: Vec<Vec2> = sub.iter().map(|&p| transform(p)).collect();
            push_polyline(
                &mut self.instances,
                &pts,
                true,
                2.0 / zoom,
                outline,
                1.0,
                Vec2::ZERO,
                0.0,
            );
        }

        if !ship.thrust.is_zero() {
            for sub in assets::THRUST_PATH {
                let pts: Vec<Vec2> = sub.iter().map(|&p| transform(p)).collect();
                push_polyline(
                    &mut self.instances,
                    &pts,
                    true,
                    2.0 / zoom,
                    [0xFF as f32 / 255.0, 0x88 as f32 / 255.0, 0.0, 1.0],
                    1.0,
                    Vec2::ZERO,
                    0.0,
                );
            }
        }

        fill_split
    }
}

/// sRGB value 0..1 -> wgpu::Color (written as-is, with no gamma conversion).
fn bytes_to_wgpu_color(c: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: c[0] as f64,
        g: c[1] as f64,
        b: c[2] as f64,
        a: c[3] as f64,
    }
}
