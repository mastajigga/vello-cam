//! vello-cam — caméra WebGPU en Rust/WASM.
//!
//! Chaîne : get_user_media(JS) -> <video> -> `copy_external_image_to_texture`
//!          -> filtres WGSL en ping-pong -> surface -> UI vectorielle vello_gpu (SrcOver).
//!
//! Capture : relecture GPU (`copy_texture_to_buffer` + `map_async`) -> PNG côté JS.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use vello_common::peniko::{
    kurbo::{Circle, Rect, RoundedRect, Shape},
    Color,
};
use vello_gpu::{
    RenderSize, RenderSettings, RenderTargetConfig, Renderer, Scene, TargetInit, TextureBindings,
};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wgpu::util::DeviceExt;
use web_sys::{HtmlCanvasElement, HtmlVideoElement};

const SHADER: &str = include_str!("shader.wgsl");

fn log(s: &str) {
    web_sys::console::log_1(&JsValue::from_str(s));
}

/// Horloge navigateur (ms), pour decomposer les latences.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Appelle une fonction JS posée sur `globalThis` (définies dans index.html).
fn js(name: &str, args: &[JsValue]) {
    let g = js_sys::global();
    match js_sys::Reflect::get(&g, &JsValue::from_str(name)) {
        Ok(v) if v.is_function() => {
            let f: js_sys::Function = v.unchecked_into();
            let arr = js_sys::Array::new();
            for a in args {
                arr.push(a);
            }
            let _ = f.apply(&JsValue::NULL, &arr);
        }
        _ => log(&format!("[vello-cam] fonction JS '{name}' absente")),
    }
}

fn status(s: &str) {
    js("velloCamStatus", &[JsValue::from_str(s)]);
}

// ---------------------------------------------------------------- uniformes

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Filters {
    brightness: f32,
    contrast: f32,
    saturation: f32,
    temperature: f32,
    sepia: f32,
    grayscale: f32,
    vignette: f32,
    grain: f32,
    blur: f32,
    aberration: f32,
    fisheye: f32,
    time: f32,
    uvscale: [f32; 2],
    texel: [f32; 2],
}

impl Filters {
    /// Étalonnage neutre : utilisé par l'auto-test.
    fn neutral() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
            sepia: 0.0,
            grayscale: 0.0,
            vignette: 0.0,
            grain: 0.0,
            blur: 0.0,
            aberration: 0.0,
            fisheye: 0.0,
            time: 0.0,
            uvscale: [1.0, 1.0],
            texel: [0.5, 0.5],
        }
    }
    /// "Look" par défaut : léger contraste, un peu de saturation.
    fn look() -> Self {
        Self {
            brightness: 0.02,
            contrast: 1.07,
            saturation: 1.06,
            ..Self::neutral()
        }
    }
    fn spatial(&self) -> bool {
        self.blur > 0.001 || self.aberration > 0.001
    }
}

fn is_bgra(f: wgpu::TextureFormat) -> bool {
    matches!(
        f,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    )
}

// ---------------------------------------------------------------- textures

struct CamTex {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    w: u32,
    h: u32,
}

fn make_target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    w: u32,
    h: u32,
    extra: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

// ---------------------------------------------------------------- App

#[derive(Copy, Clone)]
struct Chip {
    x: f32,
    y: f32,
    s: f32,
}

const NCHIP: usize = 5;

struct App {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    canvas: HtmlCanvasElement,
    video: HtmlVideoElement,
    mirror: Option<HtmlCanvasElement>,
    direct: bool,
    dpr: f32,

    sampler: wgpu::Sampler,
    uni: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    p_grade: wgpu::RenderPipeline,
    p_spatial: wgpu::RenderPipeline,

    cam: Option<CamTex>,
    ping_tex: wgpu::Texture,
    ping_view: wgpu::TextureView,
    cap_tex: wgpu::Texture,
    cap_view: wgpu::TextureView,

    renderer: Renderer,
    resources: vello_gpu::Resources,
    depth: wgpu::TextureView,
    scene: Scene,

    w: u32,
    h: u32,
    src_w: u32,
    src_h: u32,

    filters: Filters,
    on: [bool; NCHIP],
    sel: [f32; NCHIP],
    vel: [f32; NCHIP],

    chips: Vec<Chip>,
    shutter: (f32, f32, f32),
    rec: (f32, f32, f32),

    t0: f64,
    t_prev: f64,
    time: f32,
    frames: u64,
    fps: f32,
    fps_acc: f64,
    fps_n: u32,
    hud_acc: f64,

    recording: bool,
    want_capture: bool,
    /// Vrai tant qu'une capture est en vol : on arrete le rendu lourd pour que la
    /// file GPU se vide, sinon le `map_async` de relecture est affame (mesure : 35 s
    /// sous SwiftShader, contre quelques ms file vide).
    capture_busy: Rc<Cell<bool>>,
    capture_frames: u32,
    flash: f32,
    first_frame: bool,
}

impl App {
    fn layout(&mut self) {
        let dpr = self.dpr;
        let (wf, hf) = (self.w as f32, self.h as f32);
        let s = (wf * 0.052).clamp(38.0 * dpr, 56.0 * dpr);
        let gap = s * 0.26;
        let total = NCHIP as f32 * s + (NCHIP as f32 - 1.0) * gap;
        let x0 = (wf - total) * 0.5;
        let y = hf - 116.0 * dpr - s;
        self.chips = (0..NCHIP)
            .map(|i| Chip {
                x: x0 + i as f32 * (s + gap),
                y,
                s,
            })
            .collect();
        self.shutter = (wf * 0.5, hf - 56.0 * dpr, 30.0 * dpr);
        let rs = 22.0 * dpr;
        self.rec = (wf - 34.0 * dpr - rs, 40.0 * dpr, rs);
    }

    fn draw(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dst: &wgpu::TextureView,
        pipe: &wgpu::RenderPipeline,
        src: &wgpu::TextureView,
    ) {
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("filtres"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(src),
                },
            ],
        });
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("filtres"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.02,
                        g: 0.02,
                        b: 0.03,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rp.set_pipeline(pipe);
        rp.set_bind_group(0, &bg, &[]);
        rp.draw(0..3, 0..1);
    }

    /// Chaîne de filtres : grade (toujours) puis, si besoin, spatial — en ping-pong.
    fn chain(&mut self, enc: &mut wgpu::CommandEncoder, dst: &wgpu::TextureView) {
        let cam_view = match &self.cam {
            Some(c) => c.view.clone(),
            None => return,
        };
        if self.filters.spatial() {
            self.draw(enc, &self.ping_view, &self.p_grade, &cam_view);
            self.draw(enc, dst, &self.p_spatial, &self.ping_view);
        } else {
            self.draw(enc, dst, &self.p_grade, &cam_view);
        }
    }

    fn clear_pass(&self, enc: &mut wgpu::CommandEncoder, dst: &wgpu::TextureView) {
        let _ = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fond"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.031,
                        g: 0.035,
                        b: 0.047,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    fn ensure_cam(&mut self, vw: u32, vh: u32) {
        let same = self
            .cam
            .as_ref()
            .map(|c| c.w == vw && c.h == vh)
            .unwrap_or(false);
        if same {
            return;
        }
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("camera"),
            size: wgpu::Extent3d {
                width: vw,
                height: vh,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.cam = Some(CamTex {
            tex,
            view,
            w: vw,
            h: vh,
        });
        log(&format!("[vello-cam] texture caméra {vw}x{vh}"));
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.w = w.max(1);
        self.h = h.max(1);
        self.canvas.set_width(self.w);
        self.canvas.set_height(self.h);
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: self.w,
                height: self.h,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                desired_maximum_frame_latency: 2,
                view_formats: vec![],
            },
        );
        let (pt, pv) = make_target(
            &self.device,
            "ping",
            self.format,
            self.w,
            self.h,
            wgpu::TextureUsages::empty(),
        );
        self.ping_tex = pt;
        self.ping_view = pv;
        let (ct, cv) = make_target(
            &self.device,
            "capture",
            self.format,
            self.w,
            self.h,
            wgpu::TextureUsages::COPY_SRC,
        );
        self.cap_tex = ct;
        self.cap_view = cv;

        let (r, res) = Renderer::new_with(
            &self.device,
            &RenderTargetConfig {
                format: self.format,
                width: self.w.try_into().unwrap(),
                height: self.h.try_into().unwrap(),
            },
            RenderSettings::default(),
        );
        self.renderer = r;
        self.resources = res;
        self.depth = Renderer::create_depth_texture_view(
            &self.device,
            &RenderSize {
                width: self.w.try_into().unwrap(),
                height: self.h.try_into().unwrap(),
            },
        );
        self.scene = Scene::new(self.w as u16, self.h as u16);
        self.layout();
        log(&format!("[vello-cam] surface {}x{}", self.w, self.h));
    }

    fn toggle(&mut self, i: usize) {
        if i >= NCHIP {
            return;
        }
        self.on[i] = !self.on[i];
        let label = ["n&b", "sépia", "flou", "fisheye", "vignette"][i];
        status(&format!(
            "{} {}",
            label,
            if self.on[i] { "activé" } else { "coupé" }
        ));
    }

    fn hit(&mut self, x: f32, y: f32) -> bool {
        for i in 0..NCHIP {
            let c = self.chips[i];
            if x >= c.x && x <= c.x + c.s && y >= c.y && y <= c.y + c.s {
                self.toggle(i);
                return true;
            }
        }
        let (cx, cy, r) = self.shutter;
        let dx = x - cx;
        let dy = y - cy;
        if dx * dx + dy * dy <= (r * 1.35) * (r * 1.35) {
            self.want_capture = true;
            self.flash = 1.0;
            status("photo…");
            return true;
        }
        let (rx, ry, rs) = self.rec;
        if x >= rx && x <= rx + rs && y >= ry && y <= ry + rs {
            self.recording = !self.recording;
            js("velloCamRecord", &[JsValue::from_bool(self.recording)]);
            status(if self.recording {
                "enregistrement…"
            } else {
                "enregistrement arrêté"
            });
            return true;
        }
        false
    }

    // ------------------------------------------------------------ frame

    fn frame(&mut self, t_ms: f64) -> Result<(), String> {
        if self.first_frame {
            self.first_frame = false;
            self.t0 = t_ms;
            self.t_prev = t_ms;
        }
        // une capture est en vol : on ne soumet plus rien, la file se vide et le
        // readback se resout immediatement. Garde-fou si la relecture n'aboutit pas.
        if self.capture_busy.get() {
            self.capture_frames += 1;
            if self.capture_frames > 3000 {
                log(&format!(
                    "[vello-cam] t={:.0} capture: delai depasse, reprise du rendu",
                    now_ms()
                ));
                self.capture_busy.set(false);
                self.capture_frames = 0;
            }
            return Ok(());
        }

        let dt = (((t_ms - self.t_prev) / 1000.0) as f32).clamp(0.0, 0.05);
        self.t_prev = t_ms;
        self.time = ((t_ms - self.t0) / 1000.0) as f32;

        // fps
        self.fps_acc += dt as f64;
        self.fps_n += 1;
        if self.fps_acc >= 0.5 {
            self.fps = (self.fps_n as f64 / self.fps_acc) as f32;
            self.fps_acc = 0.0;
            self.fps_n = 0;
        }
        self.hud_acc += dt as f64;
        if self.hud_acc >= 0.5 {
            self.hud_acc = 0.0;
            js(
                "velloCamHud",
                &[
                    JsValue::from_f64(self.fps as f64),
                    JsValue::from_f64(self.w as f64),
                    JsValue::from_f64(self.h as f64),
                ],
            );
        }

        // ressorts des pastilles
        for i in 0..NCHIP {
            let target = if self.on[i] { 1.0 } else { 0.0 };
            let a = (target - self.sel[i]) * 320.0 - self.vel[i] * 24.0;
            self.vel[i] += a * dt;
            self.sel[i] += self.vel[i] * dt;
        }
        self.flash = (self.flash - dt * 3.2).max(0.0);

        // taille du canvas
        let dpr = self.dpr;
        let cw = (self.canvas.client_width() as f32 * dpr).max(1.0) as u32;
        let ch = (self.canvas.client_height() as f32 * dpr).max(1.0) as u32;
        if cw != self.w || ch != self.h {
            self.resize(cw, ch);
        }

        // caméra -> texture GPU
        // `direct` : <video> est copiée telle quelle dans la texture (aucun passage CPU).
        // sinon     : repli sur un canvas 2D alimenté par drawImage (marche partout).
        let mut fresh = false;
        let (vw, vh, ready) = if self.direct {
            (
                self.video.video_width(),
                self.video.video_height(),
                self.video.ready_state() >= 2,
            )
        } else {
            match &self.mirror {
                Some(c) => (c.width(), c.height(), c.width() > 1),
                None => (0, 0, false),
            }
        };
        if vw > 0 && vh > 0 && ready {
            self.ensure_cam(vw, vh);
            let src = match (&self.mirror, self.direct) {
                (_, true) => wgpu::ExternalImageSource::HTMLVideoElement(self.video.clone()),
                (Some(c), false) => wgpu::ExternalImageSource::HTMLCanvasElement(c.clone()),
                (None, false) => {
                    return Ok(());
                }
            };
            self.queue.copy_external_image_to_texture(
                &wgpu::CopyExternalImageSourceInfo {
                    source: src,
                    origin: wgpu::Origin2d::ZERO,
                    flip_y: false,
                },
                wgpu::CopyExternalImageDestInfo {
                    texture: &self.cam.as_ref().unwrap().tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                    color_space: wgpu::PredefinedColorSpace::Srgb,
                    premultiplied_alpha: false,
                },
                wgpu::Extent3d {
                    width: vw,
                    height: vh,
                    depth_or_array_layers: 1,
                },
            );
            fresh = true;
            if self.src_w != vw || self.src_h != vh {
                self.src_w = vw;
                self.src_h = vh;
                status(&format!("caméra {vw}×{vh}"));
                js("velloCamSource", &[JsValue::from_f64(vw as f64), JsValue::from_f64(vh as f64)]);
            }
        }

        // uniformes
        let f = &mut self.filters;
        f.time = self.time;
        f.grayscale = if self.on[0] { 1.0 } else { 0.0 };
        f.sepia = if self.on[1] { 1.0 } else { 0.0 };
        f.blur = if self.on[2] { 0.55 } else { 0.0 };
        f.fisheye = if self.on[3] { 0.45 } else { 0.0 };
        f.vignette = if self.on[4] { 0.85 } else { 0.0 };
        f.texel = [1.0 / self.w as f32, 1.0 / self.h as f32];
        let (ca, va) = (
            self.w as f32 / self.h as f32,
            self.src_w.max(1) as f32 / self.src_h.max(1) as f32,
        );
        f.uvscale = if ca > va {
            [1.0, va / ca]
        } else {
            [ca / va, 1.0]
        };
        self.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&self.filters));

        // ---- rendu ----
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(fr)
            | wgpu::CurrentSurfaceTexture::Suboptimal(fr) => fr,
            _ => return Ok(()),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });

        if fresh {
            self.chain(&mut enc, &view);
        } else {
            self.clear_pass(&mut enc, &view);
        }

        // UI vectorielle vello_gpu, composée PAR-DESSUS (SrcOver)
        self.build_ui();
        self.renderer
            .render(
                &self.scene,
                &mut self.resources,
                &self.device,
                &self.queue,
                &mut enc,
                &RenderSize {
                    width: self.w as u16,
                    height: self.h as u16,
                },
                &view,
                Some(&self.depth),
                &TextureBindings::new(),
                TargetInit::SrcOver,
            )
            .map_err(|e| format!("vello: {e:?}"))?;

        // capture photo : même chaîne, rendue dans une texture lisible
        let mut cap: Option<(wgpu::Buffer, u32, u32, u32)> = None;
        if self.want_capture {
            self.want_capture = false;
            log(&format!(
                "[vello-cam] t={:.0} capture demandee {}x{}",
                now_ms(),
                self.w,
                self.h
            ));
            self.capture_busy.set(true);
            self.capture_frames = 0;
            let bpr = ((self.w * 4 + 255) / 256) * 256;
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("capture"),
                size: (bpr as u64) * (self.h as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            cap = Some((buf, bpr, self.w, self.h));
        }

        self.queue.submit([enc.finish()]);
        self.queue.present(frame);

        // La capture part dans SON PROPRE command buffer, et surtout : elle ne touche
        // PAS la surface. Son achevement ne depend donc pas de la composition du canvas
        // — mesure : ~41 s avec la surface dans la meme soumission, quelques ms sans.
        if let Some((buf, bpr, w, h)) = cap.as_ref() {
            let mut enc2 = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("capture"),
                });
            self.chain_into_cap(&mut enc2, *bpr);
            enc2.copy_texture_to_buffer(
                self.cap_tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(*bpr),
                        rows_per_image: Some(*h),
                    },
                },
                wgpu::Extent3d {
                    width: *w,
                    height: *h,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit([enc2.finish()]);
            log(&format!("[vello-cam] t={:.0} soumission capture separee", now_ms()));
        }

        if let Some((buf, bpr, w, h)) = cap {
            let bgra = is_bgra(self.format);
            let busy = self.capture_busy.clone();
            log(&format!("[vello-cam] t={:.0} spawn relecture", now_ms()));
            wasm_bindgen_futures::spawn_local(async move {
                match read_buffer(&buf, bpr, w, h, bgra).await {
                    Some(rgba) => {
                        log(&format!(
                            "[vello-cam] t={:.0} readback ok: {} octets",
                            now_ms(),
                            rgba.len()
                        ));
                        js(
                            "velloCamExportPhoto",
                            &[
                                js_sys::Uint8Array::from(&rgba[..]).into(),
                                JsValue::from_f64(w as f64),
                                JsValue::from_f64(h as f64),
                            ],
                        );
                    }
                    None => {
                        log("[vello-cam] readback ECHEC");
                        status("échec de la capture");
                    }
                }
                busy.set(false);
            });
        }

        self.frames += 1;
        if self.frames == 1 {
            log("[vello-cam] premiere frame rendue");
        }
        Ok(())
    }

    fn chain_into_cap(&mut self, enc: &mut wgpu::CommandEncoder, _bpr: u32) {
        let cam_view = match &self.cam {
            Some(c) => c.view.clone(),
            None => return,
        };
        if self.filters.spatial() {
            // réutilise `ping` : la première chaîne est déjà soumise dans l'encodeur, l'ordre est conservé
            self.draw(enc, &self.ping_view, &self.p_grade, &cam_view);
            self.draw(enc, &self.cap_view, &self.p_spatial, &self.ping_view);
        } else {
            self.draw(enc, &self.cap_view, &self.p_grade, &cam_view);
        }
    }

    // ------------------------------------------------------------ UI vello

    fn build_ui(&mut self) {
        let dpr = self.dpr;
        let (wf, hf) = (self.w as f32, self.h as f32);
        self.scene.reset();

        // voiles haut/bas (faux dégradé, 10 bandes)
        let bands = 10;
        for i in 0..bands {
            let a = (1.0 - i as f32 / bands as f32) * 0.30;
            let hgt = hf * 0.13 / bands as f32;
            self.scene
                .set_paint(Color::from_rgba8(4, 6, 10, (a * 255.0) as u8));
            self.scene
                .fill_rect(&Rect::new(0.0, (i as f32 * hgt) as f64, wf as f64, (hgt * 1.4) as f64));
            self.scene
                .set_paint(Color::from_rgba8(4, 6, 10, (a * 255.0) as u8));
            self.scene.fill_rect(&Rect::new(
                0.0,
                (hf - (i as f32 + 1.0) * hgt) as f64,
                wf as f64,
                (hgt * 1.4) as f64,
            ));
        }

        // pastilles de filtres
        for i in 0..NCHIP {
            let chip = self.chips[i];
            let s = self.sel[i];
            let grow = 1.0 + 0.11 * s;
            let cs = chip.s * grow;
            let cx = chip.x + chip.s * 0.5;
            let cy = chip.y + chip.s * 0.5;
            let r = Rect::new(
                (cx - cs * 0.5) as f64,
                (cy - cs * 0.5) as f64,
                (cx + cs * 0.5) as f64,
                (cy + cs * 0.5) as f64,
            );
            // halo
            if s > 0.01 {
                let hs = cs * 1.22;
                let hr = Rect::new(
                    (cx - hs * 0.5) as f64,
                    (cy - hs * 0.5) as f64,
                    (cx + hs * 0.5) as f64,
                    (cy + hs * 0.5) as f64,
                );
                self.scene
                    .set_paint(Color::from_rgba8(56, 224, 200, (70.0 * s) as u8));
                self.scene
                    .fill_path(&RoundedRect::from_rect(hr, (hs * 0.32) as f64).to_path(0.1));
            }
            // fond de pastille
            let bg = 0.20 + 0.30 * s;
            self.scene.set_paint(Color::from_rgba8(
                14,
                18,
                26,
                (bg * 255.0) as u8,
            ));
            self.scene
                .fill_path(&RoundedRect::from_rect(r, (cs * 0.30) as f64).to_path(0.1));
            // soulignement actif
            if s > 0.01 {
                let bw = cs * 0.52 * s;
                let bar = Rect::new(
                    (cx - bw * 0.5) as f64,
                    (cy + cs * 0.5 - 3.5 * dpr) as f64,
                    (cx + bw * 0.5) as f64,
                    (cy + cs * 0.5 - 0.5 * dpr) as f64,
                );
                self.scene.set_paint(Color::from_rgba8(76, 240, 214, 255));
                self.scene
                    .fill_path(&RoundedRect::from_rect(bar, (2.0 * dpr) as f64).to_path(0.1));
            }
            self.glyph(i, cx, cy, cs * 0.30, s);
        }

        // shutter
        let (sx, sy, sr) = self.shutter;
        let breath = 1.0 + 0.035 * (self.time * 2.0).sin();
        self.scene
            .set_paint(Color::from_rgba8(255, 255, 255, 46));
        self.scene
            .fill_path(&Circle::new((sx as f64, sy as f64), (sr * breath) as f64).to_path(0.1));
        if self.recording {
            let pulse = 0.55 + 0.45 * (self.time * 5.0).sin().abs();
            self.scene.set_paint(Color::from_rgba8(
                255,
                66,
                54,
                (60.0 + 150.0 * pulse) as u8,
            ));
        } else {
            self.scene.set_paint(Color::from_rgba8(242, 246, 250, 236));
        }
        self.scene.fill_path(
            &Circle::new((sx as f64, sy as f64), (sr * 0.78) as f64).to_path(0.1),
        );

        // pastille REC
        let (rx, ry, rs) = self.rec;
        self.scene.set_paint(Color::from_rgba8(12, 14, 20, 168));
        self.scene.fill_path(
            &RoundedRect::from_rect(
                Rect::new(rx as f64, ry as f64, (rx + rs) as f64, (ry + rs) as f64),
                (rs * 0.32) as f64,
            )
            .to_path(0.1),
        );
        if self.recording {
            let pulse = 0.6 + 0.4 * (self.time * 5.0).sin().abs();
            self.scene.set_paint(Color::from_rgba8(
                255,
                66,
                54,
                (140.0 + 115.0 * pulse) as u8,
            ));
            self.scene
                .fill_path(&Circle::new(((rx + rs * 0.5) as f64, (ry + rs * 0.5) as f64), (rs * 0.24) as f64).to_path(0.1));
        } else {
            self.scene.set_paint(Color::from_rgba8(255, 66, 54, 210));
            self.scene
                .fill_path(&Circle::new(((rx + rs * 0.5) as f64, (ry + rs * 0.5) as f64), (rs * 0.30) as f64).to_path(0.1));
            self.scene.set_paint(Color::from_rgba8(12, 14, 20, 255));
            self.scene
                .fill_path(&Circle::new(((rx + rs * 0.5) as f64, (ry + rs * 0.5) as f64), (rs * 0.18) as f64).to_path(0.1));
        }

        // flash de capture
        if self.flash > 0.001 {
            self.scene
                .set_paint(Color::from_rgba8(255, 255, 255, (self.flash * 190.0) as u8));
            self.scene.fill_rect(&Rect::new(0.0, 0.0, wf as f64, hf as f64));
        }
    }

    /// Petit pictogramme qui identifie le filtre sans texte.
    fn glyph(&mut self, i: usize, cx: f32, cy: f32, r: f32, sel: f32) {
        let c = (cx as f64, cy as f64);
        match i {
            0 => {
                // noir & blanc : disque blanc + moitié noire
                self.scene.set_paint(Color::from_rgba8(245, 245, 245, 255));
                self.scene.fill_path(&Circle::new(c, r as f64).to_path(0.1));
                self.scene.push_clip_rect(&Rect::new(
                    cx as f64,
                    (cy - r) as f64,
                    (cx + r) as f64,
                    (cy + r) as f64,
                ));
                self.scene.set_paint(Color::from_rgba8(12, 14, 20, 255));
                self.scene.fill_path(&Circle::new(c, r as f64).to_path(0.1));
                self.scene.pop_clip();
            }
            1 => {
                // sépia : disque ambré + petit cœur sombre
                self.scene.set_paint(Color::from_rgba8(198, 150, 86, 255));
                self.scene.fill_path(&Circle::new(c, r as f64).to_path(0.1));
                self.scene.set_paint(Color::from_rgba8(120, 78, 38, 255));
                self.scene
                    .fill_path(&Circle::new(c, (r * 0.42) as f64).to_path(0.1));
            }
            2 => {
                // flou : cercles concentriques
                for (k, a) in [(1.0f32, 40u8), (0.68, 90), (0.36, 220)] {
                    self.scene
                        .set_paint(Color::from_rgba8(226, 240, 248, a));
                    self.scene
                        .fill_path(&Circle::new(c, (r * k) as f64).to_path(0.1));
                }
            }
            3 => {
                // fisheye : lentille
                self.scene
                    .set_paint(Color::from_rgba8(226, 240, 248, 70));
                self.scene.fill_path(&Circle::new(c, r as f64).to_path(0.1));
                self.scene.set_paint(Color::from_rgba8(14, 18, 26, 210));
                self.scene
                    .fill_path(&Circle::new(c, (r * 0.62) as f64).to_path(0.1));
                self.scene
                    .set_paint(Color::from_rgba8(76, 240, 214, 235));
                self.scene
                    .fill_path(&Circle::new(c, (r * 0.26) as f64).to_path(0.1));
            }
            _ => {
                // vignette : cadre sombre + cœur clair
                self.scene.set_paint(Color::from_rgba8(6, 8, 12, 225));
                self.scene.fill_path(
                    &RoundedRect::from_rect(
                        Rect::new(
                            (cx - r) as f64,
                            (cy - r) as f64,
                            (cx + r) as f64,
                            (cy + r) as f64,
                        ),
                        (r * 0.45) as f64,
                    )
                    .to_path(0.1),
                );
                self.scene.set_paint(Color::from_rgba8(
                    226,
                    240,
                    248,
                    (150.0 + 105.0 * sel) as u8,
                ));
                self.scene
                    .fill_path(&Circle::new(c, (r * 0.34) as f64).to_path(0.1));
            }
        }
    }
}

// ---------------------------------------------------------------- relecture GPU

async fn read_buffer(
    buf: &wgpu::Buffer,
    bpr: u32,
    w: u32,
    h: u32,
    bgra: bool,
) -> Option<Vec<u8>> {
    log(&format!("[vello-cam] t={:.0} read_buffer: map_async...", now_ms()));
    let slice = buf.slice(..);
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        slice.map_async(wgpu::MapMode::Read, move |r| {
            log(&format!("[vello-cam] map_async callback: {r:?}"));
            let _ = resolve.call0(&JsValue::NULL);
        });
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    log(&format!("[vello-cam] t={:.0} map_async resolu", now_ms()));
    let data = slice.get_mapped_range().ok()?.to_vec();
    buf.unmap();

    let row = (w * 4) as usize;
    let mut out = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        let off = y * bpr as usize;
        out.extend_from_slice(&data[off..off + row]);
    }
    if bgra {
        for px in out.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
    }
    Some(out)
}

// ---------------------------------------------------------------- auto-test

/// Rend une couleur connue à travers la passe colorimétrique et relit le pixel.
/// Prouve que le shader calcule bien : on connaît la réponse attendue.
async fn self_test(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    cw: u32,
    ch: u32,
) {
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("test"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &sm,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &sm,
            entry_point: Some("fs_grade"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // source 64x64 : rouge pur, 256 octets par ligne (alignement)
    const TS: u32 = 64;
    let src = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test-src"),
        size: wgpu::Extent3d {
            width: TS,
            height: TS,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut data = Vec::with_capacity((TS * TS * 4) as usize);
    for _ in 0..(TS * TS) {
        data.extend_from_slice(&[255u8, 0, 0, 255]);
    }
    queue.write_texture(
        src.as_image_copy(),
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TS * 4),
            rows_per_image: Some(TS),
        },
        wgpu::Extent3d {
            width: TS,
            height: TS,
            depth_or_array_layers: 1,
        },
    );
    let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());

    let (dst_tex, dst_view) = make_target(
        &device,
        "test-dst",
        format,
        TS,
        TS,
        wgpu::TextureUsages::COPY_SRC,
    );

    let bgra = is_bgra(format);
    let cases: [(&str, Filters, [f32; 3]); 3] = [
        ("identite", Filters::neutral(), [255.0, 0.0, 0.0]),
        ("n&b", Filters { grayscale: 1.0, ..Filters::neutral() }, [54.0, 54.0, 54.0]),
        ("sepia", Filters { sepia: 1.0, ..Filters::neutral() }, [100.0, 89.0, 69.0]),
    ];

    for (name, f, expect) in cases {
        let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&f),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
            ],
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (TS * TS * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&pipe);
            rp.set_bind_group(0, &bg, &[]);
            rp.draw(0..3, 0..1);
        }
        let tex = &dst_tex;
        enc.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(TS * 4),
                    rows_per_image: Some(TS),
                },
            },
            wgpu::Extent3d {
                width: TS,
                height: TS,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);

        match read_buffer(&buf, TS * 4, TS, TS, bgra).await {
            Some(px) => {
                let got = [px[0] as f32, px[1] as f32, px[2] as f32];
                let ok = got
                    .iter()
                    .zip(expect.iter())
                    .all(|(g, e)| (g - e).abs() <= 3.0);
                log(&format!(
                    "[vello-cam] AUTO-TEST {name}: obtenu ({}, {}, {}) attendu ({}, {}, {}) => {}",
                    got[0] as u32,
                    got[1] as u32,
                    got[2] as u32,
                    expect[0] as u32,
                    expect[1] as u32,
                    expect[2] as u32,
                    if ok { "OK" } else { "ECHEC" }
                ));
            }
            None => log(&format!("[vello-cam] AUTO-TEST {name}: relecture impossible")),
        }
    }
    // ---- cas 4 : relecture PLEINE TAILLE, chronometree ----
    // Mesure le cout reel du chemin de capture (texture -> buffer -> map_async)
    // sans boucle de rendu. Sert de reference pour diagnostiquer une lenteur.
    {
        let now = || {
            web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now())
                .unwrap_or(0.0)
        };
        let (big_tex, big_view) = make_target(
            device,
            "probe-big",
            format,
            cw,
            ch,
            wgpu::TextureUsages::COPY_SRC,
        );
        let bpr = ((cw * 4 + 255) / 256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe-big"),
            size: (bpr as u64) * (ch as u64),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let t0 = now();
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let _rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &big_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.2,
                            g: 0.4,
                            b: 0.6,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        enc.copy_texture_to_buffer(
            big_tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(ch),
                },
            },
            wgpu::Extent3d {
                width: cw,
                height: ch,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([enc.finish()]);
        let t_sub = now();
        let slice = buf.slice(..);
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            slice.map_async(wgpu::MapMode::Read, move |_r| {
                let _ = resolve.call0(&JsValue::NULL);
            });
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
        let t_map = now();
        let ok = slice.get_mapped_range().is_ok();
        buf.unmap();
        log(&format!(
            "[vello-cam] PROBE relecture {cw}x{ch} ({} Mo) : submit {:.0} ms, map {:.0} ms, donnees={ok}",
            (bpr as u64 * ch as u64) / 1_000_000,
            t_sub - t0,
            t_map - t_sub
        ));
    }

    log("[vello-cam] AUTO-TEST termine");
}

// ---------------------------------------------------------------- démarrage

#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    log("[vello-cam] wasm initialise (v0.2 camera)");
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            log(&format!("[vello-cam] ERREUR: {e}"));
            status(&format!("erreur : {e}"));
        }
    });
}

async fn run() -> Result<(), String> {
    let window = web_sys::window().ok_or("pas de window")?;
    let document = window.document().ok_or("pas de document")?;
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("gpu")
        .ok_or("canvas #gpu absent")?
        .dyn_into()
        .map_err(|_| "#gpu n'est pas un canvas")?;
    let video: HtmlVideoElement = document
        .get_element_by_id("cam")
        .ok_or("<video id=cam> absent")?
        .dyn_into()
        .map_err(|_| "#cam n'est pas une video")?;

    // La copie GPU directe depuis <video> n'est pas supportee partout (Dawn/SwiftShader
    // la refuse). index.html sonde la capacite au demarrage et nous transmet le verdict.
    let direct = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("velloCamDirectVideo"))
        .ok()
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let mirror: Option<HtmlCanvasElement> = document
        .get_element_by_id("copy")
        .and_then(|e| e.dyn_into::<HtmlCanvasElement>().ok());
    log(&format!(
        "[vello-cam] source camera: {}",
        if direct { "video directe" } else { "miroir 2D" }
    ));

    let dpr = window.device_pixel_ratio() as f32;
    let w = (canvas.client_width() as f32 * dpr).max(1.0) as u32;
    let h = (canvas.client_height() as f32 * dpr).max(1.0) as u32;
    canvas.set_width(w);
    canvas.set_height(h);

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        display: None,
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::from_env_or_default(),
    });
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
        .map_err(|e| format!("create_surface: {e}"))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("request_adapter: {e}"))?;
    log(&format!("[vello-cam] adaptateur: {}", adapter.get_info().name));

    let caps = surface.get_capabilities(&adapter);
    let format = [
        wgpu::TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Rgba8Unorm,
    ]
    .into_iter()
    .find(|f| caps.formats.contains(f))
    .unwrap_or(caps.formats[0]);
    log(&format!("[vello-cam] format surface: {format:?}"));

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .map_err(|e| format!("request_device: {e}"))?;

    // ---- auto-test du shader AVANT la boucle (preuve quantitative) ----
    self_test(&device, &queue, format, w, h).await;

    surface.configure(
        &device,
        &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        },
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("linear"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("filtres"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("filtres"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("filtres"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let mk = |entry: &'static str, label: &'static str| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &sm,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &sm,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };
    let p_grade = mk("fs_grade", "grade");
    let p_spatial = mk("fs_spatial", "spatial");

    let filters = Filters::look();
    let uni = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("filtres"),
        contents: bytemuck::bytes_of(&filters),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let (ping_tex, ping_view) = make_target(
        &device,
        "ping",
        format,
        w,
        h,
        wgpu::TextureUsages::empty(),
    );
    let (cap_tex, cap_view) = make_target(
        &device,
        "capture",
        format,
        w,
        h,
        wgpu::TextureUsages::COPY_SRC,
    );

    let (renderer, resources) = Renderer::new_with(
        &device,
        &RenderTargetConfig {
            format,
            width: w.try_into().unwrap(),
            height: h.try_into().unwrap(),
        },
        RenderSettings::default(),
    );
    let depth = Renderer::create_depth_texture_view(
        &device,
        &RenderSize {
            width: w.try_into().unwrap(),
            height: h.try_into().unwrap(),
        },
    );
    log("[vello-cam] renderer vello_gpu pret");

    let app = Rc::new(RefCell::new(App {
        device,
        queue,
        surface,
        format,
        canvas: canvas.clone(),
        video,
        mirror,
        direct,
        dpr,
        sampler,
        uni,
        bgl,
        p_grade,
        p_spatial,
        cam: None,
        ping_tex,
        ping_view,
        cap_tex,
        cap_view,
        renderer,
        resources,
        depth,
        scene: Scene::new(w as u16, h as u16),
        w,
        h,
        src_w: 0,
        src_h: 0,
        filters,
        on: [false; NCHIP],
        sel: [0.0; NCHIP],
        vel: [0.0; NCHIP],
        chips: Vec::new(),
        shutter: (w as f32 * 0.5, h as f32 - 56.0, 30.0),
        rec: (w as f32 - 56.0, 40.0, 22.0),
        t0: 0.0,
        t_prev: 0.0,
        time: 0.0,
        frames: 0,
        fps: 0.0,
        fps_acc: 0.0,
        fps_n: 0,
        hud_acc: 0.0,
        recording: false,
        want_capture: false,
        capture_busy: Rc::new(Cell::new(false)),
        capture_frames: 0,
        flash: 0.0,
        first_frame: true,
    }));
    app.borrow_mut().layout();

    // ---- entrées ----
    {
        let a = app.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            if let Ok(mut ap) = a.try_borrow_mut() {
                let canvas = ap.canvas.clone();
                let dpr = ap.dpr;
                let rect = canvas.get_bounding_client_rect();
                let x = ((ev.client_x() as f64 - rect.left()) * dpr as f64) as f32;
                let y = ((ev.client_y() as f64 - rect.top()) * dpr as f64) as f32;
                ap.hit(x, y);
            }
        }) as Box<dyn FnMut(_)>);
        let _ = canvas.add_event_listener_with_callback("mousedown", cb.as_ref().unchecked_ref());
        cb.forget();

        let a = app.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            if let Ok(mut ap) = a.try_borrow_mut() {
                let k = ev.key();
                match k.as_str() {
                    "1" | "2" | "3" | "4" | "5" => {
                        let i = k.as_bytes()[0] as usize - b'1' as usize;
                        ap.toggle(i);
                    }
                    " " => {
                        ap.want_capture = true;
                        ap.flash = 1.0;
                        status("photo…");
                    }
                    "r" | "R" => {
                        ap.recording = !ap.recording;
                        js("velloCamRecord", &[JsValue::from_bool(ap.recording)]);
                        status(if ap.recording {
                            "enregistrement…"
                        } else {
                            "enregistrement arrêté"
                        });
                    }
                    _ => {}
                }
            }
        }) as Box<dyn FnMut(_)>);
        let _ = window.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref());
        cb.forget();
    }

    js("velloCamReady", &[]);
    status("en attente de la caméra…");

    // ---- boucle rAF ----
    let f: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let h2 = f.clone();
    let win = window.clone();
    *g.borrow_mut() = Some(Closure::wrap(Box::new(move |t: f64| {
        match app.borrow_mut().frame(t) {
            Ok(()) => {}
            Err(e) => log(&format!("[vello-cam] frame err: {e}")),
        }
        let _ = win
            .request_animation_frame(h2.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    }) as Box<dyn FnMut(f64)>));
    let _ = window.request_animation_frame(g.borrow().as_ref().unwrap().as_ref().unchecked_ref());
    std::mem::forget(f);
    Ok(())
}
