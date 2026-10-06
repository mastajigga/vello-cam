//! UI in CSS pixels. Geometry is cached; only transforms and paints change per frame.
mod layout;
mod gesture;

use vello_common::peniko::{kurbo::{Affine, BezPath, Circle, Rect, RoundedRect, Shape}, Color, Gradient};
use vello_gpu::Scene;
use wasm_bindgen::JsCast;

pub const COUNT: usize = 10;
pub const IDS: [&str; COUNT] = ["mono", "sepia", "blur", "lens", "vignette", "prism", "photo", "record", "grid", "reset"];
const INK: Color = Color::from_rgba8(247, 248, 237, 255);
const ACCENT: Color = Color::from_rgba8(221, 246, 147, 255);
const DARK: Color = Color::from_rgba8(17, 24, 24, 235);

pub struct Ui {
    rects: [Rect; COUNT],
    labels: [web_sys::HtmlElement; COUNT],
    circle: BezPath,
    tile: BezPath,
    top: Gradient,
    bottom: Gradient,
    w: f32,
    h: f32,
    landscape: bool,
    pub gesture: gesture::Gesture,
    pub press: [f32; COUNT],
    pub focus: Option<usize>,
    grid_mix: f32,
    record_mix: f32,
    entrance: f32,
}

impl Ui {
    pub fn new() -> Self {
        let doc = web_sys::window().unwrap().document().unwrap();
        Self {
            rects: [Rect::ZERO; COUNT],
            labels: std::array::from_fn(|i| doc.get_element_by_id(IDS[i]).unwrap().unchecked_into()),
            circle: Circle::new((0.0, 0.0), 1.0).to_path(0.01),
            tile: RoundedRect::new(-1.0, -1.0, 1.0, 1.0, 0.6).to_path(0.01),
            top: Gradient::new_linear((0.0, 0.0), (0.0, 100.0)),
            bottom: Gradient::new_linear((0.0, 0.0), (0.0, 100.0)),
            w: 0.0, h: 0.0, landscape: false,
            gesture: gesture::Gesture::default(), press: [0.0; COUNT], focus: None,
            grid_mix: 0.0, record_mix: 0.0, entrance: 0.0,
        }
    }

    pub fn layout(&mut self, w: f32, h: f32) {
        self.w = w; self.h = h;
        self.gesture.clear();
        self.landscape = layout::landscape(w, h);
        self.rects = layout::controls(w, h).map(|r| Rect::new(
            r.x as f64, r.y as f64, (r.x + r.w) as f64, (r.y + r.h) as f64,
        ));
        for (i, rect) in self.rects.iter().enumerate() {
            let style = self.labels[i].style();
            for (key, value) in [("left", rect.x0), ("top", rect.y0), ("width", rect.width()), ("height", rect.height())] {
                let _ = style.set_property(key, &format!("{value}px"));
            }
        }
        self.top = Gradient::new_linear((0.0, 0.0), (0.0, 130.0))
            .with_stops([Color::from_rgba8(4, 12, 13, 190), Color::from_rgba8(4, 12, 13, 0)]);
        self.bottom = Gradient::new_linear((0.0, (h - if self.landscape { 180.0 } else { 280.0 }) as f64), (0.0, h as f64))
            .with_stops([Color::from_rgba8(4, 12, 13, 0), Color::from_rgba8(4, 12, 13, 230)]);
    }

    pub fn hit(&self, x: f32, y: f32) -> Option<usize> {
        self.rects.iter().position(|r| r.contains((x as f64, y as f64)))
    }

    pub fn sync(&self, on: &[bool; 6], recording: bool, grid: bool) {
        for i in (0..6).chain([7, 8]) {
            let active = if i < 6 { on[i] } else if i == 7 { recording } else { grid };
            let _ = self.labels[i].set_attribute("aria-pressed", if active { "true" } else { "false" });
        }
        self.labels[7].set_text_content(Some(if recording { "STOP" } else { "VIDÉO" }));
    }

    pub fn animate(&mut self, dt: f32, grid: bool, recording: bool) {
        let ease = 1.0 - (-dt * 19.0).exp();
        for i in 0..COUNT {
            let target = if self.gesture.inside && self.gesture.pointer.is_some_and(|(_, id)| id == i) { 1.0 } else { 0.0 };
            self.press[i] += (target - self.press[i]) * ease;
        }
        self.record_mix += ((if recording { 1.0 } else { 0.0 }) - self.record_mix) * ease;
        self.grid_mix += ((if grid { 1.0 } else { 0.0 }) - self.grid_mix) * (1.0 - (-dt * 10.0).exp());
        self.entrance += (1.0 - self.entrance) * (1.0 - (-dt * 7.0).exp());
    }

    fn shape(&self, scene: &mut Scene, d: f32, x: f32, y: f32, r: f32, square: bool, color: Color) {
        scene.set_transform(Affine::new([r as f64*d as f64, 0.0, 0.0, r as f64*d as f64, x as f64*d as f64, y as f64*d as f64]));
        scene.set_paint(color);
        scene.fill_path(if square { &self.tile } else { &self.circle });
        scene.set_transform(Affine::scale(d as f64));
    }

    pub fn draw(&self, scene: &mut Scene, d: f32, selected: &[f32; 6], recording: bool, time: f32, flash: f32) {
        scene.reset();
        scene.set_transform(Affine::scale(d as f64));
        let screen = Rect::new(0.0, 0.0, self.w as f64, self.h as f64);
        scene.set_paint(self.top.clone()); scene.fill_rect(&screen);
        scene.set_paint(self.bottom.clone()); scene.fill_rect(&screen);
        if self.grid_mix > 0.001 {
            scene.set_paint(Color::from_rgba8(255, 255, 255, (85.0*self.grid_mix) as u8));
            for third in [1.0, 2.0] {
                let x = self.w as f64*third/3.0; let y = self.h as f64*third/3.0;
                scene.fill_rect(&Rect::new(x, 0.0, x+0.75, self.h as f64));
                scene.fill_rect(&Rect::new(0.0, y, self.w as f64, y+0.75));
            }
        }
        for i in 0..COUNT {
            let rect = self.rects[i];
            let x = rect.center().x as f32;
            let y = if i < 6 { rect.y0 as f32 + 22.0 } else { rect.center().y as f32 - if i == 6 { 0.0 } else { 5.0 } };
            let y = y + (1.0 - self.entrance)*10.0;
            let p = self.press[i];
            let s = if i < 6 { selected[i].clamp(0.0, 1.0) } else if i == 8 { self.grid_mix } else { 0.0 };
            let radius = if i == 6 { 36.0 } else { 21.0 };
            let r = radius * (1.0 - p*0.13);
            if self.focus == Some(i) {
                self.shape(scene, d, x, y, radius + 4.0, false, ACCENT);
            }
            if i < 6 {
                self.shape(scene, d, x, y, r + 2.0*s, false, Color::from_rgba8(221, 246, 147, (240.0*s) as u8));
            }
            self.shape(scene, d, x, y, r - if i < 6 { 2.0*s } else { 0.0 }, false,
                if p > 0.02 { Color::from_rgba8(80, 99, 76, (225.0 + p*30.0) as u8) } else { DARK });
            if i < 6 {
                self.glyph(scene, d, i, x, y, 10.0*(1.0-p*0.15));
                if s > 0.01 {
                    scene.set_paint(ACCENT);
                    scene.fill_rect(&Rect::new((x-9.0*s) as f64, (rect.y1-2.0) as f64, (x+9.0*s) as f64, rect.y1));
                }
            } else if i == 6 {
                self.shape(scene, d, x, y, r, false, INK);
                self.shape(scene, d, x, y, r-2.0, false, DARK);
                self.shape(scene, d, x, y, r-6.0, false, INK);
                self.shape(scene, d, x, y, 4.0, false, Color::from_rgba8(22, 32, 29, 70));
            } else if i == 7 {
                let alpha = if recording { 185.0 + 70.0*(time*3.0).sin().abs() } else { 255.0 };
                // Crossfade the record dot into a stop square over ~160 ms.
                self.shape(scene, d, x, y, 7.0, false,
                    Color::from_rgba8(255, 110, 105, (alpha * (1.0-self.record_mix)) as u8));
                self.shape(scene, d, x, y, 8.0, true,
                    Color::from_rgba8(255, 110, 105, (alpha * self.record_mix) as u8));
            } else if i == 8 {
                scene.set_paint(if s > 0.1 { ACCENT } else { INK });
                for offset in [-4.0, 4.0] {
                    scene.fill_rect(&Rect::new((x+offset) as f64, (y-10.0) as f64, (x+offset+1.0) as f64, (y+10.0) as f64));
                    scene.fill_rect(&Rect::new((x-10.0) as f64, (y+offset) as f64, (x+10.0) as f64, (y+offset+1.0) as f64));
                }
            } else {
                self.shape(scene, d, x, y, 9.0, false, INK);
                self.shape(scene, d, x, y, 7.0, false, DARK);
                scene.set_paint(INK);
                scene.fill_rect(&Rect::new((x-10.0) as f64, (y-10.0) as f64, (x-6.0) as f64, (y-3.0) as f64));
            }
        }
        if flash > 0.001 {
            scene.set_paint(Color::from_rgba8(255, 255, 255, (flash * 190.0) as u8));
            scene.fill_rect(&screen);
        }
        scene.reset_transform();
    }

    fn glyph(&self, scene: &mut Scene, d: f32, id: usize, x: f32, y: f32, r: f32) {
        match id {
            0 => {
                self.shape(scene, d, x, y, r, false, INK);
                scene.push_clip_rect(&Rect::new(x as f64, (y-r) as f64, (x+r) as f64, (y+r) as f64));
                self.shape(scene, d, x, y, r, false, Color::from_rgba8(50, 61, 60, 255));
                scene.pop_clip();
            }
            1 => {
                self.shape(scene, d, x, y, r, false, Color::from_rgba8(228, 176, 119, 255));
                self.shape(scene, d, x, y, r*0.45, false, Color::from_rgba8(124, 78, 48, 255));
            }
            2 => for (scale, alpha) in [(1.0, 55), (0.7, 100), (0.35, 255)] {
                self.shape(scene, d, x, y, r*scale, false, Color::from_rgba8(218, 236, 241, alpha));
            },
            3 | 4 => {
                self.shape(scene, d, x, y, r, id == 4, Color::from_rgba8(219, 234, 224, 160));
                self.shape(scene, d, x, y, r*0.75, id == 4, DARK);
                self.shape(scene, d, x, y, r*0.4, false, INK);
            }
            _ => for (dx, color) in [(-4.0, Color::from_rgba8(255, 120, 137, 210)), (4.0, Color::from_rgba8(110, 206, 245, 210))] {
                self.shape(scene, d, x+dx, y, r*0.75, false, color);
            },
        }
    }
}
