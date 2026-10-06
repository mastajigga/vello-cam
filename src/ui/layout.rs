//! Pure CSS-pixel layout, testable without a browser or GPU.
#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

pub fn landscape(w: f32, h: f32) -> bool {
    w >= 520.0 && w > h && h < 540.0
}

pub fn controls(w: f32, h: f32) -> [Bounds; 10] {
    let wide = landscape(w, h);
    let columns = if wide || w < 318.0 { 3 } else { 6 };
    let size = ((w - 24.0 - (columns - 1) as f32 * 4.0) / columns as f32)
        .clamp(44.0, 56.0);
    let total = columns as f32 * (size + 4.0) - 4.0;
    let left = if wide { 16.0 } else { (w - total) / 2.0 };
    let rows = 6 / columns;
    let top = h - if wide { 20.0 } else { 116.0 } - rows as f32 * 68.0;
    let mut bounds = [Bounds { x: 0.0, y: 0.0, w: 0.0, h: 0.0 }; 10];
    for (i, rect) in bounds.iter_mut().take(6).enumerate() {
        *rect = Bounds {
            x: left + (i % columns) as f32 * (size + 4.0),
            y: top + (i / columns) as f32 * 68.0,
            w: size,
            h: 64.0,
        };
    }
    let (cx, cy) = if wide { (w - 122.0, h / 2.0 + 18.0) } else { (w / 2.0, h - 61.0) };
    bounds[6] = Bounds { x: cx - 38.0, y: cy - 38.0, w: 76.0, h: 76.0 };
    let gap = if wide { 82.0 } else { (w * 0.27).min(118.0) };
    for (i, offset) in [(7, 1.0), (8, -1.0)] {
        bounds[i] = Bounds { x: cx + offset * gap - 26.0, y: cy - 26.0, w: 52.0, h: 52.0 };
    }
    bounds[9] = Bounds { x: w - 60.0, y: 12.0, w: 48.0, h: 48.0 };
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_targets_fit_without_overlap() {
        // Dimensions of the safe canvas, including narrow portrait and landscape
        // with a home indicator already subtracted by CSS.
        for (w, h) in [(280.0, 560.0), (320.0, 568.0), (360.0, 640.0),
            (390.0, 763.0), (568.0, 299.0), (734.0, 354.0),
            (1024.0, 768.0), (1920.0, 1080.0)] {
            let rects = controls(w, h);
            for (i, r) in rects.iter().enumerate() {
                assert!(r.w >= 44.0 && r.h >= 44.0, "target {}: {}x{}", i, w, h);
                assert!(r.x >= 0.0 && r.y >= 0.0 && r.x+r.w <= w && r.y+r.h <= h,
                    "outside safe canvas: {i}: {w}x{h}: {r:?}");
                for other in rects.iter().skip(i+1) {
                    assert!(r.x+r.w <= other.x || other.x+other.w <= r.x ||
                        r.y+r.h <= other.y || other.y+other.h <= r.y,
                        "overlapping targets: {w}x{h}: {r:?}, {other:?}");
                }
            }
        }
    }

    #[test]
    fn portrait_and_landscape_reflow() {
        let portrait = controls(390.0, 844.0);
        assert_eq!(portrait[0].y, portrait[5].y);
        let wide = controls(734.0, 354.0);
        assert!(wide[3].y > wide[0].y);
        assert!(wide[6].x > wide[5].x + wide[5].w);
        let narrow = controls(280.0, 560.0);
        assert!(narrow[3].y > narrow[0].y);
    }
}
