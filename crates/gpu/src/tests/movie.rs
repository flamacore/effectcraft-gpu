//! Movie footage from the decoder's 4:2:0 planes (`walk::draw_movie`, `yuv_rgba`): the GPU
//! builds the frame from the planes and must match the CPU, which renders the frame the media
//! layer would have converted from the same planes.

use std::sync::Arc;

use effectcraft_color::Label;
use effectcraft_keyframe::Value;
use effectcraft_project::build;
use effectcraft_project::{AlphaMode, BitDepth, Footage, FootageKind, ItemId, ItemKind, LayerSource};
use effectcraft_raster::Image;
use effectcraft_render::{Backend, FootageSource, RenderOpts, Renderer, YuvFrame};
use effectcraft_time::{FrameRate, Tick};

use super::{Scene, check, diff, gpu, set, tolerance};

/// BT.709 limited range: `[y_off, y_scale, c_off, c_scale, cr_r, cr_g, cb_g, cb_b]`.
fn bt709() -> [f32; 8] {
    let (kr, kb) = (0.2126f32, 0.0722f32);
    let kg = 1.0 - kr - kb;
    let (cr_r, cb_b) = (2.0 * (1.0 - kr), 2.0 * (1.0 - kb));
    [16.0, 1.0 / 219.0, 128.0, 1.0 / 224.0, cr_r, cr_r * kr / kg, cb_b * kb / kg, cb_b]
}

/// Planes with gradients, noise and the full 0–255 range of every sample.
fn planes(w: u32, h: u32) -> YuvFrame {
    let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
    let n = |x: usize, y: usize, s: usize| ((x * 7919 + y * 104_729 + s * 31) % 256) as u8;
    let y = (0..(w * h) as usize).map(|i| n(i % w as usize, i / w as usize, 1)).collect();
    let u = (0..cw * ch).map(|i| n(i % cw, i / cw, 2)).collect();
    let v = (0..cw * ch).map(|i| n(i % cw, i / cw, 3)).collect();
    YuvFrame { width: w, height: h, y: Arc::new(y), u: Arc::new(u), v: Arc::new(v), k: bt709() }
}

/// `effectcraft-media`'s `convert_yuv` for opaque 8-bit 4:2:0, step for step.
fn reference(f: &YuvFrame) -> Image {
    let (w, h) = (f.width as usize, f.height as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let centred = |i: usize, n: usize| {
        let c = (i >> 1).min(n - 1);
        (c, if i & 1 == 0 { c.saturating_sub(1) } else { (c + 1).min(n - 1) })
    };
    let k = f.k;
    let mut img = Image::new(f.width, f.height);
    for y in 0..h {
        let (r0, r1) = centred(y, ch);
        let row = |p: &[u8]| -> Vec<f32> { (0..cw).map(|x| (p[r0 * cw + x] as f32 * 0.75 + p[r1 * cw + x] as f32 * 0.25 - k[2]) * k[3]).collect() };
        let (uc, vc) = (row(&f.u), row(&f.v));
        for x in 0..w {
            let (c, o) = centred(x, cw);
            let (u, v) = (uc[c] * 0.75 + uc[o] * 0.25, vc[c] * 0.75 + vc[o] * 0.25);
            let yy = (f.y[y * w + x] as f32 - k[0]) * k[1];
            img.set(x as u32, y as u32, [(yy + k[4] * v).clamp(0.0, 1.0), (yy - k[5] * v - k[6] * u).clamp(0.0, 1.0), (yy + k[7] * u).clamp(0.0, 1.0), 1.0]);
        }
    }
    img
}

/// `effectcraft-media`'s `half_420` (a reduced render's frame at half size), step for step.
fn reference_half(f: &YuvFrame) -> Image {
    let (w, h) = (f.width as usize, f.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let k = f.k;
    let mut img = Image::new(cw as u32, ch as u32);
    for cy in 0..ch {
        let rows = [cy.saturating_sub(1), cy, (cy + 1).min(ch - 1)];
        let col = |p: &[u8]| -> Vec<f32> {
            let d: Vec<f32> = (0..cw)
                .map(|x| ((p[rows[0] * cw + x] as f32 + p[rows[2] * cw + x] as f32) * 0.125 + p[rows[1] * cw + x] as f32 * 0.75 - k[2]) * k[3])
                .collect();
            (0..cw).map(|x| (d[x.saturating_sub(1)] + d[(x + 1).min(cw - 1)]) * 0.125 + d[x] * 0.75).collect()
        };
        let (u, v) = (col(&f.u), col(&f.v));
        for cx in 0..cw {
            let y4 = f.y[2 * cy * w + 2 * cx] as f32
                + f.y[2 * cy * w + 2 * cx + 1] as f32
                + f.y[(2 * cy + 1) * w + 2 * cx] as f32
                + f.y[(2 * cy + 1) * w + 2 * cx + 1] as f32;
            let yy = (y4 * 0.25 - k[0]) * k[1];
            let (u, v) = (u[cx], v[cx]);
            img.set(cx as u32, cy as u32, [(yy + k[4] * v).clamp(0.0, 1.0), (yy - k[5] * v - k[6] * u).clamp(0.0, 1.0), (yy + k[7] * u).clamp(0.0, 1.0), 1.0]);
        }
    }
    img
}

/// A movie whose frames are the same planes at every time; reduced frames as the media layer
/// makes them (`frame_to_image_scaled`: half size, then resampled the rest of the way).
struct Movie(Arc<YuvFrame>);

impl FootageSource for Movie {
    fn frame(&self, _item: ItemId, _f: &Footage, _t: Tick) -> Option<Arc<Image>> {
        Some(Arc::new(reference(&self.0)))
    }
    fn frame_at_size(&self, item: ItemId, f: &Footage, t: Tick, w: u32, h: u32) -> Option<Arc<Image>> {
        let p = &self.0;
        if !p.width.is_multiple_of(2) || !p.height.is_multiple_of(2) || 2 * w > p.width || 2 * h > p.height {
            return self.frame(item, f, t);
        }
        let half = reference_half(p);
        Some(Arc::new(if (half.width, half.height) == (w, h) { half } else { effectcraft_raster::resample(&half, w, h) }))
    }
    fn frame_yuv420(&self, _item: ItemId, _f: &Footage, _t: Tick) -> Option<Arc<YuvFrame>> {
        Some(self.0.clone())
    }
}

fn movie_scene(depth: BitDepth, w: u32, h: u32, effects: bool) -> Scene {
    let mut s = Scene::new(depth);
    let f = Footage {
        path: "movie.mp4".into(),
        kind: FootageKind::Video,
        width: w,
        height: h,
        pixel_aspect: 1.0,
        frame_rate: FrameRate::FPS_30,
        duration: Tick::from_seconds_f64(2.0),
        has_video: true,
        alpha: AlphaMode::Ignore,
        loop_count: 1,
        ..Default::default()
    };
    let fid = s.p.add_item("Movie", Label::Lavender, None, ItemKind::Footage(f));
    let mut l = build::layer(&mut s.p, &s.comp, "Movie", LayerSource::Footage { item: fid }, (w, h), None);
    set(&mut l, "transform/rotation", Value::Scalar(7.0));
    if effects {
        s.effect(&mut l, "ec.color.brightnesscontrast", &[("brightness", Value::Scalar(20.0)), ("contrast", Value::Scalar(15.0))]);
        s.effect(&mut l, "ec.blur.gaussian", &[("blurriness", Value::Scalar(12.0))]);
    }
    s.push(l);
    s
}

#[test]
fn movie_frames_from_planes_match_the_cpu() {
    let Some(g) = gpu() else { return };
    // Full resolution, Half and Quarter (made at half size: exactly, and resampled further) and
    // 0.6 (a full frame resampled).
    for (depth, (w, h), effects, scale) in [
        (BitDepth::Bpc8, (70, 44), false, 1.0),
        (BitDepth::Bpc8, (70, 44), true, 1.0),
        (BitDepth::Bpc32, (37, 23), true, 1.0),
        (BitDepth::Bpc32, (64, 2), false, 1.0),
        (BitDepth::Bpc8, (70, 44), true, 0.5),
        (BitDepth::Bpc32, (72, 48), false, 0.25),
        (BitDepth::Bpc32, (70, 44), false, 0.6),
    ] {
        let s = movie_scene(depth, w, h, effects);
        let src = Movie(Arc::new(planes(w, h)));
        let t = Tick::from_seconds_f64(0.5);
        let opts = RenderOpts { motion_blur: true, scale, ..Default::default() };
        let cpu = Renderer::new(&s.p, &src, RenderOpts { backend: Backend::Cpu, ..opts }).comp_frame_cpu(s.cid, t);
        let mut r = Renderer::new(&s.p, &src, RenderOpts { backend: Backend::Gpu, ..opts });
        r.accel = Some(g);
        let before = g.context().transfer_stats();
        let img = g.render(&r, s.cid, t).expect("the GPU renders the frame");
        let up = g.context().transfer_stats().upload_bytes - before.upload_bytes;
        let label = format!("{depth:?} {w}x{h} effects {effects} scale {scale}");
        // The planes went up, not a float frame (16 bytes a pixel).
        assert!(up < u64::from(w * h) * 4, "{label}: {up} bytes uploaded for a {w}x{h} movie frame");
        let mut d = diff(&cpu, &img, tolerance(depth));
        d.quantized = depth != BitDepth::Bpc32;
        check(&label, Some(d), 0.0);
    }
}
