//! Export's 8-bit 4:2:0 conversion on the GPU (`yuv.wgsl`) against the CPU reference
//! (`Yuv420::from_rgba8` of `Image::to_rgba8_over`).

use effectcraft_raster::Image;
use effectcraft_render::Yuv420;

use super::gpu;
use crate::context::Enc;

#[test]
fn yuv420_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    // Odd sizes (half chroma columns and rows, strides off the u32 grid), partial alpha and
    // values just outside 0..1.
    for (w, h) in [(37u32, 23u32), (64, 2), (1, 1), (130, 9), (1920, 1080)] {
        let mut img = Image::new(w, h);
        for (i, p) in img.data.iter_mut().enumerate() {
            let f = |k: usize| ((i * 7919 + k * 104_729) % 1000) as f32 / 900.0 - 0.05;
            let a = f(3).clamp(0.0, 1.0);
            *p = [f(0) * a, f(1) * a, f(2) * a, a];
        }
        let bg = [0.2, 0.4, 0.9];
        let want = Yuv420::from_rgba8(&img.to_rgba8_over(bg), w, h);
        let up = g.context().upload_image(&img).expect("upload");
        let got = Enc::new(g.context()).yuv420(&up, bg).expect("yuv420 readback");
        assert_eq!((got.width, got.height), (w, h));
        for (name, a, b) in [("y", &want.y, &got.y), ("u", &want.u, &got.u), ("v", &want.v, &got.v)] {
            assert_eq!(a.len(), b.len(), "{w}x{h} {name}: plane size");
            let far = a.iter().zip(b).filter(|(x, y)| x.abs_diff(**y) > 1).count();
            let near = a.iter().zip(b).filter(|(x, y)| x != y).count();
            assert_eq!(far, 0, "{w}x{h} {name}: samples more than one level from the CPU");
            // f32 division and fused multiply-adds may tip a sample sitting on a rounding
            // boundary by one level.
            assert!(near <= a.len() / 200, "{w}x{h} {name}: {near} of {} samples one level off", a.len());
        }
    }
}
