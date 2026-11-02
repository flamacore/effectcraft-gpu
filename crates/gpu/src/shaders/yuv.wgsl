// Export: the frame over the comp background as 8-bit RGB (Image::to_rgba8_over), then BT.709
// limited-range 4:2:0 (export's rgba_to_yuv420), packed into a storage buffer: `h` Y rows of
// `ys` bytes, then `ch` U rows and `ch` V rows of `cs` bytes (`ys` a multiple of 8, `cs` of 4).
// One invocation covers 4 chroma samples (8 × 2 luma pixels), so every store is a whole u32.

struct YuvParams {
    w: u32,
    h: u32,
    ys: u32,
    cs: u32,
    ch: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    bg: vec4<f32>,
}

@group(0) @binding(0) var<uniform> yp: YuvParams;
@group(0) @binding(1) var ysrc: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> yout: array<u32>;

// The 8-bit value (v · 255 + 0.5, truncated) back in 0..1, as the encoder reads it.
fn rgb8(x: u32, y: u32) -> vec3<f32> {
    let p = textureLoad(ysrc, vec2<i32>(i32(x), i32(y)), 0);
    let k = 1.0 - clamp(p.w, 0.0, 1.0);
    return floor(clamp(p.xyz + yp.bg.xyz * k, vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0 + 0.5) / 255.0;
}

// Rust's f32::round for the positive values here (WGSL's round() is half-to-even).
fn byte(v: f32) -> u32 {
    return u32(clamp(floor(v + 0.5), 1.0, 254.0));
}

@compute @workgroup_size(8, 8)
fn yuv420(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = (yp.w + 1u) / 2u;
    let cx0 = gid.x * 4u;
    let cy = gid.y;
    if (cx0 >= cw || cy >= yp.ch) {
        return;
    }
    // Y words: row 0 columns 0–3, 4–7, then row 1.
    var yw = array<u32, 4>(0u, 0u, 0u, 0u);
    var uw = 0u;
    var vw = 0u;
    for (var i = 0u; i < 4u; i++) {
        let cx = cx0 + i;
        if (cx >= cw) {
            break;
        }
        // Summed in the CPU's order: row by row, left to right.
        var a0 = 0.0;
        var a1 = 0.0;
        var n = 0.0;
        for (var dy = 0u; dy < 2u; dy++) {
            let y = cy * 2u + dy;
            if (y >= yp.h) {
                break;
            }
            for (var dx = 0u; dx < 2u; dx++) {
                let x = cx * 2u + dx;
                if (x >= yp.w) {
                    break;
                }
                let c = rgb8(x, y);
                let l = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
                let col = i * 2u + dx;
                yw[dy * 2u + col / 4u] |= byte(16.0 + 219.0 * l) << ((col % 4u) * 8u);
                a0 += (c.z - l) / 1.8556;
                a1 += (c.x - l) / 1.5748;
                n += 1.0;
            }
        }
        uw |= byte(128.0 + 224.0 * a0 / n) << (i * 8u);
        vw |= byte(128.0 + 224.0 * a1 / n) << (i * 8u);
    }
    for (var dy = 0u; dy < 2u; dy++) {
        let y = cy * 2u + dy;
        if (y < yp.h) {
            let o = y * yp.ys / 4u + gid.x * 2u;
            yout[o] = yw[dy * 2u];
            yout[o + 1u] = yw[dy * 2u + 1u];
        }
    }
    let u0 = yp.ys * yp.h / 4u;
    let plane = yp.cs * yp.ch / 4u;
    yout[u0 + cy * yp.cs / 4u + gid.x] = uw;
    yout[u0 + plane + cy * yp.cs / 4u + gid.x] = vw;
}
