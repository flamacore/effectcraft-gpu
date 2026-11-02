//! Hardware video encoding and decoding through the platform's own media framework: Media
//! Foundation on Windows, which drives the GPU's encoder and decoder (NVENC / NVDEC, AMF / VCN,
//! Quick Sync). Elsewhere, and where no hardware codec takes the stream, [`Encoder::new`] and
//! [`Decoder::open`] fail and EffectCraft keeps its pure-Rust codecs.
//!
//! The one crate allowed `unsafe` (the workspace forbids it): platform codec APIs are COM / FFI.
//! Everything unsafe stays in `mf.rs` and `mf_decode.rs`, behind this safe API.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod ahead;
#[cfg(windows)]
mod mf;
#[cfg(windows)]
mod mf_decode;

/// Set by [`turn_off`].
static OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turn the hardware codecs off for the rest of the process: from now on [`Decoder::open`] and
/// [`Encoder::new`] fail and EffectCraft keeps its pure-Rust codecs (the desktop app's CPU build).
pub fn turn_off() {
    OFF.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Why [`Decoder::open`] and [`Encoder::new`] fail after [`turn_off`].
fn turned_off() -> Result<(), String> {
    if OFF.load(std::sync::atomic::Ordering::Relaxed) { Err("hardware codecs are off in this build".into()) } else { Ok(()) }
}

/// Y'CbCr matrix of decoded frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    #[default]
    Bt709,
    Bt2020,
}

/// Transfer characteristic of decoded frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Transfer {
    #[default]
    Bt709,
    Srgb,
    Linear,
    Pq,
    Hlg,
}

/// Colour primaries of decoded frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Primaries {
    #[default]
    Bt709,
    Bt601_625,
    Bt601_525,
    Bt2020,
    P3D65,
}

/// What a decoded frame's samples mean (the stream's signalling; BT.709 limited range where it
/// says nothing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Colour {
    pub matrix: Matrix,
    pub transfer: Transfer,
    pub primaries: Primaries,
    pub full_range: bool,
}

/// A decoded 8-bit 4:2:0 picture: tight planes, `y` of width × height, `u` and `v` of
/// ⌈width / 2⌉ × ⌈height / 2⌉.
#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub y: std::sync::Arc<Vec<u8>>,
    pub u: std::sync::Arc<Vec<u8>>,
    pub v: std::sync::Arc<Vec<u8>>,
    pub colour: Colour,
    /// Presentation time and duration in 100 ns units.
    pub time: i64,
    pub duration: i64,
}

/// A hardware decoding session for one movie file's first video stream.
pub struct Decoder {
    #[cfg(windows)]
    inner: ahead::Ahead,
}

impl Decoder {
    /// Open `path` (frames of `width` × `height`, which picks the GPU's decoder or the
    /// platform's CPU one, whichever is faster for that size), or say why it can't be.
    pub fn open(path: &std::path::Path, width: u32, height: u32) -> Result<Decoder, String> {
        turned_off()?;
        #[cfg(windows)]
        {
            mf_decode::open(path, u64::from(width) * u64::from(height)).map(|inner| Decoder { inner })
        }
        #[cfg(not(windows))]
        {
            let _ = (path, width, height);
            Err("no hardware decoder on this platform".into())
        }
    }

    /// Samples read and seeks made so far (diagnostics).
    pub fn stats(&self) -> (u64, u64, i64) {
        #[cfg(windows)]
        {
            self.inner.stats()
        }
        #[cfg(not(windows))]
        {
            (0, 0, 0)
        }
    }

    /// The picture shown at `time` (100 ns units from the start of the movie): the last frame
    /// starting at or before it (the last frame of the movie after its end).
    /// Callers on several threads may ask at once.
    pub fn frame_at(&self, time: i64) -> Result<std::sync::Arc<Frame>, String> {
        #[cfg(windows)]
        {
            self.inner.frame_at(time)
        }
        #[cfg(not(windows))]
        {
            let _ = time;
            Err("no hardware decoder".into())
        }
    }
}

/// The bitstream to encode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

#[derive(Clone, Copy, Debug)]
pub struct EncoderConfig {
    pub codec: Codec,
    /// Even sizes only (4:2:0).
    pub width: u32,
    pub height: u32,
    pub fps_num: u32,
    pub fps_den: u32,
    pub bitrate_kbps: u32,
    /// Frames from one keyframe to the next.
    pub keyint: u32,
}

/// One encoded picture: its NAL units with 4-byte big-endian lengths (ISO/IEC 14496-15), the
/// parameter sets and access unit delimiters taken out (they go in the sample entry,
/// [`Encoder::parameter_sets`]). Pictures come out in presentation order (no B-frames).
#[derive(Clone, Debug)]
pub struct Packet {
    pub data: Vec<u8>,
    pub key: bool,
    /// The frame index it shows.
    pub frame: i64,
}

/// A hardware encoder session.
pub struct Encoder {
    #[cfg(windows)]
    inner: mf::Encoder,
}

impl Encoder {
    /// A hardware encoder for `cfg`, or why there is none.
    pub fn new(cfg: EncoderConfig) -> Result<Encoder, String> {
        turned_off()?;
        if cfg.width == 0 || cfg.height == 0 || !cfg.width.is_multiple_of(2) || !cfg.height.is_multiple_of(2) {
            return Err(format!("{}×{}: hardware encoders take even frame sizes", cfg.width, cfg.height));
        }
        #[cfg(windows)]
        {
            mf::Encoder::new(cfg).map(|inner| Encoder { inner })
        }
        #[cfg(not(windows))]
        {
            Err("no hardware encoder on this platform".into())
        }
    }

    /// The encoder's name ("NVIDIA H.264 Encoder MFT").
    pub fn name(&self) -> &str {
        #[cfg(windows)]
        {
            &self.inner.name
        }
        #[cfg(not(windows))]
        {
            ""
        }
    }

    /// Encode an 8-bit 4:2:0 frame (tight planes: `y` of width × height, `u` and `v` of half
    /// that each way), shown as frame `frame`. Returns the packets finished so far.
    pub fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8], frame: i64) -> Result<Vec<Packet>, String> {
        #[cfg(windows)]
        {
            self.inner.encode(y, u, v, frame)
        }
        #[cfg(not(windows))]
        {
            let _ = (y, u, v, frame);
            Err("no hardware encoder".into())
        }
    }

    /// Finish the stream: the remaining packets.
    pub fn flush(&mut self) -> Result<Vec<Packet>, String> {
        #[cfg(windows)]
        {
            self.inner.flush()
        }
        #[cfg(not(windows))]
        {
            Ok(vec![])
        }
    }

    /// The parameter sets seen in the stream (NAL units without start codes): H.264 SPS then
    /// PPS, HEVC VPS, SPS, PPS. Complete once the first packet is out.
    pub fn parameter_sets(&self) -> Vec<Vec<u8>> {
        #[cfg(windows)]
        {
            self.inner.parameter_sets()
        }
        #[cfg(not(windows))]
        {
            vec![]
        }
    }
}

/// The NAL units of an Annex B byte stream (start codes removed, trailing zero bytes trimmed).
pub fn annex_b_units(b: &[u8]) -> Vec<&[u8]> {
    let mut starts = vec![];
    let mut i = 0;
    while i + 3 <= b.len() {
        if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = starts.get(k + 1).map_or(b.len(), |&n| n - 3);
        while e > s && b[e - 1] == 0 {
            e -= 1;
        }
        if e > s {
            out.push(&b[s..e]);
        }
    }
    out
}

/// The NAL unit type: H.264's 5 bits, HEVC's 6.
pub fn nal_type(codec: Codec, nal: &[u8]) -> u8 {
    let h = nal.first().copied().unwrap_or(0);
    match codec {
        Codec::H264 => h & 0x1f,
        Codec::Hevc => (h >> 1) & 0x3f,
    }
}

/// Parameter set (or delimiter) units that go in the sample entry, not the samples.
#[cfg_attr(not(windows), allow(dead_code))]
fn is_header(codec: Codec, t: u8) -> bool {
    match codec {
        Codec::H264 => matches!(t, 7..=9),
        Codec::Hevc => matches!(t, 32..=35),
    }
}

/// An access unit as a length-prefixed sample, its parameter sets collected into `sets`
/// (replacing earlier ones of the same type).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn packetize(codec: Codec, annex_b: &[u8], sets: &mut Vec<Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::with_capacity(annex_b.len());
    for nal in annex_b_units(annex_b) {
        let t = nal_type(codec, nal);
        if is_header(codec, t) {
            if !matches!((codec, t), (Codec::H264, 9) | (Codec::Hevc, 35)) {
                sets.retain(|s| nal_type(codec, s) != t);
                sets.push(nal.to_vec());
            }
            continue;
        }
        out.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        out.extend_from_slice(nal);
    }
    // Parameter sets in their order: H.264 SPS (7), PPS (8); HEVC VPS (32), SPS (33), PPS (34).
    sets.sort_by_key(|s| nal_type(codec, s));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annex_b_splits_on_three_and_four_byte_start_codes() {
        let s = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 9, 9, 0, 0];
        let units = annex_b_units(&s);
        assert_eq!(units, vec![&[0x67, 1, 2][..], &[0x68, 3][..], &[0x65, 9, 9][..]]);
        assert!(annex_b_units(&[]).is_empty());
        assert!(annex_b_units(&[0, 0, 1]).is_empty());
    }

    #[test]
    fn packetize_moves_parameter_sets_out_and_prefixes_lengths() {
        let mut sets = vec![];
        let s = [0, 0, 0, 1, 0x09, 0xf0, 0, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x65, 9, 9];
        let p = packetize(Codec::H264, &s, &mut sets);
        assert_eq!(p, vec![0, 0, 0, 3, 0x65, 9, 9]);
        assert_eq!(sets, vec![vec![0x67, 1, 2], vec![0x68, 3]]);
        // A later SPS replaces the first.
        packetize(Codec::H264, &[0, 0, 1, 0x67, 7], &mut sets);
        assert_eq!(sets, vec![vec![0x67, 7], vec![0x68, 3]]);
    }

    /// The machine's hardware encoders, when it has them: every frame comes out, the first as a
    /// keyframe, with the parameter sets the sample entry needs.
    #[test]
    fn hardware_encoders_encode_every_frame() {
        for codec in [Codec::H264, Codec::Hevc] {
            let (w, h) = (1920usize, 1080usize);
            let cfg = EncoderConfig { codec, width: w as u32, height: h as u32, fps_num: 30, fps_den: 1, bitrate_kbps: 8000, keyint: 30 };
            let mut e = match Encoder::new(cfg) {
                Ok(e) => e,
                Err(why) => {
                    eprintln!("{codec:?}: no hardware encoder here ({why}); skipped");
                    continue;
                }
            };
            let mut packets = vec![];
            for f in 0..60i64 {
                let y: Vec<u8> = (0..w * h).map(|i| ((i % w) as i64 + f * 4) as u8).collect();
                let c = vec![128u8; (w / 2) * (h / 2)];
                packets.extend(e.encode(&y, &c, &c, f).expect("encode"));
            }
            packets.extend(e.flush().expect("flush"));
            assert_eq!(packets.len(), 60, "{codec:?} on {}: one packet a frame", e.name());
            assert!(packets[0].key, "{codec:?}: starts on a keyframe");
            assert_eq!(packets.iter().map(|p| p.frame).collect::<Vec<_>>(), (0..60).collect::<Vec<_>>(), "{codec:?}: in order");
            let sets = e.parameter_sets();
            let types: Vec<u8> = sets.iter().map(|s| nal_type(codec, s)).collect();
            assert_eq!(types, if codec == Codec::H264 { vec![7, 8] } else { vec![32, 33, 34] }, "{codec:?}: parameter sets");
        }
    }

    /// Decoder throughput on a movie named by `EFFECTCRAFT_DECODE_BENCH` (ignored by default):
    /// `cargo test --release -p effectcraft-hwcodec -- --ignored --nocapture decode_bench`.
    #[test]
    #[ignore]
    fn decode_bench() {
        let Ok(path) = std::env::var("EFFECTCRAFT_DECODE_BENCH") else { return };
        let d = Decoder::open(std::path::Path::new(&path), 0, 0).expect("open");
        let first = d.frame_at(0).expect("frame 0");
        let dur = first.duration.max(1);
        let t0 = std::time::Instant::now();
        let n = 90i64;
        // Times as the media pool asks: whole frames of the footage's rate, truncated to 100 ns.
        let mut times = vec![];
        for k in 1..=n {
            let t = k * 10_000_000 / 30;
            let f = d.frame_at(t).expect("frame");
            times.push((t, f.time));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        eprintln!("{}x{} (frame {dur} hns): {ms:.2} ms a frame sequentially ({:.0} fps), reads/seeks {:?}", first.width, first.height, 1000.0 / ms, d.stats());
        eprintln!("asked → got: {:?}", &times[..6]);
    }

    #[test]
    fn odd_sizes_are_refused() {
        let cfg = EncoderConfig { codec: Codec::H264, width: 1919, height: 1080, fps_num: 30, fps_den: 1, bitrate_kbps: 8000, keyint: 60 };
        assert!(Encoder::new(cfg).is_err());
    }
}
