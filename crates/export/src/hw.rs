//! H.264 and HEVC on the GPU's own encoder (`effectcraft-hwcodec`: NVENC, AMF or Quick Sync
//! through Media Foundation) for MP4 export, fed the 4:2:0 frames the GPU compositor converts.
//! Off with `EFFECTCRAFT_HW_ENCODE=0`; without a hardware encoder export keeps the software one.

use effectcraft_hwcodec::{Codec, Encoder, EncoderConfig};
use effectcraft_render::Yuv420;
use effectcraft_time::FrameRate;
use filmcraft_isobmff::{AvcConfig, HevcConfig, HevcNalArray, SampleEntry};

use crate::encode::{Packet, VideoEncoder};
use crate::{ExportError, Result};

/// Turns hardware encoding off (`0`).
pub(crate) const HW_ENCODE_ENV: &str = "EFFECTCRAFT_HW_ENCODE";

pub(crate) struct Hw {
    enc: Encoder,
    codec: Codec,
    w: u32,
    h: u32,
    den: i64,
    /// Packets out so far (their decode order).
    out: i64,
    scratch: Yuv420,
}

impl Hw {
    /// The hardware encoder for `codec`, or `None` (the reason logged) to use the software one.
    pub(crate) fn new(codec: Codec, w: u32, h: u32, rate: FrameRate, kbps: u32, keyint: u32) -> Option<Hw> {
        // Hardware encoders have minimum frame sizes (NVENC H.264 145 × 49) and tiny frames gain
        // nothing from them.
        if std::env::var(HW_ENCODE_ENV).is_ok_and(|v| v.trim() == "0") || w < 256 || h < 144 {
            return None;
        }
        let cfg = EncoderConfig { codec, width: w, height: h, fps_num: rate.num as u32, fps_den: rate.den as u32, bitrate_kbps: kbps, keyint };
        match Encoder::new(cfg) {
            Ok(enc) => Some(Hw { enc, codec, w, h, den: rate.den, out: 0, scratch: Yuv420::default() }),
            Err(why) => {
                log::info!(target: "effectcraft_export", "{codec:?}: software encoder ({why})");
                None
            }
        }
    }

    fn packets(&mut self, ps: Vec<effectcraft_hwcodec::Packet>) -> Vec<Packet> {
        ps.into_iter()
            .map(|p| {
                // pts − dts: zero without B-frames (the encoder is asked for none).
                let cto = ((p.frame - self.out) * self.den) as i32;
                self.out += 1;
                Packet { data: p.data, key: p.key, cto }
            })
            .collect()
    }
}

impl VideoEncoder for Hw {
    fn sample_entry(&self) -> SampleEntry {
        let sets = self.enc.parameter_sets();
        let of = |t: u8| sets.iter().filter(|s| effectcraft_hwcodec::nal_type(self.codec, s) == t).cloned().collect::<Vec<_>>();
        match self.codec {
            Codec::H264 => SampleEntry::avc(AvcConfig::new(of(7), of(8), 4), self.w as u16, self.h as u16),
            Codec::Hevc => SampleEntry::hevc(hevc_config(&of(32), &of(33), &of(34)), self.w as u16, self.h as u16),
        }
    }
    fn encode(&mut self, rgba: &[u8], index: u64) -> Result<Vec<Packet>> {
        let mut f = std::mem::take(&mut self.scratch);
        crate::encode::rgba_to_yuv420(rgba, self.w as usize, self.h as usize, &mut f.y, &mut f.u, &mut f.v);
        let r = self.encode_yuv420(&f, index);
        self.scratch = f;
        r
    }
    fn takes_yuv420(&self) -> bool {
        true
    }
    fn encode_yuv420(&mut self, yuv: &Yuv420, index: u64) -> Result<Vec<Packet>> {
        let ps = self.enc.encode(&yuv.y, &yuv.u, &yuv.v, index as i64).map_err(ExportError::Encode)?;
        Ok(self.packets(ps))
    }
    fn flush(&mut self) -> Result<Vec<Packet>> {
        let ps = self.enc.flush().map_err(ExportError::Encode)?;
        Ok(self.packets(ps))
    }
}

/// `hvcC` for 8-bit 4:2:0 from the parameter sets: the general profile, tier and level read
/// from the SPS's profile_tier_level (ISO/IEC 23008-2 §7.3.2.2, §7.3.3).
fn hevc_config(vps: &[Vec<u8>], sps: &[Vec<u8>], pps: &[Vec<u8>]) -> HevcConfig {
    // The SPS payload after its 2-byte header, emulation prevention bytes removed.
    let mut rbsp = vec![];
    let mut zeros = 0;
    for &b in sps.first().map_or(&[][..], |s| s.get(2..).unwrap_or_default()) {
        if zeros >= 2 && b == 3 {
            zeros = 0;
            continue;
        }
        zeros = if b == 0 { zeros + 1 } else { 0 };
        rbsp.push(b);
    }
    let at = |i: usize| rbsp.get(i).copied().unwrap_or(0);
    let flags = |from: usize, n: usize| (from..from + n).fold(0u64, |acc, i| (acc << 8) | u64::from(at(i)));
    let array = |nal_type: u8, nalus: &[Vec<u8>]| HevcNalArray { completeness: true, nal_type, nalus: nalus.to_vec() };
    HevcConfig {
        general_profile_space: at(1) >> 6,
        general_tier_flag: at(1) & 0x20 != 0,
        general_profile_idc: at(1) & 0x1f,
        general_profile_compatibility_flags: flags(2, 4) as u32,
        general_constraint_indicator_flags: flags(6, 6),
        general_level_idc: at(12),
        min_spatial_segmentation_idc: 0,
        parallelism_type: 0,
        chroma_format_idc: 1,
        bit_depth_luma: 8,
        bit_depth_chroma: 8,
        avg_frame_rate: 0,
        constant_frame_rate: 0,
        num_temporal_layers: ((at(0) >> 1) & 7) + 1,
        temporal_id_nested: at(0) & 1 != 0,
        length_size: 4,
        arrays: vec![array(32, vps), array(33, sps), array(34, pps)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hevc_config_reads_the_sps_profile_tier_level() {
        // Main profile (idc 1), main tier, level 4.0 (120), one sub-layer, nested; the
        // constraint flags' zeros carry an emulation prevention byte.
        let sps = vec![0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0xa0];
        let c = hevc_config(&[vec![0x40, 0x01]], &[sps], &[vec![0x44, 0x01]]);
        assert_eq!((c.general_profile_space, c.general_tier_flag, c.general_profile_idc), (0, false, 1));
        assert_eq!(c.general_profile_compatibility_flags, 0x6000_0000);
        assert_eq!(c.general_constraint_indicator_flags, 0x9000_0000_0000);
        assert_eq!(c.general_level_idc, 120);
        assert_eq!((c.num_temporal_layers, c.temporal_id_nested), (1, true));
        assert_eq!(c.arrays.iter().map(|a| a.nal_type).collect::<Vec<_>>(), vec![32, 33, 34]);
    }
}
