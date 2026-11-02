//! Media Foundation Source Reader decoding on a D3D11 video device, so the GPU's decoder
//! (NVDEC, VCN, Quick Sync) does the work; frames come back as NV12 and are split into tight
//! 4:2:0 planes. Without a D3D11 device the reader still decodes (Media Foundation's own
//! decoders, on the CPU). The reader is an [`ahead::Source`](crate::ahead::Source), which
//! decodes it ahead of the requests on a thread of its own.

use std::mem::ManuallyDrop;
use std::sync::Arc;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice, ID3D11Device, ID3D11Multithread,
};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::StructuredStorage::{PROPVARIANT, PROPVARIANT_0, PROPVARIANT_0_0, PROPVARIANT_0_0_0};
use windows::Win32::System::Variant::VT_I8;
use windows::core::{GUID, HSTRING, Interface};

use crate::ahead::{Ahead, Source};
use crate::mf::{err, startup};
use crate::{Colour, Frame, Matrix, Primaries, Transfer};

/// Frames larger than this (pixels) decode on the GPU's decoder, smaller ones on Media
/// Foundation's own (CPU, multithreaded), which writes system memory directly and is faster
/// there than the GPU's decoder and its copy back. Measured on an RTX 4080 SUPER and an 8-core
/// i9-11900, H.264: 1080p 3.9 ms a frame on the CPU vs 8 ms on the GPU; 2160p 15.6 vs 11.7 ms.
// ponytail: a fixed size; a per-machine measurement (or a zero-copy GPU path) when a slower CPU shows it.
const GPU_DECODE_ABOVE: u64 = 1920 * 1088;

/// Open `path` (frames of `pixels`, which picks the decoder: [`GPU_DECODE_ABOVE`]) decoding
/// ahead on a thread of its own.
pub(crate) fn open(path: &std::path::Path, pixels: u64) -> Result<Ahead, String> {
    // `EFFECTCRAFT_MF_D3D=0` / `1` forces Media Foundation's CPU / the GPU's decoder.
    let gpu = match std::env::var("EFFECTCRAFT_MF_D3D").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => pixels > GPU_DECODE_ABOVE,
    };
    // The other one when the first can't take the file.
    let r = Reader::open(path, gpu).or_else(|e| Reader::open(path, !gpu).map_err(|_| e))?;
    let frame = r.width * r.height * 3 / 2;
    Ahead::new(r, frame)
}

struct Reader {
    reader: IMFSourceReader,
    /// Kept alive with the reader that decodes on it.
    _device: Option<(ID3D11Device, IMFDXGIDeviceManager)>,
    width: usize,
    height: usize,
    /// Rows of the decoder's Y plane (the coded height, ≥ `height`).
    rows: usize,
    colour: Colour,
    /// Frame duration from the stream (100 ns), for samples that carry none.
    duration: i64,
    eof: bool,
    /// Samples read and seeks made (diagnostics).
    reads: u64,
    seeks: u64,
}

// SAFETY: the Source Reader and the multithread-protected D3D11 device are free-threaded COM
// objects; the reader moves to its decoding thread once, which alone uses it from then on.
unsafe impl Send for Reader {}

impl Reader {
    /// `gpu`: decode on the GPU's decoder (a D3D11 video device), else Media Foundation's CPU one.
    fn open(path: &std::path::Path, gpu: bool) -> Result<Reader, String> {
        startup()?;
        unsafe {
            let device = if gpu { Some(video_device().ok_or("no D3D11 video device")?) } else { None };
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 4).map_err(err("attributes"))?;
            let attrs = attrs.ok_or("no attributes")?;
            attrs.SetUINT32(&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, 1).map_err(err("hardware transforms"))?;
            if let Some((_, m)) = &device {
                attrs.SetUnknown(&MF_SOURCE_READER_D3D_MANAGER, m).map_err(err("D3D manager"))?;
            }
            let url = HSTRING::from(path.as_os_str());
            let reader = MFCreateSourceReaderFromURL(&url, &attrs).map_err(|e| format!("{}: {e}", path.display()))?;
            let all = MF_SOURCE_READER_ALL_STREAMS.0 as u32;
            let video = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
            reader.SetStreamSelection(all, false).map_err(err("stream selection"))?;
            reader.SetStreamSelection(video, true).map_err(|e| format!("{}: no video stream: {e}", path.display()))?;
            let nv12 = MFCreateMediaType().map_err(err("media type"))?;
            nv12.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).map_err(err("media type"))?;
            nv12.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12).map_err(err("media type"))?;
            reader.SetCurrentMediaType(video, None, &nv12).map_err(|e| format!("{}: no NV12 decode: {e}", path.display()))?;
            let mut d =
                Reader { reader, _device: device, width: 0, height: 0, rows: 0, colour: Colour::default(), duration: 0, eof: false, reads: 0, seeks: 0 };
            d.read_type()?;
            Ok(d)
        }
    }

    /// The output type's size, display area, frame rate and colour description.
    fn read_type(&mut self) -> Result<(), String> {
        unsafe {
            let t = self.reader.GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32).map_err(err("current type"))?;
            let size = t.GetUINT64(&MF_MT_FRAME_SIZE).map_err(err("frame size"))?;
            let (cw, ch) = ((size >> 32) as usize, (size & 0xffff_ffff) as usize);
            // The picture inside the coded frame (1080 of 1088 rows).
            let mut area = MFVideoArea::default();
            let shown = t
                .GetBlob(
                    &MF_MT_MINIMUM_DISPLAY_APERTURE,
                    std::slice::from_raw_parts_mut(&mut area as *mut MFVideoArea as *mut u8, size_of::<MFVideoArea>()),
                    None,
                )
                .is_ok();
            let (w, h) = if shown && area.Area.cx > 0 && area.Area.cy > 0 { (area.Area.cx as usize, area.Area.cy as usize) } else { (cw, ch) };
            self.width = w.min(cw) & !1;
            self.height = h.min(ch) & !1;
            self.rows = ch;
            let rate = t.GetUINT64(&MF_MT_FRAME_RATE).unwrap_or(0);
            let (num, den) = ((rate >> 32) as i64, (rate & 0xffff_ffff) as i64);
            self.duration = if num > 0 { 10_000_000 * den / num } else { 0 };
            let get = |k: &GUID| t.GetUINT32(k).ok().map(|v| v as i32);
            self.colour = Colour {
                matrix: match get(&MF_MT_YUV_MATRIX) {
                    Some(v) if v == MFVideoTransferMatrix_BT601.0 => Matrix::Bt601,
                    Some(v) if v == MFVideoTransferMatrix_BT2020_10.0 || v == MFVideoTransferMatrix_BT2020_10.0 + 1 => Matrix::Bt2020,
                    _ => Matrix::Bt709,
                },
                transfer: match get(&MF_MT_TRANSFER_FUNCTION) {
                    Some(v) if v == MFVideoTransFunc_sRGB.0 => Transfer::Srgb,
                    Some(v) if v == MFVideoTransFunc_10.0 => Transfer::Linear,
                    Some(v) if v == MFVideoTransFunc_2084.0 => Transfer::Pq,
                    Some(v) if v == MFVideoTransFunc_HLG.0 => Transfer::Hlg,
                    _ => Transfer::Bt709,
                },
                primaries: match get(&MF_MT_VIDEO_PRIMARIES) {
                    Some(v) if v == MFVideoPrimaries_BT470_2_SysBG.0 => Primaries::Bt601_625,
                    Some(v) if v == MFVideoPrimaries_SMPTE170M.0 => Primaries::Bt601_525,
                    Some(v) if v == MFVideoPrimaries_BT2020.0 => Primaries::Bt2020,
                    Some(v) if v == MFVideoPrimaries_DCI_P3.0 => Primaries::P3D65,
                    _ => Primaries::Bt709,
                },
                full_range: get(&MF_MT_VIDEO_NOMINAL_RANGE) == Some(MFNominalRange_0_255.0),
            };
        }
        if self.width == 0 || self.height == 0 {
            return Err("decoder reports an empty frame".into());
        }
        Ok(())
    }

    fn seek(&mut self, time: i64) -> Result<(), String> {
        let pv = PROPVARIANT {
            Anonymous: PROPVARIANT_0 {
                Anonymous: ManuallyDrop::new(PROPVARIANT_0_0 {
                    vt: VT_I8,
                    wReserved1: 0,
                    wReserved2: 0,
                    wReserved3: 0,
                    Anonymous: PROPVARIANT_0_0_0 { hVal: time.max(0) },
                }),
            },
        };
        unsafe { self.reader.SetCurrentPosition(&GUID::zeroed(), &pv) }.map_err(err("seek"))?;
        self.seeks += 1;
        self.eof = false;
        Ok(())
    }

    /// The next decoded frame (`None`: a gap or a format change, read again).
    fn read(&mut self) -> Result<Option<Arc<Frame>>, String> {
        let (mut flags, mut ts, mut sample) = (0u32, 0i64, None);
        self.reads += 1;
        unsafe { self.reader.ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0, None, Some(&mut flags), Some(&mut ts), Some(&mut sample)) }
            .map_err(err("ReadSample"))?;
        if flags & MF_SOURCE_READERF_ENDOFSTREAM.0 as u32 != 0 {
            self.eof = true;
            return Ok(None);
        }
        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
            self.read_type()?;
        }
        let Some(sample) = sample else { return Ok(None) };
        let duration = unsafe { sample.GetSampleDuration() }.ok().filter(|d| *d > 0).unwrap_or(self.duration);
        let mut f = Frame { width: self.width as u32, height: self.height as u32, colour: self.colour, time: ts, duration, ..Default::default() };
        unsafe { self.planes(&sample, &mut f) }?;
        Ok(Some(Arc::new(f)))
    }

    /// NV12 rows (at the buffer's pitch) into tight Y, U and V planes.
    unsafe fn planes(&self, sample: &IMFSample, f: &mut Frame) -> Result<(), String> {
        let (w, h, rows) = (self.width, self.height, self.rows);
        unsafe {
            let buf = sample.GetBufferByIndex(0).map_err(err("frame buffer"))?;
            let (mut p, mut pitch) = (std::ptr::null_mut(), 0i32);
            let two_d = buf.cast::<IMF2DBuffer2>().ok();
            let avail = match &two_d {
                Some(b) => {
                    let (mut start, mut len) = (std::ptr::null_mut(), 0u32);
                    b.Lock2DSize(MF2DBuffer_LockFlags_Read, &mut p, &mut pitch, &mut start, &mut len).map_err(err("frame lock"))?;
                    // Bytes from the first scanline to the end of the buffer.
                    (len as usize).saturating_sub((p as usize).saturating_sub(start as usize))
                }
                None => {
                    let mut len = 0u32;
                    buf.Lock(&mut p, None, Some(&mut len)).map_err(err("frame lock"))?;
                    pitch = w as i32;
                    len as usize
                }
            };
            // Y rows `rows` apart, then the UV rows; the last row read ends `w` bytes in.
            let need = (rows + h / 2).saturating_sub(1).saturating_mul(pitch.max(0) as usize).saturating_add(w);
            // (A negative pitch, bottom-up rows, is refused.)
            let ok = !p.is_null() && pitch > 0 && pitch as usize >= w && h >= 2 && avail >= need;
            if ok {
                let pitch = pitch as usize;
                // SAFETY: the lock gives `avail` readable bytes from `p`, and every row read below
                // ends within `need` ≤ `avail` bytes.
                let all = std::slice::from_raw_parts(p, need);
                let mut y = Vec::with_capacity(w * h);
                for r in 0..h {
                    y.extend_from_slice(&all[r * pitch..r * pitch + w]);
                }
                let (mut u, mut v) = (Vec::with_capacity(w / 2 * (h / 2)), Vec::with_capacity(w / 2 * (h / 2)));
                for r in 0..h / 2 {
                    let row = &all[(rows + r) * pitch..(rows + r) * pitch + w];
                    for [a, b] in row.as_chunks::<2>().0 {
                        u.push(*a);
                        v.push(*b);
                    }
                }
                (f.y, f.u, f.v) = (Arc::new(y), Arc::new(u), Arc::new(v));
            }
            match &two_d {
                Some(b) => b.Unlock2D().map_err(err("frame unlock"))?,
                None => buf.Unlock().map_err(err("frame unlock"))?,
            }
            if !ok {
                return Err("decoder returned an unreadable frame".into());
            }
        }
        Ok(())
    }
}

impl Source for Reader {
    fn start(&mut self) {
        // COM on the decoding thread too (the reader is free-threaded).
        let _ = startup();
    }
    fn seek(&mut self, time: i64) -> Result<(), String> {
        Reader::seek(self, time)
    }
    fn read(&mut self) -> Result<Option<Arc<Frame>>, String> {
        Reader::read(self)
    }
    fn eof(&self) -> bool {
        self.eof
    }
    fn duration(&self) -> i64 {
        self.duration
    }
    fn counts(&self) -> (u64, u64) {
        (self.reads, self.seeks)
    }
}

/// A hardware D3D11 device for the decoder, multithread-protected and wrapped in a DXGI device
/// manager; `None` without one (the reader then decodes on the CPU).
unsafe fn video_device() -> Option<(ID3D11Device, IMFDXGIDeviceManager)> {
    unsafe {
        let mut device = None;
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )
        .ok()?;
        let device = device?;
        if let Ok(mt) = device.cast::<ID3D11Multithread>() {
            let _ = mt.SetMultithreadProtected(true);
        }
        let (mut token, mut manager) = (0u32, None);
        MFCreateDXGIDeviceManager(&mut token, &mut manager).ok()?;
        let manager = manager?;
        manager.ResetDevice(&device, token).ok()?;
        Some((device, manager))
    }
}
