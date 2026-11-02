//! Media Foundation hardware encoder transforms: NV12 in, an Annex B byte stream out. Hardware
//! MFTs are asynchronous: the transform asks for input (`METransformNeedInput`) and announces
//! output (`METransformHaveOutput`) through its event queue.

use std::mem::ManuallyDrop;
use std::sync::OnceLock;

use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoTaskMemFree};
use windows::Win32::System::Variant::{VARIANT, VARIANT_0, VARIANT_0_0, VARIANT_0_0_0, VT_UI4};
use windows::core::{GUID, Interface};

use crate::ahead::STALL;
use crate::{Codec, EncoderConfig, Packet};

/// `MF_VERSION`: `MF_SDK_VERSION` (2) << 16 | `MF_API_VERSION` (0x70).
const MF_VERSION: u32 = 0x0002_0070;

pub(crate) fn err(what: &'static str) -> impl Fn(windows::core::Error) -> String {
    move |e| format!("{what}: {e}")
}

/// COM on the calling thread and Media Foundation for the process.
// ponytail: MF stays started until the process exits (no MFShutdown): encoders come and go per export.
pub(crate) fn startup() -> Result<(), String> {
    // S_FALSE (already initialised) and RPC_E_CHANGED_MODE (an STA thread) both leave COM usable.
    let _ = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    static STARTED: OnceLock<Result<(), String>> = OnceLock::new();
    STARTED.get_or_init(|| unsafe { MFStartup(MF_VERSION, MFSTARTUP_LITE) }.map_err(err("MFStartup"))).clone()
}

pub(crate) struct Encoder {
    pub(crate) name: String,
    codec: Codec,
    activate: IMFActivate,
    t: IMFTransform,
    events: IMFMediaEventGenerator,
    in_id: u32,
    out_id: u32,
    w: usize,
    h: usize,
    /// One frame in 100 ns units.
    dur: f64,
    /// Input requests not yet answered.
    need: u32,
    drained: bool,
    sets: Vec<Vec<u8>>,
    nv12: Vec<u8>,
}

impl Encoder {
    pub(crate) fn new(cfg: EncoderConfig) -> Result<Encoder, String> {
        startup()?;
        let sub = subtype(cfg.codec);
        let input = MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Video, guidSubtype: MFVideoFormat_NV12 };
        let output = MFT_REGISTER_TYPE_INFO { guidMajorType: MFMediaType_Video, guidSubtype: sub };
        let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut n = 0u32;
        let flags = MFT_ENUM_FLAG(MFT_ENUM_FLAG_HARDWARE.0 | MFT_ENUM_FLAG_SORTANDFILTER.0);
        unsafe { MFTEnumEx(MFT_CATEGORY_VIDEO_ENCODER, flags, Some(&input), Some(&output), &mut list, &mut n) }.map_err(err("MFTEnumEx"))?;
        let found: Vec<IMFActivate> = if list.is_null() {
            vec![]
        } else {
            // SAFETY: MFTEnumEx returned `n` activation objects at `list`; each is moved out once,
            // then the CoTaskMem array itself is freed.
            let items = unsafe { std::slice::from_raw_parts_mut(list, n as usize) };
            let v = items.iter_mut().filter_map(Option::take).collect();
            unsafe { CoTaskMemFree(Some(list as *const _)) };
            v
        };
        let mut why = format!("no hardware {:?} encoder", cfg.codec);
        for a in found {
            match unsafe { Self::start(&a, cfg) } {
                Ok((name, t, events, in_id, out_id)) => {
                    let dur = 1e7 * f64::from(cfg.fps_den) / f64::from(cfg.fps_num.max(1));
                    let (w, h) = (cfg.width as usize, cfg.height as usize);
                    log::info!(target: "effectcraft_hwcodec", "{:?} export on {name}", cfg.codec);
                    return Ok(Encoder {
                        name,
                        codec: cfg.codec,
                        activate: a,
                        t,
                        events,
                        in_id,
                        out_id,
                        w,
                        h,
                        dur,
                        need: 0,
                        drained: false,
                        sets: vec![],
                        nv12: vec![],
                    });
                }
                Err(e) => {
                    let _ = unsafe { a.ShutdownObject() };
                    why = e;
                }
            }
        }
        Err(why)
    }

    /// Activate and configure one encoder: rate control, GOP, no B-frames, BT.709 limited range.
    unsafe fn start(a: &IMFActivate, cfg: EncoderConfig) -> Result<(String, IMFTransform, IMFMediaEventGenerator, u32, u32), String> {
        unsafe {
            let name = friendly_name(a);
            let t: IMFTransform = a.ActivateObject().map_err(err("activate"))?;
            let attrs = t.GetAttributes().map_err(err("attributes"))?;
            if attrs.GetUINT32(&MF_TRANSFORM_ASYNC).unwrap_or(0) == 0 {
                return Err(format!("{name}: not an asynchronous transform"));
            }
            attrs.SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1).map_err(err("async unlock"))?;
            let events: IMFMediaEventGenerator = t.cast().map_err(err("event generator"))?;
            let (mut ins, mut outs) = ([0u32], [0u32]);
            // E_NOTIMPL: the streams are numbered from 0.
            let (in_id, out_id) = if t.GetStreamIDs(&mut ins, &mut outs).is_ok() { (ins[0], outs[0]) } else { (0, 0) };
            if let Ok(api) = t.cast::<ICodecAPI>() {
                set_u32(&api, &CODECAPI_AVEncCommonRateControlMode, eAVEncCommonRateControlMode_UnconstrainedVBR.0 as u32);
                set_u32(&api, &CODECAPI_AVEncCommonMeanBitRate, cfg.bitrate_kbps.saturating_mul(1000));
                set_u32(&api, &CODECAPI_AVEncMPVGOPSize, cfg.keyint.max(1));
                set_u32(&api, &CODECAPI_AVEncMPVDefaultBPictureCount, 0);
            }
            let size = (u64::from(cfg.width) << 32) | u64::from(cfg.height);
            let rate = (u64::from(cfg.fps_num) << 32) | u64::from(cfg.fps_den.max(1));
            let video = |sub: &GUID| -> windows::core::Result<IMFMediaType> {
                let m = MFCreateMediaType()?;
                m.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
                m.SetGUID(&MF_MT_SUBTYPE, sub)?;
                m.SetUINT64(&MF_MT_FRAME_SIZE, size)?;
                m.SetUINT64(&MF_MT_FRAME_RATE, rate)?;
                m.SetUINT64(&MF_MT_PIXEL_ASPECT_RATIO, (1 << 32) | 1)?;
                m.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
                // What export's frames are; the encoder signals it in the stream.
                let _ = m.SetUINT32(&MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709.0 as u32);
                let _ = m.SetUINT32(&MF_MT_VIDEO_PRIMARIES, MFVideoPrimaries_BT709.0 as u32);
                let _ = m.SetUINT32(&MF_MT_TRANSFER_FUNCTION, MFVideoTransFunc_709.0 as u32);
                let _ = m.SetUINT32(&MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235.0 as u32);
                Ok(m)
            };
            let ot = video(&subtype(cfg.codec)).map_err(err("output type"))?;
            let profile = match cfg.codec {
                Codec::H264 => eAVEncH264VProfile_High.0,
                Codec::Hevc => eAVEncH265VProfile_Main_420_8.0,
            };
            ot.SetUINT32(&MF_MT_AVG_BITRATE, cfg.bitrate_kbps.saturating_mul(1000)).map_err(err("bitrate"))?;
            ot.SetUINT32(&MF_MT_MPEG2_PROFILE, profile as u32).map_err(err("profile"))?;
            t.SetOutputType(out_id, &ot, 0).map_err(|e| format!("{name}: output type: {e}"))?;
            let it = video(&MFVideoFormat_NV12).map_err(err("input type"))?;
            t.SetInputType(in_id, &it, 0).map_err(|e| format!("{name}: NV12 input: {e}"))?;
            t.ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0).map_err(err("begin streaming"))?;
            t.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0).map_err(err("start of stream"))?;
            Ok((name, t, events, in_id, out_id))
        }
    }

    pub(crate) fn encode(&mut self, y: &[u8], u: &[u8], v: &[u8], frame: i64) -> Result<Vec<Packet>, String> {
        let (w, h) = (self.w, self.h);
        let c = (w / 2) * (h / 2);
        let (Some(y), Some(u), Some(v)) = (y.get(..w * h), u.get(..c), v.get(..c)) else {
            return Err(format!("{}: frame planes smaller than {w}×{h}", self.name));
        };
        // NV12: the Y plane, then U and V interleaved.
        self.nv12.clear();
        self.nv12.extend_from_slice(y);
        self.nv12.extend(u.iter().zip(v).flat_map(|(a, b)| [*a, *b]));
        let sample = unsafe { self.sample(frame) }.map_err(err("input sample"))?;
        let mut out = vec![];
        while self.need == 0 {
            self.pump(true, &mut out)?;
        }
        self.need -= 1;
        unsafe { self.t.ProcessInput(self.in_id, &sample, 0) }.map_err(|e| format!("{}: ProcessInput: {e}", self.name))?;
        while self.pump(false, &mut out)? {}
        Ok(out)
    }

    unsafe fn sample(&self, frame: i64) -> windows::core::Result<IMFSample> {
        unsafe {
            let len = self.nv12.len() as u32;
            let buf = MFCreateMemoryBuffer(len)?;
            let mut p = std::ptr::null_mut();
            buf.Lock(&mut p, None, None)?;
            if !p.is_null() {
                // SAFETY: the locked buffer holds at least `len` bytes (MFCreateMemoryBuffer(len)).
                std::ptr::copy_nonoverlapping(self.nv12.as_ptr(), p, self.nv12.len());
            }
            buf.Unlock()?;
            buf.SetCurrentLength(len)?;
            let s = MFCreateSample()?;
            s.AddBuffer(&buf)?;
            s.SetSampleTime((frame as f64 * self.dur).round() as i64)?;
            s.SetSampleDuration(self.dur.round() as i64)?;
            Ok(s)
        }
    }

    /// Handle one event (`wait`: wait for one, at most [`STALL`]). `false`: none was queued.
    #[allow(non_upper_case_globals)]
    fn pump(&mut self, wait: bool, out: &mut Vec<Packet>) -> Result<bool, String> {
        // Polled rather than a blocking GetEvent, which has no timeout: events come within a
        // few milliseconds, so the first 5 ms yield (a sleep would oversleep each wait), then
        // the wait sleeps a millisecond at a time.
        let start = std::time::Instant::now();
        let deadline = start + STALL;
        let ev = loop {
            match unsafe { self.events.GetEvent(MF_EVENT_FLAG_NO_WAIT) } {
                Ok(ev) => break ev,
                Err(e) if e.code() == MF_E_NO_EVENTS_AVAILABLE => {
                    if !wait {
                        return Ok(false);
                    }
                    let now = std::time::Instant::now();
                    if now > deadline {
                        return Err(format!("{}: the encoder stopped answering", self.name));
                    }
                    if now - start < std::time::Duration::from_millis(5) {
                        std::thread::yield_now();
                    } else {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                }
                Err(e) => return Err(format!("{}: {e}", self.name)),
            }
        };
        let kind = unsafe { ev.GetType() }.map_err(err("event type"))?;
        let status = unsafe { ev.GetStatus() }.map_err(err("event status"))?;
        if status.is_err() {
            return Err(format!("{}: {}", self.name, windows::core::Error::from(status)));
        }
        match MF_EVENT_TYPE(kind as i32) {
            METransformNeedInput => self.need += 1,
            METransformHaveOutput => self.output(out)?,
            METransformDrainComplete => self.drained = true,
            MEError => return Err(format!("{}: encoder error", self.name)),
            _ => {}
        }
        Ok(true)
    }

    fn output(&mut self, out: &mut Vec<Packet>) -> Result<(), String> {
        unsafe {
            let info = self.t.GetOutputStreamInfo(self.out_id).map_err(err("output stream info"))?;
            let provides = info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 | MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES.0) as u32 != 0;
            let mine = if provides {
                None
            } else {
                let b = MFCreateMemoryBuffer(info.cbSize.max(1 << 20)).map_err(err("output buffer"))?;
                let s = MFCreateSample().map_err(err("output sample"))?;
                s.AddBuffer(&b).map_err(err("output sample"))?;
                Some(s)
            };
            let mut db = [MFT_OUTPUT_DATA_BUFFER { dwStreamID: self.out_id, pSample: ManuallyDrop::new(mine), dwStatus: 0, pEvents: ManuallyDrop::new(None) }];
            let mut status = 0u32;
            let r = self.t.ProcessOutput(0, &mut db, &mut status);
            // SAFETY: each field is taken exactly once; the array is not used afterwards.
            let sample = ManuallyDrop::take(&mut db[0].pSample);
            drop(ManuallyDrop::take(&mut db[0].pEvents));
            match r {
                Ok(()) => {}
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    // The encoder renegotiated its output: take the type it offers now.
                    let t = self.t.GetOutputAvailableType(self.out_id, 0).map_err(err("new output type"))?;
                    self.t.SetOutputType(self.out_id, &t, 0).map_err(err("new output type"))?;
                    return Ok(());
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => return Ok(()),
                Err(e) => return Err(format!("{}: ProcessOutput: {e}", self.name)),
            }
            let Some(sample) = sample else { return Ok(()) };
            let key = sample.GetUINT32(&MFSampleExtension_CleanPoint).unwrap_or(0) != 0;
            let time = sample.GetSampleTime().unwrap_or(0);
            let buf = sample.ConvertToContiguousBuffer().map_err(err("output buffer"))?;
            let mut p = std::ptr::null_mut();
            let mut len = 0u32;
            buf.Lock(&mut p, None, Some(&mut len)).map_err(err("output lock"))?;
            // SAFETY: the locked buffer holds `len` valid bytes until Unlock.
            let bytes = if p.is_null() { vec![] } else { std::slice::from_raw_parts(p, len as usize).to_vec() };
            let _ = buf.Unlock();
            let data = crate::packetize(self.codec, &bytes, &mut self.sets);
            if self.sets.is_empty() {
                self.sets_from_type();
            }
            if !data.is_empty() {
                out.push(Packet { data, key, frame: (time as f64 / self.dur).round() as i64 });
            }
        }
        Ok(())
    }

    /// Parameter sets from the output type (encoders that keep them out of the stream).
    fn sets_from_type(&mut self) {
        unsafe {
            let Ok(t) = self.t.GetOutputCurrentType(self.out_id) else { return };
            let Ok(n) = t.GetBlobSize(&MF_MT_MPEG_SEQUENCE_HEADER) else { return };
            let mut b = vec![0u8; n as usize];
            if t.GetBlob(&MF_MT_MPEG_SEQUENCE_HEADER, &mut b, None).is_ok() {
                crate::packetize(self.codec, &b, &mut self.sets);
            }
        }
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<Packet>, String> {
        let mut out = vec![];
        unsafe {
            self.t.ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, self.in_id as usize).map_err(err("end of stream"))?;
            self.t.ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, self.in_id as usize).map_err(err("drain"))?;
        }
        while !self.drained {
            self.pump(true, &mut out)?;
        }
        Ok(out)
    }

    pub(crate) fn parameter_sets(&self) -> Vec<Vec<u8>> {
        self.sets.clone()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        // Asynchronous MFTs hold threads and the device until shut down.
        let _ = unsafe { self.activate.ShutdownObject() };
    }
}

fn subtype(codec: Codec) -> GUID {
    match codec {
        Codec::H264 => MFVideoFormat_H264,
        Codec::Hevc => MFVideoFormat_HEVC,
    }
}

unsafe fn friendly_name(a: &IMFActivate) -> String {
    unsafe {
        let Ok(len) = a.GetStringLength(&MFT_FRIENDLY_NAME_Attribute) else { return "hardware encoder".into() };
        let mut buf = vec![0u16; len as usize + 1];
        if a.GetString(&MFT_FRIENDLY_NAME_Attribute, &mut buf, None).is_err() {
            return "hardware encoder".into();
        }
        String::from_utf16_lossy(buf.get(..len as usize).unwrap_or_default())
    }
}

/// A ULONG codec setting; encoders that do not support one keep their default.
unsafe fn set_u32(api: &ICodecAPI, key: &GUID, v: u32) {
    let var = VARIANT {
        Anonymous: VARIANT_0 {
            Anonymous: ManuallyDrop::new(VARIANT_0_0 { vt: VT_UI4, wReserved1: 0, wReserved2: 0, wReserved3: 0, Anonymous: VARIANT_0_0_0 { ulVal: v } }),
        },
    };
    let _ = unsafe { api.SetValue(key, &var) };
}
