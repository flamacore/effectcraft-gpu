//! Decoding ahead: a thread of its own keeps a few frames decoded past the latest request
//! ([`AHEAD`]), so sequential playback and export find their next frame ready instead of
//! waiting for the decoder (decoding overlaps rendering). Platform-neutral; the decoder itself
//! is a [`Source`] (Media Foundation's Source Reader on Windows).

#![cfg_attr(not(windows), allow(dead_code))]

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use crate::Frame;

/// How long a request may wait on the decoder (or the encoder on an event) before it is given
/// up as stopped: a hung driver fails the request instead of hanging it.
pub(crate) const STALL: Duration = Duration::from_secs(30);

/// Decoded frames kept for requests a little behind the decoder (frames rendered in parallel
/// ask out of order, export's CPU path 16 at once): at most this many, and at most
/// [`RING_BYTES`] of them (20 frames at 2160p).
const RING: usize = 24;
const RING_BYTES: usize = 256 << 20;

/// A request further than this past the decoder's position seeks instead of decoding forward
/// (100 ns units: 2 s).
const SEEK_AHEAD: i64 = 20_000_000;

/// Frames decoded past the latest request.
const AHEAD: i64 = 4;

/// A decoder [`Ahead`] drives from its own thread (times in 100 ns units).
pub(crate) trait Source: Send + 'static {
    /// Called on the decoding thread before anything else.
    fn start(&mut self) {}
    /// Continue from the frame showing `time` (or an earlier one: a keyframe before it).
    fn seek(&mut self, time: i64) -> Result<(), String>;
    /// The next frame in display order; `None` at the end ([`Source::eof`]) or a gap.
    fn read(&mut self) -> Result<Option<Arc<Frame>>, String>;
    fn eof(&self) -> bool;
    /// A frame's duration.
    fn duration(&self) -> i64;
    /// Samples read and seeks made (diagnostics).
    fn counts(&self) -> (u64, u64);
}

/// What callers and the decoding thread share.
struct State {
    /// Decoded frames, oldest first, with the seek generation each was read in. A seek keeps
    /// them: a request waiting on a frame still finds it after another one seeked elsewhere.
    ring: VecDeque<(u64, Arc<Frame>)>,
    /// The furthest time asked for in this run (the frames read since the last seek): the
    /// thread decodes up to [`AHEAD`] frames past it.
    want: Option<i64>,
    /// A seek the thread has yet to make.
    seek_to: Option<i64>,
    /// The time this run's seek was asked for (`None`: no seek yet).
    from: Option<i64>,
    /// Bumped by every seek: a frame read across one is dropped.
    generation: u64,
    /// This run's first frame's start and last frame's end.
    first: Option<i64>,
    pos: Option<i64>,
    eof: bool,
    error: Option<String>,
    stop: bool,
    duration: i64,
    reads: u64,
    seeks: u64,
    /// Frames the ring holds (see [`RING`]).
    cap: usize,
}

/// What a request does next when no decoded frame shows its time.
enum Next {
    Take(Arc<Frame>),
    Seek,
    /// Wait for this run to decode it.
    Wait,
    /// The movie has no frame there.
    Fail,
}

impl State {
    /// A decoded frame showing `at` (see [`Ahead::frame_at`]).
    fn showing(&self, at: i64) -> Option<Arc<Frame>> {
        self.ring.iter().map(|(_, f)| f).find(|f| f.time <= at && at < f.time + f.duration.max(1)).cloned()
    }

    fn seek(&mut self, time: i64) {
        (self.seek_to, self.from, self.want) = (Some(time), Some(time), Some(time));
        self.generation += 1;
        (self.first, self.pos, self.eof, self.error) = (None, None, false, None);
    }

    /// For a request for `time` (`at` with rounding) that no decoded frame shows, while no
    /// seek is pending.
    fn next(&self, time: i64, at: i64) -> Next {
        let Some(from) = self.from else { return Next::Seek };
        let mut run = self.ring.iter().filter(|(g, _)| *g == self.generation).map(|(_, f)| f);
        let Some(earliest) = run.next() else {
            // A run that ended before its first frame has none for its time.
            return if !self.eof {
                Next::Wait
            } else if time >= from {
                Next::Fail
            } else {
                Next::Seek
            };
        };
        if at < earliest.time && self.first == Some(earliest.time) && time >= from {
            // The first frame after a seek can start later than asked (timestamps that do not
            // start at zero): show it rather than nothing.
            return Next::Take(earliest.clone());
        }
        if self.eof {
            // After the last frame: it stays on screen.
            return if at < earliest.time { Next::Seek } else { Next::Take(run.next_back().unwrap_or(earliest).clone()) };
        }
        // A run serves the request that seeked (from the keyframe before it, frames away)
        // before another one seeks elsewhere; two of them would otherwise take turns forever.
        if self.pos.is_none_or(|p| p <= from + self.duration.max(8) / 8) {
            return Next::Wait;
        }
        if at < earliest.time || self.pos.is_some_and(|p| time > p + SEEK_AHEAD) { Next::Seek } else { Next::Wait }
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) struct Ahead {
    shared: Arc<(Mutex<State>, Condvar)>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Ahead {
    /// Decode `src` (frames of `frame_bytes`) on a thread of its own.
    pub(crate) fn new(src: impl Source, frame_bytes: usize) -> Result<Ahead, String> {
        let cap = (RING_BYTES / frame_bytes.max(1)).clamp(AHEAD as usize + 4, RING);
        let state = State {
            ring: VecDeque::new(),
            want: None,
            seek_to: None,
            from: None,
            generation: 0,
            first: None,
            pos: None,
            eof: false,
            error: None,
            stop: false,
            duration: src.duration(),
            reads: 0,
            seeks: 0,
            cap,
        };
        let shared = Arc::new((Mutex::new(state), Condvar::new()));
        let theirs = shared.clone();
        let worker = std::thread::Builder::new()
            .name("effectcraft video decode".into())
            .spawn(move || work(src, &theirs))
            .map_err(|e| format!("decoder thread: {e}"))?;
        Ok(Ahead { shared, worker: Some(worker) })
    }

    pub(crate) fn stats(&self) -> (u64, u64, i64) {
        let s = lock(&self.shared.0);
        (s.reads, s.seeks, s.duration)
    }

    /// Requests on parallel threads arrive in any order, and one waiting looks again at every
    /// change: a request another one's seek left behind seeks back itself.
    pub(crate) fn frame_at(&self, time: i64) -> Result<Arc<Frame>, String> {
        let (m, cv) = &*self.shared;
        let mut s = lock(m);
        // The last frame starting at or before `time`, give or take rounding: the stream's
        // timestamps are rounded to 100 ns (333 333, 333 334...) and a caller's time base may
        // round the other way, so a request an eighth of a frame early still gets that frame.
        // (The media pool asks for frame centres, others for frame starts.)
        let at = time + s.duration.max(8) / 8;
        // A run that failed is retried by the next request.
        if s.error.is_some() && s.seek_to.is_none() {
            s.seek(time);
            cv.notify_all();
        }
        let deadline = Instant::now() + STALL;
        loop {
            if let Some(f) = s.showing(at) {
                return Ok(f);
            }
            if let Some(e) = &s.error {
                return Err(e.clone());
            }
            let next = if s.seek_to.is_none() { s.next(time, at) } else { Next::Wait };
            match next {
                Next::Take(f) => return Ok(f),
                Next::Fail => return Err("no frames".into()),
                Next::Seek => {
                    s.seek(time);
                    cv.notify_all();
                }
                // Within reach of this run (or the pending seek's): decode up to it.
                Next::Wait if s.from.is_some_and(|f| time >= f && time <= s.pos.unwrap_or(f) + SEEK_AHEAD) && s.want.is_none_or(|w| w < time) => {
                    s.want = Some(time);
                    cv.notify_all();
                }
                Next::Wait => {}
            }
            // A decoder that stops answering fails the request (the caller decodes another
            // way) instead of hanging it.
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err("the decoder stopped answering".into());
            }
            s = cv.wait_timeout(s, left).unwrap_or_else(PoisonError::into_inner).0;
        }
    }
}

impl Drop for Ahead {
    fn drop(&mut self) {
        // The thread ends at its next look at the state; it is not waited for, so a read stuck
        // in the driver can't hold up whoever closes the decoder.
        lock(&self.shared.0).stop = true;
        self.shared.1.notify_all();
        drop(self.worker.take());
    }
}

/// The decoding thread: seeks when asked, otherwise keeps [`AHEAD`] frames past the latest
/// request decoded, and sleeps when it is that far ahead.
fn work(mut r: impl Source, shared: &(Mutex<State>, Condvar)) {
    r.start();
    let (m, cv) = shared;
    loop {
        let (seek, generation) = {
            let mut s = lock(m);
            loop {
                if s.stop {
                    return;
                }
                if let Some(t) = s.seek_to {
                    break (Some(t), s.generation);
                }
                let room = s.want.is_some_and(|w| s.pos.is_none_or(|p| p <= w + AHEAD * s.duration.max(1)));
                if s.error.is_none() && !s.eof && room {
                    break (None, s.generation);
                }
                s = cv.wait(s).unwrap_or_else(PoisonError::into_inner);
            }
        };
        match seek {
            Some(t) => {
                let res = r.seek(t);
                let mut s = lock(m);
                (s.reads, s.seeks) = r.counts();
                // A newer seek asked for meanwhile is made next.
                if s.generation == generation {
                    s.seek_to = None;
                    if let Err(e) = res {
                        s.error = Some(e);
                    }
                }
                cv.notify_all();
            }
            None => {
                let res = r.read();
                let mut s = lock(m);
                (s.reads, s.seeks) = r.counts();
                if s.generation != generation {
                    continue;
                }
                match res {
                    Ok(Some(f)) => {
                        s.first.get_or_insert(f.time);
                        s.pos = Some(f.time + f.duration);
                        s.duration = r.duration();
                        s.ring.push_back((generation, f));
                        while s.ring.len() > s.cap {
                            s.ring.pop_front();
                        }
                    }
                    Ok(None) => s.eof |= r.eof(),
                    Err(e) => s.error = Some(e),
                }
                cv.notify_all();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 25 fps, a keyframe every 12 frames; seeks and reads take a while, like a real decoder.
    struct Fake {
        frames: i64,
        next: i64,
        eof: bool,
        delay: Duration,
        counts: (u64, u64),
    }

    const FRAME: i64 = 400_000;

    impl Source for Fake {
        fn seek(&mut self, time: i64) -> Result<(), String> {
            std::thread::sleep(self.delay);
            self.next = (time / FRAME).clamp(0, self.frames) / 12 * 12;
            self.eof = false;
            self.counts.1 += 1;
            Ok(())
        }
        fn read(&mut self) -> Result<Option<Arc<Frame>>, String> {
            std::thread::sleep(self.delay);
            self.counts.0 += 1;
            if self.next >= self.frames {
                self.eof = true;
                return Ok(None);
            }
            self.next += 1;
            Ok(Some(Arc::new(Frame { time: (self.next - 1) * FRAME, duration: FRAME, ..Default::default() })))
        }
        fn eof(&self) -> bool {
            self.eof
        }
        fn duration(&self) -> i64 {
            FRAME
        }
        fn counts(&self) -> (u64, u64) {
            self.counts
        }
    }

    fn decoder(frames: i64, ms: u64) -> Ahead {
        Ahead::new(Fake { frames, next: 0, eof: false, delay: Duration::from_millis(ms), counts: (0, 0) }, 1 << 20).expect("decoder")
    }

    /// The frame asked for (a frame centre, as the media pool asks), its own and soon.
    fn check(d: &Ahead, k: i64) {
        let t0 = Instant::now();
        let f = d.frame_at(k * FRAME + FRAME / 2).expect("frame");
        assert_eq!(f.time / FRAME, k, "request for frame {k}");
        assert!(t0.elapsed() < Duration::from_secs(5), "frame {k} took {:?}", t0.elapsed());
    }

    /// A request waiting on frames ahead keeps them coming after another request seeks the
    /// decoder back (a looping layer rendered in parallel): it was left waiting until the
    /// stall timeout.
    #[test]
    fn a_request_seeked_away_from_still_gets_its_frame() {
        let d = decoder(200, 15);
        check(&d, 120);
        std::thread::scope(|sc| {
            sc.spawn(|| check(&d, 126));
            std::thread::sleep(Duration::from_millis(20));
            sc.spawn(|| check(&d, 2));
        });
    }

    /// A request arriving while another one's seek is pending gets its own frame, not the
    /// first one decoded after that seek.
    #[test]
    fn a_request_behind_a_pending_seek_gets_its_own_frame() {
        let d = decoder(300, 30);
        std::thread::scope(|sc| {
            sc.spawn(|| check(&d, 250));
            std::thread::sleep(Duration::from_millis(10));
            sc.spawn(|| check(&d, 10));
        });
    }

    /// Parallel renders of a looping layer ask near the end and near the start at once.
    #[test]
    fn looping_parallel_requests_all_get_their_frames() {
        let d = decoder(180, 2);
        std::thread::scope(|sc| {
            for i in 0..8 {
                let d = &d;
                sc.spawn(move || {
                    for k in [175 - i, i, 170 + i % 3, 5 + i, 179 - i % 2, 20 + i] {
                        check(d, k);
                    }
                });
            }
        });
    }
}
