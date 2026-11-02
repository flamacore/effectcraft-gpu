use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::{FRAMES_AT_ONCE, Gate, TURN_WAIT};

/// The viewer renders on up to 16 threads; only [`FRAMES_AT_ONCE`] frames are on the GPU at a
/// time (16 at once exhausted a 16 GB card), and a frame's own nested entry doesn't wait.
#[test]
fn frames_take_turns_at_the_gpu() {
    let gate = Gate::default();
    let (now, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|sc| {
        for _ in 0..12 {
            sc.spawn(|| {
                let _slot = gate.enter();
                let n = now.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(n, Ordering::SeqCst);
                // Work inside the frame that enters again goes ahead at once.
                let t0 = Instant::now();
                drop(gate.enter());
                assert!(t0.elapsed() < Duration::from_millis(500));
                std::thread::sleep(Duration::from_millis(20));
                now.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    assert_eq!(most.load(Ordering::SeqCst), FRAMES_AT_ONCE);
}

/// A frame whose turn never comes goes ahead after [`TURN_WAIT`] instead of hanging.
#[test]
fn a_frame_waits_for_its_turn_only_so_long() {
    let gate = Gate::default();
    let held: Vec<_> = (0..FRAMES_AT_ONCE).map(|_| std::thread::scope(|sc| sc.spawn(|| std::mem::forget(gate.enter())).join())).collect();
    assert!(held.iter().all(Result::is_ok));
    let t0 = Instant::now();
    drop(gate.enter());
    let waited = t0.elapsed();
    assert!(waited >= TURN_WAIT && waited < TURN_WAIT * 3, "waited {waited:?}");
}
