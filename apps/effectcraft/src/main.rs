//! `effectcraft`: the desktop app's CPU build (see the library, [`effectcraft_desktop::run`]).
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

fn main() -> eframe::Result {
    effectcraft_desktop::run(false)
}
