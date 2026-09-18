use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    println!("cargo:rerun-if-changed=shaders/");
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "macos" { return; }

    let metal_file = Path::new("shaders/sampler.metal");
    if !metal_file.exists() { return; }

    let air_path = out_dir.join("sampler.air");
    let status = Command::new("xcrun")
        .args(["metal", "-c", "-target", "air64-apple-macos14.0", "-std=metal3.1", "-O2", "-o"])
        .arg(&air_path).arg(metal_file).status().expect("xcrun metal failed");
    if !status.success() { panic!("Shader compilation failed"); }

    let metallib_path = out_dir.join("sampler.metallib");
    let status = Command::new("xcrun")
        .args(["metallib", "-o"]).arg(&metallib_path).arg(&air_path)
        .status().expect("xcrun metallib failed");
    if !status.success() { panic!("metallib linking failed"); }

    println!("cargo:rustc-env=NXM_SAMPLER_METALLIB_PATH={}", metallib_path.display());
}
