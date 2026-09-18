//! Build script — generates MLX-C FFI bindings via bindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_C_PATH");
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=wrapper.h");

    // Support both MLX_C_PATH (new) and MLX_C_PREFIX (legacy) env vars
    let mlx_c_prefix =
        env::var("MLX_C_PATH")
            .or_else(|_| env::var("MLX_C_PREFIX"))
            .unwrap_or_else(|_| "/opt/homebrew/opt/mlx-c".to_owned());
    let mlx_prefix =
        env::var("MLX_PREFIX").unwrap_or_else(|_| "/opt/homebrew/opt/mlx".to_owned());

    let mlxc_lib = format!("{mlx_c_prefix}/lib/libmlxc.dylib");
    let mlx_lib = format!("{mlx_prefix}/lib/libmlx.dylib");

    let mlxc_lib_path = PathBuf::from(&mlxc_lib);
    let mlx_lib_path = PathBuf::from(&mlx_lib);

    if !mlxc_lib_path.exists() {
        // Try vendors/mlx-c
        let vendors_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../../../vendors/mlx-c");
        if vendors_path.join("lib/libmlxc.dylib").exists() {
            println!("cargo:rustc-link-search=native={}", vendors_path.join("lib").display());
            println!("cargo:rustc-link-lib=dylib=mlxc");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", vendors_path.join("lib").display());
        } else {
            panic!(
                "libmlxc.dylib not found. Install: brew install mlx-c or clone to vendors/mlx-c"
            );
        }
    } else {
        println!("cargo:rustc-link-search=native={mlx_c_prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=mlxc");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_c_prefix}/lib");
    }

    if !mlx_lib_path.exists() {
        let vendors_path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../../../vendors/mlx");
        if vendors_path.join("lib/libmlx.dylib").exists() {
            println!("cargo:rustc-link-search=native={}", vendors_path.join("lib").display());
            println!("cargo:rustc-link-lib=dylib=mlx");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", vendors_path.join("lib").display());
        }
    } else {
        println!("cargo:rustc-link-search=native={mlx_prefix}/lib");
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_prefix}/lib");
    }

    // Bindgen - find include directories
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_function("mlx_.*")
        .allowlist_function("_mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        .rustified_enum("mlx_dtype_")
        .rustified_enum("mlx_device_type_")
        .derive_default(true)
        .layout_tests(false);

    // Add include paths
    builder = builder.clang_arg("-I/opt/homebrew/include");
    if mlxc_lib_path.exists() {
        builder = builder.clang_arg(format!("-I{mlx_c_prefix}/include"));
    }
    if mlx_lib_path.exists() {
        builder = builder.clang_arg(format!("-I{mlx_prefix}/include"));
    }
    // Also check vendors
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendors_mlx_c = manifest_dir.join("../../../vendors/mlx-c/include");
    let vendors_mlx = manifest_dir.join("../../../vendors/mlx/include");
    if vendors_mlx_c.exists() {
        builder = builder.clang_arg(format!("-I{}", vendors_mlx_c.display()));
    }
    if vendors_mlx.exists() {
        builder = builder.clang_arg(format!("-I{}", vendors_mlx.display()));
    }

    let bindings = builder
        .generate()
        .expect("bindgen failed. Check mlx-c headers: brew install mlx-c or set MLX_C_PATH");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("could not write bindings.rs");

    // Strip inner attrs that bindgen 0.71+ emits
    let raw = std::fs::read_to_string(&bindings_path).unwrap();
    let cleaned: String = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("#!["))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&bindings_path, cleaned).unwrap();
}