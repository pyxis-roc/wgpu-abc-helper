// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

fn main() {
    // #[cfg(feature = "cffi")]
    // {
    //     // Get the directory of the current crate
    //     if std::env::var("CARGO_FEATURE_CFFI").is_err() {
    //         // Do nothing if the cffi feature is not enabled.
    //         return;
    //     }
    //     println!("cargo:rerun-if-changed=src/lib.rs");

    //     let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

    //     let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    //     let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    //     // Generate the C/C++ header using cbindgen
    //     let output_file = std::path::Path::new(&target_dir)
    //         .join(&profile)
    //         .join("include")
    //         .join("wgpu_constraints.hpp");
    //     cbindgen::generate(&crate_dir)
    //         .expect("Unable to generate bindings")
    //         .write_to_file(&output_file);

    //     // if let Ok(install_root) = env::var("CARGO_INSTALL_ROOT") {
    //     //     let include_dir = Path::new(&install_root).join("include");
    //     //     std::fs::create_dir_all(&include_dir).expect("Could not create include directory");
    //     //     std::fs::copy(&output_file, include_dir.join("wgpu_constraints.hpp"))
    //     //         .expect("Could not copy header file to include directory");
    //     //     let lib_dir =
    // }
    // // }
}
