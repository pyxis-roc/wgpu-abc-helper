// SPDX-FileCopyrightText: 2024-2025 University of Rochester
//
// SPDX-License-Identifier: MIT

fn main() {
    #[cfg(feature = "cffi")]
    // Now, building will just copy the header files to the target directory.
    {
        // Get the directory of the current crate
        if std::env::var("CARGO_FEATURE_CFFI").is_err() {
            // Do nothing if the cffi feature is not enabled.
            return;
        }
        println!("cargo:rerun-if-changed=ffi_headers/wgpu_abc_helper.hpp");

        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        // All we need to do is copy the files to the specified include directory.

        let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

        // Generate the C/C++ header using cbindgen
        let output_file = std::path::Path::new(&target_dir)
            .join(&profile)
            .join("include")
            .join("wgpu_abc_helper.hpp");

        std::fs::copy("ffi_headers/wgpu_abc_helper.hpp", &output_file).unwrap();
    }
}
// REUSE-IgnoreEnd
