// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

fn main() {
    #[cfg(feature = "cffi")]
    {
        // Get the directory of the current crate
        if std::env::var("CARGO_FEATURE_CFFI").is_err() {
            // Do nothing if the cffi feature is not enabled.
            return;
        }
        println!("cargo:rerun-if-changed=src/cffi.rs");

        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

        // Generate the C/C++ header using cbindgen
        let output_file = std::path::Path::new(&target_dir)
            .join(&profile)
            .join("include")
            .join("wgpu_constraints.hpp");

        // From https://github.com/mozilla/cbindgen/issues/472#issuecomment-831439826
        match cbindgen::generate(&crate_dir) {
            Ok(bindings) => bindings.write_to_file(output_file),
            // During development..
            Err(cbindgen::Error::ParseSyntaxError { .. }) => return, // ignore in favor of cargo's syntax check
            Err(err) => panic!("{:?}", err),
        };
    }
}
