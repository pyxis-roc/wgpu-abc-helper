// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
// REUSE-IgnoreStart

/* Ripped directly from cargo-expand. RUSTC_BOOTSTRAP allows us to use
unstable
*/

#[cfg(feature = "cffi")]
mod cffi_utils {
    /*!
    This is adapted directly from the `cargo-expand` crate, (which is MIT licensed).

    Basically, we need to conditionally turn on `RUSTC_BOOTSTRAP`, depending on
    whether the current toolchain allows us to use -Zunpretty=expanded.


     */
    use std::{
        env,
        ffi::{OsStr, OsString},
        iter,
        path::{Path, PathBuf},
        process::{Command, Stdio},
    };
    pub trait CommandExt {
        fn flag_value<K, V>(&mut self, k: K, v: V) -> &mut Self
        where
            K: AsRef<OsStr>,
            V: AsRef<OsStr>;
    }

    impl CommandExt for Command {
        fn flag_value<K, V>(&mut self, k: K, v: V) -> &mut Self
        where
            K: AsRef<OsStr>,
            V: AsRef<OsStr>,
        {
            let k = k.as_ref();
            let v = v.as_ref();
            if let Some(k) = k.to_str() {
                if let Some(v) = v.to_str() {
                    self.arg(format!("{k}={v}"));
                    return self;
                }
            }
            self.arg(k);
            self.arg(v);
            self
        }
    }

    pub(super) fn needs_rustc_bootstrap() -> bool {
        fn cargo_binary() -> std::ffi::OsString {
            env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
        }
        if env::var_os("RUSTC_BOOTSTRAP").is_some_and(|var| !var.is_empty()) {
            return false;
        }

        let rustc = if let Some(rustc) = env::var_os("RUSTC") {
            PathBuf::from(rustc)
        } else {
            let mut cmd = std::process::Command::new(cargo_binary());
            cmd.arg("rustc");
            cmd.arg("-Zunstable-options");
            cmd.flag_value("--print", "sysroot");
            cmd.env("RUSTC_BOOTSTRAP", "1");
            cmd.stdin(Stdio::null());
            cmd.stderr(Stdio::null());
            let Ok(output) = cmd.output() else {
                return true;
            };
            let Ok(stdout) = std::str::from_utf8(&output.stdout) else {
                return true;
            };
            let sysroot = Path::new(stdout.trim_end());
            sysroot.join("bin").join("rustc")
        };

        let rustc_wrapper = env::var_os("RUSTC_WRAPPER").filter(|wrapper| !wrapper.is_empty());
        let rustc_workspace_wrapper =
            env::var_os("RUSTC_WORKSPACE_WRAPPER").filter(|wrapper| !wrapper.is_empty());
        let mut wrapped_rustc = rustc_wrapper
            .into_iter()
            .chain(rustc_workspace_wrapper)
            .chain(iter::once(rustc.into_os_string()));

        let mut cmd = Command::new(wrapped_rustc.next().unwrap());
        cmd.args(wrapped_rustc);
        cmd.arg("-Zunpretty=expanded");
        cmd.arg("-");
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());
        let Ok(status) = cmd.status() else {
            return true;
        };
        !status.success()
    }
}

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
            .join("wgpu_abc_helper.hpp");

        // We need to turn on RUSTC_BOOTSTRAP, because cbindgen's expand uses the unstable `-Zunpretty=expanded`
        // flag of rustc, which requires unstable options.
        // So, we temporarily set the environment variable RUSTC_BOOTSTRAP to 1 just before invoking
        // cbindgen, then restore it to its original value.
        let needs_bootstrap = cffi_utils::needs_rustc_bootstrap();
        let mut old_rustc_bootstrap = None;
        if needs_bootstrap {
            old_rustc_bootstrap = std::env::var_os("RUSTC_BOOTSTRAP");
            std::env::set_var("RUSTC_BOOTSTRAP", "1");
        }

        // From https://github.com/mozilla/cbindgen/issues/472#issuecomment-831439826
        match cbindgen::generate(&crate_dir) {
            Ok(bindings) => bindings.write_to_file(output_file),
            // During development..
            Err(cbindgen::Error::ParseSyntaxError { .. }) => return, // ignore in favor of cargo's syntax check
            Err(err) => panic!("{err:?}"),
        };

        if needs_bootstrap {
            if let Some(var) = old_rustc_bootstrap {
                std::env::set_var("RUSTC_BOOTSTRAP", var);
            } else {
                std::env::remove_var("RUSTC_BOOTSTRAP");
            }
        }
    }
}
// REUSE-IgnoreEnd
