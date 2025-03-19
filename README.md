<!--
SPDX-FileCopyrightText: 2024-2025 University of Rochester

SPDX-License-Identifier: MIT
-->


# abc_helper FFI Support

abc_helper provides ffi bindings for C++.

the `ffi_examples` folder contains an example project that shows how to use this library from C++.


# Building

If you're using this library from another rust project, just add this repository as a dependency in your Cargo.toml, pointing to this github repository.

If you need to use the FFI bindings, then follow the steps below.

You'll need Rust and its toolchain - including Cargo.

https://www.rust-lang.org/tools/install

Clone this repository, and then in the cloned directory, run

`cargo build --release --features=cffi`

This will build the code and enable the FFI features. By default, the build files will be placed in `target/release`

To include and link against this library, use the directories:
include: `/path/to/wgpu-abc-helper/target/release/include`
lib:  `/path/to/wgpu-abc-helper/target/release/`

The header file is named `wgpu_abc_helper.hpp`, and the library is `abc_helper`.

See `ffi_examples/Makefile` for an example.


# Using abce-solve-json

Included is a binary capable of deserializing a module and then solving it from rust. This is meant to be used for debugging purposes to identify
why the solver is failing.

Running it requires the serialized json file with the constraints. It can be invoked via:

``cargo run --example abce-solve-json -- /path/to/serialized/module.json --idx=... --log=/path/to/log/file.log``

The optional ``--idx`` flag specifies which summary (0-based index) to evaluate the constraints for, and defaults to the last summary in the json.  
The optional ``--log`` flag specifies a path to where a log file should be written. If not specified, the log will be written to stdout.
