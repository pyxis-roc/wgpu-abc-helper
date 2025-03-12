/// Convenience macro for adding cbindgen annotations to items, in a way such that
/// the doc attribute is hidden from both rust analyzer and rustdoc.
macro_rules! cbindgen_annotate {
    ($annotation:literal $item:item) => {
        #[cfg_attr(not(any(doc, rust_analyzer)), doc = concat!("cbindgen:", $annotation))]
        $item
    };
    (($($annotation:literal),+) $item:item) => {
        $(#[cfg_attr(not(any(doc, rust_analyzer)), doc = concat!("cbindgen:", $annotation))])+
        $item
    };
}

/// Expand to a conditional return statement that
///  **returns** a [`SolverError::TypeMismatch`] if `$a` and `$b` are not the same variant.
///
/// # Arguments
/// * `$a` [`IntervalKind`]
/// * `$b` [`IntervalKind`]
///
///
/// [`SolverError::TypeMismatch`]: crate::SolverError::TypeMismatch
/// [`IntervalKind`]: crate::solvers::interval::translator::IntervalKind
macro_rules! error_if_different_variants {
    ($a:expr, $b:expr) => {{
        let a = &$a;
        let b = &$b;
        if !IntervalKind::is_same_variant(a, b) {
            return Err(SolverError::TypeMismatch {
                expected: a.variant_name(),
                have: b.variant_name(),
                file: file!(),
                line: line!(),
            });
        }
    }};
}

pub(super) use cbindgen_annotate;
pub(super) use error_if_different_variants;
