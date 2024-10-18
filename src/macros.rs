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

pub(super) use cbindgen_annotate;
