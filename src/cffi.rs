// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
#![allow(
    non_snake_case, // We use camelCase for constructors of enum variants.
    clippy::default_trait_access // Default::default() is just so much nicer.
)]

use super::{AbcType, Term};
use ffi_support::FfiStr;
use std::ffi::{c_char, CString};
use std::sync::RwLock;

use lazy_static::lazy_static;

#[allow(unused_imports)]
use crate::cbindgen_annotate;
use crate::{AbcScalar, StructField};

/// Represents a possible term, or an error code.
///
/// There is no reason to construct this enum directly, as it is returned by the FFI functions.
#[repr(C)]
pub enum MaybeTerm {
    Error(ErrorCode),
    Success(FfiTerm),
}

impl From<MaybeTerm> for Result<FfiTerm, ErrorCode> {
    fn from(m: MaybeTerm) -> Self {
        match m {
            MaybeTerm::Error(e) => Err(e),
            MaybeTerm::Success(t) => Ok(t),
        }
    }
}

impl From<Result<FfiTerm, ErrorCode>> for MaybeTerm {
    fn from(r: Result<FfiTerm, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeTerm::Success(t),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

/// Represents a possible `AbcType`, or an error code.
#[repr(C)]
pub enum MaybeAbcType {
    Error(ErrorCode),
    Success(FfiAbcType),
}
impl From<Result<FfiAbcType, ErrorCode>> for MaybeAbcType {
    fn from(r: Result<FfiAbcType, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeAbcType::Success(t),
            Err(e) => MaybeAbcType::Error(e),
        }
    }
}

/// `FfiSummary` are not created by the user. They are returned only by the library by the `endSummary()`
/// method of the helper. NOTE: The alpha version of this library's cffi module does not include the helper.
#[repr(transparent)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct FfiSummary {
    id: usize,
}

impl FfiSummary {
    #[allow(dead_code)]
    fn new<T: Into<super::Handle<super::Summary>>>(s: T) -> Result<Self, ErrorCode> {
        let mut summaries = Summaries.write().map_err(|_| ErrorCode::PoisonedLock)?;
        if let Some(id) = ReusableSummaryIds
            .lock()
            .map_err(|_| ErrorCode::PoisonedLock)?
            .pop()
        {
            summaries[id] = Some(s.into());
            Ok(FfiSummary { id })
        } else {
            let id = summaries.len();
            summaries.push(Some(s.into()));
            Ok(FfiSummary { id })
        }
    }

    /// Remove the summary from the library's collection of summaries. This
    /// can be used to free the summary when it is no longer needed.
    ///
    /// Summaries are not created by the user. They are returned only by the library by the `endSummary()`
    /// method of the helper. NOTE: The alpha version of this library's cffi module does not include the helper.
    ///
    /// However,
    ///
    /// **After calling this method, the summary must not be used.**
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if the summary was not found in the library's collection.
    /// [`ErrorCode::Success`] is returned if the summary was successfully removed.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Summaries container is poisoned.
    ///
    /// This can be thought of as the destructor for the summary, and should be called
    /// as long as the summary is no longer needed.
    ///
    /// That is, calling `delete` will just remove the summary from the library's collection.
    ///
    /// [`ErrorCode::NotFound`]: crate::ErrorCode::NotFound
    /// [`ErrorCode::Success`]: crate::ErrorCode::Success
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_summary(&self) -> ErrorCode {
        let r = self.id;
        // functional style! :D
        Summaries.write().map_or_else(
            |_| ErrorCode::PoisonedLock,
            |mut summaries| {
                summaries.get_mut(r).take().map_or_else(
                    || ErrorCode::NotFound,
                    |_| match ReusableSummaryIds.lock() {
                        Ok(mut ids) => {
                            ids.push(r);
                            ErrorCode::Success
                        }
                        Err(_) => ErrorCode::PoisonedLock,
                    },
                )
            },
        )
    }
}

pub enum MaybeSummary {
    Error(ErrorCode),
    Success(FfiSummary),
}

// Terms are really a handle to the term they contain.
lazy_static! {
    // This is a global map of terms that are created by the C API.
    // Terms use Arc, which is not FFI-safe.
    // So instead, we have a Term which is really just a handle into this map.
    static ref Terms: RwLock<Vec<Option<Term>>> = Vec::new().into();
    static ref Types: RwLock<Vec<Option<super::Handle<AbcType>>>> = Vec::new().into();
    static ref Summaries: RwLock<Vec<Option<super::Handle<super::Summary>>>> = Vec::new().into();
    static ref ReusableTermIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
    static ref ReusableTypeIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
    static ref ReusableSummaryIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
}
#[repr(C)]
pub enum ErrorCode {
    Success = 0,
    /// Indicates a panic in the Rust code.
    Panic = 1,
    /// Indicates a term that could not be found in the term arena.
    BadTerm = 2,
    /// Indicates a poisoned lock
    ///
    /// This occurs when a thread panics while holding a lock.
    PoisonedLock = 3,

    /// Indicates a type, term, or summary that does not exist in the library's collection.
    ///
    /// This means the term was either already deleted or never created.
    NotFound = 4,

    /// Indicate a null pointer was passed.
    NullPointer = 5,

    /// Indicates a passed pointer was not properly aligned.
    Alignmenterror = 6,

    /// Indicates a forbidden zero value passed as an arugment.
    ForbiddenZero = 7,

    /// Indicates that the library is at maximum capacity.
    CapacityExceeded = 8,

    /// Indicates that the wrong `AbcType` was passed.
    WrongType = 9,
}

impl super::CmpOp {
    #[must_use]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_negate(&self) -> Self {
        self.negation()
    }
}

impl super::CmpOp {
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_as_ConstraintOp(self) -> super::ConstraintOp {
        self.into()
    }
}

impl super::ConstraintOp {
    /// Conversion method to convert a `ConstraintOp` to a `CmpOp`
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_from_CmpOp(value: super::CmpOp) -> Self {
        value.into()
    }
}

/// For ffi bindings, term does not contain the actual term data, but instead a handle to it.
///
/// This handle corresponds to an index in the `Terms` vector, a static global variable in this library.
/// This is because `Term` has members that are not FFI-Safe.
#[repr(transparent)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub struct FfiTerm {
    id: usize,
}

impl FfiTerm {
    /// Like `new`, except the caller passes in the term map. Used when the caller already has a lock on the terms.
    ///
    /// # Errors
    /// If the lock on the `ReusableTermId`
    fn new_with_terms(new_term: Term, term_map: &mut Vec<Option<Term>>) -> Result<Self, ErrorCode> {
        if let Some(id) = ReusableTermIds
            .lock()
            .map_err(|_| ErrorCode::PoisonedLock)?
            .pop()
        {
            term_map[id] = Some(new_term);
            Ok(FfiTerm { id })
        } else {
            let id = term_map.len();
            term_map.push(Some(new_term));
            Ok(FfiTerm { id })
        }
    }
    /// Add the provided term to the global terms map and return a handle to it.
    ///
    /// # Errors
    /// If the lock on the global terms map is poisoned, this function will return `ErrorCode::PoisonedLock`.
    /// Reuses old IDs
    fn new(term: Term) -> Result<Self, ErrorCode> {
        // We need to get the reader of the terms
        let mut terms = Terms.write().map_err(|_| ErrorCode::PoisonedLock)?;
        Self::new_with_terms(term, &mut terms)
    }

    /// Free the term from the global terms map.
    ///
    /// # Returns
    /// `ErrorCode::Success` if the term was successfully removed.
    /// `ErrorCode::PoisonedLock` if the lock on the global terms map is poisoned.
    pub extern "C" fn abc_free_term(&self) -> ErrorCode {
        let r = self.id;
        let terms = Terms.write().map_err(|_| ErrorCode::PoisonedLock);
        let Ok(mut terms) = terms else {
            return ErrorCode::PoisonedLock;
        };
        terms[r] = None;
        match ReusableTermIds.lock() {
            Ok(mut ids) => {
                ids.push(r);
                ErrorCode::Success
            }
            Err(_) => ErrorCode::PoisonedLock,
        }
    }
}

impl TryFrom<FfiTerm> for Term {
    type Error = ErrorCode;
    fn try_from(t: FfiTerm) -> Result<Self, Self::Error> {
        // Get the map...
        let terms = Terms.read().map_err(|_| ErrorCode::PoisonedLock)?;
        match terms.get(t.id) {
            Some(Some(t)) => Ok(t.clone()),
            _ => Err(ErrorCode::BadTerm),
        }
    }
}

/*
Implementation of predicate constructors for term
*/
impl FfiTerm {
    /// Create a new unit predicate
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_unit_pred(p: Self) -> MaybeTerm {
        let p: Result<Term, ErrorCode> = p.try_into();
        match p {
            Ok(t) => Self::new(Term::new_unit_pred(&t)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }
    /// Create a Term holding the `true` predicate
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_literal_true() -> MaybeTerm {
        Self::new(Term::new_literal_true()).into()
    }

    /// Create a Term holding the `false` predicate
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_literal_false() -> MaybeTerm {
        Self::new(Term::new_literal_false()).into()
    }

    /// Creates lhs && rhs
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_logical_and(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_and(&lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs || rhs
    ///
    /// Returns a `MaybeTerm`, which is either a `Term` if the term was successfully created,
    /// or `BadTerm` if the provided terms were not valid.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_logical_or(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_or(&lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs `op` rhs
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_comparison(op: super::CmpOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_comparison(op, &lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs !t
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_not(t: Self) -> MaybeTerm {
        let t: Result<Term, ErrorCode> = t.try_into();
        match t {
            Ok(t) => Self::new(Term::new_not(&t)).into(),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

impl FfiTerm {
    /// Create a new variable Term.
    ///
    /// # Errors
    /// - [`ErrorCode::NullPointer`] is returned if the string passed is null.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[allow(clippy::needless_pass_by_value)] // We do want to pass by value here, actually.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_var(s: FfiStr) -> MaybeTerm {
        match s.as_opt_str() {
            Some(s) => Self::new(Term::new_var(s)).into(),
            None => MaybeTerm::Error(ErrorCode::NullPointer),
        }
    }

    /// Create a new `cast` expression. This corresponds to the `as` operator in WGSL.
    ///
    /// `source_term` is the term to cast, and `ty` is the type to cast it to.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if the term was not valid.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms or Types is poisoned.
    /// - [`ErrorCode::NotFound`] is returned if `ty` or `source_term` do not exist in the library's collection.
    /// - [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
                [`ErrorCode::NotFound`]: crate::ErrorCode::NotFound\n\
                [`MaybeTerm::Success`]: crate::MaybeTerm::Success\n\
                [`MaybeTerm::Error`]: crate::MaybeTerm::Error\n\
                [`ErrorCode::WrongType`]: crate::ErrorCode::WrongType"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_cast(source_term: Self, ty: FfiAbcType) -> MaybeTerm {
        Types.read().map_or_else(
            |_| MaybeTerm::Error(ErrorCode::PoisonedLock),
            |types| {
                let scalar = match types.get(ty.id) {
                    Some(Some(ty_inner)) => match ty_inner.as_ref() {
                        AbcType::Scalar(s) => *s,
                        _ => return MaybeTerm::Error(ErrorCode::WrongType),
                    },
                    _ => return MaybeTerm::Error(ErrorCode::NotFound),
                };
                // We need to get the term
                let term_map = &mut *match Terms.write() {
                    Ok(terms) => terms,
                    Err(_) => return MaybeTerm::Error(ErrorCode::PoisonedLock),
                };
                let term = match term_map.get(source_term.id) {
                    Some(Some(t)) => t.clone(),
                    _ => return MaybeTerm::Error(ErrorCode::NotFound),
                };

                Self::new_with_terms(Term::new_cast(term, scalar), term_map).into()
            },
        )
    }
}

impl FfiTerm {
    /// Get the string representation of the term.
    ///
    /// If the term is invalid, <BadTerm> is returned.
    /// Note: The returned `c_string` MUST be freed by the caller, by calling `free_string` or this will lead to a memory leak.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_term_to_cstr(self) -> *mut c_char {
        let term: Result<Term, ErrorCode> = self.try_into();
        // Unsafe guarantees here:
        // 1. We have checked that the term is valid.
        // 2. We know that a rust String does not contain a null byte in the middle, so CString::new()
        // will not fail.
        match term {
            Ok(term) => {
                let my_str = term.to_string();
                // Unsafe is OK here as a Rust string can never contain a null byte in the middle.
                let c_str = unsafe { CString::new(my_str).unwrap_unchecked() };
                c_str.into_raw()
            }
            // As above, unwrap_unchecked is safe here, as CString::new cannot
            // ever return an error from an &str.
            _ => unsafe { CString::new("<BadTerm>").unwrap_unchecked().into_raw() },
        }
    }
}

/// Free a string that was allocated by the library.
///
/// # Safety
/// The caller must ensure that the pointer passed is a valid pointer to a string allocated by the library.
/// Currently, these are only strings that are returned by [`abc_term_to_cstr`].
#[cfg_attr(
    any(doc, rust_analyzer),
    doc = "\n[`abc_term_to_cstr`]: crate::FfiTerm::abc_term_to_cstr"
)]
#[allow(unused_must_use)] // We are only dropping the string, so we don't care about the result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn abc_free_string(s: *mut c_char) {
    unsafe {
        CString::from_raw(s);
    }
}

#[repr(transparent)]
/// cffi abc type wrapper.
///
/// It is best to pass this by value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FfiAbcType {
    id: usize,
}

impl FfiAbcType {
    fn new(t: AbcType) -> Result<Self, ErrorCode> {
        let mut types = Types.write().map_err(|_| ErrorCode::PoisonedLock)?;
        if let Some(id) = ReusableTypeIds
            .lock()
            .map_err(|_| ErrorCode::PoisonedLock)?
            .pop()
        {
            types[id] = Some(t.into());
            Ok(FfiAbcType { id })
        } else {
            let id = types.len();
            types.push(Some(t.into()));
            Ok(FfiAbcType { id })
        }
    }

    /// Remove `self` from the library's collection of types. This
    /// can be used to free the type when it is no longer needed.
    ///
    /// After calling this method, the type should not be used.
    ///
    /// # Returns
    /// `ErrorCode::NotFound` is returned if the type was not found in the library's collection.
    /// `ErrorCode::Success` is returned if the type was successfully removed.
    ///
    /// This can be thought of as the destructor for the type, and should be called
    /// as long as the type is no longer needed.
    ///
    /// NB: You can safely call this method even if terms or other types or terms
    /// are made up of this type. Internally, this library uses smart pointers to share
    /// ownership of types and terms, so the type will not be deallocated until all references
    /// to it are dropped.
    ///
    ///
    /// That is, calling `delete` will just remove the type from the library's collection.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_type(&self) -> ErrorCode {
        let r = self.id;
        // functional style! :D
        Types.write().map_or_else(
            |_| ErrorCode::PoisonedLock,
            |mut types| {
                types.get_mut(r).take().map_or_else(
                    || ErrorCode::NotFound,
                    |_| match ReusableTypeIds.lock() {
                        Ok(mut ids) => {
                            ids.push(r);
                            ErrorCode::Success
                        }
                        Err(_) => ErrorCode::PoisonedLock,
                    },
                )
            },
        )
    }
}

impl FfiAbcType {
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_Scalar(scalar: AbcScalar) -> MaybeAbcType {
        Self::new(AbcType::Scalar(scalar)).into()
    }

    /// Create a new struct type. Takes a list of fields, each of which is a tuple of a string and an `AbcType`.
    ///
    /// `len` specifies how many fields are passed. It MUST be at least as long as the number of fields and types
    /// passed.
    ///
    /// # Safety
    /// The caller must ensure that the `fields` and `types` pointers are not null, and that the `len`
    /// parameter is at least as long as the number of elements and fields.
    ///
    /// This method does check that the pointers are properly aligned and not null.
    /// However, the caller must ensure that the pointers hold at least `len` elements.
    /// Otherwise, behavior is undefined.
    ///
    /// # Errors
    /// - [`ErrorCode::NullPointer`] is returned if either `fields` or `types` is null.
    /// - [`ErrorCode::Alignmenterror`] is returned if the pointers are not properly aligned.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Types container is poisoned.
    /// - [`ErrorCode::BadTerm`] is returned if any of the types passed do not exist in the library's collection.
    // This makes the doc hidden from cbindgen.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::NullPointer`]: crate::ErrorCode::NullPointer\n\
             [`ErrorCode::Alignmenterror`]: crate::ErrorCode::Alignmenterror\n\
             [`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
             [`ErrorCode::BadTerm`]: crate::ErrorCode::BadTerm"
    )]
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn abc_new_Struct(
        fields: *mut FfiStr, types: *const FfiAbcType, len: usize,
    ) -> MaybeAbcType {
        // If len is 0, then we don't even have to check against null pointers. Just return an empty struct.
        if len == 0 {
            return Self::new(AbcType::Struct {
                members: Vec::new(),
            })
            .into();
        }
        // Otherwise, we need to do the safety checks.
        if fields.is_null() || types.is_null() {
            return MaybeAbcType::Error(ErrorCode::NullPointer);
        }

        // Check for proper alignment to avoid unsafety.
        if !(fields.is_aligned() && types.is_aligned()) {
            return MaybeAbcType::Error(ErrorCode::Alignmenterror);
        }

        // Get a read lock on the types.
        let Ok(ty_map) = Types.write() else {
            return MaybeAbcType::Error(ErrorCode::PoisonedLock);
        };

        // The pointers have been checked for null and alignment, so we can safely create slices from them.
        let fields_slice = std::slice::from_raw_parts(fields, len);
        let types_slice = std::slice::from_raw_parts(types, len);

        // Make the new fields hashmap.
        let mut new_fields = Vec::with_capacity(len);

        // Iterate over the fields
        for (pos, ty) in types_slice.iter().enumerate() {
            let Some(Some(matched_ty)) = ty_map.get(ty.id) else {
                return MaybeAbcType::Error(ErrorCode::BadTerm);
            };
            let field_name = String::from(match fields_slice[pos].as_opt_str() {
                Some(s) => s,
                None => return MaybeAbcType::Error(ErrorCode::NullPointer),
            });
            new_fields.push(StructField {
                name: field_name,
                ty: matched_ty.clone(),
            });
        }

        // Create the struct type.
        Self::new(AbcType::Struct {
            members: new_fields,
        })
        .into()
    }

    /// Declare a new `SizedArray` type. `size` is the number of elements in the array. This cannot be 0.
    ///
    /// # Errors
    /// - [`ErrorCode::NotFound`] is returned if the type passed does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Types container is poisoned.
    /// - [`ErrorCode::ForbiddenZero`] is returned if the size is 0.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
                [`ErrorCode::NotFound`]: crate::ErrorCode::NotFound\n\
                [`ErrorCode::ForbiddenZero`]: crate::ErrorCode::ForbiddenZero"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_SizedArray(ty: FfiAbcType, size: u32) -> MaybeAbcType {
        // Unlock the types struct
        if size == 0 {
            return MaybeAbcType::Error(ErrorCode::ForbiddenZero);
        };

        let Ok(ty_map) = Types.write() else {
            return MaybeAbcType::Error(ErrorCode::PoisonedLock);
        };

        match ty_map.get(ty.id) {
            Some(Some(ty)) => Self::new(AbcType::SizedArray {
                ty: ty.clone(),
                // we have already checked against zero here, whjic
                size: unsafe { std::num::NonZeroU32::try_from(size).unwrap_unchecked() },
            })
            .into(),
            _ => MaybeAbcType::Error(ErrorCode::NotFound),
        }
    }

    /// Declare a new Dynamic Array type of the elements of the type passed.
    ///
    /// # Errors
    /// - [`ErrorCode::NotFound`] is returned if the type passed does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Types container is poisoned.
    /// - [`ErrorCode::ForbiddenZero`] is returned if the size is 0.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
                [`ErrorCode::NotFound`]: crate::ErrorCode::NotFound\n\
                [`ErrorCode::ForbiddenZero`]: crate::ErrorCode::ForbiddenZero"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_DynamicArray(ty: FfiAbcType) -> MaybeAbcType {
        let Ok(ty_map) = Types.write() else {
            return MaybeAbcType::Error(ErrorCode::PoisonedLock);
        };

        match ty_map.get(ty.id) {
            Some(Some(ty)) => Self::new(AbcType::DynamicArray { ty: ty.clone() }).into(),
            _ => MaybeAbcType::Error(ErrorCode::NotFound),
        }
    }
}

/// A `ValidityKind` is returned by the `check_term_validity` function.
#[repr(C)]
pub enum ValidityKind {
    /// Indicates the entity is contained in the global collection.
    Contained = 0,
    /// Indicates the entity is not contained in the global collection.
    NotContained,
    /// Indicates that the corresponding global container is poisoned.
    Poisoned,
}

/// Determine if `term` is contained in the global terms map.
///
/// If term is not found or is poisoned, returns false.
///
/// If the global terms map is poisoned,
pub extern "C" fn check_term_validity(t: FfiTerm) -> ValidityKind {
    Terms.read().map_or_else(
        |_| ValidityKind::Poisoned,
        |terms| match terms.get(t.id) {
            Some(Some(_)) => ValidityKind::Contained,
            _ => ValidityKind::NotContained,
        },
    )
}
