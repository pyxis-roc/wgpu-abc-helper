// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
#![allow(
    non_snake_case, // We use camelCase for constructors of enum variants.
    clippy::default_trait_access // Default::default() is just so much nicer.
)]

use super::{AbcType, CmpOp, Term};
use ffi_support::FfiStr;
use std::ffi::{c_char, CString};
use std::sync::RwLock;

use lazy_static::lazy_static;

#[allow(unused_imports)]
use crate::cbindgen_annotate;
use crate::{AbcScalar, Literal, StructField};

/// Represents a possible term, or an error code.
///
/// There is no reason to construct this enum directly; it meant to be returned by the FFI functions.
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
    #[no_mangle]
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
    /// Indicates an improper value
    ValueError = 2,
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
    #[no_mangle]
    pub extern "C" fn abc_negate(&self) -> Self {
        self.negation()
    }
}

impl super::CmpOp {
    #[no_mangle]
    pub extern "C" fn abc_as_ConstraintOp(self) -> super::ConstraintOp {
        self.into()
    }
}

impl super::ConstraintOp {
    /// Conversion method to convert a `ConstraintOp` to a `CmpOp`
    #[no_mangle]
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
    /// Convert the `FfiTerm` into a `Term`.
    #[inline]
    #[allow(dead_code)] // Some helper methods may not be used.
    fn into_term(self) -> Result<Term, ErrorCode> {
        self.try_into()
    }

    /// Convert the `FfiTerm` into a `Term` using the provided term map.
    /// This is used when the caller already has a lock on `Terms`.
    fn into_term_with_terms(self, term_map: &Vec<Option<Term>>) -> Result<Term, ErrorCode> {
        term_map.get(self.id).map_or_else(
            || Err(ErrorCode::NotFound),
            |t| {
                t.clone()
                    .map_or_else(|| Err(ErrorCode::NotFound), |t| Ok(t))
            },
        )
    }

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
            _ => Err(ErrorCode::NotFound),
        }
    }
}

/*
Implementation of predicate constructors for term
*/
impl FfiTerm {
    /// Create a new unit predicate
    #[no_mangle]
    pub extern "C" fn abc_new_unit_pred(p: Self) -> MaybeTerm {
        let p: Result<Term, ErrorCode> = p.try_into();
        match p {
            Ok(t) => Self::new(Term::new_unit_pred(&t)).into(),
            _ => MaybeTerm::Error(ErrorCode::NotFound),
        }
    }
    /// Create a Term holding the `true` predicate
    #[no_mangle]
    pub extern "C" fn abc_new_literal_true() -> MaybeTerm {
        Self::new(Term::new_literal_true()).into()
    }

    /// Create a Term holding the `false` predicate
    #[no_mangle]
    pub extern "C" fn abc_new_literal_false() -> MaybeTerm {
        Self::new(Term::new_literal_false()).into()
    }

    /// Creates lhs && rhs
    #[no_mangle]
    pub extern "C" fn abc_new_logical_and(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_and(&lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::NotFound),
        }
    }

    /// Constructs lhs || rhs
    ///
    /// Returns a `MaybeTerm`, which is either a `Term` if the term was successfully created,
    /// or `NotFound` if the provided terms were not valid.
    #[no_mangle]
    pub extern "C" fn abc_new_logical_or(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_or(&lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::NotFound),
        }
    }

    /// Constructs lhs `op` rhs
    #[no_mangle]
    pub extern "C" fn abc_new_comparison(op: super::CmpOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_comparison(op, &lhs, &rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::NotFound),
        }
    }

    /// Constructs !t
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[no_mangle]
    pub extern "C" fn abc_new_not(t: Self) -> MaybeTerm {
        let t: Result<Term, ErrorCode> = t.try_into();
        match t {
            Ok(t) => Self::new(Term::new_not(&t)).into(),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

/*
Implementation of term constructors for expressions
 */
impl FfiTerm {
    /// Helper method that resolves `ty` to the inner type, ensuring that it is an `AbcScalar` variant.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Types container is poisoned.
    /// - [`ErrorCode::NotFound`] is returned if the type does not exist in the library's collection.
    /// - [`ErrorCode::WrongType`] is returned if the type is not a scalar type.
    fn get_scalar_type(ty: FfiAbcType) -> Result<AbcScalar, ErrorCode> {
        let types = Types.read().map_err(|_| ErrorCode::PoisonedLock)?;
        let scalar = match types.get(ty.id) {
            Some(Some(ty_inner)) => match ty_inner.as_ref() {
                AbcType::Scalar(s) => *s,
                _ => return Err(ErrorCode::WrongType),
            },
            _ => return Err(ErrorCode::NotFound),
        };
        Ok(scalar)
    }
    /// Create a new variable Term.
    ///
    /// # Errors
    /// - [`ErrorCode::NullPointer`] is returned if the string passed is null.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[allow(clippy::needless_pass_by_value)] // We do want to pass by value here, actually.
    #[no_mangle]
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
    #[no_mangle]
    pub extern "C" fn abc_new_cast(source_term: Self, ty: FfiAbcType) -> MaybeTerm {
        let resolved_type = match Self::get_scalar_type(ty) {
            Ok(s) => s,
            Err(e) => return MaybeTerm::Error(e),
        };

        let term_map = &mut *match Terms.write() {
            Ok(terms) => terms,
            Err(_) => return MaybeTerm::Error(ErrorCode::PoisonedLock),
        };
        let term = match term_map.get(source_term.id) {
            Some(Some(t)) => t.clone(),
            _ => return MaybeTerm::Error(ErrorCode::NotFound),
        };

        Self::new_with_terms(Term::new_cast(term, resolved_type), term_map).into()
    }

    /// Create a new comparison term, e.g. `x > y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// - [`ErrorCode::NotFound`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    #[no_mangle]
    pub extern "C" fn abc_new_cmp_term(op: CmpOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs_term = match lhs.try_into() {
            Ok(term) => term,
            Err(e) => return MaybeTerm::Error(e),
        };
        let rhs_term = match rhs.try_into() {
            Ok(term) => term,
            Err(e) => return MaybeTerm::Error(e),
        };
        Self::new(Term::new_cmp_op(op, &lhs_term, &rhs_term)).into()
    }

    /// Create a new index access term, e.g. `x[y]`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `base` or `index` do not exist in the library's collection.
    #[no_mangle]
    pub extern "C" fn abc_new_index_access(base: Self, index: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let base_term = match base.into_term_with_terms(terms) {
            Ok(term) => term,
            Err(e) => return MaybeTerm::Error(e),
        };
        let index_term = match index.into_term_with_terms(terms) {
            Ok(term) => term,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_index_access(&base_term, &index_term), terms).into()
    }

    /// Create a new struct access term, e.g. `x.y`.
    ///
    /// # Arguments
    /// - `base`: The base term whose field is being accessed
    /// - `field`: The name of the field being accessed.
    /// - `ty`: The type of the struct being accessed. This is needed for term validation.
    /// - `field_idx`: The index of the field in the structure being accessed.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term or type could not be found.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `base` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms or Types container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_struct_access(
        base: Self, field: FfiStr, ty: FfiAbcType, field_idx: usize,
    ) -> MaybeTerm {
        let ty = match ty.try_into() {
            Ok(ty) => ty,
            Err(e) => return MaybeTerm::Error(e),
        };

        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let base = match base.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let Some(field) = field.into_opt_string() else {
            return MaybeTerm::Error(ErrorCode::NullPointer);
        };

        Self::new_with_terms(Term::new_struct_access(&base, field, ty, field_idx), terms).into()
    }

    /// Create a new splat term, e.g. `vec3(x)`.
    ///
    /// A `splat` is just shorthand for a vector of size `size` where each element is `term`.
    ///
    /// # Arguments
    /// - `term`: The term to splat.
    /// - `size`: The number of elements in the vector. Must be between 2 and 4  (inclusive).
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::ValueError`] is returned if `size` is not between 2 and 4.
    #[no_mangle]
    pub extern "C" fn abc_new_splat(term: Self, size: u32) -> MaybeTerm {
        if size < 2 || size > 4 {
            return MaybeTerm::Error(ErrorCode::ValueError);
        }
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_splat(term, size), terms).into()
    }

    /// Create a new literal term.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a new `Literal` variant of `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_literal(lit: Literal) -> MaybeTerm {
        Self::new(Term::new_literal(lit)).into()
    }

    /// Create a binary operation term, e.g. `x + y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_binary_op(op: super::BinaryOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let lhs = match lhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match rhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_binary_op(op, &lhs, &rhs), terms).into()
    }

    /// Create a new unary operation term, e.g. `-x`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_unary_op(op: super::UnaryOp, term: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_unary_op(op, &term), terms).into()
    }

    /// Create a new term corresponding to wgsl's [`max`](https://www.w3.org/TR/WGSL/#max-float-builtin) bulitin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_max(lhs: Self, rhs: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let lhs = match lhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match rhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_max(&lhs, &rhs), terms).into()
    }

    /// Create a new term corresponding to wgsl's [`min`](https://www.w3.org/TR/WGSL/#min-float-builtin) builtin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_min(lhs: Self, rhs: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let lhs = match lhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match rhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_min(&lhs, &rhs), terms).into()
    }

    /// Create a new term akin to wgsl's [`select`](https://www.w3.org/TR/WGSL/#select-builtin) bulitin.
    ///
    /// ### Note: In this method, the condition is the first argument, while it is the last argument in the WGSL builtin.
    ///
    ///
    /// # Arguments
    /// - `iftrue`: The expression this term evaluates to if `predicate` is true
    /// - `iffalse`: The expression this term evaluates to if `predicate` is false
    /// - `Predicate`: The term that determines the resolution of `iftrue` or `iffalse`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `lhs`, `m`, or `rhs` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_select(iftrue: Self, iffalse: Self, predicate: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let iftrue = match iftrue.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let iffalse = match iffalse.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let predicate = match predicate.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_select(&iftrue, &iffalse, &predicate), terms).into()
    }

    /// Helper method for creating new vector terms in a DRY fashion. Also works for matrices.
    fn new_vec_helper(
        components: &[Self], ty: FfiAbcType, is_matrix: bool,
    ) -> Result<Self, ErrorCode> {
        let resolved_type = Self::get_scalar_type(ty)?;

        let Ok(ref mut terms) = Terms.write() else {
            return Err(ErrorCode::PoisonedLock);
        };

        let mut resolved_components: Vec<Term> = Vec::with_capacity(components.len());
        for component in components {
            resolved_components.push(component.into_term_with_terms(terms)?);
        }

        let new_term = if is_matrix {
            Term::new_vector(&resolved_components, resolved_type)
        } else {
            Term::new_vector(&resolved_components, resolved_type)
        };

        Self::new_with_terms(new_term, terms).map_err(|_| ErrorCode::PoisonedLock)
    }

    /// Create a new term corresponding to wgsl's `vec2` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either of the terms or the type does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_vec2(term_0: Self, term_1: Self, ty: FfiAbcType) -> MaybeTerm {
        Self::new_vec_helper(&[term_0, term_1], ty, false).into()
    }

    /// Create a new term corresponding to wgsl's `vec3` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `term_3`: The third term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either of the terms or the type does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_vec3(
        term_0: Self, term_1: Self, term_2: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[term_0, term_1, term_2], ty, false).into()
    }

    /// Create a new term corresponding to wgsl's `vec4` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `term_2`: The third term in the vector
    /// - `term_3`: The fourth term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either of the terms or the type does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_vec4(
        term_0: Self, term_1: Self, term_2: Self, term_3: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[term_0, term_1, term_2, term_3], ty, false).into()
    }

    /// Create a new `array_length` term corresponding to the `arrayLength` operator in WGSL.
    /// Note that `wgsl` only defines this method for dynamically sized arrays. This method is
    /// not valid for fixed-sized arrays, matrices, or vectors.
    ///
    /// ### Notes
    /// Do not use this term to add a constraint on the length of an array. Instead, invoke the solver's `mark_length` method.
    /// This method is only meant to be used in the case that `arrayLength` is spelled in wgsl source.
    ///
    /// # Arguments
    /// - `term`: The array term that this expression is being applied to.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_array_length(term: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::make_array_length(&term), terms).into()
    }

    /// Create a new `store` term.
    ///
    /// There is no explicit wgsl method that this corresponds to. However, it must be used when writing to array in
    /// order to respect SSA requirements. This method creates a new term where each array element is the same as the
    /// original array, except for the element at `index`, which is `value`.
    ///
    /// # Arguments
    /// - `term`: The array term that is being written to.
    /// - `index`: The index of the array that is being written to.
    /// - `value`: The value that is being written to the array.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_store(term: Self, index: Self, value: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let index = match index.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let value = match value.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_store(term, index, value), terms).into()
    }

    /// Create a new struct store term.
    ///
    /// There is no explicit wgsl method that this corresponds to. However, it must be used when writing to a struct in
    /// order to respect SSA requirements. This method creates a new term where each field is the same as the
    /// original struct, except for the field at `field_idx`, which is `value`.
    ///
    /// # Arguments
    /// - `term`: The struct term that is being written to.
    /// - `field_idx`: The index of the field that is being written to.
    /// - `value`: The value that is being written to the struct.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_struct_store(term: Self, field_idx: usize, value: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let value = match value.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_struct_store(term, field_idx, value), terms).into()
    }

    /// Create a new absolute value term corresponding to the [`abs`](https://www.w3.org/TR/WGSL/#abs-float-builtin) operator in WGSL.
    ///
    /// # Arguments
    /// - `term`: The term that is being passed to the `abs` operator.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if `term` does not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_abs(term: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let term = match term.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_abs(&term), terms).into()
    }

    /// Create a new `pow` term corresponding to the [`pow`](https://www.w3.org/TR/WGSL/#pow-builtin) builtin in WGSL.
    ///
    /// # Arguments
    /// - `base`: The base of the power operation.
    /// - `exponent`: The exponent of the power operation.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `base` or `exponent` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_pow(base: Self, exponent: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let base = match base.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let exponent = match exponent.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_pow(&base, &exponent), terms).into()
    }

    /// Create a new term corresponding to wgsl's [`dot`](https://www.w3.org/TR/WGSL/#dot-builtin) builtin.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the dot product.
    /// - `rhs`: The right-hand side of the dot product.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    #[no_mangle]
    pub extern "C" fn abc_new_dot(lhs: Self, rhs: Self) -> MaybeTerm {
        let Ok(ref mut terms) = Terms.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };

        let lhs = match lhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match rhs.into_term_with_terms(terms) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        Self::new_with_terms(Term::new_dot(&lhs, &rhs), terms).into()
    }

    /// Create a new `mat2x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec2`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec2`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat2x2(row_0: Self, row_1: Self, ty: FfiAbcType) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1], ty, true).into()
    }

    /// Create a new `mat2x3` term
    ///
    /// The components of the matrix must correspond to `vec3` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat2x3(row_0: Self, row_1: Self, ty: FfiAbcType) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1], ty, true).into()
    }

    /// Create a new `mat2x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat2x4(row_0: Self, row_1: Self, ty: FfiAbcType) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1], ty, true).into()
    }

    /// Create a new `mat3x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec2`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec2`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat3x2(
        row_0: Self, row_1: Self, row_2: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
    }

    /// Create a new `mat3x3` term
    ///
    /// The components of the matrix must correspond to `vec3` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat3x3(
        row_0: Self, row_1: Self, row_2: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
    }

    /// Create a new `mat3x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat3x4(
        row_0: Self, row_1: Self, row_2: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
    }

    /// Create a new `mat4x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat4x2(
        row_0: Self, row_1: Self, row_2: Self, row_3: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true).into()
    }

    /// Create a new `mat4x3` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat4x3(
        row_0: Self, row_1: Self, row_2: Self, row_3: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true).into()
    }

    /// Create a new `mat4x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// [`ErrorCode::NotFound`] is returned if any of the rows or the type do not exist in the library's collection.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms container is poisoned.
    /// [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[no_mangle]
    pub extern "C" fn abc_new_mat4x4(
        row_0: Self, row_1: Self, row_2: Self, row_3: Self, ty: FfiAbcType,
    ) -> MaybeTerm {
        Self::new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true).into()
    }
}

impl FfiTerm {
    /// Get the string representation of the term.
    ///
    /// If the term is invalid, <NotFound> is returned.
    /// Note: The returned `c_string` MUST be freed by the caller, by calling `free_string` or this will lead to a memory leak.
    #[no_mangle]
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
            _ => unsafe { CString::new("<NotFound>").unwrap_unchecked().into_raw() },
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
#[no_mangle]
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
    #[no_mangle]
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
    #[no_mangle]
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
    /// - [`ErrorCode::NotFound`] is returned if any of the types passed do not exist in the library's collection.
    // This makes the doc hidden from cbindgen.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::NullPointer`]: crate::ErrorCode::NullPointer\n\
             [`ErrorCode::Alignmenterror`]: crate::ErrorCode::Alignmenterror\n\
             [`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
             [`ErrorCode::NotFound`]: crate::ErrorCode::NotFound"
    )]
    #[no_mangle]
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
                return MaybeAbcType::Error(ErrorCode::NotFound);
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
    #[no_mangle]
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
    #[no_mangle]
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

impl TryFrom<FfiAbcType> for super::Handle<AbcType> {
    type Error = ErrorCode;
    fn try_from(t: FfiAbcType) -> Result<Self, Self::Error> {
        // Get the map...
        let types = Types.read().map_err(|_| ErrorCode::PoisonedLock)?;
        match types.get(t.id) {
            Some(Some(t)) => Ok(t.clone()),
            _ => Err(ErrorCode::NotFound),
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

/// Determine if `term` is contained in the global terms map, returning `ValidityKind`.
pub extern "C" fn check_term_validity(t: FfiTerm) -> ValidityKind {
    Terms.read().map_or_else(
        |_| ValidityKind::Poisoned,
        |terms| match terms.get(t.id) {
            Some(Some(_)) => ValidityKind::Contained,
            _ => ValidityKind::NotContained,
        },
    )
}
