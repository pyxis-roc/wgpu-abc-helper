// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
#![allow(non_snake_case)]

use super::{AbcType, Term};
use ffi_support::FfiStr;
use std::ffi::{c_char, CStr, CString};
use std::sync::RwLock;

use lazy_static::lazy_static;

#[allow(unused_imports)]
use crate::cbindgen_annotate;
use crate::AbcScalar;

/// Represents a possible term, or an error code.
///
/// There is no reason to construct this enum directly, as it is returned by the FFI functions.
#[repr(C)]
pub enum MaybeTerm {
    Error(ErrorCode),
    Term(FfiTerm),
}

impl From<Result<FfiTerm, ErrorCode>> for MaybeTerm {
    fn from(r: Result<FfiTerm, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeTerm::Term(t),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

/// Represents a possible AbcType, or an error code.
#[repr(C)]
pub enum MaybeAbcType {
    Error(ErrorCode),
    AbcType(FfiAbcType),
}
impl From<Result<FfiAbcType, ErrorCode>> for MaybeAbcType {
    fn from(r: Result<FfiAbcType, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeAbcType::AbcType(t),
            Err(e) => MaybeAbcType::Error(e),
        }
    }
}

// Terms are really a handle to the term they contain.
lazy_static! {
    // This is a global map of terms that are created by the C API.
    // Terms use Arc, which is not FFI-safe.
    // So instead, we have a Term which is really just a handle into this map.
    static ref Terms: RwLock<Vec<Option<Term>>> = Vec::new().into();
    static ref Types: RwLock<Vec<Option<AbcType>>> = Vec::new().into();
    static ref ReusableTermIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
    static ref ReusableTypeIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
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

    /// Indicates a type or term that does not exist in the library's collection.
    ///
    /// This means the term was either already deleted or never created.
    NotFound = 4,

    /// Indicate a null pointer was passed.
    NullPointer = 5,
}

impl<'a> super::CmpOp {
    #[no_mangle]
    pub extern "C" fn negate(&self) -> Self {
        self.negation()
    }
}

impl super::CmpOp {
    #[no_mangle]
    pub extern "C" fn as_ConstraintOp(self) -> super::ConstraintOp {
        self.into()
    }
}

impl super::ConstraintOp {
    /// Conversion method to convert a ConstraintOp to a CmpOp
    #[no_mangle]
    pub extern "C" fn from_CmpOp(value: super::CmpOp) -> Self {
        value.into()
    }
}

cbindgen_annotate! {
""
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
/// For ffi bindings, term does not contain the actual term data, but instead a handle to it.
///
/// This handle corresponds to an index in the `Terms` vector, a static global variable in this library.
/// This is because `Term` has members that are not FFI-Safe.
pub struct FfiTerm {
    id: usize,
}
}

impl FfiTerm {
    /// Used internally to manage terms.
    /// Reuses old IDs
    fn new(term: Term) -> Result<Self, ErrorCode> {
        // We need to get the reader of the terms
        let mut terms = Terms.write().map_err(|_| ErrorCode::PoisonedLock)?;
        if let Some(id) = ReusableTermIds
            .lock()
            .map_err(|_| ErrorCode::PoisonedLock)?
            .pop()
        {
            terms[id] = Some(term);
            Ok(FfiTerm { id })
        } else {
            let id = terms.len();
            terms.push(Some(term));
            return Ok(FfiTerm { id });
        }
    }

    pub extern "C" fn delete(&self) -> ErrorCode {
        let r = self.id;
        let mut terms = Terms.write().unwrap();
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
    #[no_mangle]
    pub extern "C" fn new_unit_pred(p: Self) -> MaybeTerm {
        let p: Result<Term, ErrorCode> = p.try_into();
        match p {
            Ok(t) => Self::new(Term::new_unit_pred(t)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }
    /// Create a Term holding the `true` predicate
    #[no_mangle]
    pub extern "C" fn new_literal_true() -> MaybeTerm {
        Self::new(Term::new_literal_true()).into()
    }

    /// Create a Term holding the `false` predicate
    #[no_mangle]
    pub extern "C" fn new_literal_false() -> MaybeTerm {
        Self::new(Term::new_literal_false()).into()
    }

    /// Creates lhs && rhs
    #[no_mangle]
    pub extern "C" fn new_logical_and(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_and(lhs, rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs || rhs
    ///
    /// Returns a `MaybeTerm`, which is either a `Term` if the term was successfully created,
    /// or `BadTerm` if the provided terms were not valid.
    #[no_mangle]
    pub extern "C" fn new_logical_or(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_logical_or(lhs, rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs `op` rhs
    #[no_mangle]
    pub extern "C" fn new_comparison(op: super::CmpOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<Term, ErrorCode> = lhs.try_into();
        let rhs: Result<Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(Term::new_comparison(op, lhs, rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs !t
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[no_mangle]
    pub extern "C" fn new_not(t: Self) -> MaybeTerm {
        let t: Result<Term, ErrorCode> = t.try_into();
        match t {
            Ok(t) => Self::new(Term::new_not(t)).into(),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

impl FfiTerm {
    /// Get the string representation of the term.
    ///
    /// If the term is invalid, <BadTerm> is returned.
    /// Note: The returned c_string MUST be freed by the caller, by calling `free_string` or this will lead to a memory leak.
    #[no_mangle]
    pub extern "C" fn to_c_str(self) -> *const c_char {
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

#[allow(unused_must_use)] // We are only dropping the string, so we don't care about the result.
#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        CString::from_raw(s);
    }
}

#[repr(C)]
/// cffi abc type wrapper.
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
            types[id] = Some(t);
            Ok(FfiAbcType { id })
        } else {
            let id = types.len();
            types.push(Some(t));
            Ok(FfiAbcType { id })
        }
    }

    /// Remove the AbcType from the library's collection of types. This
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
    pub extern "C" fn delete(&self) -> ErrorCode {
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

/* */
impl FfiAbcType {
    #[no_mangle]
    pub extern "C" fn new_Scalar(scalar: AbcScalar) -> MaybeAbcType {
        Self::new(AbcType::Scalar(scalar)).into()
    }
    /*
    /// Create a new struct type. Takes a list of fields, each of which is a tuple of a string and an AbcType.
    ///
    /// `len` specifies how many fields are passed. It MUST be at least as long a the number of fields and types
    /// passed.
    ///
    /// # Safety
    /// The caller must ensure that the `fields` and `types` pointers are valid, and that the `len` parameter
    /// is correct.
    // #[no_mangle]
    // pub extern "C" fn new_Struct(
    //     fields: *const FfiStr,
    //     types: *const FfiAbcType,
    //     len: usize,
    // ) -> MaybeAbcType {
    //     if fields.is_null() || types.is_null() {
    //         return MaybeAbcType::Error(ErrorCode::NullPointer);
    //     }
    //     let fields: Result<Vec<(String, AbcType)>, ErrorCode> = fields.try_into();
    //     match fields {
    //         Ok(fields) => Self::new(AbcType::Struct(fields)).into(),
    //         Err(e) => MaybeAbcType::Error(e),
    //     }
    // }
     */
}
