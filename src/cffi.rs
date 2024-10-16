// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
#![allow(non_snake_case)]

use std::ffi::{c_char, CString};
use std::sync::RwLock;

use lazy_static::lazy_static;

// use ffi_support::FfiStr;

/// cbindgen:derive-const-casts
#[repr(C)]
pub enum MaybeTerm {
    Error(ErrorCode),
    Term(Term),
}

impl From<Result<Term, ErrorCode>> for MaybeTerm {
    fn from(r: Result<Term, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeTerm::Term(t),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

// Terms are really a handle to the term they contain.
lazy_static! {
    // This is a global map of terms that are created by the C API.
    // Terms use Arc, which is not FFI-safe.
    // So instead, we have a Term which is really just a handle into this map.
    static ref Terms: RwLock<Vec<Option<super::Term>>> = Vec::new().into();
    static ref Types: RwLock<Vec<Option<super::AbcType>>> = Vec::new().into();
    static ref ReusableIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
}
#[repr(C)]
pub enum ErrorCode {
    Success = 0,
    Panic = 1,
    /// Indicates a term that could not be found in the term arena.
    BadTerm = 2,
    PoisonedLock = 3,
}

impl<'a> super::CmpOp {
    #[no_mangle]
    pub extern "C" fn negate(&self) -> Self {
        self.negation()
    }
}

impl super::CmpOp {
    pub extern "C" fn as_ConstraintOp(self) -> super::ConstraintOp {
        self.into()
    }
}

impl super::ConstraintOp {
    #[no_mangle]
    pub extern "C" fn from_CmpOp(value: super::CmpOp) -> Self {
        value.into()
    }
}

#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
pub struct Term {
    id: usize,
}

impl Term {
    /// Used internally to manage terms.
    /// Reuses old IDs
    fn new(term: super::Term) -> Result<Self, ErrorCode> {
        // We need to get the reader of the terms
        let mut terms = Terms.write().map_err(|_| ErrorCode::PoisonedLock)?;
        if let Some(id) = ReusableIds
            .lock()
            .map_err(|_| ErrorCode::PoisonedLock)?
            .pop()
        {
            terms[id] = Some(term);
            Ok(Term { id })
        } else {
            let id = terms.len();
            terms.push(Some(term));
            return Ok(Term { id });
        }
    }

    pub extern "C" fn delete(&self) -> ErrorCode {
        let r = self.id;
        let mut terms = Terms.write().unwrap();
        terms[r] = None;
        match ReusableIds.lock() {
            Ok(mut ids) => {
                ids.push(r);
                ErrorCode::Success
            }
            Err(_) => ErrorCode::PoisonedLock,
        }
    }
}

impl TryFrom<Term> for super::Term {
    type Error = ErrorCode;
    fn try_from(t: Term) -> Result<Self, Self::Error> {
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
impl Term {
    /// Create a new unit predicate
    #[no_mangle]
    pub extern "C" fn new_unit_pred(p: Self) -> MaybeTerm {
        let p: Result<super::Term, ErrorCode> = p.try_into();
        match p {
            Ok(t) => Self::new(super::Term::new_unit_pred(t)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }
    /// Create a Term holding the `true` predicate
    #[no_mangle]
    pub extern "C" fn new_literal_true() -> MaybeTerm {
        Self::new(super::Term::new_literal_true()).into()
    }

    /// Create a Term holding the `false` predicate
    #[no_mangle]
    pub extern "C" fn new_literal_false() -> MaybeTerm {
        Self::new(super::Term::new_literal_false()).into()
    }

    /// Creates lhs && rhs
    #[no_mangle]
    pub extern "C" fn new_logical_and(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<super::Term, ErrorCode> = lhs.try_into();
        let rhs: Result<super::Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(super::Term::new_logical_and(lhs, rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs || rhs
    ///
    /// Returns a `MaybeTerm`, which is either a `Term` if the term was successfully created,
    /// or `BadTerm` if the provided terms were not valid.
    #[no_mangle]
    pub extern "C" fn new_logical_or(lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<super::Term, ErrorCode> = lhs.try_into();
        let rhs: Result<super::Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(super::Term::new_logical_or(lhs, rhs)).into(),
            _ => MaybeTerm::Error(ErrorCode::BadTerm),
        }
    }

    /// Constructs lhs `op` rhs
    #[no_mangle]
    pub extern "C" fn new_comparison(op: super::CmpOp, lhs: Self, rhs: Self) -> MaybeTerm {
        let lhs: Result<super::Term, ErrorCode> = lhs.try_into();
        let rhs: Result<super::Term, ErrorCode> = rhs.try_into();
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => Self::new(super::Term::new_comparison(op, lhs, rhs)).into(),
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
        let t: Result<super::Term, ErrorCode> = t.try_into();
        match t {
            Ok(t) => Self::new(super::Term::new_not(t)).into(),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

impl Term {
    /// Get the string representation of the term.
    /// Note: The returned c_string MUST be freed by the caller, or this will lead to a memory leak.
    #[no_mangle]
    pub extern "C" fn to_c_str(self) -> *mut c_char {
        let term: Result<super::Term, ErrorCode> = self.try_into();
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
