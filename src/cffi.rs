// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
#![allow(
    non_snake_case, // We use camelCase for constructors of enum variants.
    clippy::default_trait_access // Default::default() is just so much nicer.
)]

use super::{AbcType, CmpOp, Term};
use ffi_support::FfiStr;
use lazy_static::lazy_static;
use std::ffi::{c_char, CString};
use std::sync::RwLock;

#[allow(unused_imports)]
use crate::cbindgen_annotate;
use crate::{AbcScalar, Literal, StructField};

/*************************
 *   Macro Definitions   *
 ************************/

/// Expands to a reference to the context object.
///
/// # Arguments
/// - The first argument is the context object to get.
/// - The second argument is an identifier that will hold the `RwLockWriteGuard`.
/// - The third argument is the identifier that will hold the mutable reference to the `ContextInner`.
///
/// This is a macro instead of a function as the RwLockWriteGuard must be in the same scope as the mutable reference to the `ContextInner`.
/// This is also just boilerplate that is repeated in many functions in order to access the `ContextInner` from the context handle.
macro_rules! get_context_mut {
    (@maybe_term, $context_obj:expr, $container_var:ident, $result_var:ident $(,)?) => {
        let Ok(mut $container_var) = FfiContexts.write() else {
            return MaybeTerm::Error(ErrorCode::PoisonedLock);
        };
        let Some(Some($result_var)) = $container_var.get_mut($context_obj.id) else {
            return MaybeTerm::Error(ErrorCode::InvalidContext);
        };
    };

    (@maybe_type, $context_obj:expr, $container_var:ident, $result_var:ident $(,)?) => {
        let Ok(mut $container_var) = FfiContexts.write() else {
            return MaybeAbcType::Error(ErrorCode::PoisonedLock);
        };
        let Some(Some($result_var)) = $container_var.get_mut($context_obj.id) else {
            return MaybeAbcType::Error(ErrorCode::InvalidContext);
        };
    };

    (@plain, $context_obj:expr, $container_var:ident, $result_var:ident $(,)?) => {
        let Ok(mut $container_var) = FfiContexts.write() else {
            return ErrorCode::PoisonedLock;
        };
        let Some(Some($result_var)) = $container_var.get_mut($context_obj.id) else {
            return ErrorCode::InvalidContext;
        };
    };

    (@result, $context_obj:expr, $container_var:ident, $result_var:ident $(,)?) => {
        let Ok(mut $container_var) = FfiContexts.write() else {
            return Err(ErrorCode::PoisonedLock);
        };
        let Some(Some($result_var)) = $container_var.get_mut($context_obj.id) else {
            return Err(ErrorCode::InvalidContext);
        };
    };
}

/*************************
 * End Macro Definitions *
 ************************/

// Global variables that are required for the ffi functions.

lazy_static! {
    // This is a global collection of contexts that are created by the ffi API.
    // This indirection protects contexts from improper interference over the ffi boundary,
    // as the actual ContextInner object is never exposed.
    static ref FfiContexts: RwLock<Vec<Option<ContextInner>>> = Vec::new().into();
    static ref ReusableContextIds: std::sync::Mutex<Vec<usize>> = Vec::new().into();
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

    /// Indicates a type that does not exist in the context.
    ///
    /// This means the term was either already deleted or never created.
    InvalidTerm = 4,

    /// Indicates that a type does not exist in the context.
    ///
    /// This means the type was either already deleted or never created.
    InvalidType = 5,

    /// Indicates that a summary does not exist in the context.
    ///
    /// This means the summary was either already deleted or never created.
    InvalidSummary = 6,

    /// Indicate a null pointer was passed.
    NullPointer = 7,

    /// Indicates a passed pointer was not properly aligned.
    Alignmenterror = 8,

    /// Indicates a forbidden zero value passed as an arugment.
    ForbiddenZero = 9,

    /// Indicates that the library is at maximum capacity.
    CapacityExceeded = 10,

    /// Indicates that the wrong `AbcType` was passed.
    WrongType = 11,

    /// Indicates that the context does not exist in the global context map.
    InvalidContext = 12,
}

/// Represents a possible term, or an error code.
///
/// There is no reason to construct this enum directly; it is meant to be returned by the FFI functions.
#[repr(C)]
pub enum MaybeTerm {
    Error(ErrorCode),
    Success(FfiTerm),
}

#[repr(C)]
pub enum MaybeContext {
    Error(ErrorCode),
    Success(Context),
}

impl From<MaybeContext> for Result<Context, ErrorCode> {
    fn from(m: MaybeContext) -> Self {
        match m {
            MaybeContext::Error(e) => Err(e),
            MaybeContext::Success(c) => Ok(c),
        }
    }
}

impl From<Result<Context, ErrorCode>> for MaybeContext {
    fn from(r: Result<Context, ErrorCode>) -> Self {
        match r {
            Ok(c) => MaybeContext::Success(c),
            Err(e) => MaybeContext::Error(e),
        }
    }
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

cbindgen_annotate! { "cbindgen:derive-eq"
/// `FfiSummary` are not created by the user. They are returned only by the library by the `endSummary()`
/// method of the helper. NOTE: The alpha version of this library's cffi module does not include the helper.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct FfiSummary {
    id: usize,
}
}

pub enum MaybeSummary {
    Error(ErrorCode),
    Success(FfiSummary),
}

cbindgen_annotate! { "cbindgen:derive-eq"
/// A context refers to a collection of terms, types, and summaries for a single compilation unit.
///
/// You must use different contexts for different compilation units.
/// When you are done with a context, it is important to call `abc_free_context` to free the resources.
#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]

pub struct Context {
    pub(crate) id: usize,
}}
/// Implements the methods for Context which exist in the FFI api.
impl Context {
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_context() -> MaybeContext {
        let (Ok(mut contexts), Ok(mut ids)) = (FfiContexts.write(), ReusableContextIds.lock())
        else {
            return MaybeContext::Error(ErrorCode::PoisonedLock);
        };
        let id = match ids.pop() {
            Some(id) => id,
            None => contexts.len(),
        };

        let new = Context { id };
        contexts.push(Some(ContextInner {
            terms: Vec::new(),
            types: Vec::new(),
            summaries: Vec::new(),
            reusable_term_ids: Vec::new(),
            reusable_type_ids: Vec::new(),
            reusable_summary_ids: Vec::new(),
        }));
        MaybeContext::Success(new)
    }
    /// Free the context from the global context map.
    ///
    /// Freeing a context will deallocate all of the memory that has been
    /// associated with it and allow its id to be reused. Attempting to use
    /// the `Context` after it has been freed may modify a different context
    /// if its id has been reused.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_context(context: Context) -> ErrorCode {
        let Ok(mut contexts) = FfiContexts.write() else {
            return ErrorCode::PoisonedLock;
        };
        match contexts.get_mut(context.id) {
            Some(t @ Some(_)) => {
                *t = None;
                let Ok(mut ids) = ReusableContextIds.lock() else {
                    return ErrorCode::PoisonedLock;
                };
                ids.push(context.id);
                ErrorCode::Success
            }
            _ => ErrorCode::InvalidContext,
        }
    }

    /// Free the term from the term map.
    ///
    /// Freeing a term will remove it from the global terms map and allow its id
    /// to be reused. This will *not* break any terms that refer to this term.
    /// For example, if `x` is the term being freed, then a term for
    /// the expression `x + 1` will continue to work as expected.
    ///
    /// # Returns
    /// `ErrorCode::Success` if the term was successfully removed.
    /// `ErrorCode::PoisonedLock` if the lock on the global terms map is poisoned.
    /// `ErrorCode::InvalidContext` if the provided context does not exist in the global context map.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_term(self, term: FfiTerm) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.free_term_impl(term)
    }

    /// Free the term from the term map.
    ///
    /// Freeing a type will remove it from the type map and allow its id to be
    /// reused. This will *not* break any terms or summaries that have used this
    /// type.
    ///
    /// # Returns
    /// `ErrorCode::Success` if the term was successfully removed.
    /// `ErrorCode::PoisonedLock` if the lock on the global context is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_type(self, ty: FfiAbcType) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.free_type_impl(ty)
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
    /// # Returns
    /// `ErrorCode::Success` if the summary was successfully removed, otherwise an error described below.
    ///
    /// # Errors
    /// `ErrorCode::SummaryNotFound` is returned if the summary was not found in the library's collection.
    /// `ErrorCode::PoisonedLock` is returned if the lock on the contexts map is poisoned.
    /// `ErrorCode::InvalidContext` is returned if the provided context does not exist in the global context map.
    ///
    /// This can be thought of as the destructor for the summary, and should be called
    /// as long as the summary is no longer needed.
    ///
    /// That is, calling `delete` will just remove the summary from the library's collection.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_summary(self, summary: FfiSummary) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        match context.summaries.get_mut(summary.id) {
            Some(t @ Some(_)) => {
                *t = None;
                context.reusable_summary_ids.push(summary.id);
                ErrorCode::Success
            }
            _ => ErrorCode::InvalidContext,
        }
    }
}

/// This is the internal representation of the context.
///
/// Context objects are never exposed over the ffi boundary, and are only accessed via the `Context` handle.
struct ContextInner {
    terms: Vec<Option<Term>>,
    types: Vec<Option<super::Handle<AbcType>>>,
    summaries: Vec<Option<super::Handle<super::Summary>>>,
    reusable_term_ids: Vec<usize>,
    reusable_type_ids: Vec<usize>,
    reusable_summary_ids: Vec<usize>,
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

cbindgen_annotate!("cbindgen:derive-eq"
#[repr(C)]
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Hash)]
/// For ffi bindings, term does not contain the actual term data, but instead a handle to it.
///
/// This handle corresponds to an index in the `Terms` vector, a static global variable in this library.
/// This is because `Term` has members that are not FFI-Safe.
pub struct FfiTerm {
    id: usize,
}
);

impl ContextInner {
    /// Implementation of `new_summary` for the `ContextInner` struct.
    fn new_summary_impl<T: Into<super::Handle<super::Summary>>>(
        &mut self, s: T,
    ) -> Result<FfiSummary, ErrorCode> {
        if let Some(id) = self.reusable_summary_ids.pop() {
            self.summaries[id] = Some(s.into());
            Ok(FfiSummary { id })
        } else {
            let id = self.summaries.len();
            self.summaries.push(Some(s.into()));
            Ok(FfiSummary { id })
        }
    }

    /// Get the `Term` from the `FfiTerm`.
    fn get_term(&self, ffi_term: FfiTerm) -> Result<&Term, ErrorCode> {
        match self.terms.get(ffi_term.id) {
            Some(Some(t)) => Ok(t),
            _ => Err(ErrorCode::InvalidTerm),
        }
    }

    /// Check if the term exists in the context.
    fn has_term(&self, term: FfiTerm) -> bool {
        matches!(self.terms.get(term.id), Some(Some(_)))
    }

    fn new_term(&mut self, t: Term) -> FfiTerm {
        if let Some(id) = self.reusable_term_ids.pop() {
            self.terms[id] = Some(t.into());
            FfiTerm { id }
        } else {
            let id = self.terms.len();
            self.terms.push(Some(t.into()));
            FfiTerm { id }
        }
    }

    /// Implementation of the `free_term` method for the `ContextInner` struct.
    fn free_term_impl(&mut self, term: FfiTerm) -> ErrorCode {
        let id = term.id;
        if id >= self.terms.len() {
            return ErrorCode::InvalidTerm;
        }
        match self.terms[id].take() {
            Some(_) => {
                self.reusable_term_ids.push(id);
                ErrorCode::Success
            }
            None => ErrorCode::InvalidTerm,
        }
    }

    /// Get the type from the `FfiAbcType` handle as an `AbcScalar`.
    ///
    /// # Errors
    /// - [`ErrorCode::WrongType`] is returned if the type is not a scalar type.
    /// - [`ErrorCode::InvalidType`] is returned if the type does not exist in this context.
    fn get_scalar_type(&self, ty: FfiAbcType) -> Result<AbcScalar, ErrorCode> {
        let scalar = match self.types.get(ty.id) {
            Some(Some(ty_inner)) => match ty_inner.as_ref() {
                AbcType::Scalar(s) => *s,
                _ => return Err(ErrorCode::WrongType),
            },
            _ => return Err(ErrorCode::InvalidType),
        };
        Ok(scalar)
    }

    /// Add the provided type to this context and return its handle.
    fn new_type(&mut self, ty: AbcType) -> FfiAbcType {
        if let Some(id) = self.reusable_type_ids.pop() {
            self.types[id] = Some(ty.into());
            FfiAbcType { id }
        } else {
            let id = self.types.len();
            self.types.push(Some(ty.into()));
            FfiAbcType { id }
        }
    }

    /// Get the type from the `FfiAbcType` handle.
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidType`] is returned if the type does not exist in this context.
    fn get_type(&self, ty: FfiAbcType) -> Result<&super::Handle<AbcType>, ErrorCode> {
        match self.types.get(ty.id) {
            Some(Some(ty)) => Ok(ty),
            _ => Err(ErrorCode::InvalidType),
        }
    }

    /// Implementation of the `free_type` method for the `ContextInner` struct.
    fn free_type_impl(&mut self, term: FfiAbcType) -> ErrorCode {
        let id = term.id;
        if id >= self.types.len() {
            return ErrorCode::InvalidTerm;
        }
        match self.types[id].take() {
            Some(_) => {
                self.reusable_type_ids.push(id);
                ErrorCode::Success
            }
            None => ErrorCode::InvalidTerm,
        }
    }

    /// Get the summary from the `FfiSummary` handle.
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if the summary does not exist in the library's collection.
    #[allow(dead_code)]
    fn get_summary(
        &self, summary: FfiSummary,
    ) -> Result<&super::Handle<super::Summary>, ErrorCode> {
        match self.summaries.get(summary.id) {
            Some(Some(s)) => Ok(s),
            _ => Err(ErrorCode::InvalidTerm),
        }
    }

    /// Implementation of the `free_summary` method for the `ContextInner` struct.
    #[allow(dead_code)]
    fn free_summary_impl(&mut self, summary: FfiSummary) -> ErrorCode {
        let id = summary.id;
        if id >= self.summaries.len() {
            return ErrorCode::InvalidTerm;
        }
        match self.summaries[id].take() {
            Some(_) => {
                self.reusable_summary_ids.push(id);
                ErrorCode::Success
            }
            None => ErrorCode::InvalidTerm,
        }
    }
}

/// Removes all contexts from the global context map. This is the only way to resolve a `PoisonedLock` error.
pub extern "C" fn reset_contexts() {
    match FfiContexts.write() {
        Err(mut e) => {
            e.get_mut().clear();
            FfiContexts.clear_poison();
        }
        Ok(mut contexts) => {
            contexts.clear();
        }
    }

    match ReusableContextIds.lock() {
        Err(mut e) => {
            e.get_mut().clear();
            ReusableContextIds.clear_poison();
        }
        Ok(mut ids) => {
            ids.clear();
        }
    }
}

/*
Implementation of predicate constructors for term
*/
impl Context {
    /// Create a new unit predicate. Must only be used on the `Variable` variant of `Term` or the following `Expression` variants
    /// - A `Select` where the `iftrue` and `iffalse` are booleans.
    /// - A `FieldAccess` where the accessed field is a `bool`
    /// - An `AccessIndex` on an array of `bool`s
    /// - A `cast` to a boolean type.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] if the term does not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] if the provided context does not exist in the library's collection.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_unit_pred(self, term: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let Ok(resolved_term) = context.get_term(term) else {
            return MaybeTerm::Error(ErrorCode::InvalidTerm);
        };
        let new_term = Term::new_unit_pred(resolved_term);
        MaybeTerm::Success(context.new_term(new_term))
    }
    /// Create a Term holding the `true` predicate
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_literal_true(self) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let new_term = Term::new_literal_true();
        MaybeTerm::Success(context.new_term(new_term))
    }

    /// Create a Term holding the `false` predicate
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_literal_false(self) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        MaybeTerm::Success(context.new_term(Term::new_literal_false()))
    }

    /// Constructs the Predicate term `lhs && rhs`.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the logical and. Must be a predicate term.
    /// - `rhs`: The right-hand side of the logical and. Must be a predicate term.
    ///
    /// Use [`abc_new_unit_pred`] to convert variables and expression terms to predicates that can be used in this function.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] if either `lhs` or `rhs` do not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] if the provided context does not exist in the library's collection.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`abc_new_unit_pred`]: crate::cffi::Context::abc_new_unit_pred"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_logical_and(self, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let lhs = context.get_term(lhs);
        let rhs = context.get_term(rhs);
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                MaybeTerm::Success(context.new_term(Term::new_logical_and(&lhs, &rhs)))
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
    }

    /// Constructs the Predicate Term `lhs || rhs`.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the logical or. Must be a predicate term.
    /// - `rhs`: The right-hand side of the logical or. Must be a predicate term.
    ///
    /// Use [`abc_new_unit_pred`] to convert variables and expression terms to predicates that can be used in this function.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] if either `lhs` or `rhs` do not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] if the provided context does not exist in the library's collection.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`abc_new_unit_pred`]: crate::cffi::Context::abc_new_unit_pred"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_logical_or(self, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let lhs = context.get_term(lhs);
        let rhs = context.get_term(rhs);
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                MaybeTerm::Success(context.new_term(Term::new_logical_or(&lhs, &rhs)))
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
    }

    /// Constructs the predicate term `lhs op rhs`
    ///
    /// # Arguments
    /// - `op`: The comparison operator to use.
    /// - `lhs`: The left-hand side of the comparison. Must be a predicate term.
    /// - `rhs`: The right-hand side of the comparison.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`abc_new_unit_pred`]: crate::cffi::Context::abc_new_unit_pred"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_comparison(
        self, op: super::CmpOp, lhs: FfiTerm, rhs: FfiTerm,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let lhs = context.get_term(lhs);
        let rhs = context.get_term(rhs);
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                MaybeTerm::Success(context.new_term(Term::new_comparison(op, &lhs, &rhs))).into()
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
    }

    /// Constructs the predicate term `!t`
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if `t` does not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_not(self, t: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        match context.get_term(t) {
            Ok(t) => MaybeTerm::Success(context.new_term(Term::new_not(&t))).into(),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

impl Context {
    /// Helper method that resolves `ty` to the inner type, ensuring that it is an `AbcScalar` variant.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Types container is poisoned.
    /// - [`ErrorCode::InvalidTerm`] is returned if the type does not exist in the library's collection.
    /// - [`ErrorCode::WrongType`] is returned if the type is not a scalar type.

    /// Create a new variable term, with the provided name.
    ///
    /// It is important to remember that terms must have unique names! All terms
    /// are required to use SSA naming within a context. This includes terms defined within their own summaries.
    ///
    /// # Errors
    /// - [`ErrorCode::NullPointer`] is returned if the string passed is null.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    #[allow(clippy::needless_pass_by_value)] // We do want to pass by value here, actually.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_var(self, s: FfiStr) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        match s.as_opt_str() {
            Some(s) => MaybeTerm::Success(context.new_term(Term::new_var(s))),
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
    /// - [`ErrorCode::InvalidType`] is returned if `ty` is not found in the context.
    /// - [`ErrorCode::InvalidTerm`] is returned if `source_term` is not found in the context.
    /// - [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    #[cfg_attr(
        any(doc, rust_analyzer),
        doc = "\n[`ErrorCode::PoisonedLock`]: crate::ErrorCode::PoisonedLock\n\
                [`ErrorCode::InvalidTerm`]: crate::ErrorCode::InvalidTerm\n\
                [`MaybeTerm::Success`]: crate::MaybeTerm::Success\n\
                [`MaybeTerm::Error`]: crate::MaybeTerm::Error\n\
                [`ErrorCode::WrongType`]: crate::ErrorCode::WrongType"
    )]
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_cast(self, source_term: FfiTerm, ty: FfiAbcType) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let resolved_type = match context.get_scalar_type(ty) {
            Ok(s) => s,
            Err(e) => return MaybeTerm::Error(e),
        };

        let Ok(term) = context.get_term(source_term) else {
            return MaybeTerm::Error(ErrorCode::InvalidTerm);
        };

        MaybeTerm::Success(context.new_term(Term::new_cast(term.clone(), resolved_type)))
    }

    /// Create a new comparison term, e.g. `x > y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_cmp_term(self, op: CmpOp, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let lhs = context.get_term(lhs);
        let rhs = context.get_term(rhs);
        match (lhs, rhs) {
            (Ok(lhs), Ok(rhs)) => {
                MaybeTerm::Success(context.new_term(Term::new_cmp_op(op, &lhs, &rhs))).into()
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
    }

    /// Create a new index access term, e.g. `x[y]`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `base` or `index` do not exist in the context.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_index_access(self, base: FfiTerm, index: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let base = context.get_term(base);
        let index = context.get_term(index);
        match (base, index) {
            (Ok(base), Ok(index)) => {
                MaybeTerm::Success(context.new_term(Term::new_index_access(base, index))).into()
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
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
    /// `ErrorCode::InvalidTerm` if `base` does not exist in the context.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms or Types container is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_struct_access(
        self, base: FfiTerm, field: FfiStr, ty: FfiAbcType, field_idx: usize,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let ty = match context.get_type(ty) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };
        let base = match context.get_term(base) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let Some(field) = field.into_opt_string() else {
            return MaybeTerm::Error(ErrorCode::NullPointer);
        };

        MaybeTerm::Success(context.new_term(Term::new_struct_access(
            &base,
            field,
            ty.clone(),
            field_idx,
        )))
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
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the library's collection.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::ValueError` if `size` is not between 2 and 4.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_splat(self, term: FfiTerm, size: u32) -> MaybeTerm {
        if !(2..=4).contains(&size) {
            return MaybeTerm::Error(ErrorCode::ValueError);
        }
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };
        MaybeTerm::Success(context.new_term(Term::new_splat(term.clone(), size)))
    }

    /// Create a new literal term.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a new `Literal` variant of `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_literal(self, lit: Literal) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        MaybeTerm::Success(context.new_term(Term::new_literal(lit)))
    }

    /// Create a binary operation term, e.g. `x + y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the library's collection.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_binary_op(
        self, op: super::BinaryOp, lhs: FfiTerm, rhs: FfiTerm,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let lhs = match context.get_term(lhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match context.get_term(rhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_binary_op(op, &lhs, &rhs)))
    }

    /// Create a new unary operation term, e.g. `-x`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_unary_op(self, op: super::UnaryOp, term: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_unary_op(op, &term)))
    }

    /// Create a new term corresponding to wgsl's [`max`](https://www.w3.org/TR/WGSL/#max-float-builtin) bulitin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_max(self, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let lhs = match context.get_term(lhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match context.get_term(rhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_max(&lhs, &rhs)))
    }

    /// Create a new term corresponding to wgsl's [`min`](https://www.w3.org/TR/WGSL/#min-float-builtin) builtin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_min(self, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let lhs = match context.get_term(lhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match context.get_term(rhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_min(&lhs, &rhs)))
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
    /// `ErrorCode::InvalidTerm` if either `lhs`, `m`, or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_select(
        self, iftrue: FfiTerm, iffalse: FfiTerm, predicate: FfiTerm,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let iftrue = match context.get_term(iftrue) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let iffalse = match context.get_term(iffalse) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let predicate = match context.get_term(predicate) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_select(&iftrue, &iffalse, &predicate)))
    }

    /// Helper method for creating new vector terms in a DRY fashion. Also works for matrices.
    fn new_vec_helper(
        self, components: &[FfiTerm], ty: FfiAbcType, is_matrix: bool,
    ) -> Result<FfiTerm, ErrorCode> {
        get_context_mut!(@result, self, contexts, context);
        let resolved_type = context.get_scalar_type(ty)?;

        let mut resolved_components: Vec<Term> = Vec::with_capacity(components.len());
        for component in components {
            resolved_components.push(context.get_term(*component)?.clone());
        }

        let new_term = if is_matrix {
            Term::new_matrix(&resolved_components, resolved_type)
        } else {
            Term::new_vector(&resolved_components, resolved_type)
        };

        Ok(context.new_term(new_term))
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
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_vec2(
        self, term_0: FfiTerm, term_1: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[term_0, term_1], ty, false).into()
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
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::InvalidType` if `ty` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_vec3(
        self, term_0: FfiTerm, term_1: FfiTerm, term_2: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[term_0, term_1, term_2], ty, false)
            .into()
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
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::InvalidType` if `ty` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_vec4(
        self, term_0: FfiTerm, term_1: FfiTerm, term_2: FfiTerm, term_3: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[term_0, term_1, term_2, term_3], ty, false)
            .into()
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
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_array_length(self, term: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::make_array_length(&term)))
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
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_store(
        self, term: FfiTerm, index: FfiTerm, value: FfiTerm,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let index = match context.get_term(index) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let value = match context.get_term(value) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_store(
            term.clone(),
            index.clone(),
            value.clone(),
        )))
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
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_struct_store(
        self, term: FfiTerm, field_idx: usize, value: FfiTerm,
    ) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let value = match context.get_term(value) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_struct_store(
            term.clone(),
            field_idx,
            value.clone(),
        )))
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
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_abs(self, term: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let term = match context.get_term(term) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_abs(term)))
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
    /// `ErrorCode::InvalidTerm` if either `base` or `exponent` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_pow(self, base: FfiTerm, exponent: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let base = match context.get_term(base) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let exponent = match context.get_term(exponent) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_pow(base, exponent)))
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
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_dot(self, lhs: FfiTerm, rhs: FfiTerm) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);

        let lhs = match context.get_term(lhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        let rhs = match context.get_term(rhs) {
            Ok(t) => t,
            Err(e) => return MaybeTerm::Error(e),
        };

        MaybeTerm::Success(context.new_term(Term::new_dot(&lhs, &rhs)))
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat2x2(
        self, row_0: FfiTerm, row_1: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat2x3(
        self, row_0: FfiTerm, row_1: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat2x4(
        self, row_0: FfiTerm, row_1: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat3x2(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat3x3(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat3x4(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2], ty, true).into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat4x2(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, row_3: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true)
            .into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat4x3(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, row_3: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true)
            .into()
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
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_mat4x4(
        self, row_0: FfiTerm, row_1: FfiTerm, row_2: FfiTerm, row_3: FfiTerm, ty: FfiAbcType,
    ) -> MaybeTerm {
        self.new_vec_helper(&[row_0, row_1, row_2, row_3], ty, true)
            .into()
    }
}

impl Context {
    /// Get the string representation of the term.
    ///
    /// If the term is invalid, `<NotFound>` is returned.
    /// Note: The returned string must be freed by calling `abc_free_string` or this will lead to a memory leak.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_term_to_cstr(self, term: FfiTerm) -> *mut c_char {
        let Ok(contexts) = FfiContexts.read() else {
            return unsafe { CString::new("<NotFound>").unwrap_unchecked().into_raw() };
        };
        let Some(Some(context)) = contexts.get(self.id) else {
            return unsafe { CString::new("<NotFound>").unwrap_unchecked().into_raw() };
        };
        let term = context.get_term(term);
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn abc_free_string(s: *mut c_char) {
    unsafe {
        CString::from_raw(s);
    }
}

#[repr(C)]
/// cffi abc type wrapper.
///
/// This is just a wrapper around a usize that provides a safe interface to the types that have been defined in the library.
///
/// Note that type ids that have been freed via the `abc_free_type` will be reused for types created later.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FfiAbcType {
    id: usize,
}

impl Context {
    /// Create a new `scalar` type from an `AbcScalar`.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_Scalar_type(self, scalar: AbcScalar) -> MaybeAbcType {
        get_context_mut!(@maybe_type, self, contexts, context);
        MaybeAbcType::Success(context.new_type(AbcType::Scalar(scalar)))
    }

    /// Create a new struct type. Takes a list of fields, each of which is a tuple of a string and an `AbcType`.
    ///
    /// `num_fields` specifies how many fields are passed. It MUST be at least as long as the number of fields and types
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
    /// - `ErrorCode::NullPointer` if either `fields` or `types` is null.
    /// - `ErrorCode::Alignmenterror` if the pointers are not properly aligned.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidType` if any of the types passed do not exist the context.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn abc_new_Struct_type(
        self, fields: *mut FfiStr, types: *const FfiAbcType, num_fields: usize,
    ) -> MaybeAbcType {
        // If len is 0, then we don't even have to check against null pointers. Just return an empty struct.
        if num_fields == 0 {
            get_context_mut!(@maybe_type, self, contexts, context);
            return MaybeAbcType::Success(context.new_type(AbcType::Struct {
                members: Vec::new(),
            }));
        }
        // Otherwise, we need to do the safety checks.
        if fields.is_null() || types.is_null() {
            return MaybeAbcType::Error(ErrorCode::NullPointer);
        }

        // Check for proper alignment to avoid unsafety.
        if !(fields.is_aligned() && types.is_aligned()) {
            return MaybeAbcType::Error(ErrorCode::Alignmenterror);
        }

        get_context_mut!(@maybe_type, self, contexts, context);

        // The pointers have been checked for null and alignment, so we can safely create slices from them.
        let fields_slice = std::slice::from_raw_parts(fields, num_fields);
        let types_slice = std::slice::from_raw_parts(types, num_fields);

        // Make the new fields hashmap.
        let mut new_fields = Vec::with_capacity(num_fields);

        // Iterate over the fields
        for (pos, ty) in types_slice.iter().enumerate() {
            let Ok(matched_ty) = context.get_type(*ty) else {
                return MaybeAbcType::Error(ErrorCode::InvalidType);
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
        MaybeAbcType::Success(context.new_type(AbcType::Struct {
            members: new_fields,
        }))
    }

    /// Declare a new `SizedArray` type. `size` is the number of elements in the array. This cannot be 0.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidType` is returned if the type passed does not exist in the context.
    /// - `ErrorCode::PoisonedLock` is returned if the lock on the global contexts is poisoned.
    /// - `ErrorCode::ForbiddenZero` is returned if the size is 0.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_SizedArray_type(self, ty: FfiAbcType, size: u32) -> MaybeAbcType {
        // Unlock the types struct
        if size == 0 {
            return MaybeAbcType::Error(ErrorCode::ForbiddenZero);
        };

        get_context_mut!(@maybe_type, self, contexts, context);

        match context.get_type(ty) {
            Ok(ty) => MaybeAbcType::Success(context.new_type(AbcType::SizedArray {
                ty: ty.clone(),
                // Safety: We checked against 0 above.
                size: unsafe { std::num::NonZeroU32::try_from(size).unwrap_unchecked() },
            })),
            _ => MaybeAbcType::Error(ErrorCode::InvalidType),
        }
    }

    /// Declare a new Dynamic Array type of the elements of the type passed.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidType` is returned if the type passed does not exist in the context.
    /// - `ErrorCode::PoisonedLock` is returned if the lock on the global contexts is poisoned.
    /// - `ErrorCode::ForbiddenZero` is returned if the size is 0.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_new_DynamicArray_type(self, ty: FfiAbcType) -> MaybeAbcType {
        get_context_mut!(@maybe_type, self, contexts, context);

        match context.get_type(ty) {
            Ok(ty) => {
                MaybeAbcType::Success(context.new_type(AbcType::DynamicArray { ty: ty.clone() }))
            }
            _ => MaybeAbcType::Error(ErrorCode::InvalidType),
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

    InvalidContext,
}

/// Determine if `term` is contained in the context, returning `ValidityKind`.
impl Context {
    pub extern "C" fn check_term_validity(self, t: FfiTerm) -> ValidityKind {
        let Ok(contexts) = FfiContexts.read() else {
            return ValidityKind::Poisoned;
        };
        let Some(Some(context)) = contexts.get(self.id) else {
            return ValidityKind::Poisoned;
        };
        if context.has_term(t) {
            ValidityKind::Contained
        } else {
            ValidityKind::NotContained
        }
    }
}
