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
use std::num::NonZeroUsize;
use std::sync::{LazyLock, RwLock};

#[allow(unused_imports)]
use crate::cbindgen_annotate;
use crate::{
    AbcScalar, AssumptionOp, BinaryOp, ConstraintHelper, ConstraintInterface, ConstraintOp,
    Literal, StructField, SummaryId, Var,
};

/* ***********************
 *   Macro Definitions   *
 ************************/

/// Expands to a reference to the context object.
///
/// # Arguments
/// - The first argument is the context object to get.
/// - The second argument is an identifier that will hold the `RwLockWriteGuard`.
/// - The third argument is the identifier that will hold the mutable reference to the `ContextInner`.
///
/// This is a macro instead of a function as the `RwLockWriteGuard` must be in the same scope as the mutable reference to the `ContextInner`.
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

    (@maybe_summary, $context_obj:expr, $container_var:ident, $result_var:ident $(,)?) => {
        let Ok(mut $container_var) = FfiContexts.write() else {
            return MaybeSummary::Error(ErrorCode::PoisonedLock);
        };
        let Some(Some($result_var)) = $container_var.get_mut($context_obj.id) else {
            return MaybeSummary::Error(ErrorCode::InvalidContext);
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
// This is a global collection of contexts that are created by the ffi API.
// This indirection protects contexts from improper interference over the ffi boundary,
// as the actual ContextInner object is never exposed.
#[allow(non_upper_case_globals)]
static FfiContexts: LazyLock<RwLock<Vec<Option<ContextInner>>>> =
    LazyLock::new(|| RwLock::new(Vec::new()));
#[allow(non_upper_case_globals)]
static ReusableContextIds: LazyLock<std::sync::Mutex<Vec<usize>>> =
    LazyLock::new(|| Vec::new().into());

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

    /// Indicates a term that does not exist in the context.
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
    AlignmentError = 8,

    /// Indicates a forbidden zero value passed as an arugment.
    ForbiddenZero = 9,

    /// Indicates that the library is at maximum capacity.
    CapacityExceeded = 10,

    /// Indicates that the wrong `AbcType` was passed.
    WrongType = 11,

    /// Indicates that the context does not exist in the global context map.
    InvalidContext = 12,

    /// Indicates that a file error occurred.
    FileError = 13,

    /// An error occured during serialization
    SerializationError = 14,

    /// An error indicating the operation is not supported for the method.
    UnsupportedOperation = 15,

    /// Indicates that the predicate stack is empty
    PredicateStackEmpty = 16,

    /// Indicates that there is no active summary when one is required.
    NotInSummary = 17,

    /// Indicates an attempt to declare the type of the same term multiple times.
    DuplicateType = 18,

    /// Indicates that the requested operation is not implemented
    NotImplemented = 19,

    /// Indicates an attempt to use a `break` or `continue` outside of a loop context.
    NotInLoopContext = 20,

    /// Indicates that the maximum loop depth has been exceeded.
    MaxLoopDepthExceeded = 21,

    //// Indicates that the empty term was used in a disallowed context.
    EmptyTerm = 22,

    /// Indicates that the arguments to a method call were incorrect.
    InvalidArguments = 23,

    /// Indicates an attempt to assign the return value of a function that does not return a value.
    NoReturnValue = 24,

    /// Indicates that the loop condition is not supported.
    UnsupportedLoopCondition = 25,

    /// Indicates the type is not supported for the requested operation.
    UnsupportedType = 26,

    /// Indicates that an additional assumption on the same boundary of a term was attempted.
    DuplicateAssumption = 27,

    /// Indicates an error that went wrong during the solving process.
    SolverError = 28,

    /// Indicates the number of constraints exceeded the maximum allowed.
    ConstraintLimitExceeded = 29,

    /// Indicates the number of summaries exceeded the maximum allowed.
    SummaryLimitExceeded = 30,
}

impl From<crate::ConstraintError> for ErrorCode {
    fn from(e: crate::ConstraintError) -> Self {
        use crate::ConstraintError as E;
        match e {
            E::UnsupportedLoopOperation => ErrorCode::UnsupportedOperation,
            E::PredicateStackEmpty => ErrorCode::PredicateStackEmpty,
            E::DuplicateType => ErrorCode::DuplicateType,
            E::NotImplemented(..) => ErrorCode::NotImplemented,
            E::NotInLoopContext => ErrorCode::NotInLoopContext,
            E::MaxLoopDepthExceeded => ErrorCode::MaxLoopDepthExceeded,
            E::EmptyTerm => ErrorCode::EmptyTerm,
            E::InvalidArguments => ErrorCode::InvalidArguments,
            E::NoReturnValue => ErrorCode::NoReturnValue,
            E::UnsupportedLoopCondition(_) => ErrorCode::UnsupportedLoopCondition,
            E::UnsupportedType(_) => ErrorCode::UnsupportedType,
            E::DuplicateAssumption => ErrorCode::DuplicateAssumption,
            E::SolverError(_) => ErrorCode::SolverError,
            E::ConstraintLimitExceeded => ErrorCode::ConstraintLimitExceeded,
            E::SummaryLimitExceeded => ErrorCode::SummaryLimitExceeded,
            E::InvalidSummary | E::SummaryError => ErrorCode::InvalidSummary,
        }
    }
}

impl From<Result<(), crate::ConstraintError>> for ErrorCode {
    fn from(r: Result<(), crate::ConstraintError>) -> Self {
        match r {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }
}

impl From<Result<(), ErrorCode>> for ErrorCode {
    fn from(r: Result<(), ErrorCode>) -> Self {
        r.err().unwrap_or(ErrorCode::Success)
    }
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

/// Represents a possible summary, or an error code.
#[repr(C)]
pub enum MaybeSummary {
    Error(ErrorCode),
    Success(FfiSummary),
}

impl From<Result<FfiSummary, ErrorCode>> for MaybeSummary {
    fn from(r: Result<FfiSummary, ErrorCode>) -> Self {
        match r {
            Ok(t) => MaybeSummary::Success(t),
            Err(e) => MaybeSummary::Error(e),
        }
    }
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

/// Get the `empty` term.
///
/// The empty term is used to represent the absence of a term.
/// It is used in several methods that require a term to indicate that the term is unimportant.
///
/// For example, it must be passed when invoking a method that does not return a value.
///
/// Attempting to delete the empty term will do nothing, but will not result in an error.
/// That is, the empty term is *always* valid, in any context.
#[unsafe(no_mangle)]
pub extern "C" fn abc_get_empty_term() -> FfiTerm {
    FfiTerm { id: 0 }

}

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
        let context = Some(ContextInner {
            terms: vec![Some(Term::Empty)],
            types: Vec::new(),
            summaries: Vec::new(),
            reusable_term_ids: Vec::new(),
            reusable_type_ids: Vec::new(),
            reusable_summary_ids: Vec::new(),
            constraint_helper: crate::helper::ConstraintHelper::default(),
        });
        if id < contexts.len() {
            contexts[id] = context;
        } else {
            contexts.push(context);
        }
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
        if term.id == 0 {
            return ErrorCode::Success;
        }
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
    summaries: Vec<Option<SummaryId>>,
    reusable_term_ids: Vec<NonZeroUsize>,
    reusable_type_ids: Vec<usize>,
    reusable_summary_ids: Vec<usize>,
    constraint_helper: crate::helper::ConstraintHelper,
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
    #[allow(clippy::unnecessary_wraps)] // Future proofing
    fn new_summary_impl(&mut self, s: SummaryId) -> Result<FfiSummary, ErrorCode> {
        if let Some(id) = self.reusable_summary_ids.pop() {
            self.summaries[id] = Some(s);
            Ok(FfiSummary { id })
        } else {
            let id = self.summaries.len();
            self.summaries.push(Some(s));
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
            let id = id.get();
            self.terms[id] = Some(t);
            FfiTerm { id }
        } else {
            let id = self.terms.len();
            self.terms.push(Some(t));
            FfiTerm { id }
        }
    }

    /// Implementation of the `free_term` method for the `ContextInner` struct.
    fn free_term_impl(&mut self, term: FfiTerm) -> ErrorCode {
        let id = term.id;
        if id == 0 {
            return ErrorCode::Success;
        }
        if id >= self.terms.len() {
            return ErrorCode::InvalidTerm;
        }
        match self.terms[id].take() {
            Some(_) => {
                // Safety: We checked against id being 0 above.
                self.reusable_term_ids
                    .push(unsafe { NonZeroUsize::new_unchecked(id) });
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
    fn new_type(&mut self, ty: AbcType) -> Result<FfiAbcType, ErrorCode> {
        let ty = self
            .constraint_helper
            .declare_type(ty)
            .map_err(|_| ErrorCode::WrongType)?;
        if let Some(id) = self.reusable_type_ids.pop() {
            self.types[id] = Some(ty);
            Ok(FfiAbcType { id })
        } else {
            let id = self.types.len();
            self.types.push(Some(ty));
            Ok(FfiAbcType { id })
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
    fn get_summary(&self, summary: FfiSummary) -> Result<SummaryId, ErrorCode> {
        match self.summaries.get(summary.id) {
            Some(Some(s)) => Ok(*s),
            _ => Err(ErrorCode::InvalidSummary),
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
                MaybeTerm::Success(context.new_term(Term::new_logical_and(lhs, rhs)))
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
                MaybeTerm::Success(context.new_term(Term::new_logical_or(lhs, rhs)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
                MaybeTerm::Success(context.new_term(Term::new_comparison(op, lhs, rhs)))
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
            Ok(t) => MaybeTerm::Success(context.new_term(Term::new_not(t))),
            Err(e) => MaybeTerm::Error(e),
        }
    }
}

impl Context {
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

        match s.into_opt_string() {
            Some(s) => {
                let var = Var { name: s };
                let term = match context.constraint_helper.declare_var(var) {
                    Ok(t) => t,
                    Err(e) => return MaybeTerm::Error(e.into()),
                };
                MaybeTerm::Success(context.new_term(term))
            }
            None => MaybeTerm::Error(ErrorCode::NullPointer),
        }
    }

    /// Mark a variable term as [uniform](https://www.w3.org/TR/WGSL/#address-spaces-uniform).
    ///
    /// Arguments:
    /// - `var`: The variable term to mark as uniform. Must be a term previously declared as a variable via `abc_new_var`.
    ///
    /// Constraints that would be able to be eliminated in the case that
    /// it knew the value of every uniform variable, and the sizes of all runtime-sized arrays,
    /// will have their "No" answers replaced with "Maybe" answers.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] is returned if the provided term does not exist in the context.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    /// - [`ErrorCode::WrongType`] is returned if the term passed is not a variable term.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_uniform(self, var: FfiTerm) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.mark_uniform_var_impl(var)
    }

    /// Create a new `cast` expression. This corresponds to the `as` operator in WGSL.
    ///
    /// `source_term` is the term to cast, and `ty` is the type to cast it to.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if the term was not valid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
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
                MaybeTerm::Success(context.new_term(Term::new_cmp_op(op, lhs, rhs)))
            }
            _ => MaybeTerm::Error(ErrorCode::InvalidTerm),
        }
    }

    /// Create a new index access term, e.g. `x[y]`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
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
                MaybeTerm::Success(context.new_term(Term::new_index_access(base, index)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term or type could not be found.
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
            base,
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_binary_op(op, lhs, rhs)))
    }

    /// Create a new unary operation term, e.g. `-x`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_unary_op(op, term)))
    }

    /// Create a new term corresponding to wgsl's [`max`](https://www.w3.org/TR/WGSL/#max-float-builtin) bulitin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_max(lhs, rhs)))
    }

    /// Create a new term corresponding to wgsl's [`min`](https://www.w3.org/TR/WGSL/#min-float-builtin) builtin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_min(lhs, rhs)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_select(predicate, iftrue, iffalse)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::make_array_length(term)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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

        MaybeTerm::Success(context.new_term(Term::new_dot(lhs, rhs)))
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
    /// A `MaybeTerm` which is either an `FfiTerm` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
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
        let Ok(ty) = context.new_type(AbcType::Scalar(scalar)) else {
            return MaybeAbcType::Error(ErrorCode::WrongType);
        };
        MaybeAbcType::Success(ty)
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
    /// - `ErrorCode::AlignmentError` if the pointers are not properly aligned.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidType` if any of the types passed do not exist the context.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn abc_new_Struct_type(
        self, fields: *mut FfiStr, types: *const FfiAbcType, num_fields: usize,
    ) -> MaybeAbcType {
        // If len is 0, then we don't even have to check against null pointers. Just return an empty struct.
        if num_fields == 0 {
            get_context_mut!(@maybe_type, self, contexts, context);
            let Ok(ty) = context.new_type(AbcType::Struct {
                members: Vec::new(),
            }) else {
                return MaybeAbcType::Error(ErrorCode::WrongType);
            };
            return MaybeAbcType::Success(ty);
        }
        // Otherwise, we need to do the safety checks.
        if fields.is_null() || types.is_null() {
            return MaybeAbcType::Error(ErrorCode::NullPointer);
        }

        // Check for proper alignment to avoid unsafety.
        if !(fields.is_aligned() && types.is_aligned()) {
            return MaybeAbcType::Error(ErrorCode::AlignmentError);
        }

        get_context_mut!(@maybe_type, self, contexts, context);

        // The pointers have been checked for null and alignment, so we can safely create slices from them.
        let fields_slice = unsafe { std::slice::from_raw_parts(fields, num_fields) };
        let types_slice = unsafe { std::slice::from_raw_parts(types, num_fields) };

        // Make the new fields hashmap.
        let mut members = Vec::with_capacity(num_fields);

        // Iterate over the fields
        for (pos, ty) in types_slice.iter().enumerate() {
            let Ok(matched_ty) = context.get_type(*ty) else {
                return MaybeAbcType::Error(ErrorCode::InvalidType);
            };
            let field_name = String::from(match fields_slice[pos].as_opt_str() {
                Some(s) => s,
                None => return MaybeAbcType::Error(ErrorCode::NullPointer),
            });
            members.push(StructField {
                name: field_name,
                ty: matched_ty.clone(),
            });
        }

        // Create the struct type.
        let Ok(ty) = context.new_type(AbcType::Struct { members }) else {
            return MaybeAbcType::Error(ErrorCode::WrongType);
        };
        MaybeAbcType::Success(ty)
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
        }

        get_context_mut!(@maybe_type, self, contexts, context);

        match context.get_type(ty) {
            Ok(ty) => {
                let Ok(ty) = context.new_type(AbcType::SizedArray {
                    ty: ty.clone(),
                    // Safety: We checked against 0 above.
                    size: unsafe { std::num::NonZeroU32::try_from(size).unwrap_unchecked() },
                }) else {
                    return MaybeAbcType::Error(ErrorCode::WrongType);
                };
                MaybeAbcType::Success(ty)
            }
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
                let Ok(ty) = context.new_type(AbcType::DynamicArray { ty: ty.clone() }) else {
                    return MaybeAbcType::Error(ErrorCode::WrongType);
                };
                MaybeAbcType::Success(ty)
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

/* Solver */
// Section of methods to allow for cffi to interface with the solver.
// They belong to the context.

/// The error codes that can be returned by `abc_solve_constraints` method.
#[repr(C)]
pub enum SolverErrorCode {
    /// Summary is invalid.
    InvalidSummary = 1,
    /// Unsupported type for an operation
    UnsupportedType = 2,
    /// Multiple assumptions for the same term.
    DuplicateAssignmentError = 3,

    InvalidOp = 4,

    /// Multiple assignments to the same term.
    /// The only term that this can occur for is `ret`.
    ///
    /// `Ret` is special, it is the only term that may be assigned to more than once.
    SsaViolation = 5,

    TypeMismatch = 6,

    Unexpected = 7,

    Unsupported = 8,

    /// Top level constraints can not be satisfied.
    DeadCode = 9,

    /// The lock on the global contexts is poisoned.
    PoisonedLock = 10,

    /// The context does not exist.
    InvalidContext = 11,
}

use crate::solvers::interval::{IntervalError, SolverError, SolverResult};

impl From<SolverError> for SolverErrorCode {
    fn from(e: SolverError) -> Self {
        use IntervalError as IE;
        use SolverError as SE;
        match e {
            SE::InvalidSummary => SolverErrorCode::InvalidSummary,
            SE::UnsupportedType => SolverErrorCode::UnsupportedType,
            SE::DuplicateAssignmentError(_) => SolverErrorCode::DuplicateAssignmentError,
            SE::IntervalError(IE::InvalidBinOp(_, _, _) | IE::InvalidOp(_, _)) => {
                SolverErrorCode::InvalidOp
            }
            SE::SsaViolation(_) => SolverErrorCode::SsaViolation,
            SE::IntervalError(IE::IncompatibleTypes) | SE::TypeMismatch { .. } => {
                SolverErrorCode::TypeMismatch
            }
            SE::Unexpected(_) => SolverErrorCode::Unexpected,
            SE::Unsupported(_, _, _) => SolverErrorCode::Unsupported,
            SE::DeadCode => SolverErrorCode::DeadCode,
        }
    }
}

/// The solution to the constraints.
#[repr(C)]
pub enum MaybeSolution {
    Success(ConstraintSolution),
    Error(SolverErrorCode),
}

/// The result of a constraint.
#[repr(C)]
pub struct ConstraintResult(u32, SolverResult);

#[repr(C)]
pub struct ConstraintSolution {
    /// The size of the results array.
    len: usize,
    results: *const ConstraintResult,
}

impl From<(u32, SolverResult)> for ConstraintResult {
    fn from((id, result): (u32, SolverResult)) -> Self {
        Self(id, result)
    }
}
impl From<&(u32, SolverResult)> for ConstraintResult {
    fn from((id, result): &(u32, SolverResult)) -> Self {
        Self(*id, *result)
    }
}
impl From<ConstraintResult> for (u32, SolverResult) {
    fn from(result: ConstraintResult) -> Self {
        (result.0, result.1)
    }
}
impl ConstraintSolution {
    /// Free the solution vector. After calling this method, the solution is no longer valid
    /// This must be called to avoid a memory leak.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_free_solution(solution: Self) {
        #[allow(clippy::drop_non_drop)] // Explicit > implicit
        drop(solution);
    }
}

impl Context {
    /// Solve the constraints for the provided summary.
    ///
    /// Note that the returned solution *must be freed* by calling `abc_free_solution` otherwise
    /// a memory leak will occur.
    ///
    /// This will return an `FfiSolution` which contains the results.
    /// The results are in the form (id, result). `id` corresponds to the
    /// id of the constraint that was provided when the constraint was added.
    /// the `result` corresponds to the result of the constraint. It is `Yes`
    /// if the constraint is always satisfied. In this case, the bounds check
    /// can be removed. `No` means that the constraint could not
    /// be proven to always be satisfied. In this case, the bounds check
    /// must be kept. `Maybe` means that the constraint can be tested for
    /// satisfiability if the system is provided with concrete values for
    /// all uniform variables and array lengths.
    ///
    /// In this current implementation, there is not a distinction
    /// between checks that are always violated and checks that may be violated.
    ///
    ///
    /// # Errors
    /// - `SolverErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `SolverErrorCode::InvalidContext` if the context does not exist.
    /// - `SolverErrorCode::InvalidSummary` if the summary does not exist in the context.
    /// - Many other errors may be returned that occur during the solving process.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_solve_constraints(self, summary: FfiSummary) -> MaybeSolution {
        let Ok(contexts) = FfiContexts.read() else {
            return MaybeSolution::Error(SolverErrorCode::PoisonedLock);
        };
        let Some(Some(context)) = contexts.get(self.id) else {
            return MaybeSolution::Error(SolverErrorCode::InvalidContext);
        };

        // Get the `id` from the summary.
        let Ok(id) = context.get_summary(summary) else {
            return MaybeSolution::Error(SolverErrorCode::InvalidSummary);
        };

        let result = match context.constraint_helper.solve(id) {
            Ok(r) => r,
            Err(e) => return MaybeSolution::Error(e.into()),
        };

        let result: Vec<ConstraintResult> = ConstraintHelper::solution_to_result(&result)
            .iter()
            .map(Into::into)
            .collect();

        let ffi_vec = ConstraintSolution {
            len: result.len(),
            results: result.as_ptr(),
        };

        std::mem::forget(result);

        MaybeSolution::Success(ffi_vec)
    }

    /// Serializes the constraint system to json.
    /// The input should be a file path where the serialized json will be written.
    #[unsafe(no_mangle)]
    #[allow(clippy::needless_pass_by_value)]
    pub extern "C" fn abc_serialize_constraints(self, path: FfiStr) -> ErrorCode {
        use serde::Serialize;

        let Ok(contexts) = FfiContexts.read() else {
            return ErrorCode::PoisonedLock;
        };

        let Some(Some(context)) = contexts.get(self.id) else {
            return ErrorCode::InvalidContext;
        };

        let Some(path) = path.as_opt_str() else {
            return ErrorCode::NullPointer;
        };

        let Ok(f) = std::fs::File::create(path) else {
            return ErrorCode::FileError;
        };

        let mut serializer = serde_json::Serializer::new(f);

        let Ok(()) = context
            .constraint_helper
            .get_module()
            .serialize(&mut serializer)
        else {
            return ErrorCode::SerializationError;
        };

        ErrorCode::Success
    }

    /// Mark a variable as a loop variable.
    ///
    /// A loop variable is any variable that is updated within a loop. Supported operations
    /// are limited to binary operations. For best results, the increment / decrement term must be
    /// a constnat.
    ///
    /// # Arguments
    /// - `var`: The induction variable.
    /// - `init`: The expression the induction variable is initialized to.
    /// - `update_rhs`: The expression that the term is incremented / decremented by
    /// - `update_op`: The operation that is used to update the term.
    ///
    /// Note that `inc_term` and `inc_op` are parts of the expression with the induction term.
    /// That is, only expressions of the form `a = a op b` are allowed, where `a` is the induction variable.
    /// This method should come **before** the `begin_loop` call.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if any of the terms passed are invalid.
    /// - `ErrorCode::UnsupportedOperation` if the operation is not supported for loops.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_loop_variable(
        self, var: FfiTerm, init: FfiTerm, update_rhs: FfiTerm, update_op: BinaryOp,
    ) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        match context.mark_loop_variable_impl(var, init, update_rhs, update_op) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e,
        }
    }

    /// Add the constraints of the summary by invoking the function call with the proper arguments.
    ///
    ///
    /// # Arguments
    /// - `summary`: The summary for the function being invoked
    /// - `args`: An array containing the arguments that are passed to the function.
    /// - `num_args`: The size of the `args` array.
    /// - `return_dest`: A fresh variable term that will hold the return value. If the return value is not used,
    ///   then this must be the `Empty` term.
    ///
    /// This will add the constraints of the summary to the constraint system, with
    /// the arguments substituted with `args`, and the return value substituted with `return_dest`.
    ///
    /// # Safety
    /// The caller must ensure that `args` holds at least `num_args` elements.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `args` is null and `num_args` is not 0.
    /// - `ErrorCode::AlignmentError` if `args` is not properly aligned.
    /// - `ErrorCode::InvalidTerm` if any of the terms passed do not exist in the context.
    /// - `ErrorCode::InvalidSummary` if the summary does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn abc_new_call(
        self, summary: FfiSummary, args: *const FfiTerm, num_args: u32, return_dest: FfiTerm,
    ) -> MaybeTerm {
        let resolved_args: &[FfiTerm] = if num_args != 0 {
            if args.is_null() {
                return MaybeTerm::Error(ErrorCode::NullPointer);
            }
            if !args.is_aligned() {
                return MaybeTerm::Error(ErrorCode::AlignmentError);
            }
            // Safety: We have checked that the pointer is not null and is aligned.
            // and it is the user's responsibility to ensure that `num_args` is at most the
            // length of the array.
            unsafe { std::slice::from_raw_parts(args, num_args as usize) }
        } else {
            &[]
        };

        get_context_mut!(@maybe_term, self, contexts, context);

        let return_dest = if return_dest.id == 0 {
            None
        } else {
            Some(return_dest)
        };

        context
            .make_call_impl(summary, resolved_args, return_dest)
            .into()
    }

    /// Begins a loop context.
    ///
    /// A loop context is akin to a predicate context. It allows for the convenience methods for `break` and `continue`
    /// to be properly handled. Additionally, all updates to variables within are marked as range constraints.
    ///
    /// The `condition` should correspond to a boolean term where one of the operands has been marked as a loop variable.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::UnsupportedLoopCondition` if the condition is not a `Predicate` or `Var` term.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_begin_loop(self, condition: FfiTerm) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        match context.begin_loop_impl(condition) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e,
        }
    }

    /// End a loop context.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInLoopcontext` if there has been no matching call to `begin_loop`.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_end_loop(self) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.constraint_helper.end_loop().into()
    }

    /// Mark an assumption.
    ///
    /// Assumptions are invariants that must hold at all times.
    /// At solving time, these differ from constraints in that they are not inverted
    /// to test for satisfiability.
    ///
    /// # Arguments
    /// - `lhs`: The left hand side of the assumption.
    /// - `op`: The operation that is used to compare the terms.
    /// - `rhs`: The right hand side of the assumption.
    ///
    /// There can only be one assumption per direction for each term. That is,
    /// each term may *either* have one equality assumption (:=) *or* one
    ///  inequality assumption per direction (one less / less equal, one
    /// greater / greater equal). Violating this will result in a
    /// `DuplicateAssumption` error.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if any term does not exist in the context.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::DuplicateAssumption` if `lhs` has already had an assumption on the specified boundary.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_add_assumption(
        self, lhs: FfiTerm, op: AssumptionOp, rhs: FfiTerm,
    ) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.add_assumption_impl(lhs, rhs, op).into()
    }

    /// Begin a summary block.
    ///
    /// A summary block corresponds to a function definition. All constraints
    /// that are added within the summary block are considered to be constraints
    /// of the function.
    ///
    /// A summary block must be ended with a call to `end_summary`, which will
    /// provide the `FfiSummary` that can be used to refer to it.
    ///
    /// # Arguments
    /// - `name`: The name of the function.
    /// - `num_args`: The number of arguments the function takes.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `name` is null.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::SummaryLimitExceeded` if the number of summaries exceeds the limit.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_begin_summary(self, name: FfiStr, num_args: u8) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        let Some(name) = name.into_opt_string() else {
            return ErrorCode::NullPointer;
        };
        context
            .constraint_helper
            .begin_summary(name, num_args)
            .into()
    }

    /// End a summary block.
    ///
    /// This will return the `FfiSummary` that can be used to refer to the summary.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInSummary` if there has been no matching call to `begin_summary`.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_end_summary(self) -> MaybeSummary {
        get_context_mut!(@maybe_summary, self, contexts, context);
        context.end_summary_impl().into()
    }

    /// Add an argument to the active summary.
    ///
    /// The `begin_summary` method must be called before this method.
    ///
    /// # Arguments
    /// - `name`: The name of the argument.
    /// - `ty`: The type of the argument.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the argument was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `name` is null.
    /// - `ErrorCode::InvalidType` if the type does not exist in the context.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInSummary` if there has been no matching call to `begin_summary`.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_add_argument(self, name: FfiStr, ty: FfiAbcType) -> MaybeTerm {
        get_context_mut!(@maybe_term, self, contexts, context);
        let Some(name) = name.into_opt_string() else {
            return MaybeTerm::Error(ErrorCode::NullPointer);
        };
        context.add_argument_impl(name, ty).into()
    }

    /// Mark a break statement. (Currently unimplemented)
    ///
    /// # Errors
    /// `ConstraintError::NotImplemented`
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_break(self) -> ErrorCode {
        ErrorCode::NotImplemented
    }

    /// Mark a continue statement. (Currently unimplemented)
    ///
    /// # Errors
    /// `ConstraintError::NotImplemented`
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_continue(self) -> ErrorCode {
        ErrorCode::NotImplemented
    }

    /// Mark the type of the provided term.
    ///
    /// # Errors
    /// [`DuplicateType`] if the type of the term has already been marked.
    ///
    /// [`DuplicateType`]: crate::ConstraintError::DuplicateType
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_type(self, term: FfiTerm, ty: FfiAbcType) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.mark_type_impl(term, ty)
    }

    /// Begin a predicate block. This indicates to the solver
    /// that all expressions that follow can be filtered by the predicate.
    /// Any constraint that falls within a predicate becomes a soft constraint
    ///
    /// In other words, it would be as if all constraints were of the form
    /// ``p -> c``
    /// Nested predicate blocks end up composing the predicates. E.g.,
    /// ``begin_predicate_block(p1)`` followed by ``begin_predicate_block(p2)``
    /// would mark all constraints as ``p1 && p2 -> c``
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_begin_predicate_block(self, condition: FfiTerm) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.begin_predicate_block_impl(condition)
    }

    /// End the active predicate block.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::PredicateStackEmpty` if there is no active predicate block.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_end_predicate_block(self) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.constraint_helper.end_predicate_block().into()
    }

    /// Add a constraint to the system that narrows the domain of `term`.
    ///
    /// Any active predicates are automatically applied.
    ///
    /// There can only be one constraint per boundary side for each term. That is,
    /// each term may *either* have one equality constraint (:=) *or* one
    /// inequality constraint per side (one less / less equal, one
    /// greater / greater equal). Violating this will result in a
    /// `DuplicateConstraint` error.
    ///
    /// # Arguments
    /// - `lhs`: The left hand side of the constraint.
    /// - `op`: The operation of the constraint.
    /// - `rhs`: The right hand side of the constraint.
    /// - `id`: The identifier that the constraint system will use to refer to
    ///   the constraint. Results will reference this id.
    ///
    ///
    /// # Errors
    /// - `TypeMismatch` if the type of `term` is different from the type of
    ///   `rhs` and `op` is `ConstraintOp::Assign`
    /// - `MaxConstraintCountExceeded` if the maximum number of constraints has
    ///   been exceeded.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_add_constraint(
        self, lhs: FfiTerm, op: ConstraintOp, rhs: FfiTerm, id: u32,
    ) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.add_constraint_impl(lhs, op, rhs, id)
    }

    /// Mark a return statement.
    ///
    /// # Arguments
    /// - `retval`: The term that corresponds to the return value. If no value
    ///   is returned, this *must* be the `Empty` term.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_return(self, retval: FfiTerm) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.mark_return_impl(retval)
    }

    /// Mark the length of an array's dimension. Only valid for dynamically
    /// sized arrays.
    ///
    ///
    /// It is *strongly* preferred to use the type system to mark the variable as
    /// a `SizedArray` type.
    /// This should *only* be used for dynamic arrays whose size is determined later.
    ///
    /// # Arguments
    /// - `term`: The array term to mark the dimension of
    /// - `dim`: The term that corresponds to the 0-based dimension of the array.
    /// - `size`: The size of the array.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_array_length(self, term: FfiTerm, dim: u8, size: u64) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.mark_length_impl(term, dim, size)
    }

    /// Mark the range of a runtime constant. Sugar for a pair of assumptions
    /// (var >= min) and (var <= max).
    ///
    /// The `lower` and `upper` terms must be literals of the same type.
    ///
    /// ## Notes
    /// - The current predicate block is ignored, though the constraints *are*
    ///   added to the active summary (or global if no summary is active)
    /// - This is not meant to be used for loop variables.
    ///   Use `mark_loop_variable` for that.
    ///
    /// # Arguments
    /// - `term`: The term to mark the range of.
    /// - `lower`: The lower bound of the range.
    /// - `upper`: The upper bound of the range.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::WrongType` if the two literals are not the same variant.
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_range(
        self, term: FfiTerm, lower: Literal, upper: Literal,
    ) -> ErrorCode {
        if std::mem::discriminant(&lower) != std::mem::discriminant(&upper) {
            return ErrorCode::WrongType;
        }
        get_context_mut!(@plain, self, contexts, context);
        context.mark_range_impl(term, lower, upper)
    }

    /// Mark the return type for the active summary.
    ///
    /// It is not necessary to call this method for functions that do not return
    /// a value.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::InvalidSummary` if there is no active summary
    /// - `ErrorCode::DuplicateType` if the return type has already been marked for the summary
    #[unsafe(no_mangle)]
    pub extern "C" fn abc_mark_return_type(self, ty: FfiAbcType) -> ErrorCode {
        get_context_mut!(@plain, self, contexts, context);
        context.mark_return_type_impl(ty)
    }
}

impl ContextInner {
    /// Implementation of the `mark_uniform` method for `ContextInner`.
    fn mark_uniform_var_impl(&mut self, term: FfiTerm) -> ErrorCode {
        let Some(Some(term)) = self.terms.get(term.id) else {
            return ErrorCode::InvalidTerm;
        };
        if !term.is_var() {
            return ErrorCode::WrongType;
        }
        match self.constraint_helper.mark_uniform_var(term) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }
    /// Implementation of the `mark_loop_variable` method for `ContextInner`.
    fn mark_loop_variable_impl(
        &mut self, var: FfiTerm, init: FfiTerm, update_rhs: FfiTerm, update_op: BinaryOp,
    ) -> Result<(), ErrorCode> {
        let Some(Some(var)) = self.terms.get(var.id) else {
            return Err(ErrorCode::InvalidTerm);
        };
        let Some(Some(init)) = self.terms.get(init.id) else {
            return Err(ErrorCode::InvalidTerm);
        };
        let Some(Some(update_rhs)) = self.terms.get(update_rhs.id) else {
            return Err(ErrorCode::InvalidTerm);
        };

        self.constraint_helper
            .mark_loop_variable(var, init, update_rhs, update_op)
            .map_err(|_| ErrorCode::UnsupportedOperation)?;
        Ok(())
    }

    fn make_call_impl(
        &mut self, summary: FfiSummary, args: &[FfiTerm], return_dest: Option<FfiTerm>,
    ) -> Result<FfiTerm, ErrorCode> {
        let Some(Some(summary)) = self.summaries.get(summary.id) else {
            return Err(ErrorCode::InvalidSummary);
        };

        let mut resolved_args = Vec::with_capacity(args.len());
        for arg in args {
            match self.terms.get(arg.id) {
                Some(Some(arg)) => resolved_args.push(arg.clone()),
                _ => return Err(ErrorCode::InvalidTerm),
            }
        }

        let return_dest = match return_dest {
            Some(return_dest) => match self.terms.get(return_dest.id) {
                Some(Some(return_dest)) => Some(return_dest),
                _ => {
                    return Err(ErrorCode::InvalidTerm)
                },
            },
            None => None,
        };

        self.constraint_helper
            .make_call(*summary, resolved_args, return_dest)
            .map_or_else(|e| Err(e.into()), |t| Ok(self.new_term(t)))
    }

    fn begin_loop_impl(&mut self, condition: FfiTerm) -> Result<(), ErrorCode> {
        let Some(Some(condition)) = self.terms.get(condition.id) else {
            return Err(ErrorCode::InvalidTerm);
        };

        self.constraint_helper
            .begin_loop(condition)
            .map_err(Into::into)
    }

    fn add_assumption_impl(
        &mut self, lhs: FfiTerm, rhs: FfiTerm, op: AssumptionOp,
    ) -> Result<(), ErrorCode> {
        let Some(Some(lhs)) = self.terms.get(lhs.id) else {
            return Err(ErrorCode::InvalidTerm);
        };
        let Some(Some(rhs)) = self.terms.get(rhs.id) else {
            return Err(ErrorCode::InvalidTerm);
        };

        self.constraint_helper
            .add_assumption(lhs, op, rhs)
            .map_err(Into::into)
    }

    fn end_summary_impl(&mut self) -> Result<FfiSummary, ErrorCode> {
        match self.constraint_helper.end_summary() {
            Err(e) => Err(e.into()),
            Ok(summary) => self.new_summary_impl(summary),
        }
    }

    fn add_argument_impl(&mut self, name: String, ty: FfiAbcType) -> Result<FfiTerm, ErrorCode> {
        let ty = match self.types.get(ty.id) {
            Some(Some(ty)) => ty.clone(),
            _ => return Err(ErrorCode::InvalidType),
        };
        match self.constraint_helper.add_argument(name, &ty) {
            Err(e) => Err(e.into()),
            Ok(term) => Ok(self.new_term(term)),
        }
    }

    fn mark_type_impl(&mut self, term: FfiTerm, ty: FfiAbcType) -> ErrorCode {
        let Some(Some(term)) = self.terms.get(term.id) else {
            return ErrorCode::InvalidTerm;
        };

        let Some(Some(ty)) = self.types.get(ty.id) else {
            return ErrorCode::InvalidType;
        };

        match self.constraint_helper.mark_type(term, ty) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn mark_return_type_impl(&mut self, ty: FfiAbcType) -> ErrorCode {
        let Some(Some(ty)) = self.types.get(ty.id) else {
            return ErrorCode::InvalidType;
        };

        match self.constraint_helper.mark_return_type(ty) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn add_constraint_impl(
        &mut self, term: FfiTerm, op: ConstraintOp, rhs: FfiTerm, id: u32,
    ) -> ErrorCode {
        let Some(Some(term)) = self.terms.get(term.id) else {
            return ErrorCode::InvalidTerm;
        };
        let Some(Some(rhs)) = self.terms.get(rhs.id) else {
            return ErrorCode::InvalidTerm;
        };

        match self.constraint_helper.add_constraint(term, op, rhs, id) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn mark_return_impl(&mut self, retval: FfiTerm) -> ErrorCode {
        let term = if retval.id == 0 {
            None
        } else {
            match self.terms.get(retval.id) {
                Some(Some(term)) => Some(term.clone()),
                _ => return ErrorCode::InvalidTerm,
            }
        };

        match self.constraint_helper.mark_return(term) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn mark_length_impl(&mut self, term: FfiTerm, dim: u8, size: u64) -> ErrorCode {
        let Some(Some(term)) = self.terms.get(term.id) else {
            return ErrorCode::InvalidTerm;
        };

        match self.constraint_helper.mark_length(term, dim, size) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn mark_range_impl(&mut self, var: FfiTerm, min: Literal, max: Literal) -> ErrorCode {
        let Some(Some(var)) = self.terms.get(var.id) else {
            return ErrorCode::InvalidTerm;
        };

        match self.constraint_helper.mark_range(var, min, max) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }

    fn begin_predicate_block_impl(&mut self, condition: FfiTerm) -> ErrorCode {
        let Some(Some(condition)) = self.terms.get(condition.id) else {
            return ErrorCode::InvalidTerm;
        };

        match self.constraint_helper.begin_predicate_block(condition) {
            Ok(()) => ErrorCode::Success,
            Err(e) => e.into(),
        }
    }
}
