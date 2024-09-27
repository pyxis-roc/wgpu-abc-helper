// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

/*! Array Bounds Checking Helper

The crate is meant to provide a friendly interface for working with the array bounds checking constraint system.

The central structure is a `ConstraintHelper`, which corresponds to a wgsl module.

The constraint helper uses a scheme similar to [naga](https://docs.rs/naga/latest/naga/).

A constraint helper consists of:
- [`Summary`]s corresponding to each function and entry point in the module.
- [`Var`]s corresponding to each global variable in the module.
- [`Type`]s


## Function Summaries

`AbcHelper`'s summaries capture the constraints within a function that must be met
in order for the array accesses to be in bounds. It also generates constraints that
narrow the return value of the function.

When the constraint solver sees a function call, it applies the constraints from the summary
while narrowing the domain of the arguments passed to the function based on its own constraints.


## Control Flow

As with any static analysis, proper handling of loops is tricky.
In an effort to avoid overapproximating the constraints, the constraint helper
provides a few mechanisms to handle loops.


[`Summary`]: crate::Summary
[`Type`]: crate::AbcType
[`Var`]: crate::Var
[`Expression`]: crate::AbcExpression
[`ConstraintHelper`]: crate::ConstraintHelper
*/

use std::sync::Arc;
// use std::rc::Rc

type FastHashMap<K, V> = rustc_hash::FxHashMap<K, V>;

use lazy_static::lazy_static;

use log::info as log_info;

// For right now, we are using handles. Later on, we might switch to an arena with actual handles.
pub type Handle<T> = Arc<T>;

#[derive(Clone, Debug, thiserror::Error)]
pub enum ConstraintError {
    #[error("Predicate stack is empty")]
    PredicateStackEmpty,
    #[error("Active summary is not set")]
    SummaryError,
    #[error("This is not yet implemented")]
    NotImplemented(String),
    #[error("Attempt to end a loop when no loop is active")]
    NotInLoopContext,
    #[error("Maximum loop depth exceeded")]
    MaxLoopDepthExceeded,
    #[error("Empty expression in disallowed context.")]
    EmptyExpression,
}

/// An opaque marker that is provided when specifying expressions to relate them, when necessary.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct OpaqueMarker<T>
where
    T: Clone,
    T: std::fmt::Debug,
{
    payload: T,
}

// For right now, everything will be a string... We will not do type checking.
#[derive(Clone, Debug)]
pub struct Var {
    /// The name of the variable
    pub name: String,
    // pub marker: OpaqueMarker<T>,
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(strum_macros::Display, Debug, Clone, Copy)]
#[repr(C)]
pub enum UnaryOp {
    #[strum(to_string = "-")]
    Minus,
}

#[derive(strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum BinaryOp {
    #[strum(to_string = "+")]
    Plus,
    #[strum(to_string = "-")]
    Minus,
    #[strum(to_string = "*")]
    Times,
    #[strum(to_string = "%")]
    Mod,
    #[strum(to_string = "//")]
    Div,

    // The following are probably not supported by our constraint system.
    #[strum(to_string = "&")]
    BitAnd,
    #[strum(to_string = "|")]
    BitOr,
    #[strum(to_string = "^")]
    BitXor,

    // These can probably be supported if the lhs or rhs is a constant.
    #[strum(to_string = "<<")]
    Shl,
    #[strum(to_string = ">>")]
    Shr,
}

#[derive(strum_macros::Display, Debug, Clone, Copy)]
#[repr(C)]
pub enum CmpOp {
    #[strum(to_string = "==")]
    Eq,
    #[strum(to_string = "!=")]
    Neq,
    #[strum(to_string = "<")]
    Lt,
    #[strum(to_string = ">")]
    Gt,
    #[strum(to_string = "<=")]
    Leq,
    #[strum(to_string = ">=")]
    Geq,
}

impl CmpOp {
    /// Negate the predicate operation
    #[must_use]
    pub fn negation(&self) -> Self {
        match self {
            CmpOp::Eq => CmpOp::Neq,
            CmpOp::Neq => CmpOp::Eq,
            CmpOp::Lt => CmpOp::Geq,
            CmpOp::Gt => CmpOp::Leq,
            CmpOp::Leq => CmpOp::Gt,
            CmpOp::Geq => CmpOp::Lt,
        }
    }
}

/// A constraint operation is any comparison operator OR an assignment.
#[derive(strum_macros::Display, Debug, Clone)]
#[repr(C)]
pub enum ConstraintOp {
    #[strum(to_string = "=")]
    Assign,
    #[strum(to_string = "{0}")]
    Cmp(CmpOp),
    #[strum(to_string = "UnaryConstraint")]
    Unary,
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Assign {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        rhs: Term,
    },

    Cmp {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        op: CmpOp,
        rhs: Term,
    },

    Expression {
        guard: Option<Handle<Predicate>>,
        term: Term,
    },
}

impl Constraint {
    /// Return the guard portion of the constraint
    fn guard(&self) -> Option<Handle<Predicate>> {
        match self {
            Constraint::Assign { guard, .. }
            | Constraint::Cmp { guard, .. }
            | Constraint::Expression { guard, .. } => guard.clone(),
        }
    }
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.guard() {
            Some(guard) => write!(f, "{{{guard}}} "),
            None => Ok(()),
        }?;
        match self {
            Constraint::Assign { lhs, rhs, .. } => write!(f, "{lhs} = {rhs}"),
            Constraint::Cmp { lhs, op, rhs, .. } => write!(f, "{lhs} {op} {rhs}"),
            Constraint::Expression { term, .. } => write!(f, "{term}"),
        }
    }
}

#[derive(strum_macros::Display, Debug, Clone)]
#[repr(C)]
pub enum Predicate {
    // Conjunction of two predicates, e.g. x && y
    #[strum(to_string = "({0}) && ({1})")]
    And(Handle<Predicate>, Handle<Predicate>),
    /// Disjunction of two predicates, e.g. x || y
    #[strum(to_string = "({0}) || ({1})")]
    Or(Handle<Predicate>, Handle<Predicate>),
    /// The negation of a predicate, e.g. !x
    #[strum(to_string = "!({0})")]
    Not(Handle<Predicate>),
    /// A comparison expression, e.g., x == y
    #[strum(to_string = "({1}) {0} ({2})")]
    Comparison(CmpOp, Term, Term),
    /// A single variable, e.g. x. Variable should be a boolean.
    #[strum(to_string = "{0}")]
    Unit(Term),

    /// The literal False predicate
    #[strum(to_string = "false")]
    False,

    /// The literal True predicate.
    #[strum(to_string = "true")]
    True,
}

impl From<&Arc<Predicate>> for Term {
    fn from(pred: &Arc<Predicate>) -> Self {
        Term::Predicate(pred.clone())
    }
}

impl From<&Term> for Term {
    fn from(term: &Term) -> Self {
        term.clone()
    }
}

impl From<Arc<Predicate>> for Term {
    fn from(pred: Arc<Predicate>) -> Self {
        Term::Predicate(pred)
    }
}

impl From<Predicate> for Term {
    fn from(pred: Predicate) -> Self {
        Term::Predicate(pred.into())
    }
}

impl From<Term> for Arc<Predicate> {
    fn from(term: Term) -> Self {
        match term {
            Term::Predicate(pred) => pred,
            _ => Arc::new(Predicate::Unit(term)),
        }
    }
}

impl From<Term> for Predicate {
    fn from(term: Term) -> Self {
        match term {
            Term::Predicate(pred) => pred.as_ref().clone(),
            _ => Predicate::Unit(term),
        }
    }
}

impl From<&Term> for Predicate {
    fn from(term: &Term) -> Self {
        Predicate::Unit(term.clone())
    }
}

impl Predicate {
    pub fn new_and<T, U>(lhs: T, rhs: U) -> Handle<Self>
    where
        T: Into<Handle<Self>>,
        U: Into<Handle<Self>>,
    {
        let (lhs, rhs) = (lhs.into(), rhs.into());
        match (lhs.as_ref(), rhs.as_ref()) {
            (Predicate::True, _) => rhs,
            (_, Predicate::True) => lhs,
            (Predicate::False, _) | (_, Predicate::False) => Predicate::False.into(),
            _ => Predicate::And(lhs, rhs).into(),
        }
    }

    pub fn new_or<T, U>(lhs: T, rhs: U) -> Handle<Self>
    where
        T: Into<Handle<Self>>,
        U: Into<Handle<Self>>,
    {
        let (lhs, rhs) = (lhs.into(), rhs.into());
        match (lhs.as_ref(), rhs.as_ref()) {
            (Predicate::False, _) => rhs,
            (_, Predicate::False) => lhs,
            (Predicate::True, _) | (_, Predicate::True) => Predicate::True.into(),
            _ => Predicate::Or(lhs, rhs).into(),
        }
    }
    pub fn new_not<T>(pred: T) -> Handle<Self>
    where
        T: Into<Handle<Self>>,
    {
        let pred: Handle<Self> = pred.into();
        match pred.as_ref() {
            Predicate::Not(t) => t.clone(),
            Predicate::True => Predicate::False.into(),
            Predicate::False => Predicate::True.into(),
            Predicate::Comparison(op, l, r) => {
                Predicate::Comparison(op.negation(), l.clone(), r.clone()).into()
            }
            _ => Predicate::Not(pred).into(),
        }
    }

    #[must_use]
    pub fn new_comparison(op: CmpOp, lhs: Term, rhs: Term) -> Self {
        Predicate::Comparison(op, lhs, rhs)
    }

    pub fn new_unit<T: Into<Term>>(var: T) -> Self {
        Predicate::Unit(var.into())
    }
}

#[derive(Debug)]
pub enum AbcExpression {
    BinaryOp(BinaryOp, Term, Term),
    /// A select expression, e.g., select(x, y, z)
    Select(Term, Term, Term),
    ArrayLength(Term),
    /// A function call, e.g., foo(x, y)
    /// This should correspond to a function that has been defined...
    Call {
        func: Handle<Summary>,
        args: Vec<Term>,
    },

    /// Cast a term to a scalar type, e.g. `i32(x)`
    Cast(Term, AbcScalar),

    /// Access a member of a struct, e.g. `x.y`
    FieldAccess {
        base: Term,
        ty: Handle<AbcType>,
        fieldname: String,
    },

    /// Access an element of an array, e.g. `x[3]`
    IndexAccess {
        base: Term,
        index: Term,
    },

    /// The empty expression. This is meant to be used as a placeholder for constraint operations that have no expression.
    /// Displaying this is therefore a parse error.
    Empty,
}

impl std::fmt::Display for AbcExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbcExpression::BinaryOp(op, lhs, rhs) => write!(f, "{lhs} {op} {rhs}"),
            AbcExpression::Select(pred, then_expr, else_expr) => {
                write!(f, "select({pred}, {then_expr}, {else_expr})")
            }
            AbcExpression::ArrayLength(var) => write!(f, "length({var})"),
            AbcExpression::Call { func, args } => {
                write!(
                    f,
                    "{}({})",
                    func,
                    args.iter()
                        .map(std::string::ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            AbcExpression::Cast(expr, ty) => write!(f, "cast({expr}, {ty})"),
            AbcExpression::FieldAccess {
                base, fieldname, ..
            } => {
                write!(f, "{base}.{fieldname}")
            }
            AbcExpression::IndexAccess { base, index } => write!(f, "{base}[{index}]"),
            AbcExpression::Empty => write!(f, "%PARSE_ERROR"),
        }
    }
}

// pub struct PredicateBlock {
//     /// The predicate that guards this block
//     guard: Handle<Predicate>,
//     /// The statements in this block.
//     statements: Vec<String>,
// }

// Design: We have a helper class. This helper class holds the variables.

/// Provides an interface to define a type in the constraint system.
#[derive(Clone, Debug)]
pub enum AbcType {
    // A user defined compound type.
    Struct {
        members: FastHashMap<String, Handle<AbcType>>,
    },

    /// An array with a known size.
    SizedArray {
        ty: Handle<AbcType>,
        size: std::num::NonZeroU32,
    },
    // An array with an unknown size.
    // what is an array with an override size?
    DynamicArray {
        ty: Handle<AbcType>,
    },
    Scalar(AbcScalar), // Vector { ty: Handle<AbcType>, size: u32 }, // Just means we can swizzle...

    /// A value that doesn't exist
    ///
    /// Currently used as the type of variables that cannot be used, but whose expressions are needed
    NoneType,
}

impl std::fmt::Display for AbcType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbcType::Struct { members } => {
                write!(f, "{{")?;
                for (name, ty) in members {
                    write!(f, "{name}: {ty}, ")?;
                }
                write!(f, "}}")
            }
            AbcType::SizedArray { ty, size } => write!(f, "[{ty}; {size}]"),
            AbcType::DynamicArray { ty } => write!(f, "[{ty}]"),
            AbcType::Scalar(scalar) => f.write_str(&scalar.to_string()),
            AbcType::NoneType => f.write_str("NoneType"),
        }
    }
}

/// Represents a scalar type. These are builtin types that have assumed
/// bounds on them. E.g., an i32 is assumed to have bounds -2^31 to 2^31 - 1.
///
/// Note that for Sint, Uint, and Float, the width is in **bytes**.
#[derive(Clone, Copy, Debug)]
pub enum AbcScalar {
    /// Signed integer type. The width is in bytes.
    Sint(u8),
    /// Unsigned integer type. The width is in bytes.
    Uint(u8),
    /// IEEE-754 Floating point type.
    Float(u8),
    /// Boolean type.
    Bool,

    /// Abstract integer type. That is, an integer type with unknown bounds.
    AbstractInt,

    /// Abstract floating point type. That is, a floating point type with unknown bounds.
    AbstractFloat,
}

impl std::fmt::Display for AbcScalar {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbcScalar::Sint(width) => write!(f, "i{}", width * 8),
            AbcScalar::Uint(width) => write!(f, "u{}", width * 8),
            AbcScalar::Float(width) => write!(f, "f{}", width * 8),
            AbcScalar::Bool => write!(f, "bool"),
            AbcScalar::AbstractInt => write!(f, "abstract int"),
            AbcScalar::AbstractFloat => write!(f, "abstract float"),
        }
    }
}

/// A summary is akin to a function reference.
/// For right now, we are just storing the name and the nargs...
#[derive(Clone, Debug)]
pub struct Summary {
    pub name: String,
    pub args: Vec<Term>,
    pub return_type: Handle<AbcType>,

    // TODO: Make use of constraints.
    // This will eventually store the constraints that are associated with the summary.
    // However, in the current implementation, it is going to do nothing.
    pub constraints: Vec<Constraint>,
}

/// Displaying a Summary just shows the name of the function.
impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

impl Summary {
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn nargs(&self) -> u8 {
        self.args.len() as u8
    }
    #[must_use]
    pub fn new(name: String, nargs: u8) -> Self {
        Summary {
            name,
            args: Vec::with_capacity(nargs as usize),
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
        }
    }

    /// Add an argument to the summary.
    pub(crate) fn add_argument(&mut self, arg: Term) {
        self.args.push(arg);
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

// Ret is a special Arc that is guaranteed to hold a predicate

lazy_static! {
    static ref RET: Term = Term::Var(Arc::new(Var {
        name: "@ret".to_string()
    }));
    pub static ref NONETYPE: Arc<AbcType> = Arc::new(AbcType::NoneType);
    pub static ref EMPTY_TERM: Term = Term::Expr(Arc::new(AbcExpression::Empty));
}

#[allow(clippy::missing_errors_doc)]
pub trait ConstraintInterface {
    /// Can be used to reference an element of the constraint system.
    type Handle<T>: Clone;
    /// The error type for the constraint interface
    type E;

    /// Get the handle to an empty expression.
    ///
    /// This is provided since empty expression is meant to be a singleton.
    fn empty_expression(&self) -> Term;

    /// Get the handle for the `NoneType`
    fn none_type(&self) -> Self::Handle<AbcType>;

    /// Add an expression into the constraint system.
    fn add_expression(&mut self, expr: AbcExpression) -> Result<Term, Self::E>;

    /// Add an argument to the function summary
    ///
    /// Name is the name of the argument.
    /// Returns a var handle.
    fn add_argument(&mut self, name: String, ty: Self::Handle<AbcType>) -> Result<Term, Self::E>;

    /// Makes a call expression.
    ///
    /// # Arguments
    /// * `func` - A handle to the function that is being invoked
    /// * `args` - The arguments to the function
    /// * `into` - If the result of the function is used, this is the variable that holds it.
    ///
    /// This returns a handle to the expression that allows it to be used in other expressions
    ///
    /// # Errors
    /// The implementation should return an error if the function called does not reside in the arena.
    fn make_call(
        &mut self,
        func: Self::Handle<Summary>,
        args: Vec<Term>,
        into: Option<Term>,
    ) -> Result<Term, Self::E>;

    /// Add a new constraint to the constraint system. Any active predicates are applied to the constraint to "filter" the domain of its expression.
    ///
    /// Note: These will need to be desugared.
    ///
    /// # Errors
    ///
    fn add_constraint(&mut self, lhs: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E>;

    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E>;

    /// Add a new constraint to the constraint system, marked by `source`
    fn add_tracked_constraint<T: std::fmt::Debug>(
        &mut self,
        lhs: Term,
        op: ConstraintOp,
        rhs: Term,
        source: &T,
    ) -> Result<(), Self::E>;

    /// Declare the variable in the constraint system.
    ///
    /// This passes by value since the variable is replaced with a handle to it.
    fn declare_var(&mut self, name: Var) -> Result<Term, Self::E>;

    /// Mark the type of a variable.
    /// If the type is an array, this will also mark the number of dimensions.
    /// If the type is an array with a fixed size, this will also mark the size of the array.
    fn mark_type(&mut self, term: Term, ty: self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the length of a variable. The type of the variable must be a dynamic array.
    fn mark_length(&mut self, var: Term, dim: u8, size: u64) -> Result<(), Self::E>;

    /// Mark a constraint, which is a predicate.
    fn mark_assumption(&mut self, assumption: Self::Handle<Predicate>) -> Result<(), Self::E>;

    /// Begin a predicate block. This indicates to the solver
    /// that all expressions that follow can be filtered by the predicate.
    /// Any constraint that falls within a predicate becomes a soft constraint
    ///
    /// In other words, it would be as if all constraints were of the form ``p -> c``
    /// Nested predicate blocks end up composing the predicates. E.g.,
    /// ``begin_predicate_block(p1)`` followed by ``begin_predicate_block(p2)`` would
    /// mark all constraints as ``p1 && p2 -> c``
    /// That is, when determining if the constraint is violated, the solver
    /// will essentially check p -> c.
    fn begin_predicate_block(&mut self, p: Term) -> Result<(), Self::E>;

    /// End the active predicate block.
    ///
    /// If there was a return statement within the block, then all future constraints are marked as ``!p -> c``
    ///
    /// # Errors
    /// Returns an error if there is no active predicate block.
    fn end_predicate_block(&mut self) -> Result<(), Self::E>;

    /// A summary block corresponds to a function.
    /// This will return a handle that can be used to access the summary for a function.
    ///
    /// # Errors
    /// Returns an error if there is already an active summary.
    fn begin_summary(&mut self, name: String, nargs: u8) -> Result<(), Self::E>;

    /// Ends the current summary, returning a handle that references it.
    ///
    /// # Errors
    /// Returns an error if there is no active summary.
    fn end_summary(&mut self) -> Result<Self::Handle<Summary>, Self::E>;

    /// Marks a return statement
    ///
    ///
    /// All future constraints that are added will be marked as `[!p] -> [c]`
    ///
    /// When an IR system sees a return statement, the
    /// Proper handling of control flow is difficult.
    /// Rather than require the user of this interface to precisely
    /// model control flow and its impact on the constraint system,
    /// the interface provides a way to mark a point at which the return occurs.
    /// When this occurs within a block, the IR system must end the current block immediately after
    /// the return. If this occured within a predicate block (e.g., an If statement), then
    /// [`ConstraintInterface::end_predicate_block`] must be called immediately after the return.
    ///
    /// At a low-level, what this function does is add a
    /// "global" predicate block to all future statements that appear
    /// The current predicate block is immediately popped.
    /// From then on, any time a predicate block is closed, it is immediately enclosed
    /// within a new predicate block that inverts the conditions that were popped.
    ///
    fn mark_return(&mut self, retval: Option<Term>) -> Result<(), Self::E>;

    fn mark_return_type(&mut self, ty: Self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the beginning of a loop.
    fn begin_loop<T>(&mut self, condition: T) -> Result<(), Self::E>
    where
        T: AsRef<Predicate>;

    /// Mark the end of a loop.
    fn end_loop(&mut self) -> Result<(), Self::E>;

    /// This is sugar for marking a predicate block with a single entry...
    fn mark_break(&mut self) -> Result<(), Self::E>;

    /// Marks a continue statement
    fn mark_continue(&mut self) -> Result<(), Self::E>;

    /// Mark the range of a term.
    fn mark_range<T>(&mut self, var: Term, low: T, high: T) -> Result<(), Self::E>
    where
        T: ToString;

    // Marks a call to a function.
}

// I need a container for my terms...

// Finally, I need a container for my expressions?

// Then I can give all of these things to the helper?

/// A term is used to capture both variables and expressions
///
/// It makes marking constraints easier, as this allows
/// for both to be used interchangably.\
///
/// It also simplifies the logic for storing references to variables.
///
/// Sometimes, we would like to be able to use a variable as a constraint.
#[derive(Clone, Debug, strum_macros::Display)]
pub enum Term {
    #[strum(to_string = "{0}")]
    Expr(Handle<AbcExpression>),
    #[strum(to_string = "{0}")]
    Var(Handle<Var>),
    #[strum(to_string = "{0}")]
    Literal(String),
    Predicate(Handle<Predicate>),
}

// `true` and `false` are predicates
// A predicate is really any expression that evaluates to a boolean.
impl From<bool> for Term {
    fn from(val: bool) -> Self {
        Self::Predicate(if val {
            Predicate::True.into()
        } else {
            Predicate::False.into()
        })
    }
}

impl From<u32> for Term {
    fn from(val: u32) -> Self {
        Term::Literal(val.to_string())
    }
}

impl From<i32> for Term {
    fn from(val: i32) -> Self {
        Term::Literal(val.to_string())
    }
}

impl From<u64> for Term {
    fn from(val: u64) -> Self {
        Term::Literal(val.to_string())
    }
}

impl From<i64> for Term {
    fn from(val: i64) -> Self {
        Term::Literal(val.to_string())
    }
}

impl From<f32> for Term {
    fn from(val: f32) -> Self {
        Term::Literal(val.to_string())
    }
}

impl From<f64> for Term {
    fn from(val: f64) -> Self {
        Term::Literal(val.to_string())
    }
}

/*
Implementation of predicate constructors for term
*/

impl Term {
    /// Creates lhs && rhs
    #[must_use]
    pub fn new_logical_and(lhs: Term, rhs: Term) -> Self {
        Term::Predicate(Predicate::new_and(lhs, rhs))
    }

    /// Constructs lhs || rhs
    #[must_use]
    pub fn new_logical_or(lhs: Term, rhs: Term) -> Self {
        Term::Predicate(Predicate::new_or(lhs, rhs))
    }

    /// Constructs lhs `op` rhs
    #[must_use]
    pub fn new_comparison(op: CmpOp, lhs: Term, rhs: Term) -> Self {
        Term::Predicate(Predicate::new_comparison(op, lhs, rhs).into())
    }

    /// Constructs !t
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[must_use]
    pub fn new_not(t: Term) -> Self {
        Term::Predicate(match t {
            Term::Predicate(pred) => Predicate::new_not(pred),
            _ => Predicate::new_not(Predicate::new_unit(t)),
        })
    }
}

impl Term {
    /// Constructs a new `Term::Var`
    #[must_use]
    pub fn new_var<T>(var: T) -> Self
    where
        T: Into<Handle<Var>>,
    {
        Term::Var(var.into())
    }

    #[must_use]
    pub fn new_expr<T>(expr: T) -> Self
    where
        T: Into<Handle<AbcExpression>>,
    {
        Term::Expr(expr.into())
    }

    #[must_use]
    pub fn new_call(func: Handle<Summary>, args: Vec<Term>) -> Self {
        Term::Expr(AbcExpression::Call { func, args }.into())
    }
    /// Wrapper around making a new predicate and placing the predicate in this expression.
    #[must_use]
    pub fn new_cmp_op(op: CmpOp, lhs: Term, rhs: Term) -> Self {
        Term::Predicate(Predicate::new_comparison(op, lhs, rhs).into())
    }

    #[must_use]
    pub fn new_index_access(base: Term, index: Term) -> Self {
        AbcExpression::IndexAccess { base, index }.into()
    }

    /// Creates a new field access expression
    #[must_use]
    pub fn new_struct_access(base: Term, fieldname: String, ty: Handle<AbcType>) -> Self {
        AbcExpression::FieldAccess {
            base,
            fieldname,
            ty,
        }
        .into()
    }

    #[must_use]
    pub fn new_literal<T>(lit: &T) -> Self
    where
        T: ToString,
    {
        Self::Literal(lit.to_string())
    }

    #[must_use]
    pub fn new_binary_op(op: BinaryOp, lhs: Term, rhs: Term) -> Self {
        AbcExpression::BinaryOp(op, lhs, rhs).into()
    }

    #[must_use]
    pub fn new_select(pred: Term, then_expr: Term, else_expr: Term) -> Self {
        // In select, term *should* be a predicate.
        // Otherwise, we have to make it into one.
        let pred = Term::Predicate(match pred {
            Term::Predicate(p) => p.clone(),
            _ => Predicate::new_unit(pred).into(),
        });
        AbcExpression::Select(pred, then_expr, else_expr).into()
    }

    #[must_use]
    pub fn make_array_length(var: Term) -> Self {
        AbcExpression::ArrayLength(var).into()
    }
}

impl From<AbcExpression> for Term {
    fn from(expr: AbcExpression) -> Self {
        Term::Expr(Arc::new(expr))
    }
}

impl From<Var> for Term {
    fn from(var: Var) -> Self {
        Term::Var(Arc::new(var))
    }
}

impl From<Handle<AbcExpression>> for Term {
    fn from(expr: Handle<AbcExpression>) -> Self {
        Term::Expr(expr)
    }
}

impl From<Handle<Var>> for Term {
    fn from(var: Handle<Var>) -> Self {
        Term::Var(var)
    }
}

#[derive(Default)]
pub struct ConstraintHelper {
    /// For right now, the handles are just Arcs.
    /// This is so that they have an easy time printing
    ///
    /// When we make the interface with the constraint solver itself, we will use a real handle, like naga's arena.
    ///
    /// Thus, we don't actually need containers for the things we hold.
    /// We would just need to be
    /// Track the active predicate context, if we are in one.
    /// If the stack is empty, we are not in a predicate stack.
    predicate_stack: Vec<Handle<Predicate>>,

    /// When popping a predicate stack, if we had a return, then we add the inverse of the condition to the global predicate stack's expression list
    /// For example, if we had
    /// ```wgsl
    /// if (x > 0) {
    ///    y = 1;
    ///    if (x < 10) {
    ///      y_1 = 2;
    ///      return;
    ///    }
    ///    w_1 = 14;
    ///    if (z <= 11 ) {
    ///      w_2 = 9;
    ///      return;
    ///    }
    ///    y_2 = 3;
    /// }
    /// y_3 = 4;
    /// ```
    /// Then we would have the following constraints:
    ///
    /// ```wgsl
    /// {x > 0} y = 1
    /// { x < 10 } y_1 = 2
    /// {!( x < 10 ) } w_1 = 14
    /// {!( x < 10 ) && z <= 11 } w_2 = 9
    /// {!(x < 10 || z <= 11)} y_2 = 3
    /// {!(x < 10 || z <= 11)} y_3 = 4
    /// ```
    /// E.g., if we hit a return, then the new return predicate becomes the `or` of the existing one.
    return_predicate: Option<Handle<Predicate>>,

    /// The set of predicates that are appended to all future constraints.
    ///
    /// This is used in conjunction with the `had_return` field.
    /// As soon as a return is marked, its predicate is appended to the `permanent_predicate` field.
    permanent_predicate: Option<Handle<Predicate>>,

    /// Summaries can't be nested, so we only need to track what the active one is.
    /// When we pop a summary, we clear out the predicate stack.
    active_summary: Option<Summary>,

    // Statements are the set of statements that comprise the constraint system...
    statements: Vec<String>,

    /// Summaries contain the set of summaries that have previously been parsed.
    summaries: Vec<Handle<Summary>>,

    types: Vec<Handle<AbcType>>,

    /// Global constraints not tied to a function
    global_constraints: Vec<Constraint>,

    /// Expression counters.
    /// This is a map from an expression to the counter for its current value.
    #[allow(dead_code)]
    expression_ssa_counter: FastHashMap<Handle<AbcExpression>, u32>,

    /// The number of loop layers that are active.
    ///
    /// Incremented when a loop begins, decremented when a loop ends.
    loop_depth: u8,
}

impl From<String> for Var {
    fn from(name: String) -> Self {
        Var { name }
    }
}

impl From<&str> for Var {
    fn from(name: &str) -> Self {
        Var {
            name: name.to_string(),
        }
    }
}

impl ConstraintHelper {
    /// Creates the predicate guard, if there should be one.
    ///
    /// # Returns
    /// `None` if there should be no predicate guard.
    fn make_guard(&self) -> Option<Handle<Predicate>> {
        let pred_block_pred = self.join_predicate_stack();
        match (&self.permanent_predicate, pred_block_pred) {
            (Some(perm), Some(pred)) => Some(Predicate::new_and(perm.clone(), pred)),
            (Some(ref p), None) | (None, Some(ref p)) => Some(p.clone()),
            (None, None) => None,
        }
    }

    fn write(&mut self, s: String) {
        println!("{}", &s);
        self.statements.push(s);
    }

    /// Append a comment after the end of the last statement.
    fn append_last(&mut self, s: String) {
        println!("/* {} */", &s);
        if let Some(o) = self.statements.last_mut() {
            o.push(' ');
            o.push_str(&s);
        } else {
            self.write(s);
        }
    }

    #[allow(unused)]
    /// Mark the length of a dimension
    /// Not sure if this is even needed, honestly.
    fn mark_ndim(&mut self, term: &Term, ndim: u8) {
        self.write(format!("ndim({term}) = {ndim}"));
    }

    /// Write the constraints for the module to the provided stream
    ///
    /// # Errors
    /// Propagates any errors encountered when writing to the provided `stream`
    pub fn write_to_stream(
        &self,
        stream: &mut impl std::io::Write,
    ) -> Result<usize, std::io::Error> {
        stream.write(self.statements.join("\n").as_bytes())
    }

    /// Join the predicate stack together, returning a predicate which is the conjunction of all predicates in the stack.
    fn join_predicate_stack(&self) -> Option<Handle<Predicate>> {
        self.predicate_stack
            .iter()
            .cloned()
            .reduce(Predicate::new_and)
    }
}

impl ConstraintInterface for ConstraintHelper {
    type Handle<T> = std::sync::Arc<T>;
    type E = ConstraintError;

    /// A reference to the empty expression in this constraint system.
    ///
    /// The empty expression is meant to be a singleton, and this provides a wrapper to reference it.
    fn empty_expression(&self) -> Term {
        EMPTY_TERM.clone()
    }
    /// A reference to the `NoneType` in this constraint system.
    ///
    /// Nonetype is meant to be a singleton, and this provides a wrapper to reference it.
    fn none_type(&self) -> Self::Handle<AbcType> {
        NONETYPE.clone()
    }

    /// Define a type that is used in the summary.
    ///
    /// This returns a handle that can be used to refer to the type.
    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E> {
        let ty: Self::Handle<AbcType> = ty.into();
        self.types.push(ty.clone());
        Ok(ty)
    }

    /// Add the constraints of the summary by invoking the function call with the proper arguments.
    ///
    /// This will add the constraints of the summary to the constraint system.
    fn make_call(
        &mut self,
        func: Self::Handle<Summary>,
        args: Vec<Term>,
        into: Option<Term>,
    ) -> Result<Term, Self::E> {
        // We have to add the constraints from the summary.
        // For now, we don't do this, but we will have to on the desugaring pass.
        let call = self.add_expression(AbcExpression::Call {
            func: func.clone(),
            args,
        })?;

        if let Some(into) = into {
            // Turn the var into an expression, and return said expression.
            self.mark_type(into.clone(), func.return_type.clone())?;
            self.add_constraint(into.clone(), ConstraintOp::Assign, call)?;
            Ok(into)
        } else {
            // At this point, we need to add the constraints of all of the terms in the function.
            self.add_constraint(call.clone(), ConstraintOp::Unary, EMPTY_TERM.clone())?;
            Ok(call)
        }
    }

    fn add_argument(&mut self, name: String, ty: Self::Handle<AbcType>) -> Result<Term, Self::E> {
        let mut argname = String::with_capacity(name.len() + 1);
        argname.push('@');
        argname.push_str(&name);
        let var = self.declare_var(Var { name: argname })?;
        self.mark_type(var.clone(), ty)?;
        match self.active_summary {
            Some(ref mut summary) => {
                summary.add_argument(var.clone());
                Ok(var)
            }
            None => Err(ConstraintError::SummaryError),
        }
    }

    /// Add an expression to the constraint system, returning a handle to it.
    ///
    /// # Errors
    /// Never (always returns an Ok)
    /// fn
    fn add_expression(&mut self, expr: AbcExpression) -> Result<Term, Self::E> {
        Ok(Term::Expr(expr.into()))
    }

    /// Create a new variable, returning a handle to that variable.
    ///
    /// The interface for variables may change in the future so it is recommended to use this instead..
    /// We won't do anything about expressions
    /// These will have to be created.
    fn declare_var(&mut self, name: Var) -> Result<Term, ConstraintError> {
        Ok(Term::Var(Arc::new(name)))
    }

    fn mark_break(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("break".to_string()))
    }

    fn mark_continue(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("continue".to_string()))
    }

    /// Mark the type of the variable.
    fn mark_type(&mut self, var: Term, ty: Self::Handle<AbcType>) -> Result<(), Self::E> {
        // If we are already given a handle, then we just use that
        // Otherwise, if we are given a string, then we create a new var.

        // if let Some(arc_var) = Arc::downcast::<Var>(var)
        self.write(format!("type({var}) = {ty}"));
        Ok(())
    }

    /// Mark the return type for the active summary.
    ///
    /// # Errors
    /// Returns [`ConstraintError::SummaryError`] if there is no active summary.
    fn mark_return_type(&mut self, ty: Self::Handle<AbcType>) -> Result<(), Self::E> {
        self.write(format!("type(@ret) = {ty}"));
        match self.active_summary {
            None => Err(ConstraintError::SummaryError),
            Some(ref mut summary) => {
                summary.return_type = ty;
                Ok(())
            }
        }
    }

    fn add_tracked_constraint<T: std::fmt::Debug>(
        &mut self,
        term: Term,
        op: ConstraintOp,
        rhs: Term,
        source: &T,
    ) -> Result<(), Self::E> {
        // self.write(format!("/* {:?} */", source));
        self.add_constraint(term, op, rhs)?;
        self.append_last(format!("{source:?}"));
        // Then, just write the source
        Ok(())
    }

    /// Add a constraint to the system that narrows the domain of `term`
    ///
    /// Any active predicates are applied.
    ///
    /// If source is given and not none, then the source is used to provide context for the constraint.
    /// It really should implement the `ToString` trait.
    fn add_constraint(&mut self, term: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E> {
        // build the predicate. To start with, we have the permanent predicate
        // Then, we have the AND of the current predicate stack.

        let guard = self.make_guard();
        let new_constraint = match op {
            ConstraintOp::Unary => Constraint::Expression { guard, term },
            // An empty rhs is not allowed unless this is a unary constraint.
            _ if matches!(rhs, Term::Expr(ref e) if matches!(e.as_ref(), AbcExpression::Empty)) => {
                return Err(ConstraintError::EmptyExpression);
            }
            ConstraintOp::Assign => Constraint::Assign {
                lhs: term,
                rhs,
                guard,
            },
            ConstraintOp::Cmp(op) => Constraint::Cmp {
                guard,
                lhs: term,
                op,
                rhs,
            },
            // Allow sus to add constraints in the future without having to implement them.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(ConstraintError::NotImplemented(format!(
                    "ConstraintOp::{op:?}"
                )))
            }
        };
        self.write(new_constraint.to_string());
        // Now we add the constraint.
        match self.active_summary {
            Some(ref mut summary) => &mut summary.constraints,
            None => &mut self.global_constraints,
        }
        .push(new_constraint);
        Ok(())
    }

    /// When we encounter a return, push the current predicate on the current, if there is a current predicate stack...
    fn mark_return(&mut self, retval: Option<Term>) -> Result<(), ConstraintError> {
        if self.active_summary.is_none() {
            return Err(ConstraintError::SummaryError);
        }
        // Add the constraint on retval, if there is a retval.
        if let Some(expr) = retval {
            self.add_constraint(RET.clone(), ConstraintOp::Assign, expr)?;
        }
        // It is likely an error if the return predicate already exists and is not None,
        // But just in case, we will make the return predicate the `or` of the current predicate stack...
        let pred_ctx = self
            .predicate_stack
            .last()
            .cloned()
            .unwrap_or(Arc::new(Predicate::True));
        match &self.return_predicate {
            None => {
                self.return_predicate = Some(pred_ctx.clone());
            }
            Some(pred) => {
                self.return_predicate = Some(Predicate::new_or(pred.clone(), pred_ctx.clone()));
            }
        };
        Ok(())
    }

    #[allow(unused_variables)]
    /// Mark an assumption, which is a predicate about a primitive type.
    fn mark_assumption(&mut self, assumption: Handle<Predicate>) -> Result<(), ConstraintError> {
        Err(ConstraintError::NotImplemented(
            "mark_assumption".to_string(),
        ))
    }

    /// Mark the length of an array's dimension. Used for arrays with a fixed size.
    fn mark_length(&mut self, var: Term, dim: u8, size: u64) -> Result<(), ConstraintError> {
        self.write(format!("length({var}, {dim}) = {size}"));
        Ok(())
    }

    /// Mark the range of a variable. This works as an assumption when the range of a variable is fixed.
    fn mark_range<T>(&mut self, var: Term, low: T, high: T) -> Result<(), ConstraintError>
    where
        T: ToString,
    {
        self.write(format!(
            "range({var}) \\in {{{}, {}}}",
            low.to_string(),
            high.to_string()
        ));
        Ok(())
    }

    /// Push a predicate block onto the stack
    fn begin_predicate_block(&mut self, p: Term) -> Result<(), ConstraintError> {
        // In this form, every subsequent constraint we add is guarded by the conjunction of each predicate...
        let pred: Self::Handle<Predicate> = match p {
            Term::Predicate(p) => p,
            _ => Predicate::new_unit(p).into(),
        };
        log_info!("Beginning predicate block: {}", pred.as_ref());
        self.predicate_stack.push(pred);
        Ok(())
    }

    /// Ends the previous predicate block.
    ///
    /// Returns an error if the predicate stack is empty.
    fn end_predicate_block(&mut self) -> Result<(), ConstraintError> {
        // Predicate block is sugar. It disappears, so there should be no use for it...
        log_info!("Popping predicate");
        // Make the permanent predicate the negation of the current predicate.

        // If we currently have a return predicate, then the permanent predicate becomes the And of the current permanent predicate and the negation of the return predicate.
        // When we do this, the return predicate becomes the permanent predicate, right?
        // This is because we would do !(a || b) which would translate to !(a) && !(b)
        if let Some(p) = self.return_predicate.take() {
            let p = Predicate::new_not(p);
            self.permanent_predicate = self
                .permanent_predicate
                .clone()
                .map_or(Some(p.clone()), move |perm| {
                    Predicate::new_and(perm, p).into()
                });
        };

        self.predicate_stack
            .pop()
            .map_or(Err(ConstraintError::PredicateStackEmpty), |_| Ok(()))
    }

    /// Begins a loop context.
    ///
    /// When we are inside of a loop context, any update to a variable is marked as a range constraint.
    /// It also allows for special handling of break and continue statements.
    /// HOWEVER, for the time being, we can't do any of this fancy handling
    /// So we are just going to emit a "`begin_loop(condition)`" statement, and will assume this is a part of the constraint system.
    fn begin_loop<T>(&mut self, condition: T) -> Result<(), ConstraintError>
    where
        T: AsRef<Predicate>,
    {
        self.write(format!("begin_loop({})", condition.as_ref()));
        // We begin the predicate block here...
        // self.begin_predicate_block(condition);
        if self.loop_depth == u8::MAX {
            return Err(ConstraintError::MaxLoopDepthExceeded);
        }
        self.loop_depth += 1;
        Ok(())
    }

    fn end_loop(&mut self) -> Result<(), ConstraintError> {
        self.write("end_loop()".to_string());
        if self.loop_depth == 0 {
            return Err(ConstraintError::NotInLoopContext);
        }
        self.loop_depth -= 1;
        Ok(())
    }

    /// Begin a summary block.
    #[inline]
    fn begin_summary(&mut self, name: String, nargs: u8) -> Result<(), ConstraintError> {
        // Mini optimization for allocating the perfect amount of characters needed for the string
        let str_len = 17
            + name.len()
            + match nargs {
                0..=9 => 1,
                10..=99 => 2,
                100..=255 => 3,
            };

        // The most efficient way to construct a string in rust.
        let mut param_list_str = String::with_capacity(str_len);

        param_list_str.push_str("begin_summary(");
        param_list_str.push_str(&name);
        param_list_str.push_str(", ");
        param_list_str.push_str(&nargs.to_string());
        param_list_str.push(')');
        self.write(param_list_str);

        // Create a new summary
        let new_summary = Summary {
            name,
            args: Vec::with_capacity(nargs as usize),
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
        };
        self.active_summary = Some(new_summary);

        Ok(())
    }

    /// End a summary block.
    fn end_summary(&mut self) -> Result<Self::Handle<Summary>, ConstraintError> {
        let summary = self
            .active_summary
            .take()
            .ok_or(ConstraintError::SummaryError)?;
        let mut fmt_str = String::with_capacity(13 + summary.name.len());
        fmt_str.push_str("end_summary(");
        fmt_str.push_str(&summary.name);
        fmt_str.push(')');
        self.write(fmt_str);

        // Clear the state from the current summary.
        self.return_predicate = None;
        self.permanent_predicate = None;
        self.predicate_stack.clear();

        let summary = Arc::new(summary);
        self.summaries.push(summary.clone());
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn constraint_helper<'a>() -> ConstraintHelper {
        ConstraintHelper::default()
    }

    fn fresh_var(constraint_helper: &mut ConstraintHelper, name: String) -> Term {
        constraint_helper.declare_var(name.into()).unwrap()
    }

    fn check_constraint_output(helper: ConstraintHelper, expected: &str) {
        let mut stream = Vec::new();
        helper.write_to_stream(&mut stream).unwrap();
        assert_eq!(String::from_utf8(stream).unwrap(), expected.to_string());
    }

    #[rstest]
    fn test_mark_type(mut constraint_helper: ConstraintHelper) {
        let var = fresh_var(&mut constraint_helper, "x".to_string());
        // mark the type
        let ty = constraint_helper
            .declare_type(AbcType::Scalar(AbcScalar::Sint(4)))
            .unwrap();
        assert!(constraint_helper.mark_type(var, ty).is_ok());
        check_constraint_output(constraint_helper, "type(x) = i32");
    }

    #[rstest]
    fn test_mark_ndim(mut constraint_helper: ConstraintHelper) {
        let x = fresh_var(&mut constraint_helper, "x".to_string());
        constraint_helper.mark_ndim(&x, 3);
        check_constraint_output(constraint_helper, "ndim(x) = 3");
    }

    #[fixture]
    fn var_x() -> Var {
        Var {
            name: "x".to_string(),
        }
    }

    #[rstest]
    fn test_predicate_not(var_x: Var) {
        let p = Predicate::new_not(Predicate::new_unit(var_x));
        assert_eq!(p.to_string(), "!(x)");
    }

    #[rstest]
    fn test_predicate_not_not(var_x: Var) {
        let p = Predicate::new_not(Predicate::new_unit(var_x));
        let p2 = Predicate::new_not(p);
        assert_eq!(p2.to_string(), "x");
    }

    // Test terms.
    #[rstest]
    fn test_term_from_var(var_x: Var) {
        let term = Term::from(var_x.clone());
        assert_eq!(term.to_string(), "x");
    }
}

// /// FFI bindings for the rust library.
// #[cfg(feature = "cffi")]
// pub mod ffi_bindings;
