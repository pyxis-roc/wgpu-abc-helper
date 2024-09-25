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

AbcHelper's summaries capture the constraints within a function that must be met
in order for the array accesses to be in bounds. It also generates constraints that
narrow the return value of the function.

When the constraint solver sees a function call, it applies the constraints from the summary
while narrowing the domain of the arguments passed to the function based on its own constraints.


## Control Flow

As with any static analysis, proper handling of loops is tricky.
In an effort to avoid overapproximating the constraints, the constraint helper
provides a few mechanisms to handle loops.


[`Summary`]: ::Summary
[`Type`]: ::AbcType
[`Var`]: ::Var
[`Expression`]: ::AbcExpression
[`ConstraintHelper`]: ::ConstraintHelper
*/

use std::sync::Arc;
// use std::rc::Rc

type FastHashMap<K, V> = rustc_hash::FxHashMap<K, V>;

use lazy_static::lazy_static;

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

#[derive(strum_macros::Display, Debug, Clone, Copy)]
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
        lhs: Handle<AbcExpression>,
        rhs: Handle<AbcExpression>,
    },

    Cmp {
        guard: Option<Handle<Predicate>>,
        lhs: Handle<AbcExpression>,
        op: CmpOp,
        rhs: Handle<AbcExpression>,
    },

    Expression {
        guard: Option<Handle<Predicate>>,
        term: Handle<AbcExpression>,
    },
}

impl Constraint {
    /// Return the guard portion of the constraint
    fn guard(&self) -> Option<Handle<Predicate>> {
        match self {
            Constraint::Assign { guard, .. } => guard.clone(),
            Constraint::Cmp { guard, .. } => guard.clone(),
            Constraint::Expression { guard, .. } => guard.clone(),
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
    Comparison(CmpOp, Handle<AbcExpression>, Handle<AbcExpression>),
    /// A single variable, e.g. x. Variable should be a boolean.
    #[strum(to_string = "{0}")]
    Unit(Handle<Var>),

    Expression(Handle<AbcExpression>),

    /// The literal False predicate
    #[strum(to_string = "false")]
    False,

    /// The literal True predicate.
    #[strum(to_string = "true")]
    True,
}

impl Predicate {
    /// Extract the predicate from an expression.
    pub fn from_expression(expr: AbcExpression) -> Handle<Self> {
        match expr {
            AbcExpression::Pred(pred) => pred.clone(),
            AbcExpression::Var(var) => Predicate::Unit(var).into(),
            AbcExpression::Literal(t) if t == "true" => Predicate::True.into(),
            AbcExpression::Literal(t) if t == "false" => Predicate::False.into(),
            _ => Predicate::Expression(expr.into()).into(),
        }
    }
    /// Extract the predicate from an expression handle.
    pub fn from_expression_handle(expr: Handle<AbcExpression>) -> Handle<Self> {
        match expr.as_ref() {
            AbcExpression::Pred(pred) => pred.clone(),
            AbcExpression::Var(var) => Predicate::Unit(var.clone()).into(),
            _ => Predicate::Expression(expr.clone()).into(),
        }
    }
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
            _ => Predicate::Not(pred).into(),
        }
    }

    pub fn new_comparison<T, U>(op: CmpOp, lhs: T, rhs: U) -> Self
    where
        T: Into<Handle<AbcExpression>>,
        U: Into<Handle<AbcExpression>>,
    {
        Predicate::Comparison(op, lhs.into(), rhs.into())
    }

    pub fn new_unit<T>(var: T) -> Self
    where
        T: Into<Handle<Var>>,
    {
        Predicate::Unit(var.into())
    }
}

#[derive(strum_macros::Display, Debug)]
pub enum AbcExpression {
    #[strum(to_string = "{0}")]
    Var(Handle<Var>),
    #[strum(to_string = "{0}")]
    Literal(String),
    #[strum(to_string = "({1} {0} {2})")]
    BinaryOp(BinaryOp, Handle<AbcExpression>, Handle<AbcExpression>),
    #[strum(to_string = "select({0}, {1}, {2})")]
    Select(
        Handle<Predicate>,
        Handle<AbcExpression>,
        Handle<AbcExpression>,
    ),
    #[strum(to_string = "{0}")]
    Pred(Handle<Predicate>),
    #[strum(to_string = "length({0})")]
    ArrayLength(Handle<AbcExpression>),
    /// A function call, e.g., foo(x, y)
    /// This should correspond to a function that has been defined...
    #[strum(to_string = "{func}({args:?})")]
    Call {
        func: Handle<Summary>,
        args: Vec<Handle<AbcExpression>>,
    },
    #[strum(to_string = "cast({0}, {1})")]
    Cast(Handle<AbcExpression>, AbcScalar),
    #[strum(to_string = "{base}.{fieldname}")]
    FieldAccess {
        base: Handle<AbcExpression>,
        ty: Handle<AbcType>,
        fieldname: String,
    },
    #[strum(to_string = "{base}[{index}]")]
    IndexAccess {
        base: Handle<AbcExpression>,
        index: Handle<AbcExpression>,
    },

    /// The empty expression. This is meant to be used as a placeholder for constraint operations that have no expression.
    /// Displaying this is therefore a parse error.
    #[strum(to_string = "%PARSE_ERROR")]
    Empty,
}

impl From<Handle<Var>> for AbcExpression {
    fn from(var: Handle<Var>) -> Self {
        AbcExpression::Var(var.clone())
    }
}

impl From<Var> for AbcExpression {
    fn from(var: Var) -> Self {
        AbcExpression::Var(Arc::new(var))
    }
}

impl From<Predicate> for AbcExpression {
    fn from(pred: Predicate) -> Self {
        AbcExpression::Pred(pred.into())
    }
}

impl From<Handle<Predicate>> for AbcExpression {
    fn from(pred: Handle<Predicate>) -> Self {
        AbcExpression::Pred(pred)
    }
}

impl AbcExpression {
    pub fn new_call(func: Handle<Summary>, args: Vec<Handle<AbcExpression>>) -> Self {
        AbcExpression::Call { func, args }
    }
    /// Wrapper around making a new predicate and placing the predicate in this expression.
    pub fn new_cmp_op(op: CmpOp, lhs: Handle<AbcExpression>, rhs: Handle<AbcExpression>) -> Self {
        AbcExpression::Pred(Predicate::new_comparison(op, lhs, rhs).into())
    }

    pub fn new_index_access(base: Handle<AbcExpression>, index: Handle<AbcExpression>) -> Self {
        AbcExpression::IndexAccess { base, index }
    }

    /// Creates a new field access expression
    pub fn new_struct_access(
        base: Handle<AbcExpression>,
        fieldname: String,
        ty: Handle<AbcType>,
    ) -> Self {
        AbcExpression::FieldAccess {
            base,
            fieldname,
            ty,
        }
    }
    pub fn new_var<T>(var: T) -> Self
    where
        T: Into<Handle<Var>>,
    {
        AbcExpression::Var(var.into())
    }

    pub fn new_literal<T>(lit: T) -> Self
    where
        T: ToString,
    {
        AbcExpression::Literal(lit.to_string())
    }

    pub fn new_binary_op<T, U>(op: BinaryOp, lhs: T, rhs: U) -> Self
    where
        T: Into<Handle<AbcExpression>>,
        U: Into<Handle<AbcExpression>>,
    {
        AbcExpression::BinaryOp(op, lhs.into(), rhs.into())
    }

    pub fn new_select<T, U, V>(pred: T, then_expr: U, else_expr: V) -> Self
    where
        T: Into<Handle<Predicate>>,
        U: Into<Handle<AbcExpression>>,
        V: Into<Handle<AbcExpression>>,
    {
        AbcExpression::Select(pred.into(), then_expr.into(), else_expr.into())
    }

    pub fn new_pred<T>(pred: T) -> Self
    where
        T: Into<Handle<Predicate>>,
    {
        AbcExpression::Pred(pred.into())
    }

    pub fn make_array_length<T>(var: T) -> Self
    where
        T: Into<Arc<AbcExpression>>,
    {
        AbcExpression::ArrayLength(var.into())
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
                for (name, ty) in members.iter() {
                    write!(f, "{}: {}, ", name, ty)?;
                }
                write!(f, "}}")
            }
            AbcType::SizedArray { ty, size } => write!(f, "[{}; {}]", ty, size),
            AbcType::DynamicArray { ty } => write!(f, "[{}]", ty),
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
    pub nargs: u8,
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
    pub fn new(name: String, nargs: u8) -> Self {
        Summary {
            name,
            nargs,
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
        }
    }

    /// Initialize a sumamry with a capacity for the number of constraints.
    pub fn with_capacity(name: String, nargs: u8, capacity: usize) -> Self {
        Summary {
            name,
            nargs,
            constraints: Vec::with_capacity(capacity),
            return_type: NONETYPE.clone(),
        }
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

// Ret is a special Arc that is guaranteed to hold a predicate

lazy_static! {
    static ref RET: Arc<Var> = Arc::new(Var {
        name: "@ret".to_string()
    });
    static ref RET_EXPR: Arc<AbcExpression> = Arc::new(AbcExpression::Var(RET.clone()));
    pub static ref NONETYPE: Arc<AbcType> = Arc::new(AbcType::NoneType);
    pub static ref EMPTY_EXPR: Arc<AbcExpression> = Arc::new(AbcExpression::Empty);
}

pub trait ConstraintInterface {
    /// Can be used to reference an element of the constraint system.
    type Handle<T>: Clone;
    /// The error type for the constraint interface
    type E;

    /// Get the handle to an empty expression.
    ///
    /// This is provided since empty expression is meant to be a singleton.
    fn empty_expression(&self) -> Self::Handle<AbcExpression>;

    /// Get the handle for the NoneType
    fn none_type(&self) -> Self::Handle<AbcType>;

    /// Add an expression into the constraint system.
    fn add_expression(
        &mut self,
        expr: AbcExpression,
    ) -> Result<Self::Handle<AbcExpression>, Self::E>;

    /// Makes a call expression.
    ///
    /// # Arguments
    /// * `func` - A handle to the function that is being invoked
    /// * `args` - The arguments to the function
    /// * `into` - If the result of the function is used, this is the variable that holds it.
    ///
    /// This returns a handle to the expression that allows it to be used in other expressions
    fn make_call(
        &mut self,
        func: Self::Handle<Summary>,
        args: Vec<Self::Handle<AbcExpression>>,
        into: Option<Self::Handle<Var>>,
    ) -> Result<Self::Handle<AbcExpression>, Self::E>;

    /// Add a new constraint to the constraint system. Any active predicates are applied to the constraint to "filter" the domain of its expression.
    ///
    /// Note: These will need to be desugared.
    fn add_constraint(
        &mut self,
        lhs: Self::Handle<AbcExpression>,
        op: ConstraintOp,
        rhs: Self::Handle<AbcExpression>,
    ) -> Result<(), Self::E>;

    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E>;

    /// Add a new constraint to the constraint system, marked by `source`
    fn add_tracked_constraint<T: std::fmt::Debug>(
        &mut self,
        lhs: Self::Handle<AbcExpression>,
        op: ConstraintOp,
        rhs: Self::Handle<AbcExpression>,
        source: &T,
    ) -> Result<(), Self::E>;

    /// Declare the variable in the constraint system.
    ///
    /// This passes by value since the variable is replaced with a handle to it.
    fn declare_var(&mut self, name: Var) -> Result<Self::Handle<Var>, Self::E>;

    /// Mark the type of a variable.
    /// If the type is an array, this will also mark the number of dimensions.
    /// If the type is an array with a fixed size, this will also mark the size of the array.
    fn mark_type(
        &mut self,
        var: Self::Handle<Var>,
        ty: self::Handle<AbcType>,
    ) -> Result<(), Self::E>;

    /// Mark the length of a variable. The type of the variable must be a dynamic array.
    fn mark_length(&mut self, var: Self::Handle<Var>, dim: u8, size: u64) -> Result<(), Self::E>;

    /// Mark a constraint, which is a predicate.
    fn mark_assumption(&mut self, assumption: Self::Handle<Predicate>) -> Result<(), Self::E>;

    /// Begin a predicate block. This indicates to the solver
    /// that all expressions that follow can be filtered by the predicate.
    /// Any constraint that falls within a predicate becomes a soft constraint
    ///
    /// In other words, it would be as if all constraints were of the form [p] -> [c]
    /// Nested predicate blocks end up composing the predicates. E.g.,
    /// `begin_predicate_block(p1)`` followed by `begin_predicate_block(p2)` would
    /// mark all constraints as [p1 && p2] -> [c]
    /// That is, when determining if the constraint is violated, the solver
    /// will essentially check p -> c.
    fn begin_predicate_block(&mut self, p: Self::Handle<Predicate>) -> Result<(), Self::E>;

    /// End the active predicate block.
    ///
    /// If there was a return statement within the block, then all future constraints are marked as [!p] -> c
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
    fn mark_return(&mut self, retval: Option<Self::Handle<AbcExpression>>) -> Result<(), Self::E>;

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

    /// Mark the range of a variable.
    fn mark_range<T>(&mut self, var: Handle<Var>, low: T, high: T) -> Result<(), Self::E>
    where
        T: ToString;

    // Marks a call to a function.
}

// I need a container for my terms...

// Finally, I need a container for my expressions?

// Then I can give all of these things to the helper?

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

    /// The types we know about..
    ///

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
    fn mark_ndim(&mut self, var: Handle<Var>, ndim: u8) {
        self.write(format!("ndim({var}) = {ndim}"));
    }

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
    fn empty_expression(&self) -> Self::Handle<AbcExpression> {
        EMPTY_EXPR.clone()
    }
    /// A reference to the NoneType in this constraint system.
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
        args: Vec<Self::Handle<AbcExpression>>,
        into: Option<Self::Handle<Var>>,
    ) -> Result<Self::Handle<AbcExpression>, Self::E> {
        // We have to add the constraints from the summary.
        // For now, we don't do this, but we will have to on the desugaring pass.
        let call = self.add_expression(AbcExpression::new_call(func.clone(), args))?;

        if let Some(into) = into {
            // Turn the var into an expression, and return said expression.
            let var_as_expr = self.add_expression(AbcExpression::Var(into.clone()))?;
            self.mark_type(into.clone(), func.return_type.clone())?;
            self.add_constraint(var_as_expr.clone(), ConstraintOp::Assign, call)?;
            Ok(var_as_expr)
        } else {
            // At this point, we need to add the constraints of all of the terms in the function.
            self.add_constraint(call.clone(), ConstraintOp::Unary, EMPTY_EXPR.clone())?;
            Ok(call)
        }
    }

    /// Add an expression to the constraint system, returning a handle to it.
    ///
    /// # Errors
    /// Never (always returns an Ok)
    fn add_expression(
        &mut self,
        expr: AbcExpression,
    ) -> Result<Self::Handle<AbcExpression>, Self::E> {
        Ok(Arc::new(expr))
    }
    /// Create a new variable, returning a handle to that variable.
    ///
    /// The interface for variables may change in the future so it is recommended to use this instead..
    /// We won't do anything about expressions
    /// These will have to be created.
    fn declare_var(&mut self, name: Var) -> Result<Self::Handle<Var>, ConstraintError> {
        Ok(Arc::new(Var {
            name: name.to_string(),
        }))
    }

    fn mark_break(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("break".to_string()))
    }

    fn mark_continue(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("continue".to_string()))
    }

    /// Mark the type of the variable, returning a handle to that variable.
    /// Var can be either an existing Var, a handle to a Var, or a string. If it is a string, then a new var is created.
    /// Otherwise, the existing var is used.
    /// Right now, type expects a string. This may be changed in the future
    /// Sized arrays are specified as [ty; N] where N is the size of the array.
    /// Nested arrays are specified as [[ty; N]; M] where N is the size of the inner array and M is the size of the outer array.
    fn mark_type(
        &mut self,
        var: Self::Handle<Var>,
        ty: Self::Handle<AbcType>,
    ) -> Result<(), Self::E> {
        // If we are already given a handle, then we just use that
        // Otherwise, if we are given a string, then we create a new var.

        // if let Some(arc_var) = Arc::downcast::<Var>(var)
        self.write(format!("type({var}) = {}", ty));
        Ok(())
    }

    /// Mark the return type for the active summary.
    ///
    /// # Errors
    /// Returns [`ConstraintError::SummaryError`] if there is no active summary.
    fn mark_return_type(&mut self, ty: Self::Handle<AbcType>) -> Result<(), Self::E> {
        self.write(format!("type(@ret) = {}", ty));
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
        term: Self::Handle<AbcExpression>,
        op: ConstraintOp,
        rhs: Self::Handle<AbcExpression>,
        source: &T,
    ) -> Result<(), Self::E> {
        // self.write(format!("/* {:?} */", source));
        self.add_constraint(term, op, rhs)?;
        self.append_last(format!("{:?}", source));
        // Then, just write the source
        Ok(())
    }

    /// Add a constraint to the system that narrows the domain of `term`
    ///
    /// Any active predicates are applied.
    ///
    /// If source is given and not none, then the source is used to provide context for the constraint.
    /// It really should implement the ToString trait.
    fn add_constraint(
        &mut self,
        term: Self::Handle<AbcExpression>,
        op: ConstraintOp,
        rhs: Self::Handle<AbcExpression>,
    ) -> Result<(), Self::E> {
        // build the predicate. To start with, we have the permanent predicate
        // Then, we have the AND of the current predicate stack.

        let guard = self.make_guard();
        let new_constraint = match op {
            ConstraintOp::Unary => Constraint::Expression { guard, term },
            // An empty rhs is not allowed unless this is a unary constraint.
            _ if matches!(rhs.as_ref(), AbcExpression::Empty) => {
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
                    "ConstraintOp::{:?}",
                    op
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
    fn mark_return(
        &mut self,
        retval: Option<Self::Handle<AbcExpression>>,
    ) -> Result<(), ConstraintError> {
        if self.active_summary.is_none() {
            return Err(ConstraintError::SummaryError);
        }
        // Add the constraint on retval, if there is a retval.
        if let Some(expr) = retval {
            self.add_constraint(
                AbcExpression::new_var(RET.clone()).into(),
                ConstraintOp::Assign,
                expr,
            )?;
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
    fn mark_length(&mut self, var: Handle<Var>, dim: u8, size: u64) -> Result<(), ConstraintError> {
        self.write(format!("length({var}, {dim}) = {size}"));
        Ok(())
    }

    /// Mark the range of a variable. This works as an assumption when the range of a variable is fixed.
    fn mark_range<T>(&mut self, var: Handle<Var>, low: T, high: T) -> Result<(), ConstraintError>
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
    fn begin_predicate_block(&mut self, p: Handle<Predicate>) -> Result<(), ConstraintError> {
        // In this form, every subsequent constraint we add is guarded by the conjunction of each predicate...
        self.write(format!("begin_predicate_block({})", p.as_ref()));
        self.predicate_stack.push(p.clone());
        Ok(())
    }

    /// Sugar for a predicate block with a single entry.
    ///
    /// Provided for conveniently handling wgsl's select<...> syntax.
    // fn push_select(
    //     &mut self,
    //     pred: Handle<Predicate>,
    //     then_expr: Handle<AbcExpression>,
    //     else_expr: Handle<AbcExpression>,
    // ) {
    //     self.write("{pred} ".to_string());
    //     self.write(format!("select({pred}, {then_expr}, {else_expr})"));
    // }

    /// Ends the previous predicate block.
    ///
    /// Returns an error if the predicate stack is empty.
    fn end_predicate_block(&mut self) -> Result<(), ConstraintError> {
        // Predicate block is sugar. It disappears, so there should be no use for it...
        self.write("pop_predicate()".to_string());
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
    /// When we are inside of a loop context, any update to a variable is marked as a range constraint.
    /// It also allows for special handling of break and continue statements.
    /// HOWEVER, for the time being, we can't do any of this fancy handling
    /// So we are just going to emit a "begin_loop(condition)" statement, and will assume this is a part of the constraint system.
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
        let str_len = 15
            + name.len()
            + match nargs {
                0 => 0,
                1..=9 => 6 * (nargs as usize),
                10..=99 => 54 + 7 * (nargs as usize - 9),
                100..=255 => 684 + 8 * (nargs as usize - 99),
            };

        // The most efficient way to construct a string in rust.
        let mut param_list_str = String::with_capacity(str_len);

        param_list_str.push_str("begin_summary(");
        param_list_str.push_str(&name);

        for i in 1..=nargs {
            param_list_str.push_str(", arg");
            param_list_str.push_str(&i.to_string());
        }
        // Create a new summary
        let new_summary = Summary {
            name,
            nargs,
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

    fn fresh_var(constraint_helper: &mut ConstraintHelper, name: String) -> Handle<Var> {
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
        constraint_helper.mark_ndim(x, 3);
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
}

// /// FFI bindings for the rust library.
// #[cfg(feature = "cffi")]
// pub mod ffi_bindings;
