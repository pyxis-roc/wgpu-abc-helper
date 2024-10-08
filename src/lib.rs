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

// Clippy lints
#![allow(clippy::must_use_candidate)]

use std::{fmt::Write, sync::Arc};
// use std::rc::Rc

type FastHashMap<K, V> = rustc_hash::FxHashMap<K, V>;

use lazy_static::lazy_static;

/// Objects that derive this trait mean they support replacing terms within them with other terms.
trait SubstituteTerm {
    /// This should always return a clone of `to` if `self.is_identical(from)` is true.
    #[must_use]
    fn substitute(&self, from: &Term, to: &Term) -> Self;

    /// Substitute multiple terms at once.
    #[must_use]
    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self;
}

// For right now, we are using handles. Later on, we might switch to an arena with actual handles.

pub type Handle<T> = Arc<T>;

mod helper;
pub use helper::*;

/// An opaque marker that is provided when specifying expressions to relate them, when necessary.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct OpaqueMarker<T>
where
    T: Clone + std::fmt::Debug,
{
    payload: T,
}

impl<T> OpaqueMarker<T>
where
    T: Clone + std::fmt::Debug,
{
    pub fn new(payload: T) -> Self {
        Self { payload }
    }
}

// For right now, everything will be a string... We will not do type checking.

/// A variable. Used by terms to refer to a persistent, mutable value.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Var {
    /// The name of the variable
    pub name: String,
}

/// Create a new variable with the name
impl From<String> for Var {
    fn from(name: String) -> Self {
        Self { name }
    }
}

impl std::fmt::Display for Var {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// A unary operation. Currently only supports unary minus.
#[derive(strum_macros::Display, Debug, Clone, Copy)]
pub enum UnaryOp {
    #[strum(to_string = "-")]
    Minus,
}

#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// A comparison operator used by predicates.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(strum_macros::Display, Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ConstraintOp {
    #[strum(to_string = "=")]
    Assign,
    #[strum(to_string = "{0}")]
    Cmp(CmpOp),
    #[strum(to_string = "UnaryConstraint")]
    Unary,
}

impl From<CmpOp> for ConstraintOp {
    fn from(value: CmpOp) -> Self {
        Self::Cmp(value)
    }
}

/// Constraints are the building blocks of the constraint system.
/// They establish relationships between terms that limit their domain.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    /// An assignment constraint, e.g. x = y
    Assign {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        rhs: Term,
    },

    /// A comparison constraint, e.g. length(x) < y
    Cmp {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        op: CmpOp,
        rhs: Term,
    },

    /// An expression constraint, e.g. x (In this case, the expression must be a predicate term.)
    Expression {
        guard: Option<Handle<Predicate>>,
        term: Term,
    },
}

// Expands to the match that calls itself on the fields within.
macro_rules! constraint_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::Assign { guard, lhs, rhs } => Self::Assign {
                guard: guard.as_ref().map(|f| f.$name($($args),*).into()),
                lhs: lhs.$name($($args),*),
                rhs: rhs.$name($($args),*),
            },
            Self::Cmp { guard, lhs, op, rhs } => Self::Cmp {
                guard: guard.as_ref().map(|f| f.$name($($args),*).into()),
                lhs: lhs.$name($($args),*),
                op: *op,
                rhs: rhs.$name($($args),*),
            },
            Self::Expression { guard, term } => Self::Expression {
                guard: guard.as_ref().map(|f| f.$name($($args),*).into()),
                term: term.$name($($args),*),
            },
        }
    };
}

impl SubstituteTerm for Constraint {
    fn substitute(&self, from: &Term, to: &Term) -> Self {
        if from.is_identical(to) {
            return self.clone();
        }
        constraint_sub! {self, substitute, (from, to)}
    }

    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self {
        if mapping.len() == 1 {
            return self.substitute(mapping[0].0, mapping[0].1);
        }
        constraint_sub!(self, substitute_multi, (mapping))
    }
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

#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(strum_macros::Display, Debug, Clone, PartialEq)]
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

impl From<&bool> for Term {
    fn from(val: &bool) -> Self {
        Term::Predicate(if *val {
            Predicate::True.into()
        } else {
            Predicate::False.into()
        })
    }
}

impl From<bool> for Predicate {
    fn from(val: bool) -> Self {
        if val {
            Predicate::True
        } else {
            Predicate::False
        }
    }
}

impl From<&bool> for Predicate {
    fn from(val: &bool) -> Self {
        if *val {
            Predicate::True
        } else {
            Predicate::False
        }
    }
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

    pub fn new_unit<T: Into<Term>>(var: T) -> Handle<Self> {
        let var = var.into();
        match var {
            Term::Predicate(p) => p.clone(),
            _ => Predicate::Unit(var).into(),
        }
    }
}

/// Enum for different expression kinds
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub enum AbcExpression {
    /// A binary operator, e.g., x + y
    BinaryOp(BinaryOp, Term, Term),
    /// A select expression, e.g., select(x, y, z)
    Select(Term, Term, Term),

    /// Splat, aka a vector with the same value repeated n times.
    Splat(Term, u32),
    /// The expression for the length of an array, e.g., ArrayLength(x)
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
    IndexAccess { base: Term, index: Term },

    /// The empty expression. This is meant to be used as a placeholder for constraint operations that have no expression.
    /// Displaying this is therefore a parse error.
    Empty,
    /// A store expression, which represents an assignment to an array.
    ///
    /// A store resolves to a copy of the base array, by value,
    /// with the value at the index replaced by the new value.
    /// This allows the domain of the
    Store {
        base: Term,
        index: Term,
        value: Term,
    },

    /// A store expression, which represents an assignment to a struct field.
    ///
    /// A struct store resolves to a copy of the base struct, by value,
    /// with the value at the field replaced by the new value.
    /// We keep the type here to simplify our analysis. We don't want to have
    /// to go look up what the type of the `term` is every time we have a struct
    /// access to figure out which field we are updating..
    StructStore {
        base: Term,
        fieldname: String,
        ty: Handle<AbcType>,
        value: Term,
    },
}

//
macro_rules! expression_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::Empty => Self::Empty,
            Self::Cast(t, s) => Self::Cast(t.$name($($args),*), *s),
            Self::ArrayLength(t) => Self::ArrayLength(t.$name($($args),*)),
            Self::BinaryOp(op, l, r) => {
                Self::BinaryOp(*op, l.$name($($args),*), r.$name($($args),*))
            }
            Self::Call { func, args } => {
                let mut new_args = Vec::with_capacity(args.len());
                for arg in args {
                    new_args.push(arg.$name($($args),*))
                }
                Self::Call {
                    func: func.clone(),
                    args: new_args,
                }
            }
            Self::FieldAccess {
                base,
                ty,
                fieldname,
            } => Self::FieldAccess {
                base: base.$name($($args),*),
                ty: ty.clone(),
                fieldname: fieldname.clone(),
            },

            Self::Splat(t, v) => Self::Splat(t.$name($($args),*), *v),

            Self::Select(t1, t2, t3) => Self::Select(
                t1.$name($($args),*),
                t2.$name($($args),*),
                t3.$name($($args),*),
            ),
            Self::IndexAccess { base, index } => Self::IndexAccess {
                base: base.$name($($args),*),
                index: index.$name($($args),*),
            },
            Self::Store { base, index, value } => Self::Store {
                base: base.$name($($args),*),
                index: index.$name($($args),*),
                value: value.$name($($args),*),
            },
            Self::StructStore {
                base,
                fieldname,
                ty,
                value,
            } => Self::StructStore {
                base: base.$name($($args),*),
                fieldname: fieldname.clone(),
                ty: ty.clone(),
                value: value.$name($($args),*),
            },
        }
    }
}

impl SubstituteTerm for AbcExpression {
    /// Create a new `AbcExpression` from `self` where all instances of `from` have been replaced by `to`
    fn substitute(&self, from: &Term, to: &Term) -> Self {
        // There's nothing to substitute if `from` and `to` are already the same.
        if from.is_identical(to) {
            return (*self).clone();
        }
        expression_sub! {self, substitute, (from, to)}
    }
    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self {
        if mapping.is_empty() {
            return self.substitute(mapping[0].0, mapping[0].1);
        }
        expression_sub! {self, substitute_multi, (mapping)}
    }
}

macro_rules! predicate_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::False | Self::True => $self.clone(),
            Self::And(l, r) => {
                Self::And(l.$name($($args),*).into(), r.$name($($args),*).into())
            }
            Self::Or(l, r) => {
                Self::Or(l.$name($($args),*).into(), r.$name($($args),*).into())
            }
            Self::Comparison(op, t1, t2) => {
                Self::Comparison(*op, t1.$name($($args),*), t2.$name($($args),*))
            }
            Self::Not(p) => Self::Not(p.$name($($args),*).into()),
            Self::Unit(t) => Self::Unit(t.$name($($args),*)),
        }
    }
}

impl SubstituteTerm for Predicate {
    fn substitute(&self, from: &Term, to: &Term) -> Self {
        if from.is_identical(to) {
            return self.clone();
        }
        predicate_sub! {self, substitute, (from, to)}
    }

    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self {
        if mapping.len() == 1 {
            return self.substitute(mapping[0].0, mapping[0].1);
        }
        predicate_sub! {self, substitute_multi, (mapping)}
    }
}

impl std::fmt::Display for AbcExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbcExpression::Splat(expr, size) => {
                f.write_char('<')?;
                f.write_str(&expr.to_string())?;
                f.write_str(&expr.to_string())?;
                for _ in 1..*size {
                    f.write_str(", ")?;
                    f.write_str(&expr.to_string())?;
                }
                f.write_char('>')
            }
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
            AbcExpression::Store { base, index, value } => {
                write!(f, "store({base}, {index}, {value})")
            }
            AbcExpression::StructStore {
                base,
                fieldname,
                value,
                ..
            } => {
                write!(f, "store_field({base}, {fieldname}, {value})")
            }
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
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
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

impl AbcType {
    pub fn is_none_type(&self) -> bool {
        matches!(self, AbcType::NoneType)
    }
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
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Summary {
    pub name: String,
    pub args: Vec<Term>,
    pub return_type: Handle<AbcType>,

    pub(self) ret_term: Term,

    /// Constraints are predicates that must hold for the summary to be valid
    pub constraints: Vec<Constraint>,

    /// Assumptions are predicates that filter out invalid domains
    ///
    /// They mark things like assignment
    ///
    /// These encode information such as variable assignments.
    pub assumptions: Vec<Constraint>,
}

/// Displaying a Summary just shows the name of the function.
impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

impl Summary {
    /// Return the number of arguments the function takes.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn nargs(&self) -> u8 {
        self.args.len() as u8
    }

    /// Create a new summary with the given name and number of arguments.
    #[must_use]
    pub fn new(name: String, nargs: u8) -> Self {
        Summary {
            name,
            args: Vec::with_capacity(nargs as usize),
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
            assumptions: Vec::new(),
            ret_term: Term::Empty,
        }
    }

    /// Add an argument to the summary.
    pub(crate) fn add_argument(&mut self, arg: Term) {
        self.args.push(arg);
    }

    /// Add a constraint to the summary.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add an assumption to the summary.
    pub fn add_assumption(&mut self, assumption: Constraint) {
        self.assumptions.push(assumption);
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

/// A literal, ripped directly from naga's literal, with the `bool` literal dropped
/// The bool literal turns into [`Predicate::True`] or [`Predicate::False`]
///
/// [`Predicate::True`]: crate::Predicate::True
/// [`Predicate::False`]: crate::Predicate::False
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, strum_macros::Display)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum Literal {
    #[strum(to_string = "{0}")]
    /// May not be NaN or infinity.
    F64(f64),
    #[strum(to_string = "{0}")]
    F32(f32),
    #[strum(to_string = "{0}")]
    U32(u32),
    #[strum(to_string = "{0}")]
    I32(i32),
    #[strum(to_string = "{0}")]
    U64(u64),
    #[strum(to_string = "{0}")]
    I64(i64),
    #[strum(to_string = "{0}")]
    AbstractInt(i64),
    #[strum(to_string = "{0}")]
    AbstractFloat(f64),
}

macro_rules! from_lit_impl {
    ($ty:ty, $variant:path) => {
        impl From<$ty> for Literal {
            fn from(v: $ty) -> Self {
                $variant(v)
            }
        }
    };
    ($ty:ty, $as:ty, $variant:path) => {
        impl From<$ty> for Literal {
            fn from(v: $ty) -> Self {
                $variant(<$as>::from(v))
            }
        }
    };
}

// Macros reduce boilerplate.
from_lit_impl!(u8, u32, Literal::U32);
from_lit_impl!(u16, u32, Literal::U32);
from_lit_impl!(u32, Literal::U32);
from_lit_impl!(u64, Literal::U64);
from_lit_impl!(i8, i32, Literal::I32);
from_lit_impl!(i16, i32, Literal::I32);
from_lit_impl!(i32, Literal::I32);
from_lit_impl!(i64, Literal::I64);
from_lit_impl!(f32, Literal::F32);
from_lit_impl!(f64, Literal::F64);
from_lit_impl!(std::num::NonZeroU32, u32, Literal::U32);
from_lit_impl!(std::num::NonZeroI32, i32, Literal::I32);
from_lit_impl!(std::num::NonZeroI64, i64, Literal::I64);
from_lit_impl!(std::num::NonZeroU64, u64, Literal::U64);

impl From<std::num::NonZeroU8> for Literal {
    fn from(value: std::num::NonZeroU8) -> Self {
        Self::U32(u32::from(u8::from(value)))
    }
}

impl From<std::num::NonZeroU16> for Literal {
    fn from(value: std::num::NonZeroU16) -> Self {
        Self::U32(u32::from(u16::from(value)))
    }
}

impl From<std::num::NonZeroI8> for Literal {
    fn from(value: std::num::NonZeroI8) -> Self {
        Self::I32(i32::from(i8::from(value)))
    }
}

impl From<std::num::NonZeroI16> for Literal {
    fn from(value: std::num::NonZeroI16) -> Self {
        Self::I32(i32::from(i16::from(value)))
    }
}

/// A term is used to capture both variables and expressions
///
/// It makes marking constraints easier, as this allows
/// for both to be used interchangably.
///
/// It also simplifies the logic for storing references to variables.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[derive(Clone, Debug, strum_macros::Display, PartialEq)]
pub enum Term {
    #[strum(to_string = "{0}")]
    Expr(Handle<AbcExpression>),
    #[strum(to_string = "{0}")]
    Var(Handle<Var>),
    #[strum(to_string = "{0}")]
    Literal(Literal),
    #[strum(to_string = "{0}")]
    Predicate(Handle<Predicate>),
    /// An empty term or predicate.
    Empty,
}

impl Term {
    /// Return whether this is the empty term.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Term::Empty)
    }
    /// Determine whether `self` and `with` have identical structure and references.
    ///
    /// This is different from equality in that it uses `Arc::ptr_eq` to check its contents.
    fn is_identical(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Expr(a), Self::Expr(b)) => Arc::ptr_eq(a, b),
            (Self::Var(a), Self::Var(b)) => Arc::ptr_eq(a, b),
            (Self::Predicate(a), Self::Predicate(b)) => Arc::ptr_eq(a, b),
            (Self::Literal(a), Self::Literal(b)) => a == b,
            (Self::Empty, Self::Empty) => true,
            _ => false,
        }
    }
}

impl SubstituteTerm for Term {
    /// Creates a new term where all references to `self` have been replaced with `to`
    ///
    /// If `self` and `from` are identical, this method just returns `with`
    #[must_use]
    fn substitute(&self, from: &Term, with: &Term) -> Self {
        if self.is_identical(from) {
            println!("Found a match for {self:?}, replacing with {with:?}");
            with.clone()
        } else {
            match self {
                Self::Expr(a) => Self::Expr(a.substitute(from, with).into()),
                Self::Var(_) | Self::Literal(_) | Self::Empty => self.clone(),
                Self::Predicate(p) => Self::Predicate(p.substitute(from, with).into()),
            }
        }
    }

    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self {
        if mapping.len() == 1 {
            return self.substitute(mapping[0].0, mapping[0].1);
        }
        // Otherwise, if we are identical to one of the terms being replaced, then return what it is replaced with.
        if let Some(t) = mapping.iter().find(|p| p.0.is_identical(self)) {
            println!("Found a match for {:?}, replacing with {:?}", self, t.1);
            t.1.clone()
        } else {
            match self {
                Self::Expr(a) => Self::Expr(a.substitute_multi(mapping).into()),
                Self::Var(_) | Self::Literal(_) | Self::Empty => self.clone(),
                Self::Predicate(p) => Self::Predicate(p.substitute_multi(mapping).into()),
            }
        }
    }
}

// `true` and `false` are predicates
// A predicate is really any expression that evaluates to a boolean.
impl From<bool> for Term {
    /// Converts the `bool` to [`Predicate::True`] or [`Predicate::False`]
    ///
    /// [`Predicate::True`]: crate::Predicate::True
    /// [`Predicate::False`]: crate::Predicate::False
    fn from(val: bool) -> Self {
        Self::Predicate(if val {
            Predicate::True.into()
        } else {
            Predicate::False.into()
        })
    }
}

// Blanket implementation for converting anything that can be converted to a
//literal into a term.
impl<T: Into<Literal>> From<T> for Term {
    fn from(val: T) -> Self {
        Term::Literal(val.into())
    }
}

/*
Implementation of predicate constructors for term
*/
impl Term {
    /// Create a new unit predicate term.
    #[must_use]
    pub fn new_unit_pred(p: Term) -> Self {
        Term::Predicate(Predicate::new_unit(p))
    }
    /// Create a term holding the `true` predicate
    #[must_use]
    pub fn new_literal_true() -> Self {
        Term::Predicate(Predicate::True.into())
    }

    /// Create a term holding the `false` predicate
    #[must_use]
    pub fn new_literal_false() -> Self {
        Term::Predicate(Predicate::False.into())
    }

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
        T: Into<Var>,
    {
        Term::Var(Handle::new(var.into()))
    }

    #[must_use]
    pub fn new_expr<T>(expr: T) -> Self
    where
        T: Into<Handle<AbcExpression>>,
    {
        Term::Expr(expr.into())
    }

    /// Create a cast expression
    #[must_use]
    pub fn new_cast(term: Term, ty: AbcScalar) -> Term {
        Term::Expr(AbcExpression::Cast(term, ty).into())
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
        Term::Expr(
            AbcExpression::FieldAccess {
                base,
                fieldname,
                ty,
            }
            .into(),
        )
    }

    /// Create a new splat expresion.
    #[must_use]
    pub fn new_splat(term: Term, size: u32) -> Self {
        Term::Expr(AbcExpression::Splat(term, size).into())
    }

    /// Create a new literal.
    ///
    /// This does not work for true/false.
    #[must_use]
    pub fn new_literal<T>(lit: T) -> Self
    where
        T: Into<Literal>,
    {
        Self::Literal(lit.into())
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
            _ => Predicate::new_unit(pred),
        });
        AbcExpression::Select(pred, then_expr, else_expr).into()
    }

    #[must_use]
    pub fn make_array_length(var: Term) -> Self {
        AbcExpression::ArrayLength(var).into()
    }

    #[must_use]
    pub fn new_store(base: Term, index: Term, value: Term) -> Self {
        AbcExpression::Store { base, index, value }.into()
    }

    #[must_use]
    pub fn new_struct_store(
        base: Term,
        fieldname: String,
        ty: Handle<AbcType>,
        value: Term,
    ) -> Self {
        Term::Expr(
            AbcExpression::StructStore {
                base,
                fieldname,
                ty,
                value,
            }
            .into(),
        )
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

impl From<&str> for Var {
    fn from(name: &str) -> Self {
        Var {
            name: name.to_string(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn var_x() -> Term {
        Term::new_var("x")
    }

    #[fixture]
    fn var_y() -> Term {
        Term::new_var("y")
    }

    #[fixture]
    fn literal_true() -> Term {
        Term::Predicate(Predicate::True.into())
    }

    #[fixture]
    fn literal_false() -> Term {
        Term::Predicate(Predicate::False.into())
    }

    #[rstest]
    fn test_predicate_not(var_x: Term) {
        let p = Predicate::new_not(Predicate::new_unit(var_x));
        assert_eq!(p.to_string(), "!(x)");
    }

    /// Ensure that negating an already negated predicate just removes the negation.
    #[rstest]
    fn test_predicate_not_not(var_x: Term) {
        let p = Predicate::new_not(Predicate::new_unit(var_x));
        let p2 = Predicate::new_not(p);
        assert_eq!(p2.to_string(), "x");
    }

    // Test terms.
    #[rstest]
    fn test_term_from_var(var_x: Term) {
        let term = var_x.clone();
        assert_eq!(term.to_string(), "x");
    }

    #[rstest]
    fn test_predicate_and(var_x: Term, var_y: Term) {
        let p = Predicate::new_and(var_x, var_y);
        assert_eq!(p.to_string(), "(x) && (y)");
    }

    #[rstest]
    fn test_predicate_or(var_x: Term, var_y: Term) {
        let p = Predicate::new_or(Predicate::new_unit(var_x), Predicate::new_unit(var_y));
        assert_eq!(p.to_string(), "(x) || (y)");
    }

    #[rstest]
    fn test_predicate_comparison(var_x: Term, var_y: Term) {
        let p = Predicate::new_comparison(CmpOp::Eq, var_x, var_y);
        assert_eq!(p.to_string(), "(x) == (y)");
    }

    #[rstest]
    fn test_term_new_literal_true() {
        let term = Term::new_literal_true();
        assert_eq!(term.to_string(), "true");
    }

    #[rstest]
    fn test_term_new_literal_false() {
        let term = Term::new_literal_false();
        assert_eq!(term.to_string(), "false");
    }

    #[rstest]
    fn test_term_new_logical_and(var_x: Term) {
        let term = Term::new_logical_and(var_x.clone(), var_x);
        assert_eq!(term.to_string(), "(x) && (x)");
    }

    /// Ensure `true` or x = x
    #[rstest]
    fn test_or_with_true(literal_true: Term) {
        // Get some predicate., e.g. 'x'
        let any_pred = Term::new_var(Var {
            name: "X".to_string(),
        });
        let term = Term::new_logical_or(literal_true, any_pred);
        // This should be the exact same inner.
        assert_eq!(term.to_string(), "true");
    }

    #[rstest]
    fn test_term_new_comparison(var_x: Term) {
        let var_y = Var {
            name: "y".to_string(),
        };
        let term = Term::new_comparison(CmpOp::Eq, var_x, Term::from(var_y));
        assert_eq!(term.to_string(), "(x) == (y)");
    }

    #[rstest]
    fn test_term_new_not(var_x: Term) {
        let term = Term::new_not(var_x);
        assert_eq!(term.to_string(), "!(x)");
    }

    #[rstest]
    fn test_term_new_expr() {
        let expr =
            AbcExpression::BinaryOp(BinaryOp::Plus, Term::new_literal(1), Term::new_literal(2));
        let term = Term::new_expr(expr);
        assert_eq!(term.to_string(), "1 + 2");
    }

    #[rstest]
    fn test_term_new_cast() {
        let term = Term::new_cast(Term::new_literal(1), AbcScalar::Sint(4));
        assert_eq!(term.to_string(), "cast(1, i32)");
    }

    #[rstest]
    fn test_term_new_call() {
        let summary = Summary::new("foo".to_string(), 2);
        let term = Term::new_call(
            Arc::new(summary),
            vec![Term::new_literal(1), Term::new_literal(2)],
        );
        assert_eq!(term.to_string(), "foo(1, 2)");
    }

    #[rstest]
    fn test_term_new_index_access() {
        let base = Term::new_var(Var::from("arr"));
        let index = Term::new_literal(0);
        let term = Term::new_index_access(base, index);
        assert_eq!(term.to_string(), "arr[0]");
    }

    #[rstest]
    fn test_term_new_struct_access() {
        let base = Term::new_var(Var::from("obj"));
        let ty = Arc::new(AbcType::NoneType);
        let term = Term::new_struct_access(base, "field".to_string(), ty);
        assert_eq!(term.to_string(), "obj.field");
    }

    #[rstest]
    fn test_term_new_select() {
        let pred = Term::new_literal_true();
        let then_expr = Term::new_literal(1);
        let else_expr = Term::new_literal(0);
        let term = Term::new_select(pred, then_expr, else_expr);
        assert_eq!(term.to_string(), "select(true, 1, 0)");
    }

    #[rstest]
    fn test_term_make_array_length() {
        let var = Term::new_var(Var::from("arr"));
        let term = Term::make_array_length(var);
        assert_eq!(term.to_string(), "length(arr)");
    }

    #[rstest]
    fn test_term_substitute(var_x: Term, var_y: Term) {
        let term = Term::new_logical_and(var_x.clone(), var_y.clone());
        let term2 = term.substitute(&var_x, &var_y);
        assert_eq!(term2.to_string(), "(y) && (y)");
    }

    #[rstest]
    fn test_term_substitute_multi(var_x: Term, var_y: Term) {
        let z_term = Term::new_var(Var {
            name: "z".to_string(),
        });
        let w_term = Term::new_var(Var {
            name: "w".to_string(),
        });
        let term = Term::new_logical_and(var_x.clone(), var_y.clone());
        let term2 = term.substitute_multi(&[(&var_x, &z_term), (&var_y, &w_term)]);
        assert_eq!(term2.to_string(), "(z) && (w)");
    }

    #[rstest]
    fn test_constraint_substitute_multi(var_x: Term, var_y: Term) {
        let z_term = Term::new_var(Var {
            name: "z".to_string(),
        });
        let w_term = Term::new_var(Var {
            name: "w".to_string(),
        });
        let constraint = Constraint::Cmp {
            guard: None,
            op: CmpOp::Eq,
            lhs: var_x.clone(),
            rhs: var_y.clone(),
        };
        let constraint2 = constraint.substitute_multi(&[(&var_x, &z_term), (&var_y, &w_term)]);
        assert_eq!(constraint2.to_string(), "z == w");
    }
}
