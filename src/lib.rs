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
#![allow(clippy::must_use_candidate, clippy::default_trait_access)]
#![allow(dead_code)]

use std::{
    borrow::Borrow,
    fmt::Write,
    sync::{Arc, LazyLock},
};

type FastHashMap<K, V> = rustc_hash::FxHashMap<K, V>;
type FastHashSet<K> = rustc_hash::FxHashSet<K>;

#[doc(hidden)]
mod macros;
#[doc(hidden)]
use macros::cbindgen_annotate;

pub mod solvers;
use serde::{Deserialize, Serialize};
pub use solvers::interval::SolverError;

pub use solvers::interval::{
    translator::IntervalKind, BoolInterval, I32Interval, U32Interval, U64Interval,
};

use helper::AssumptionSerializer;

/// Get the version of the abc package, as a string.
#[allow(dead_code)]
pub(crate) fn abc_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Objects that derive this trait mean they support replacing terms within them with other terms.
trait SubstituteTerm {
    /// This should always return a clone of `to` if `self.is_identical(from)` is true.
    fn substitute(&self, from: &Term, to: &Term) -> Self;

    /// Substitute multiple terms at once.
    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self;
}

// For right now, we are using handles that are just Arcs. Later on, we might switch to an arena with actual handles.
// We also might switch to using Rc instead of Arc.

pub type Handle<T> = Arc<T>;

mod helper;
#[allow(unused_imports)] // This is a re epxort
pub use helper::*;

/// The maximum number of constraints that can be added per summary.
pub const CONSTRAINT_LIMIT: usize = u32::MAX as usize;

/// The maximum number of summaries that can be added to a constraint helper.
pub const SUMMARY_LIMIT: usize = (u32::MAX - 1) as usize;

/// Unique identifier for a constraint.
///
/// When a constraint is added to the constraint system, it is assigned a unique identifier that can be used to track the constraint.
///
/// When solving constraints, this identifier will be used to report the results.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstraintId {
    /// The function index that the constraint is associated with.
    func: u32,
    /// The constraint index within the function.
    idx: u32,
}

impl std::fmt::Display for ConstraintId {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({}, {})", self.func, self.idx)
    }
}

// For right now, everything will be a string... We will not do type checking.

/// A variable. Used by terms to refer to a persistent, mutable value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(
    strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
#[cfg_attr(feature = "cffi", repr(C))]
pub enum UnaryOp {
    #[strum(to_string = "-")]
    Minus,
}
impl UnaryOp {
    #[inline]
    const fn variant_name(self) -> &'static str {
        match self {
            UnaryOp::Minus => "Minus",
        }
    }
}

#[cfg_attr(feature = "cffi", repr(C))]
#[derive(
    strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize,
)]
pub enum BinaryOp {
    #[strum(to_string = "+")]
    #[serde(rename = "+")]
    Plus,
    #[strum(to_string = "-")]
    #[serde(rename = "-")]
    Minus,
    #[strum(to_string = "*")]
    #[serde(rename = "*")]
    Times,
    #[strum(to_string = "%")]
    #[serde(rename = "%")]
    Mod,
    #[strum(to_string = "//")]
    #[serde(rename = "//")]
    Div,

    // The following are probably not supported by our constraint system.
    #[strum(to_string = "&")]
    #[serde(rename = "&")]
    BitAnd,
    #[strum(to_string = "|")]
    #[serde(rename = "|")]
    BitOr,
    #[strum(to_string = "^")]
    #[serde(rename = "^")]
    BitXor,

    // These can probably be supported if the lhs or rhs is a constant.
    #[strum(to_string = "<<")]
    #[serde(rename = "<<")]
    Shl,
    #[strum(to_string = ">>")]
    #[serde(rename = ">>")]
    Shr,
}

impl BinaryOp {
    #[inline]
    const fn variant_name(self) -> &'static str {
        match self {
            BinaryOp::BitAnd => "BitAnd",
            BinaryOp::BitOr => "BitOr",
            BinaryOp::BitXor => "BitXor",
            BinaryOp::Div => "Div",
            BinaryOp::Minus => "Minus",
            BinaryOp::Mod => "Mod",
            BinaryOp::Plus => "Plus",
            BinaryOp::Shl => "Shl",
            BinaryOp::Shr => "Shr",
            BinaryOp::Times => "Times",
        }
    }
}

cbindgen_annotate! {
"derive-const-casts"
#[doc = "A comparison operator used by ccates."]
#[cfg_attr(feature = "cffi", repr(C))]
#[derive(strum_macros::Display, Debug, Clone, Copy, PartialEq, Eq, Hash, strum_macros::AsRefStr, Serialize, Deserialize)]
pub enum CmpOp {
    #[strum(to_string = "==")]
    #[serde(rename = "==")]
    Eq = 0,
    #[serde(rename = "!=")]
    #[strum(to_string = "!=")]
    Neq,
    #[strum(to_string = "<")]
    #[serde(rename = "<")]
    Lt,
    #[strum(to_string = ">")]
    #[serde(rename = ">")]
    Gt,
    #[strum(to_string = "<=")]
    #[serde(rename = "<=")]
    Leq,
    #[serde(rename = ">=")]
    #[strum(to_string = ">=")]
    Geq,
}
}
impl CmpOp {
    /// Construct a new comparison operator that represents
    /// the opposite of the operator.
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
#[derive(
    strum_macros::Display,
    Debug,
    Clone,
    Copy,
    Eq,
    PartialEq,
    Hash,
    strum_macros::AsRefStr,
    Serialize,
    Deserialize,
)]
#[cfg_attr(feature = "cffi", repr(C))]
pub enum AssumptionOp {
    #[strum(to_string = ":=")]
    Assign,
    #[strum(to_string = "<")]
    Lt,
    #[strum(to_string = ">")]
    Gt,
    #[strum(to_string = "<=")]
    Leq,
    #[strum(to_string = ">=")]
    Geq,
}

impl From<AssumptionOp> for CmpOp {
    fn from(op: AssumptionOp) -> Self {
        match op {
            AssumptionOp::Lt => Self::Lt,
            AssumptionOp::Gt => Self::Gt,
            AssumptionOp::Leq => Self::Leq,
            AssumptionOp::Geq => Self::Geq,
            AssumptionOp::Assign => Self::Eq,
        }
    }
}

impl TryFrom<CmpOp> for AssumptionOp {
    type Error = ();
    /// Attempt to cast the comparison operator to an assumption comparison operator.
    fn try_from(op: CmpOp) -> Result<Self, Self::Error> {
        match op {
            CmpOp::Lt => Ok(AssumptionOp::Lt),
            CmpOp::Gt => Ok(AssumptionOp::Gt),
            CmpOp::Leq => Ok(AssumptionOp::Leq),
            CmpOp::Geq => Ok(AssumptionOp::Geq),
            CmpOp::Eq => Ok(AssumptionOp::Assign),
            CmpOp::Neq => Err(()),
        }
    }
}

/// A constraint operation is any comparison operator OR an assignment.
#[derive(strum_macros::Display, Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "cffi", repr(C))]
pub enum ConstraintOp {
    #[strum(to_string = "UnaryConstraint")]
    Unary,
    #[strum(to_string = "{0}")]
    Cmp(CmpOp),
}

impl From<CmpOp> for ConstraintOp {
    fn from(op: CmpOp) -> Self {
        ConstraintOp::Cmp(op)
    }
}

/// Constraints are the building blocks of the constraint system.
/// They establish relationships between terms that limit their domain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, strum_macros::EnumIs, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Constraint {
    /// A comparison constraint, e.g. length(x) < y
    Cmp {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        op: CmpOp,
        rhs: Term,
    },

    /// An identity constraint, e.g. `x`.  In this case, `Term` _must_ be a predicate variant.
    #[serde(rename = "identity")]
    Identity {
        guard: Option<Handle<Predicate>>,
        term: Term,
    },
}

impl std::fmt::Display for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get_guard() {
            Some(guard) => write!(f, "{{{guard}}} "),
            None => Ok(()),
        }?;
        match self {
            Constraint::Cmp { lhs, op, rhs, .. } => write!(f, "{lhs} {op} {rhs}"),
            Constraint::Identity { term, .. } => write!(f, "{term}"),
        }
    }
}

// An assumption `Inequality` is a special case of
#[derive(strum_macros::EnumIs, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Assumption {
    /// An assignment assumption, e.g. x = y
    Assign {
        guard: Option<Handle<Predicate>>,
        lhs: Term,
        rhs: Term,
    },

    // An inequality assumption REQUIRES that `rhs` is a literal and that there is no `guard`
    /// An inequality assumption, e.g. x < y. `y` is always a literal.
    Inequality {
        lhs: Term,
        lower: Option<(Term, bool)>,
        upper: Option<(Term, bool)>,
    },
}

impl std::fmt::Display for Assumption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg_attr(any(), rustfmt::skip)]
        match self {
            Self::Assign { guard: None, lhs, rhs } => write!(f, "assume({lhs} := {rhs})"),
            Self::Assign { guard: Some(g), lhs, rhs} => write!(f, "{g} assume({lhs} := {rhs})"),
            Self::Inequality { lhs, lower: None, upper: Some((upper, true)) } => write!(f, "assume({lhs} <= {upper})"),
            Self::Inequality { lhs, lower: None, upper: Some((upper, false)) } => write!(f, "assume({lhs} < {upper})"),
            Self::Inequality { lhs, lower: Some((lower, true)), upper: None } => write!(f, "assume({lhs} >= {lower})"),
            Self::Inequality { lhs, lower: Some((lower, false)), upper: None } => write!(f, "assume({lhs} > {lower})"),
            Self::Inequality { lhs, lower: Some((lower, true)), upper: Some((upper, true)) } => write!(f, "assume({lower} <= {lhs} <= {upper})"),
            Self::Inequality { lhs, lower: Some((lower, true)), upper: Some((upper, false)) } => write!(f, "assume({lower} <= {lhs} < {upper})"),
            Self::Inequality { lhs, lower: Some((lower, false)), upper: Some((upper, true)) } => write!(f, "assume({lower} < {lhs} <= {upper})"),
            Self::Inequality { lhs, lower: Some((lower, false)), upper: Some((upper, false)) } => write!(f, "assume({lower} < {lhs} < {upper})"),
            Self::Inequality { lhs, .. } => write!(f, "assume(?? <= {lhs} <= ??)")
        }
    }
}

impl Assumption {
    #[allow(dead_code)]
    fn get_guard(&self) -> Option<Handle<Predicate>> {
        match self {
            Assumption::Assign { guard, .. } => guard.clone(),
            Assumption::Inequality { .. } => None,
        }
    }
    /// Merges `other` into `self`.
    /// The guards for `other` are ignored.
    /// Also assumes that `self` and `other` have the same `lhs`.
    fn merge(&mut self, other: &Self) -> bool {
        use Assumption::Inequality;
        let Inequality {
            lower: ref mut low_a,
            upper: ref mut up_a,
            ..
        } = *self
        else {
            return false;
        };
        let Inequality {
            lower: ref low_b,
            upper: ref up_b,
            ..
        } = other
        else {
            return false;
        };

        if up_b.is_some() {
            up_a.clone_from(up_b);
        }
        if low_b.is_some() {
            low_a.clone_from(low_b);
        }
        true
    }
    pub fn get_lhs(&self) -> &Term {
        match self {
            Self::Assign { lhs, .. } | Self::Inequality { lhs, .. } => lhs,
        }
    }

    pub fn get_rhs(&self) -> Option<&Term> {
        match self {
            Self::Assign { rhs, .. } => Some(rhs),
            Self::Inequality { .. } => None,
        }
    }

    /// Set the lower bound for the term.
    ///
    /// If the term is not an inequality, or is but has already set `lower`, then this function returns false.
    #[allow(dead_code)]
    fn set_lower<T: Into<Term>>(&mut self, term: T, inclusive: bool) -> bool {
        let term = term.into();
        let Self::Inequality {
            lower: ref mut lower @ None,
            ..
        } = *self
        else {
            return false;
        };
        let bound = Some((term, inclusive));
        *lower = bound;
        true
    }

    /// Set the upper bound for the term.
    ///
    /// If the term is not an inequality, or is but has already set `upper`, then this function returns false.
    #[allow(dead_code)]
    fn set_upper<T: Into<Term>>(&mut self, term: T, inclusive: bool) -> bool {
        let term = term.into();
        let Self::Inequality {
            upper: ref mut upper @ None,
            ..
        } = *self
        else {
            return false;
        };
        let bound = Some((term, inclusive));
        *upper = bound;
        true
    }
}

// Expands to the match that calls itself on the fields within.
macro_rules! constraint_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::Cmp { guard, lhs, op, rhs } => Self::Cmp {
                guard: guard.as_ref().map(|f| f.$name($($args),*).into()),
                lhs: lhs.$name($($args),*),
                op: *op,
                rhs: rhs.$name($($args),*),
            },
            Self::Identity { guard, term } => Self::Identity {
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

macro_rules! assumption_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::Assign { guard, lhs, rhs } => Self::Assign {
                guard: guard.as_ref().map(|f| f.$name($($args),*).into()),
                lhs: lhs.$name($($args),*),
                rhs: rhs.$name($($args),*),
            },
            Self::Inequality { lhs, lower, upper } => Self::Inequality {
                lhs: lhs.$name($($args),*),
                lower: lower.clone(),
                upper: upper.clone(),
            },
        }
    };
}

impl SubstituteTerm for Assumption {
    fn substitute(&self, from: &Term, to: &Term) -> Self {
        if from.is_identical(to) {
            return self.clone();
        }
        assumption_sub!(self, substitute, (from, to))
    }
    fn substitute_multi(&self, mapping: &[(&Term, &Term)]) -> Self {
        if mapping.len() == 1 {
            return self.substitute(mapping[0].0, mapping[0].1);
        }
        assumption_sub!(self, substitute_multi, (mapping))
    }
}

impl Constraint {
    /// Return the guard portion of the constraint
    fn get_guard(&self) -> Option<Handle<Predicate>> {
        match self {
            Constraint::Cmp { guard, .. } | Constraint::Identity { guard, .. } => guard.clone(),
        }
    }

    /// Get the guard for the constraint as a reference
    fn get_guard_ref(&self) -> Option<&Handle<Predicate>> {
        match self {
            Constraint::Cmp { guard, .. } | Constraint::Identity { guard, .. } => guard.as_ref(),
        }
    }
}

cbindgen_annotate! {"ignore"
#[derive(
    strum_macros::Display,
    Debug,
    Clone,
    PartialEq,
    Eq,
    Hash,
    strum_macros::EnumIs,
    strum_macros::EnumTryAs,
    strum_macros::AsRefStr,
    Serialize,
    Deserialize
)]
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
}

impl Predicate {
    /// Get a handle to the true predicate.
    pub(crate) fn mk_true() -> Handle<Self> {
        static TRUE: LazyLock<Handle<Predicate>> = LazyLock::new(|| Predicate::True.into());
        TRUE.clone()
    }

    #[allow(dead_code)] // This might be used in the future...
    /// Get a handle to the false predicate.
    pub(crate) fn mk_false() -> Handle<Self> {
        static FALSE: LazyLock<Handle<Predicate>> = LazyLock::new(|| Predicate::False.into());
        FALSE.clone()
    }

    /// Return a set consisting of the child predicates, in `Handle` form, of this `And` or `Or` predicate in the conjunction / disjunction.
    /// For example, if `self` is `(a && b)` && (`c` && `d`), this returns `{a, b, c, d}`
    pub(crate) fn get_children_set_handles(&self) -> FastHashSet<Handle<Predicate>> {
        fn iter_and_terms(term: &Handle<Predicate>, terms: &mut FastHashSet<Handle<Predicate>>) {
            if let Predicate::And(a, b) = term.as_ref() {
                iter_and_terms(a, terms);
                iter_and_terms(b, terms);
            } else {
                terms.insert(term.clone());
            }
        }
        fn iter_or_terms(term: &Handle<Predicate>, terms: &mut FastHashSet<Handle<Predicate>>) {
            if let Predicate::Or(a, b) = term.as_ref() {
                iter_or_terms(a, terms);
                iter_or_terms(b, terms);
            } else {
                terms.insert(term.clone());
            }
        }
        let mut terms = FastHashSet::default();
        match self {
            Predicate::And(a, b) => {
                iter_and_terms(a, &mut terms);
                iter_and_terms(b, &mut terms);
            }
            Predicate::Or(a, b) => {
                iter_or_terms(a, &mut terms);
                iter_or_terms(b, &mut terms);
            }
            _ => (),
        }

        terms
    }

    /// Return a set consisting of the child predicates of this `And` or `Or` predicate in the conjunction / disjunction.
    /// For example, if `self` is `(a && b)` && (`c` && `d`), this returns `{a, b, c, d}`
    pub(crate) fn get_children_set(&self) -> FastHashSet<&Predicate> {
        fn iter_and_terms<'a>(term: &'a Handle<Predicate>, terms: &mut FastHashSet<&'a Predicate>) {
            if let Predicate::And(a, b) = term.as_ref() {
                iter_and_terms(a, terms);
                iter_and_terms(b, terms);
            } else {
                terms.insert(term.borrow());
            }
        }
        fn iter_or_terms<'a>(term: &'a Handle<Predicate>, terms: &mut FastHashSet<&'a Predicate>) {
            if let Predicate::Or(a, b) = term.as_ref() {
                iter_or_terms(a, terms);
                iter_or_terms(b, terms);
            } else {
                terms.insert(term.borrow());
            }
        }
        let mut terms = FastHashSet::default();
        match self {
            Predicate::And(a, b) => {
                iter_and_terms(a, &mut terms);
                iter_and_terms(b, &mut terms);
            }
            Predicate::Or(a, b) => {
                iter_or_terms(a, &mut terms);
                iter_or_terms(b, &mut terms);
            }
            _ => (),
        }

        terms
    }
}

impl AsRef<Predicate> for Predicate {
    fn as_ref(&self) -> &Predicate {
        self
    }
}

impl Predicate {
    /// Constant `FALSE` predicate.
    pub const FALSE: Predicate = Predicate::False;
    /// Constant `TRUE` predicate.
    pub const TRUE: Predicate = Predicate::True;
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

impl From<Handle<Predicate>> for Term {
    fn from(pred: Handle<Predicate>) -> Self {
        Term::Predicate(pred)
    }
}

impl From<Predicate> for Term {
    fn from(pred: Predicate) -> Self {
        Term::Predicate(pred.into())
    }
}

impl From<Term> for Handle<Predicate> {
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
    /// Constructs a new `And` term from two Predicates.
    pub fn new_and<T, U>(lhs: T, rhs: U) -> Handle<Self>
    where
        T: Into<Handle<Self>>,
        U: Into<Handle<Self>>,
    {
        let (lhs, rhs) = (lhs.into(), rhs.into());

        // Check if the `rhs` is something like (a && b && c && d && e) ..
        // In which case, if `a` is equal to any of the terms, we
        // can just return the `rhs`.
        // That is, we always keep our terms in simplified form.
        // We continually loop while the current term is `And`.

        match (lhs.as_ref(), rhs.as_ref()) {
            (a, b) if a == b => a.clone().into(),
            (Predicate::True, _) => rhs,
            (_, Predicate::True) => lhs,
            (Predicate::False, _) | (_, Predicate::False) => Predicate::False.into(),
            (Predicate::And(a, b), Predicate::And(c, d)) => {
                if b == c || b == d {
                    Predicate::And(a.clone(), rhs).into()
                } else if a == c || a == d {
                    Predicate::And(b.clone(), rhs).into()
                } else {
                    // `And` terms are always normalized.
                    // That is, we never want to have (a && b) && (c && d).
                    // We only ever have a && (b && (c && d)).
                    Predicate::And(a.clone(), Self::new_and(b.clone(), rhs)).into()
                }
            }
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
            // a `or` a is the same as just `a`.
            (Predicate::False, _) => rhs,
            (_, Predicate::False) => lhs,
            (Predicate::True, _) | (_, Predicate::True) => Predicate::True.into(),
            (a, b) if a == b => a.clone().into(),
            (Predicate::Or(a, b), Predicate::Or(c, d)) => {
                if b == c || b == d {
                    Predicate::new_or(a.clone(), rhs)
                } else if a == c || a == d {
                    Predicate::new_or(b.clone(), rhs)
                } else {
                    Predicate::new_or(a.clone(), Self::new_or(b.clone(), rhs))
                }
            }
            _ => Predicate::Or(lhs, rhs).into(),
        }
    }

    /// Make a new `Not` predicate.
    ///
    /// Note that this constructor always simplifies its arguments,
    /// applying de Morgan's laws to convert `!And(a, b)` to `Or(!a, !b)` and `!Or(a, b)` to `And(!a, !b)`.
    ///
    /// It also reverses the comparison operator. That means that this
    /// function is not safe to use for floating point terms, as
    /// Not(a < b) is going to be converted to a >= b, which would be violated
    /// for NaN comparisons.
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
            Predicate::And(a, b) => {
                Predicate::new_or(Self::new_not(a.clone()), Self::new_not(b.clone()))
            }
            Predicate::Or(a, b) => {
                Predicate::new_and(Self::new_not(a.clone()), Self::new_not(b.clone()))
            }
            Predicate::Unit(_) => Predicate::Not(pred).into(),
        }
    }

    /// Constructs a new comparison predicate.
    ///
    /// This constructor will eagerly evaluate the comparison of scalars.
    #[must_use]
    pub fn new_comparison(op: CmpOp, lhs: &Term, rhs: &Term) -> Self {
        use std::cmp::Ordering;
        // If this is a scalar....
        let ordering = match (lhs, rhs) {
            (Term::Literal(Literal::F32(a)), Term::Literal(Literal::F32(b))) => {
                match a.partial_cmp(b) {
                    Some(ordering) => ordering,
                    None => return (op == CmpOp::Neq).into(),
                }
            }
            (Term::Literal(Literal::I32(a)), Term::Literal(Literal::I32(b))) => a.cmp(b),
            (Term::Literal(Literal::U32(a)), Term::Literal(Literal::U32(b))) => a.cmp(b),
            (Term::Literal(Literal::I64(a)), Term::Literal(Literal::I64(b))) => a.cmp(b),
            _ => {
                return Predicate::Comparison(op, lhs.clone(), rhs.clone());
            }
        };

        // We do scalar evaluation here.

        match (ordering, op) {
            (Ordering::Equal, CmpOp::Eq | CmpOp::Leq | CmpOp::Geq)
            | (Ordering::Less, CmpOp::Lt | CmpOp::Leq)
            | (Ordering::Greater, CmpOp::Gt | CmpOp::Geq)
            | (Ordering::Less | Ordering::Greater, CmpOp::Neq) => Predicate::True,
            _ => Predicate::False,
        }
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, strum_macros::EnumIs, Serialize, Deserialize)]
pub enum AbcExpression {
    /// A Vector is a homogenous collection of terms.
    ///
    /// The terms within are snapshotted.
    Vector {
        /// Components of the vector
        components: Vec<Term>,
        /// The innermost type.
        ty: crate::AbcScalar,
    },

    /// A matrix is a 2d homogenous collection of terms.
    ///
    /// The terms in the matrix must be vectors.
    Matrix {
        /// The terms, always of `Vector`.
        components: Vec<Term>,
        ty: crate::AbcScalar,
    },
    /// A unary operator applied to a term
    ///
    /// Currently only supports unary minus.
    UnaryOp(UnaryOp, Term),
    /// A binary operator, e.g., x + y
    BinaryOp(BinaryOp, Term, Term),
    /// A select expression, e.g., select(x, y, z)
    ///
    /// This is of the form `Select(cond, iftrue, iffalse)`
    Select(Term, Term, Term),

    /// Splat, aka a vector with the same value repeated n times.
    Splat(Term, u32),
    /// The expression for the length of an array, e.g., ArrayLength(x)
    ArrayLength(Term),

    /// The expression for the length of a specific dimension of an array, e.g., (ArrayLength(x, 1)) or (ArrayLength(x, 2))
    ArrayLengthDim(Term, std::num::NonZeroU8),

    /// Cast a term to a scalar type, e.g. `i32(x)`
    Cast(Term, AbcScalar),

    /// Access a member of a struct, e.g. `x.y`
    FieldAccess {
        base: Term,
        ty: Handle<AbcType>,
        fieldname: String,
        field_idx: usize,
    },

    /// Access an element of an array, e.g. `x[3]`
    IndexAccess { base: Term, index: Term },

    /// A store expression, which represents an assignment to an array.
    ///
    /// A store resolves to a copy of the base array, by value,
    /// with the value at the index replaced by the new value.
    Store {
        base: Term,
        index: Term,
        value: Term,
    },

    /// A store expression, which represents an assignment to a struct field.
    ///
    /// A struct store resolves to a copy of the base struct, by value,
    /// with the value at the field replaced by the new value.
    /// Here, `field_idx` corresponds to the 0-based index of the field in the struct.
    StructStore {
        base: Term,
        field_idx: usize,
        value: Term,
    },

    // The following are operator-like expressions..
    /// Max of two terms
    Max(Term, Term),
    /// Min of two terms
    Min(Term, Term),
    /// Absolute value
    Abs(Term),
    /// Pow expression
    Pow { base: Term, exponent: Term },
    /// Dot product
    Dot(Term, Term),
}

impl AbcExpression {
    pub const fn variant_name(&self) -> &'static str {
        use AbcExpression::{
            Abs, ArrayLength, ArrayLengthDim, BinaryOp, Cast, Dot, FieldAccess, IndexAccess,
            Matrix, Max, Min, Pow, Select, Splat, Store, StructStore, UnaryOp, Vector,
        };
        match *self {
            Vector { .. } => "Vector",
            Matrix { .. } => "Matrix",
            UnaryOp(..) => "UnaryOp",
            BinaryOp(..) => "BinaryOp",
            Select(..) => "Select",
            Splat(..) => "Splat",
            ArrayLength(..) => "ArrayLength",
            ArrayLengthDim(..) => "ArrayLengthDim",
            Cast(..) => "Cast",
            FieldAccess { .. } => "FieldAccess",
            IndexAccess { .. } => "IndexAccess",
            Store { .. } => "Store",
            StructStore { .. } => "StructStore",
            Max(..) => "Max",
            Min(..) => "Min",
            Abs(..) => "Abs",
            Pow { .. } => "Pow",
            Dot(..) => "Dot",
        }
    }
    /// Assuming that `self` is a series of index accesses,
    /// return the base term being indexed into and the 1-based nest level of the access.
    ///
    /// E.g., if this is `x[0][1][2]`, this will return `(x, 3)`.
    /// If this is not an index access, returns `None`.
    ///
    /// This will also return `None` if the nest level exceeds `u8::MAX`.
    pub fn get_index_and_nest(&self) -> Option<(&Term, u8)> {
        if !self.is_index_access() {
            return None;
        }

        let mut nest_level = 1;
        // Increment nest level as long as base is an index access.
        let mut curr = self;
        while let Self::IndexAccess {
            base: term @ Term::Expr(base),
            ..
        } = curr
        {
            curr = base.as_ref();
            if !curr.is_index_access() {
                return Some((term, nest_level));
            }
            nest_level = nest_level.checked_add(1u8)?;
        }

        None
    }
}

impl AbcExpression {
    /// Returns true if any of the following are true:
    ///
    /// - The expression is a literal or is comprised entirely of literals, array lengths, or terms appearing in the "uniforms" set.
    /// - The expression is a Min(a, b) where this method returns true for either `a` or `b`.
    /// - The expression's sub-terms are all uniform.
    fn only_uniforms(&self, uniforms: &FastHashSet<Term>) -> bool {
        #[allow(clippy::match_same_arms)]
        match self {
            // For min and min only, we short-circuit if either side is a uniform.
            // This is useful because we can limit the upper bound of the right hand side.
            Self::Min(a, b) => a.only_uniforms(uniforms) || b.only_uniforms(uniforms),
            Self::Abs(a) | Self::Splat(a, _) | AbcExpression::UnaryOp(_, a) => {
                a.only_uniforms(uniforms)
            }
            Self::Pow {
                base: a,
                exponent: b,
            }
            | Self::Dot(a, b)
            | Self::Max(a, b)
            | Self::BinaryOp(_, a, b) => a.only_uniforms(uniforms) && b.only_uniforms(uniforms),
            // If the base is uniform, then any access into it is a uniform..
            AbcExpression::FieldAccess { base, .. } => base.only_uniforms(uniforms),
            AbcExpression::Store { .. }
            | AbcExpression::StructStore { .. }
            | AbcExpression::Matrix { .. }
            | AbcExpression::Vector { .. } => false,
            AbcExpression::Select(cond, a, b) => {
                cond.only_uniforms(uniforms)
                    && a.only_uniforms(uniforms)
                    && b.only_uniforms(uniforms)
            }
            AbcExpression::ArrayLength(_) | AbcExpression::ArrayLengthDim(_, _) => true,
            AbcExpression::Cast(a, ty) => {
                matches!(*ty, AbcScalar::I32 | AbcScalar::U32) && a.only_uniforms(uniforms)
            }
            AbcExpression::IndexAccess { .. } => false,
        }
    }

    #[allow(clippy::match_same_arms)]
    fn only_literals(&self) -> bool {
        match self {
            Self::Vector { components, .. } | Self::Matrix { components, .. } => {
                components.iter().all(Term::only_literals)
            }
            Self::UnaryOp(_, t) => t.only_literals(),
            Self::Cast(t, _) => t.only_literals(),
            Self::ArrayLength(t) => t.only_literals(),
            Self::ArrayLengthDim(t, _) => t.only_literals(),
            Self::BinaryOp(_, l, r) => l.only_literals() && r.only_literals(),
            Self::FieldAccess { base, .. } => base.only_literals(),
            Self::Splat(t, _) => t.only_literals(),
            Self::Select(c, a, b) => c.only_literals() && a.only_literals() && b.only_literals(),
            Self::IndexAccess { base, index } => base.only_literals() && index.only_literals(),
            Self::Store { base, index, value } => {
                base.only_literals() && index.only_literals() && value.only_literals()
            }
            Self::StructStore { base, value, .. } => base.only_literals() && value.only_literals(),
            Self::Max(a, b) | Self::Min(a, b) => a.only_literals() && b.only_literals(),
            Self::Abs(t) => t.only_literals(),
            Self::Pow { base, exponent } => base.only_literals() && exponent.only_literals(),
            Self::Dot(a, b) => a.only_literals() && b.only_literals(),
        }
    }
}

impl Term {
    /// Return whether the term is comprised entirely of literals, array length terms, and terms appearing in the "uniforms" set.
    fn only_uniforms(&self, uniforms: &FastHashSet<Term>) -> bool {
        if self.is_array_length_like() || uniforms.contains(self) {
            return true;
        }
        if let Self::Expr(e) = self {
            return e.only_uniforms(uniforms);
        }
        if let Self::Predicate(p) = self {
            return p.only_uniforms(uniforms);
        }
        if let Self::Literal(l) = self {
            return l.is_u_32() || l.is_i_32() || l.is_i_64() || l.is_u_64();
        }

        false
    }

    fn only_literals(&self) -> bool {
        match self {
            Self::Literal(_) => true,
            Self::Expr(e) => e.only_literals(),
            Self::Var(_) | Self::Empty => false,
            Self::Predicate(p) => p.only_literals(),
        }
    }
}

impl Predicate {
    fn only_uniforms(&self, uniforms: &FastHashSet<Term>) -> bool {
        match self {
            Predicate::False | Predicate::True => true,
            Predicate::And(a, b) | Predicate::Or(a, b) => {
                a.only_uniforms(uniforms) && b.only_uniforms(uniforms)
            }
            Predicate::Comparison(_, a, b) => {
                a.only_uniforms(uniforms) && b.only_uniforms(uniforms)
            }
            Predicate::Not(a) => a.only_uniforms(uniforms),
            Predicate::Unit(a) => a.only_uniforms(uniforms),
        }
    }

    fn only_literals(&self) -> bool {
        match self {
            Predicate::False | Predicate::True => true,
            Predicate::And(a, b) | Predicate::Or(a, b) => a.only_literals() && b.only_literals(),
            Predicate::Comparison(_, a, b) => a.only_literals() && b.only_literals(),
            Predicate::Not(a) => a.only_literals(),
            Predicate::Unit(a) => a.only_literals(),
        }
    }
}
macro_rules! expression_sub {
    ($self:ident, $name:ident, ($($args:expr),*)) => {
        match $self {
            Self::ArrayLengthDim(t, dim) => Self::ArrayLengthDim(t.$name($($args),*), *dim),
            Self::Vector{components, ty} => Self::Vector {
                components: components.iter().map(|t| t.$name($($args),*)).collect(),
                ty: *ty
            },
            Self::Matrix{components, ty} => Self::Matrix {
                components: components.iter().map(|t| t.$name($($args),*)).collect(),
                ty: *ty
            },
            Self::UnaryOp(op, t) => Self::UnaryOp(*op, t.$name($($args),*)),
            Self::Cast(t, s) => Self::Cast(t.$name($($args),*), *s),
            Self::ArrayLength(t) => Self::ArrayLength(t.$name($($args),*)),
            Self::BinaryOp(op, l, r) => {
                Self::BinaryOp(*op, l.$name($($args),*), r.$name($($args),*))
            }
            Self::FieldAccess {
                base,
                ty,
                fieldname,
                field_idx,
            } => Self::FieldAccess {
                base: base.$name($($args),*),
                ty: ty.clone(),
                fieldname: fieldname.clone(),
                field_idx: *field_idx,
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
                field_idx,
                value,
            } => Self::StructStore {
                base: base.$name($($args),*),
                field_idx: field_idx.clone(),
                value: value.$name($($args),*),
            },
            Self::Max(a, b) => Self::Max(
                a.$name($($args),*),
                b.$name($($args),*),
            ),
            Self::Min(a, b)  => Self::Min(
                a.$name($($args),*),
                b.$name($($args),*),
            ),
            Self::Abs(t) => Self::Abs(t.$name($($args),*)),
            Self::Pow { base, exponent } => Self::Pow {
                base: base.$name($($args),*),
                exponent: exponent.$name($($args),*),
            },
            Self::Dot(a, b) => Self::Dot (
                a.$name($($args),*),
                b.$name($($args),*),
            )
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
        if mapping.len() == 1 {
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
            Self::Splat(expr, size) => {
                f.write_char('<')?;
                f.write_str(expr.as_ref())?;
                f.write_str(expr.as_ref())?;
                for _ in 1..*size {
                    f.write_str(", ")?;
                    f.write_str(expr.as_ref())?;
                }
                f.write_char('>')
            }
            Self::UnaryOp(op, term) => write!(f, "{op}({term})"),
            Self::Vector { components, .. } | Self::Matrix { components, .. } => {
                if components.is_empty() {
                    f.write_str("< >")
                } else {
                    f.write_str("")?;
                    f.write_str(components.first().unwrap().as_ref())?;
                    for term in components.get(1..).unwrap_or_default() {
                        f.write_str(", ")?;
                        f.write_str(term.as_ref())?;
                    }
                    f.write_char('>')
                }
            }
            Self::ArrayLengthDim(t, dim) => {
                write!(f, "length({t}, dim={dim})")
            }
            Self::BinaryOp(op, lhs, rhs) => write!(f, "{lhs} {op} {rhs}"),
            Self::Select(pred, then_expr, else_expr) => {
                write!(f, "select({pred}, {then_expr}, {else_expr})")
            }
            Self::ArrayLength(var) => write!(f, "length({var})"),
            Self::Cast(expr, ty) => write!(f, "cast({expr}, {ty})"),
            Self::FieldAccess {
                base, fieldname, ..
            } => {
                write!(f, "{base}.{fieldname}")
            }
            Self::IndexAccess { base, index } => write!(f, "{base}[{index}]"),
            Self::Store { base, index, value } => {
                write!(f, "store({base}, {index}, {value})")
            }
            Self::StructStore {
                base,
                field_idx,
                value,
                ..
            } => {
                write!(f, "store_field({base}, {field_idx}, {value})")
            }
            Self::Max(a, b) => write!(f, "max({a}, {b})"),
            Self::Min(a, b) => write!(f, "min({a}, {b})"),
            Self::Abs(t) => write!(f, "abs({t})"),
            Self::Pow { base, exponent } => write!(f, "pow({base}, {exponent})"),
            Self::Dot(a, b) => write!(f, "dot({a}, {b})"),
        }
    }
}

/// A struct field is a pair of a name and a type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StructField {
    name: String,
    ty: Handle<AbcType>,
}

impl StructField {
    /// Create a new struct field with the given name and type.
    #[must_use]
    pub fn new(name: String, ty: Handle<AbcType>) -> Self {
        Self { name, ty }
    }

    /// Return a reference to the name of the field
    #[must_use]
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Return a reference to the type of the field
    #[must_use]
    pub fn get_ty(&self) -> &Handle<AbcType> {
        &self.ty
    }
}

impl<T, U> From<(T, U)> for StructField
where
    T: Into<String>,
    U: Into<Handle<AbcType>>,
{
    fn from((name, ty): (T, U)) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
        }
    }
}

/// Provides an interface to define a type in the constraint system.
#[derive(Clone, Debug, PartialEq, Eq, Hash, strum_macros::EnumIs, Serialize, Deserialize)]
pub enum AbcType {
    // A user defined compound type.
    Struct {
        members: Vec<StructField>,
    },

    /// An array with a known size.
    SizedArray {
        ty: Handle<AbcType>,
        size: std::num::NonZeroU32,
    },

    /// An array with an unknown size.
    DynamicArray {
        ty: Handle<AbcType>,
    },

    /// A builtin scalar type.
    Scalar(AbcScalar),

    /// A value that doesn't exist
    ///
    /// Currently used as the type of variables that cannot be used, but whose expressions are needed
    NoneType,
}

#[allow(non_upper_case_globals)]
impl AbcType {
    /// Create a handle to a new `u32` type.
    pub fn mk_u32() -> Handle<AbcType> {
        static AbcType_U32: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Handle::new(AbcType::Scalar(AbcScalar::U32)));
        AbcType_U32.clone()
    }

    /// Create a handle to a new `i32` type.
    pub fn mk_i32() -> Handle<AbcType> {
        static AbcType_I32: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::I32)));
        AbcType_I32.clone()
    }

    /// Create a handle to a new `u64` type.
    pub fn mk_u64() -> Handle<AbcType> {
        static AbcType_U64: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::U64)));
        AbcType_U64.clone()
    }

    /// Create a handle to a new `i64` type.
    pub fn mk_i64() -> Handle<AbcType> {
        static AbcType_I64: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::I64)));
        AbcType_I64.clone()
    }

    /// Create a handle to a new `u16` type.
    pub fn mk_u16() -> Handle<AbcType> {
        static AbcType_U16: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::U16)));
        AbcType_U16.clone()
    }

    pub fn mk_i16() -> Handle<AbcType> {
        static AbcType_I16: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::I16)));
        AbcType_I16.clone()
    }

    pub fn mk_f32() -> Handle<AbcType> {
        static AbcType_F32: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::Float(4))));
        AbcType_F32.clone()
    }

    pub fn mk_f64() -> Handle<AbcType> {
        static AbcType_F64: LazyLock<Handle<AbcType>> =
            LazyLock::new(|| Arc::new(AbcType::Scalar(AbcScalar::Float(8))));
        AbcType_F64.clone()
    }

    pub const fn variant_name(&self) -> &'static str {
        match self {
            AbcType::Struct { .. } => "Struct",
            AbcType::SizedArray { .. } => "SizedArray",
            AbcType::DynamicArray { .. } => "DynamicArray",
            AbcType::Scalar(..) => "Scalar",
            AbcType::NoneType => "NoneType",
        }
    }

    /// Get the nested type from the given depth.
    /// If this is not a `SizedArray` or `Array`, then this will return `None`.
    pub fn get_nested_type(&self, depth: u8) -> Option<&Handle<AbcType>> {
        let mut curr_ty: &AbcType = self;
        for _ in 0..depth {
            curr_ty = match curr_ty {
                AbcType::SizedArray { ty, .. } | AbcType::DynamicArray { ty } => ty.as_ref(),
                _ => return None,
            }
        }

        match curr_ty {
            AbcType::SizedArray { ty, .. } | AbcType::DynamicArray { ty } => Some(ty),
            _ => None,
        }
    }
}

impl From<AbcScalar> for AbcType {
    fn from(scalar: AbcScalar) -> Self {
        Self::Scalar(scalar)
    }
}

impl TryFrom<AbcType> for AbcScalar {
    type Error = &'static str;

    fn try_from(value: AbcType) -> Result<Self, Self::Error> {
        match value {
            AbcType::Scalar(scalar) => Ok(scalar),
            _ => Err("Not a scalar"),
        }
    }
}

/// An `AbcType` can be converted from anything that can iterate over a pair of `string, type` pairs.
///
/// N.B. This trait implementation allows for the type to be constructed from a `HashMap`.
/// However, the order of the fields in a `HashMap` is not guaranteed to be preserved.
impl<K, V, T> From<T> for AbcType
where
    K: Into<String>,
    V: Into<Handle<AbcType>>,
    T: IntoIterator<Item = (K, V)>,
{
    fn from(fields: T) -> Self {
        Self::Struct {
            members: fields
                .into_iter()
                .map(|(name, ty)| StructField {
                    name: name.into(),
                    ty: ty.into(),
                })
                .collect(),
        }
    }
}

impl AbcType {
    pub fn new_struct(members: Vec<StructField>) -> Self {
        AbcType::Struct { members }
    }
}

impl std::fmt::Display for AbcType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AbcType::Struct { members } => {
                write!(f, "{{")?;
                for StructField { name, ty } in members {
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
#[cfg_attr(feature = "cffi", repr(C))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

impl AbcScalar {
    /// Convenience alias for a 32-bit unsigned integer.
    pub const U32: AbcScalar = AbcScalar::Uint(4);
    /// Convenience alias for a 32-bit signed integer.
    pub const I32: AbcScalar = AbcScalar::Sint(4);
    /// Convenience alias for a 64-bit unsigned integer.
    pub const U64: AbcScalar = AbcScalar::Uint(8);
    /// Convenience alias for a 64-bit signed integer.
    pub const I64: AbcScalar = AbcScalar::Sint(8);
    /// Convenience alias for a 16-bit unsigned integer.
    pub const U16: AbcScalar = AbcScalar::Uint(2);
    /// Convenience alias for a 16-bit signed integer.
    pub const I16: AbcScalar = AbcScalar::Sint(2);
    /// Convenience alias for a 32-bit floating point number.
    pub const F32: AbcScalar = AbcScalar::Float(4);
    /// Convenience alias for a 64-bit floating point number.
    pub const F64: AbcScalar = AbcScalar::Float(8);
}

// Aliases for common types.

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    pub name: String,
    pub args: Vec<Term>,
    pub return_type: Handle<AbcType>,

    ret_term: Term,

    /// Constraints are predicates that must hold for the summary to be valid
    pub constraints: Vec<(Constraint, u32)>,

    /// Assumptions are predicates that filter out invalid domains
    ///
    /// They mark things like assignment
    ///
    /// These encode information such as variable assignments.
    #[serde(with = "AssumptionSerializer")]
    pub assumptions: FastHashMap<Term, Assumption>,
}

/// Displaying a Summary just shows the name of the function.
impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// Expands to a macro that merges the assumption into the map, returning whether
/// or not the merge was possible.
macro_rules! add_assumption_impl {
    ($map:expr, $assumption:expr $(,)?) => {{
        use std::collections::hash_map::Entry;
        let lhs = $assumption.get_lhs();
        match $map.entry(lhs.clone()) {
            Entry::Occupied(mut entry) => {
                // If this assumption exists, then we merge it with other if possible.
                entry.get_mut().merge(&$assumption)
            }
            Entry::Vacant(entry) => {
                entry.insert($assumption);
                true
            }
        }
    }};
}
pub(crate) use add_assumption_impl;

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
            assumptions: FastHashMap::default(),
            ret_term: Term::Empty,
        }
    }

    /// Add an argument to the summary.
    pub(crate) fn add_argument(&mut self, arg: &Term) {
        self.args.push(arg.clone());
    }

    /// Add a constraint to the summary.
    pub fn add_constraint(&mut self, constraint: &Constraint, id: u32) {
        self.constraints.push((constraint.clone(), id));
    }

    /// Add an assumption to the summary.
    #[inline]
    pub fn add_assumption(&mut self, assumption: Assumption) -> bool {
        add_assumption_impl!(self.assumptions, assumption)
    }
}

static RET: std::sync::LazyLock<Term> = std::sync::LazyLock::new(|| {
    Term::Var(Arc::new(Var {
        name: "@ret".to_string(),
    }))
});
static NONETYPE: std::sync::LazyLock<Handle<AbcType>> =
    std::sync::LazyLock::new(|| Handle::new(AbcType::NoneType));

/// A literal, ripped directly from naga's literal, with the `bool` literal dropped
/// The bool literal turns into [`Predicate::True`] or [`Predicate::False`]
///
/// [`Predicate::True`]: crate::Predicate::True
/// [`Predicate::False`]: crate::Predicate::False
#[derive(
    Debug,
    Clone,
    Copy,
    PartialOrd,
    strum_macros::Display,
    strum_macros::EnumIs,
    Serialize,
    Deserialize,
)]
#[cfg_attr(feature = "cffi", repr(C))]
pub enum Literal {
    #[strum(to_string = "{0}")]
    /// May not be NaN or infinity, as defined by the wgsl spec..
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

impl Literal {
    pub fn get_scalar_kind(&self) -> AbcScalar {
        match self {
            Self::F64(_) | Self::AbstractFloat(_) => AbcScalar::F64,
            Self::F32(_) => AbcScalar::F32,
            Self::U32(_) => AbcScalar::U32,
            Self::I32(_) | Self::AbstractInt(_) => AbcScalar::I32,
            Self::U64(_) => AbcScalar::U64,
            Self::I64(_) => AbcScalar::I64,
        }
    }

    pub fn get_type(&self) -> Handle<AbcType> {
        match self {
            Self::F64(_) | Self::AbstractFloat(_) => AbcType::mk_f64(),
            Self::F32(_) => AbcType::mk_f32(),
            Self::U32(_) => AbcType::mk_u32(),
            Self::I32(_) | Self::AbstractInt(_) => AbcType::mk_i32(),
            Self::U64(_) => AbcType::mk_u64(),
            Self::I64(_) => AbcType::mk_i64(),
        }
    }
}

impl Literal {
    pub const fn variant_name(&self) -> &'static str {
        match self {
            Self::F64(_) => "F64",
            Self::F32(_) => "F32",
            Self::U32(_) => "U32",
            Self::I32(_) => "I32",
            Self::U64(_) => "U64",
            Self::I64(_) => "I64",
            Self::AbstractInt(_) => "AbstractInt",
            Self::AbstractFloat(_) => "AbstractFloat",
        }
    }
}

impl PartialEq for Literal {
    /// Equality comparisons for floating point types use their bit representation.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F64(a), Self::F64(b)) | (Self::AbstractFloat(a), Self::AbstractFloat(b)) => {
                a.to_bits() == b.to_bits()
            }
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::U32(a), Self::U32(b)) => a == b,
            (Self::I32(a), Self::I32(b)) => a == b,
            (Self::U64(a), Self::U64(b)) => a == b,
            (Self::I64(a), Self::I64(b)) | (Self::AbstractInt(a), Self::AbstractInt(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Literal {}

impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::F64(v) | Self::AbstractFloat(v) => v.to_bits().hash(state),
            Self::F32(v) => v.to_bits().hash(state),
            Self::U32(v) => v.hash(state),
            Self::I32(v) => v.hash(state),
            Self::U64(v) => v.hash(state),
            Self::I64(v) | Self::AbstractInt(v) => v.hash(state),
        }
    }
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
///
/// Note that cloning `Term` is quite cheap, as all of its members either store Arcs or are extremely small.
#[derive(
    Clone,
    Debug,
    strum_macros::Display,
    PartialEq,
    Eq,
    Hash,
    strum_macros::EnumIs,
    strum_macros::EnumTryAs,
    strum_macros::AsRefStr,
    Serialize,
    Deserialize,
)]
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
    /// Get a reference to the contained expression, otherwise return `None`
    pub const fn try_get_expr(&self) -> Option<&Handle<AbcExpression>> {
        match self {
            Self::Expr(expr) => Some(expr),
            _ => None,
        }
    }

    /// Get a reference to the contained literal, otherwise return `None`
    pub const fn try_get_literal(&self) -> Option<&Literal> {
        match self {
            Self::Literal(lit) => Some(lit),
            _ => None,
        }
    }

    /// Get a reference to the contained predicate, otherwise return `None`
    pub const fn try_get_predicate(&self) -> Option<&Handle<Predicate>> {
        match self {
            Self::Predicate(pred) => Some(pred),
            _ => None,
        }
    }

    /// Get a reference to the contained variable, otherwise return `None`
    pub const fn try_get_var(&self) -> Option<&Handle<Var>> {
        match self {
            Self::Var(var) => Some(var),
            _ => None,
        }
    }
    pub const fn variant_name(&self) -> &'static str {
        match self {
            Self::Expr(_) => "Expr",
            Self::Var(_) => "Var",
            Self::Literal(_) => "Literal",
            Self::Predicate(_) => "Predicate",
            Self::Empty => "Empty",
        }
    }
    /// Determine whether `self` and `with` have identical structure and references.
    ///
    /// This is different from equality in that it requires `Arc::ptr_eq` to check its contents.
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

    /// If `term` is an expression, gets the nest level.
    #[inline]
    pub fn get_index_and_nest_level(&self) -> Option<(&Term, u8)> {
        match self {
            Self::Expr(expr) => expr.get_index_and_nest(),
            _ => None,
        }
    }

    /// Return whether `term` is an array length or array length dim expression.
    fn is_array_length_like(&self) -> bool {
        match self {
            Self::Expr(expr) => expr.is_array_length() || expr.is_array_length_dim(),
            _ => false,
        }
    }
}

impl SubstituteTerm for Term {
    /// Creates a new term where all references to `self` have been replaced with `to`
    ///
    /// If `self` and `from` are identical, this method just returns `with`
    fn substitute(&self, from: &Term, with: &Term) -> Self {
        if self == from {
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
        if let Some(t) = mapping.iter().find(|p| p.0 == self) {
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
// literal into a term.
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
    pub fn new_unit_pred(p: &Term) -> Self {
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
    pub fn new_logical_and(lhs: &Term, rhs: &Term) -> Self {
        Term::Predicate(Predicate::new_and(lhs.clone(), rhs.clone()))
    }

    /// Constructs lhs || rhs
    #[must_use]
    pub fn new_logical_or(lhs: &Term, rhs: &Term) -> Self {
        Term::Predicate(Predicate::new_or(lhs.clone(), rhs.clone()))
    }

    /// Constructs lhs `op` rhs
    #[must_use]
    pub fn new_comparison(op: CmpOp, lhs: &Term, rhs: &Term) -> Self {
        Term::Predicate(Predicate::new_comparison(op, lhs, rhs).into())
    }

    /// Constructs !t
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    #[must_use]
    pub fn new_not(t: &Term) -> Self {
        Term::Predicate(match *t {
            Term::Predicate(ref pred) => Predicate::new_not(pred.clone()),
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

    /// Wrapper around making a new predicate and placing the predicate in this expression.
    #[must_use]
    pub fn new_cmp_op(op: CmpOp, lhs: &Term, rhs: &Term) -> Self {
        Term::Predicate(Predicate::new_comparison(op, lhs, rhs).into())
    }

    #[must_use]
    pub fn new_index_access(base: &Term, index: &Term) -> Self {
        AbcExpression::IndexAccess {
            base: base.clone(),
            index: index.clone(),
        }
        .into()
    }

    /// Creates a new field access expression
    #[must_use]
    pub fn new_struct_access(
        base: &Term, fieldname: String, ty: Handle<AbcType>, field_idx: usize,
    ) -> Self {
        Term::Expr(
            AbcExpression::FieldAccess {
                base: base.clone(),
                fieldname,
                ty,
                field_idx,
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
    pub fn new_vector(terms: &[Term], ty: AbcScalar) -> Self {
        let components = terms.to_vec();
        AbcExpression::Vector { components, ty }.into()
    }

    #[must_use]
    pub fn new_matrix(terms: &[Term], ty: AbcScalar) -> Self {
        let components = terms.to_vec();
        AbcExpression::Matrix { components, ty }.into()
    }

    #[must_use]
    pub fn new_unary_op(op: UnaryOp, t: &Term) -> Self {
        AbcExpression::UnaryOp(op, t.clone()).into()
    }

    #[must_use]
    pub fn new_binary_op(op: BinaryOp, lhs: &Term, rhs: &Term) -> Self {
        AbcExpression::BinaryOp(op, lhs.clone(), rhs.clone()).into()
    }

    #[must_use]
    pub fn new_select(pred: &Term, then_expr: &Term, else_expr: &Term) -> Self {
        // In select, term *should* be a predicate.
        // Otherwise, we have to make it into one.
        let pred = Term::Predicate(match *pred {
            Term::Predicate(ref p) => p.clone(),
            _ => Predicate::new_unit(pred),
        });
        AbcExpression::Select(pred.clone(), then_expr.clone(), else_expr.clone()).into()
    }

    #[must_use]
    #[inline]
    pub fn make_array_length(var: &Term) -> Self {
        AbcExpression::ArrayLength(var.clone()).into()
    }

    #[must_use]
    pub fn make_array_length_dim(var: &Term, dim: std::num::NonZeroU8) -> Self {
        AbcExpression::ArrayLengthDim(var.clone(), dim).into()
    }

    #[must_use]
    #[inline]
    pub fn new_store(base: Term, index: Term, value: Term) -> Self {
        AbcExpression::Store { base, index, value }.into()
    }

    #[must_use]
    pub fn new_struct_store(base: Term, field_idx: usize, value: Term) -> Self {
        Term::Expr(
            AbcExpression::StructStore {
                base,
                field_idx,
                value,
            }
            .into(),
        )
    }

    #[must_use]
    #[inline]
    pub fn new_abs(term: &Term) -> Self {
        // Multiple `abs` on the exact same term are stripped.
        if let Term::Expr(ref t) = *term {
            if t.as_ref().is_abs() {
                return term.clone();
            }
        }
        Term::Expr(AbcExpression::Abs(term.clone()).into())
    }

    /// Construct a new `max` expression.
    #[must_use]
    pub fn new_max(a: &Term, b: &Term) -> Self {
        // `Max` on two literals is evaluated.
        Term::Expr(AbcExpression::Max(a.clone(), b.clone()).into())
    }

    /// Construct a new min expression.
    #[must_use]
    pub fn new_min(a: &Term, b: &Term) -> Self {
        Term::Expr(AbcExpression::Min(a.clone(), b.clone()).into())
    }

    /// Construct a new pow expression.
    #[must_use]
    pub fn new_pow(base: &Term, exponent: &Term) -> Self {
        Term::Expr(
            AbcExpression::Pow {
                base: base.clone(),
                exponent: exponent.clone(),
            }
            .into(),
        )
    }

    /// Construct a new dot expression.
    #[must_use]
    pub fn new_dot(a: &Term, b: &Term) -> Self {
        Term::Expr(AbcExpression::Dot(a.clone(), b.clone()).into())
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
        let p = Predicate::new_comparison(CmpOp::Eq, &var_x, &var_y);
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
        let term = Term::new_logical_and(&var_x, &var_x);
        assert_eq!(term.to_string(), "x");
    }

    /// Ensure `true` or x = x
    #[rstest]
    fn test_or_with_true(literal_true: Term) {
        // Get some predicate., e.g. 'x'
        let any_pred = Term::new_var(Var {
            name: "X".to_string(),
        });
        let term = Term::new_logical_or(&literal_true, &any_pred);
        // This should be the exact same inner.
        assert_eq!(term.to_string(), "true");
    }

    #[rstest]
    fn test_term_new_comparison(var_x: Term) {
        let var_y = Var {
            name: "y".to_string(),
        };
        let term = Term::new_comparison(CmpOp::Eq, &var_x, &Term::from(var_y));
        assert_eq!(term.to_string(), "(x) == (y)");
    }

    #[rstest]
    fn test_term_new_not(var_x: Term) {
        let term = Term::new_not(&var_x);
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
    fn test_term_new_index_access() {
        let base = Term::new_var(Var::from("arr"));
        let index = Term::new_literal(0);
        let term = Term::new_index_access(&base, &index);
        assert_eq!(term.to_string(), "arr[0]");
    }

    #[rstest]
    fn test_term_new_struct_access() {
        let base = Term::new_var(Var::from("obj"));
        let ty = Arc::new(AbcType::NoneType);
        let term = Term::new_struct_access(&base, "field".to_string(), ty, 0);
        assert_eq!(term.to_string(), "obj.field");
    }

    #[rstest]
    fn test_term_new_select() {
        let pred = Term::new_literal_true();
        let then_expr = Term::new_literal(1);
        let else_expr = Term::new_literal(0);
        let term = Term::new_select(&pred, &then_expr, &else_expr);
        assert_eq!(term.to_string(), "select(true, 1, 0)");
    }

    #[rstest]
    fn test_term_make_array_length() {
        let var = Term::new_var(Var::from("arr"));
        let term = Term::make_array_length(&var);
        assert_eq!(term.to_string(), "length(arr)");
    }

    #[rstest]
    fn test_term_substitute(var_x: Term, var_y: Term) {
        let term = Term::new_logical_and(&var_x, &var_y);
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
        let term = Term::new_logical_and(&var_x, &var_y);
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

// This is intentionally placed at the end of the file in order for cbindgen to properly declare the bindings in this file
// before the bindings in the other files, which require these to be defined first.
#[cfg(feature = "cffi")]
pub mod cffi;
