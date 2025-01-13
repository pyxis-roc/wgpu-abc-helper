/*!
Translates constraints from the system into intervals.

 The order of constraints in the system must be agnostic.
To properly account for this, intervals are captured with `RefCell` in order to allow for
terms that are assigned to be equal to each other to have the same interval.
x = a
a = x
Then the interval for `x` should be [4, 4]
And the interval for `a` should be [4, 4].
That means that what we *really* want are references to the interval.
That is, our intervals will always correspond with SSA names.
If they don't correspond to SSA names, then we can't do anything.
This is because the order of the constraints is supposed to be agnostic.
So when we see `x = a`, how are we supposed to know what `a` is?
That means that we HAVE to use Rc<RefCell> for this.
That allows us to basically have smart pointers to the interval.*/

#![allow(unused)]
use std::borrow::{Borrow, Cow};
use std::collections::hash_map::Entry;
use std::hint::unreachable_unchecked;
use std::{array, default};

#[cfg(feature = "deserialize")]
use serde::Deserialize;
#[cfg(feature = "serialize")]
use serde::Serialize;

use super::ops::IntervalCast;
use super::{
    ops::{
        Intersect, IntersectAssign, IntervalAbs, IntervalAdd, IntervalDiv, IntervalEq, IntervalGe,
        IntervalGt, IntervalLe, IntervalLt, IntervalMax, IntervalMin, IntervalMod, IntervalMul,
        IntervalNe, IntervalNeg, IntervalSub, Union, UnionAssign,
    },
    solver, I64Interval, U64Interval, WrappedInterval,
};
use super::{BoolInterval, I32Interval, Interval, IntervalError, U32Interval};

use crate::solvers::interval::{self, IntervalBoundary};
use crate::{
    helper::ConstraintModule, AbcExpression, AbcScalar, AbcType, FastHashMap, Handle, Term,
};
use crate::{
    Assumption, BinaryOp, CmpOp, Constraint, ConstraintId, ConstraintOp, FastHashSet, Literal,
    Predicate, StructField,
};
use log::warn;
use strum::VariantNames;
use thiserror;

mod resolver;
use resolver::Resolver;

const I32_MAX_AS_I64: i64 = i32::MAX as i64;

#[cfg_attr(feature = "serialize", derive(Serialize))]
#[cfg_attr(feature = "deserialize", derive(Deserialize))]
#[derive(Debug, thiserror::Error, Clone)]
pub enum SolverError {
    #[error("Invalid Summary")]
    InvalidSummary,
    #[error("Unsupported type")]
    UnsupportedType,
    #[error("Multiple assumptions for the same term")]
    DuplicateAssignmentError(Term),

    #[error("Interval Error: {0}")]
    IntervalError(#[from] super::IntervalError),

    /// Multiple assignments to the same term.
    /// The only term that this can occur for is `ret`.
    ///
    /// `Ret` is special, it is the only term that may be assigned to more than once.
    #[error("SSA Violation: {0}")]
    SsaViolation(Term),

    #[error("Type mismatch: expected {expected}, have: {have}")]
    TypeMismatch {
        expected: &'static str,
        have: &'static str,
    },
    #[error("Unexpected: {0}")]
    Unexpected(&'static str),
    #[error("Unsupported operation: {0}")]
    Unsupported(&'static str),
    #[error("Top-level assumptions cannot be satisfied.")]
    DeadCode,
}

/// Wrapper for intervals of specific kinds.
///
/// Primarily used to allow intervals corresponding to different value types to be stored in the same map.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum_macros::AsRefStr,
    strum_macros::VariantNames,
    strum_macros::EnumIs,
)]
pub enum IntervalKind {
    I32(I32Interval),
    I64(I64Interval),
    U64(U64Interval),
    U32(U32Interval),
    Bool(BoolInterval),
    Top,
}

impl IntervalKind {
    pub fn pretty_print(&self) -> impl std::fmt::Display {
        match *self {
            Self::Bool(b) => b.pretty_print().to_string(),
            Self::I64(ref i) => i.pretty_print().to_string(),
            Self::I32(ref i) => i.pretty_print().to_string(),
            Self::U32(ref u) => u.pretty_print().to_string(),
            Self::U64(ref u) => u.pretty_print().to_string(),
            Self::Top => "T".to_string(),
        }
    }
}

impl From<bool> for IntervalKind {
    /// Convert `true` to `IntervalKind::Bool(BoolInterval::True)`, and `false` to `IntervalKind::Bool(BoolInterval::False)`.
    fn from(value: bool) -> Self {
        IntervalKind::Bool(value.into())
    }
}

/// Implement `From<Range<T>>` for `IntervalKind` for each type in the provided list.
macro_rules! from_range_impl {
    ($(($ty:ty, $variant:ident)),*) => {
        $(
            impl From<std::ops::Range<$ty>> for IntervalKind {
                fn from(range: std::ops::Range<$ty>) -> Self {
                    Self::$variant(range.into())
                }
            }
        )*
    }
}

from_range_impl! {(i32, I32), (i64, I64), (u32, U32), (u64, U64)}

impl std::fmt::Display for IntervalKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntervalKind::I32(interval) => write!(f, "{interval}"),
            IntervalKind::I64(interval) => write!(f, "{interval}"),
            IntervalKind::U32(interval) => write!(f, "{interval}"),
            IntervalKind::U64(interval) => write!(f, "{interval}"),
            IntervalKind::Bool(interval) => write!(f, "{interval}"),
            IntervalKind::Top => write!(f, "Top"),
        }
    }
}

impl IntervalKind {
    fn from_inequality_assumption() -> Self {
        todo!()
    }
}

impl IntervalKind {
    /// Return true if the contained interval is `top`.
    ///
    /// This means it is either `IntervalKind::Top`, or the wrapped interval is `Interval::TOP`.
    #[inline]
    fn matches_top(&self) -> bool {
        match self {
            Self::Top => true,
            Self::I32(interval) => interval.is_top(),
            Self::I64(interval) => interval.is_top(),
            Self::U32(interval) => interval.is_top(),
            Self::U64(interval) => interval.is_top(),
            Self::Bool(interval) => interval.is_top(),
        }
    }

    #[inline]
    fn variant_name(&self) -> &'static str {
        match self {
            Self::I32(_) => "I32",
            Self::I64(_) => "I64",
            Self::U32(_) => "U32",
            Self::U64(_) => "U64",
            Self::Bool(_) => "Bool",
            Self::Top => "Top",
        }
    }
    /// Return whether the two intervals are the same variant (or one is TOP).
    #[inline]
    fn is_same_variant(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (IntervalKind::Top, _)
                | (_, IntervalKind::Top)
                | (IntervalKind::I32(_), IntervalKind::I32(_))
                | (IntervalKind::I64(_), IntervalKind::I64(_))
                | (IntervalKind::U32(_), IntervalKind::U32(_))
                | (IntervalKind::U64(_), IntervalKind::U64(_))
                | (IntervalKind::Bool(_), IntervalKind::Bool(_))
        )
    }
    fn is_unit(&self) -> bool {
        match *self {
            Self::I32(ref interval) => interval.is_unit(),
            Self::I64(ref interval) => interval.is_unit(),
            Self::U32(ref interval) => interval.is_unit(),
            Self::U64(ref interval) => interval.is_unit(),
            Self::Bool(interval) => {
                interval == BoolInterval::True || interval == BoolInterval::False
            }
            Self::Top => false,
        }
    }
    /// Return whether the interval contains the provided literal.
    ///
    /// If the interval is of the wrong type
    #[inline]
    fn contains(&self, literal: &Literal) -> bool {
        match (self, literal) {
            (&Self::Top, _) => true,
            (Self::I32(interval), Literal::I32(literal)) => interval.has_value(*literal),
            (Self::I64(interval), Literal::I64(literal)) => interval.has_value(*literal),
            (Self::U32(interval), Literal::U32(literal)) => interval.has_value(*literal),
            (Self::U64(interval), Literal::U64(literal)) => interval.has_value(*literal),
            _ => false,
        }
    }
    /// Return whether the lower bound is the maximum value for its type.
    fn lower_is_max(&self) -> bool {
        match self {
            Self::I32(interval) => interval.get_lower() == (i32::MAX, false),
            Self::I64(interval) => interval.get_lower() == (i64::MAX, false),
            Self::U32(interval) => interval.get_lower() == (u32::MAX, false),
            Self::U64(interval) => interval.get_lower() == (u64::MAX, false),
            Self::Bool(interval) => interval == &BoolInterval::True,
            Self::Top => false,
        }
    }

    /// Return whether the lower bound is the minimum value for its type.
    fn lower_is_min(&self) -> bool {
        match self {
            Self::I32(interval) => interval.get_lower() == (i32::MIN, false),
            Self::I64(interval) => interval.get_lower() == (i64::MIN, false),
            Self::U32(interval) => interval.get_lower() == (u32::MIN, false),
            Self::U64(interval) => interval.get_lower() == (u64::MIN, false),
            Self::Bool(interval) => interval == &BoolInterval::False,
            Self::Top => true,
        }
    }

    /// Return whether the upper bound is the minimum value for its type.
    fn upper_is_min(&self) -> bool {
        match self {
            Self::I32(interval) => interval.get_upper() == (i32::MIN, false),
            Self::I64(interval) => interval.get_upper() == (i64::MIN, false),
            Self::U32(interval) => interval.get_upper() == (u32::MIN, false),
            Self::U64(interval) => interval.get_upper() == (u64::MIN, false),
            Self::Bool(interval) => interval == &BoolInterval::False,
            Self::Top => false,
        }
    }

    /// Return whether the upper bound is the maximum value for its type.
    fn upper_is_max(&self) -> bool {
        match self {
            Self::I32(interval) => interval.get_upper() == (i32::MAX, false),
            Self::I64(interval) => interval.get_upper() == (i64::MAX, false),
            Self::U32(interval) => interval.get_upper() == (u32::MAX, false),
            Self::U64(interval) => interval.get_upper() == (u64::MAX, false),
            Self::Bool(interval) => interval == &BoolInterval::True,
            Self::Top => true,
        }
    }

    /// Return a new interval that can be used to intersect with another to represent
    /// an interval that is greater equal than `self`.
    ///
    /// If `self` is not empty and its lower bound is not MAX, this returns an interval
    /// of the form `[self.lower(), MAX]`. Otherwise, this returns the empty interval.
    fn as_gt_interval(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.lower_is_max() {
            return IntervalKind::U32(U32Interval::Empty);
        }
        macro_rules! arm_impl {
            ($variant:ident, $kind:ty, $interval:expr, $ty:ty) => {
                IntervalKind::$variant(<$kind>::new_concrete(
                    $interval.get_lower().0 + 1,
                    <$ty>::MAX,
                ))
            };
        }
        match self {
            Self::I32(interval) => arm_impl!(I32, I32Interval, interval, i32),
            Self::I64(interval) => arm_impl!(I64, I64Interval, interval, i64),
            Self::U32(interval) => arm_impl!(U32, U32Interval, interval, u32),
            Self::U64(interval) => arm_impl!(U64, U64Interval, interval, u64),
            Self::Bool(_) | Self::Top => self.clone(),
        }
    }

    /// Return a new interval that can be used to intersect with another to represent
    /// an interval that is greater than or equal than `self`.
    ///
    /// If `self` is not empty, this returns an interval of the form `[self.lower(), MAX]`
    /// Otherwise, this returns the empty interval.
    fn as_ge_interval(&self) -> Self {
        if self.is_empty() || self.lower_is_min() || self.upper_is_max() {
            return self.clone();
        }
        macro_rules! arm_impl {
            ($variant:ident, $kind:ty, $interval:expr, $ty:ty) => {
                <$kind>::new_concrete($interval.get_lower().0, <$ty>::MAX).into()
            };
        }
        match self {
            Self::I32(interval) => arm_impl!(I32, I32Interval, interval, i32),
            Self::I64(interval) => arm_impl!(I64, I64Interval, interval, i64),
            Self::U32(interval) => arm_impl!(U32, U32Interval, interval, u32),
            Self::U64(interval) => arm_impl!(U64, U64Interval, interval, u64),
            Self::Bool(_) | Self::Top => self.clone(),
        }
    }

    /// Return a new interval that can be used to intersect with another to represent
    /// an interval that is less than `self`.
    ///
    /// If `self` is not empty and upper is not MIN, this returns an interval of the form `[MIN, self.upper() - 1]`
    /// Otherwise, this returns the empty interval.
    fn as_lt_interval(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.upper_is_min() {
            return IntervalKind::U32(U32Interval::Empty);
        }
        macro_rules! arm_impl {
            ($variant:ident, $kind:ty, $interval:expr, $ty:ty) => {
                IntervalKind::$variant(<$kind>::new_concrete(
                    <$ty>::MIN,
                    $interval.get_upper().0 - 1,
                ))
            };
        }
        match self {
            Self::I32(interval) => arm_impl!(I32, I32Interval, interval, i32),
            Self::I64(interval) => arm_impl!(I64, I64Interval, interval, i64),
            Self::U32(interval) => arm_impl!(U32, U32Interval, interval, u32),
            Self::U64(interval) => arm_impl!(U64, U64Interval, interval, u64),
            Self::Bool(_) | Self::Top => self.clone(),
        }
    }

    /// Return a new interval that can be used to intersect with another to represent
    /// an interval that is less than or equal than `self`.
    ///
    /// If `self` is not empty, this returns an interval of the form `[MIN, self.upper()]`.
    /// Otherwise, this returns the empty interval.
    fn as_le_interval(&self) -> Self {
        if self.is_empty() || self.upper_is_min() || self.lower_is_max() {
            return self.clone();
        }
        macro_rules! arm_impl {
            ($variant:ident, $kind:ty, $interval:expr, $ty:ty) => {
                <$kind>::new_concrete($interval.get_upper().0, <$ty>::MAX).into()
            };
        }
        match self {
            Self::I32(interval) => arm_impl!(I32, I32Interval, interval, i32),
            Self::I64(interval) => arm_impl!(I64, I64Interval, interval, i64),
            Self::U32(interval) => arm_impl!(U32, U32Interval, interval, u32),
            Self::U64(interval) => arm_impl!(U64, U64Interval, interval, u64),
            Self::Bool(_) | Self::Top => self.clone(),
        }
    }
}

/// Generate `TryFrom<IntervalKind>` for `variant`
macro_rules! IntervalKind_try_from_impl {
    ($($ty:ty => $variant:ident, $try_name:ident);* $(;)?) => {
        $(impl TryFrom<IntervalKind> for $ty {
            type Error = IntervalError;
            fn try_from(value: IntervalKind) -> Result<Self, Self::Error> {
                match value {
                    IntervalKind::$variant(interval) => Ok(interval),
                    IntervalKind::Top => Ok(<$ty>::TOP),
                    _ => Err(IntervalError::IncompatibleTypes),
                }
            }
        }
        impl IntervalKind {
            fn $try_name(&self) -> Option<&$ty> {
                match *self {
                    IntervalKind::$variant(ref interval) => Some(interval),
                    IntervalKind::Top => Some(&<$ty>::TOP),
                    _ => None,
                }
            }
        }
    )*
    }
}

IntervalKind_try_from_impl! {
    I32Interval => I32, try_as_i32_interval;
    U32Interval => U32, try_as_bool_interval;
    BoolInterval => Bool, try_as_bool;
    I64Interval => I64, try_as_i64_interval;
    U64Interval => U64, try_as_u64_interval;
}

impl IntervalKind {
    pub const fn try_as_i32(&self) -> Option<&I32Interval> {
        match *self {
            Self::I32(ref interval) => Some(interval),
            IntervalKind::Top => Some(&I32Interval::TOP),
            _ => None,
        }
    }
}

/// `IntervalKind` is a wrapper, and operations on intervals of the same kind are delegated to that operation
/// applied to each of the intervals.
///
/// This macro expands to a match statement that will apply the operation to the intervals of the same kind.
///
/// It also includes a special invocation for `Top` intervals,
/// and allows for additional arms for other cases.
macro_rules! same_arm_impl {
    ($a:ident, $b:ident, $orig:ident, $self:expr, $rhs:expr, $answer:expr,
        $top_answer:expr,
        $bool_answer:expr,
        $($arm:pat => $body:expr),*
        $(,)?
    ) => {
        {
            use IntervalKind as IK;
            match ($self, $rhs) {
                (IK::I32(ref $a), IK::I32(ref $b)) => {$answer},
                (IK::Top, $orig @ IK::I32(ref $a)) | ($orig @ IK::I32(ref $a), IK::Top) => {$top_answer}

                (IK::U32(ref $a), IK::U32(ref $b)) => {$answer},
                (IK::Top, $orig @ IK::U32(ref $a)) | ($orig @ IK::U32(ref $a), IK::Top) => {$top_answer}

                (IK::I64(ref $a), IK::I64(ref $b)) => {$answer},
                (IK::Top, $orig @ IK::I64(ref $a)) | ($orig @ IK::I64(ref $a), IK::Top) => {$top_answer}

                (IK::U64(ref $a), IK::U64(ref $b)) => {$answer},
                (IK::Top, $orig @ IK::U64(ref $a)) | ($orig @ IK::U64(ref $a), IK::Top) => {$top_answer}

                (IK::Bool(ref $a), IK::Bool(ref $b)) => {$bool_answer},
                (IK::Top, $orig @ IK::Bool(ref $a)) | ($orig @ IK::Bool(ref $a), IK::Top) => {$top_answer},
                $($arm => $body),*
            }
        }
    };
}
impl IntervalKind {
    /// Return whether the contained interval is empty.
    fn is_empty(&self) -> bool {
        match *self {
            IntervalKind::Top => false,
            IntervalKind::U32(ref interval) => interval.is_empty(),
            IntervalKind::I32(ref interval) => interval.is_empty(),
            IntervalKind::Bool(ref interval) => interval.is_empty(),
            IntervalKind::I64(ref interval) => interval.is_empty(),
            IntervalKind::U64(ref interval) => interval.is_empty(),
        }
    }
    /// Invokes the intervals' [`interval_union`] method on the provided intervals.
    /// # Errors
    /// If the intervals are different variants (and one of them is not Top), then this returns [`IntervalError::IncompatibleTypes`].
    ///
    /// [`interval_union`]: WrappedInterval::interval_union
    fn interval_union(&self, rhs: &Self) -> Result<Self, IntervalError> {
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_union(b).into()),
            Ok(IK::Top),
            Ok(a.union(*b).into()),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            _ => Err(IntervalError::IncompatibleTypes),
        )
    }

    /// Invokes the intervals' [`interval_add`] method on the provided intervals.
    ///
    /// # Errors
    /// If the intervals are different variants (and one of them is not Top), then this returns [`IntervalError::IncompatibleTypes`].
    ///
    /// [`interval_add`]: WrappedInterval::interval_add
    fn interval_add(&self, rhs: &Self) -> Result<Self, IntervalError> {
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_add(b).into()),
            Ok(IK::Top),
            Err(IntervalError::InvalidOp("Add", "Bool")),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            (a, b) => Err(IntervalError::InvalidBinOp("add", a.variant_name(), b.variant_name())),
        )
    }

    fn interval_sub(&self, rhs: &Self) -> Result<Self, IntervalError> {
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_sub(b).into()),
            Ok(IK::Top),
            Err(IntervalError::InvalidOp("Sub", "Bool")),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            _ => Err(IntervalError::IncompatibleTypes),
        )
    }

    /// Invokes the intervals' [`interval_mul`] method on the provided intervals.
    /// # Errors
    /// If the intervals are different variants (and one of them is not Top), then this returns [`IntervalError::IncompatibleTypes`].
    ///
    /// [`interval_mul`]: WrappedInterval::interval_mul
    fn interval_mul(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if self.is_empty() || rhs.is_empty() {
            return Ok(IntervalKind::U32(U32Interval::Empty));
        }
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_mul(b).into()),
            Ok(IK::Top),
            Err(IntervalError::InvalidOp("Mul", "Bool")),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            (a, b) => Err(IntervalError::InvalidBinOp("Multiplication", a.variant_name(), b.variant_name())),
        )
    }

    /// Invokes the interval's `interval_abs` method.
    fn interval_abs(&self) -> std::borrow::Cow<Self> {
        use std::borrow::Cow;
        let res = Cow::Borrowed(self);
        match self {
            // we do nothing to empty intervals.
            _ if self.is_empty() => res,
            // These intervals have no negative values, so `abs` does nothing.
            Self::Bool(_) | Self::U32(_) | Self::U64(_) => res,
            Self::I32(interval) => {
                let low_bound = std::cmp::max(0, interval.get_lower().0);
                Cow::Owned(I32Interval::new_concrete(low_bound, interval.get_upper().0).into())
            }
            Self::I64(interval) => {
                let low_bound = std::cmp::max(0, interval.get_lower().0);
                Cow::Owned(I64Interval::new_concrete(low_bound, interval.get_upper().0).into())
            }
            Self::Top => Cow::Owned(IntervalKind::Top),
        }
    }

    fn interval_div(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if self.is_empty() || rhs.is_empty() {
            return Ok(IntervalKind::U32(U32Interval::Empty));
        }
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_div(b).into()),
            Ok(IK::Top),
            Err(IntervalError::InvalidOp("Div", "Bool")),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            (a, b) => Err(IntervalError::InvalidBinOp("Division", a.variant_name(), b.variant_name())),
        )
    }

    fn interval_mod(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if self.is_empty() || rhs.is_empty() {
            return Ok(IntervalKind::U32(U32Interval::Empty));
        }
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_mod(b).into()),
            Ok(IK::Top),
            Err(IntervalError::InvalidOp("Mod", "Bool")),
            (IntervalKind::Top, IntervalKind::Top) => Ok(IK::Top),
            (a, b) => Err(IntervalError::InvalidBinOp("Modulo", a.variant_name(), b.variant_name())),
        )
    }

    fn interval_max(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if self.is_bool() || rhs.is_bool() {
            return Err(IntervalError::InvalidOp("Max", "Bool"));
        }
        if self.is_empty() {
            return Ok(self.clone());
        }
        if rhs.is_empty() {
            return Ok(rhs.clone());
        }
        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_max(b).into()),
            // Max of `a` and `top` is a's lower and b
            Ok(orig.as_ge_interval()),
            Err(IntervalError::InvalidOp("Max", "Bool")),
            (IK::Top, IK::Top) => Ok(IK::Top),
            (a, b) => Err(IntervalError::InvalidBinOp("Max", a.variant_name(), b.variant_name()))
        )
    }

    /// Casts the interval to `u32`, with the same semantics as wgsl's [`u32`](https://gpuweb.github.io/gpuweb/wgsl/#u32-builtin) builtin.
    fn interval_cast_u32(&self) -> Self {
        match self {
            Self::I32(interval) => Self::U32(interval.interval_cast()),
            Self::U32(_) => self.clone(),
            Self::Bool(interval) => Self::U32(interval.interval_cast()),
            _ => {
                log::warn!(
                    "Casting from {:?} to u32 results in an overapproximation",
                    self.variant_name()
                );
                Self::Top
            }
        }
    }

    /// Casts the interval to `i32`, with the same semantics as wgsl's [`i32`](https://gpuweb.github.io/gpuweb/wgsl/#i32-builtin) builtin.
    fn interval_cast_i32(&self) -> Self {
        match self {
            Self::I32(_) => self.clone(),
            Self::U32(interval) => Self::I32(interval.interval_cast()),
            Self::Bool(interval) => Self::I32(interval.interval_cast()),
            _ => {
                log::warn!(
                    "Casting from {:?} to i32 results in an overapproximation",
                    self.variant_name()
                );
                Self::Top
            }
        }
    }

    /// Apply wgsl's `bool()` builtin on the contained interval.
    fn interval_cast_bool(&self) -> Self {
        if self.is_empty() {
            return self.clone();
        }
        if self.matches_top() {
            return IntervalKind::Bool(BoolInterval::Unknown);
        }
        match self {
            Self::I32(interval) => Self::Bool(interval.interval_cast()),
            Self::I64(interval) => Self::Bool(interval.interval_cast()),
            Self::U32(interval) => Self::Bool(interval.interval_cast()),
            Self::U64(interval) => Self::Bool(interval.interval_cast()),
            Self::Bool(_) => self.clone(),
            Self::Top => IntervalKind::Bool(BoolInterval::Unknown),
        }
    }

    fn interval_min(&self, rhs: &Self) -> Result<Self, IntervalError> {
        if self.is_bool() || rhs.is_bool() {
            return Err(IntervalError::InvalidOp("Min", "Bool"));
        }
        if self.is_empty() {
            return Ok(self.clone());
        }
        if rhs.is_empty() {
            return Ok(rhs.clone());
        }

        same_arm_impl!(
            a, b, orig, self, rhs,
            Ok(a.interval_min(b).into()),
            Ok(orig.as_le_interval()),
            Err(IntervalError::InvalidOp("Min", "Bool")),
            (IK::Top, IK::Top) => Ok(IK::Top),
            _ => Err(IntervalError::IncompatibleTypes),
        )
    }

    fn interval_eq(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                interval_a.interval_eq(interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                interval_a.interval_eq(interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                interval_a.interval_eq(interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                interval_a.interval_eq(interval_b)
            }
            (IntervalKind::Bool(interval_a), IntervalKind::Bool(interval_b)) => {
                interval_a.interval_eq(*interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }

    fn interval_ne(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                interval_a.interval_ne(interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                interval_a.interval_ne(interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                interval_a.interval_ne(interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                interval_a.interval_ne(interval_b)
            }
            (IntervalKind::Bool(interval_a), IntervalKind::Bool(interval_b)) => {
                interval_a.interval_neq(*interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }

    fn interval_ge(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                <I32Interval as IntervalGe>::interval_ge(interval_a, interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                <U32Interval as IntervalGe>::interval_ge(interval_a, interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                <I64Interval as IntervalGe>::interval_ge(interval_a, interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                <U64Interval as IntervalGe>::interval_ge(interval_a, interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }

    fn interval_gt(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                <I32Interval as IntervalGt>::interval_gt(interval_a, interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                <U32Interval as IntervalGt>::interval_gt(interval_a, interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                <I64Interval as IntervalGt>::interval_gt(interval_a, interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                <U64Interval as IntervalGt>::interval_gt(interval_a, interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }

    fn interval_le(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                <I32Interval as IntervalLe>::interval_le(interval_a, interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                <U32Interval as IntervalLe>::interval_le(interval_a, interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                <I64Interval as IntervalLe>::interval_le(interval_a, interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                <U64Interval as IntervalLe>::interval_le(interval_a, interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }

    fn interval_lt(&self, other: &Self) -> BoolInterval {
        match (self, other) {
            (IntervalKind::I32(interval_a), IntervalKind::I32(interval_b)) => {
                <I32Interval as IntervalLt>::interval_lt(interval_a, interval_b)
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(interval_b)) => {
                <U32Interval as IntervalLt>::interval_lt(interval_a, interval_b)
            }
            (IntervalKind::I64(interval_a), IntervalKind::I64(interval_b)) => {
                <I64Interval as IntervalLt>::interval_lt(interval_a, interval_b)
            }
            (IntervalKind::U64(interval_a), IntervalKind::U64(interval_b)) => {
                <U64Interval as IntervalLt>::interval_lt(interval_a, interval_b)
            }
            (IntervalKind::I32(interval_a), IntervalKind::U32(interval_b)) => {
                if interval_a.is_empty() || interval_b.is_empty() {
                    return BoolInterval::Unknown;
                };
                let Ok(a_upper) = u32::try_from(interval_a.get_upper().0) else {
                    return BoolInterval::False;
                };
                let Ok(a_lower) = u32::try_from(interval_a.get_lower().0) else {
                    return BoolInterval::Empty;
                };

                U32Interval::new_concrete(a_lower, a_upper).interval_lt(interval_b)
            }
            _ => BoolInterval::Unknown,
        }
    }
}

impl Default for IntervalKind {
    fn default() -> Self {
        Self::TOP
    }
}

impl IntervalKind {
    const TOP: IntervalKind = IntervalKind::Top;
}

macro_rules! interval_from_impl {
    ($($ty:ty => $variant:ident);* $(;)?) => {
        $(impl From<$ty> for IntervalKind {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }

        })*
    }
}
interval_from_impl! {
    I32Interval => I32;
    U32Interval => U32;
    BoolInterval => Bool;
    I64Interval => I64;
    U64Interval => U64;
}

impl IntervalKind {
    /// Return the intersection `self` with `other`
    fn intersection(&self, other: &Self) -> Result<Self, IntervalError> {
        if other.is_empty() {
            return Ok(self.clone());
        }

        match (self, other) {
            (IntervalKind::Top, other) | (other, IntervalKind::Top) => Ok(other.clone()),
            (IntervalKind::Bool(ref interval_a), IntervalKind::Bool(ref interval_b)) => {
                Ok(IntervalKind::Bool(interval_a.intersection(*interval_b)))
            }
            (IntervalKind::I32(ref interval_a), IntervalKind::I32(ref interval_b)) => Ok(
                IntervalKind::I32(interval_a.interval_intersection(interval_b)),
            ),
            (IntervalKind::U32(ref interval_a), IntervalKind::U32(ref interval_b)) => Ok(
                IntervalKind::U32(interval_a.interval_intersection(interval_b)),
            ),
            (a, b) => Err(IntervalError::InvalidBinOp(
                "Interval Intersection",
                a.variant_name(),
                b.variant_name(),
            )),
        }
    }

    /// In-place update `self` to be the intersection with `other.`
    fn intersect(&mut self, other: &Self) -> Result<(), IntervalError> {
        if self.is_top() {
            self.clone_from(other);
            return Ok(());
        }
        if other.is_top() {
            return Ok(());
        }
        match (self, other) {
            (IntervalKind::Bool(ref mut interval_a), IntervalKind::Bool(ref interval_b)) => {
                *interval_a = interval_a.intersection(*interval_b);
            }
            (IntervalKind::I32(ref mut interval_a), IntervalKind::I32(ref interval_b)) => {
                interval_a.interval_intersect(interval_b);
            }
            (IntervalKind::U32(ref mut interval_a), IntervalKind::U32(ref interval_b)) => {
                interval_a.interval_intersect(interval_b);
            }
            (a, b) => {
                return Err(IntervalError::InvalidBinOp(
                    "Interval Intersect",
                    a.variant_name(),
                    b.variant_name(),
                ))
            }
        }

        Ok(())
    }
}

impl IntervalKind {
    pub fn as_u32(&self) -> Option<&U32Interval> {
        match self {
            IntervalKind::U32(interval) => Some(interval),
            IntervalKind::Top => Some(&U32Interval::TOP),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&I32Interval> {
        match self {
            IntervalKind::I32(interval) => Some(interval),
            IntervalKind::Top => Some(&I32Interval::TOP),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<BoolInterval> {
        match self {
            IntervalKind::Bool(interval) => Some(*interval),
            IntervalKind::Top => Some(BoolInterval::Unknown),
            _ => None,
        }
    }
}

type Error = &'static str;

/// Attempt to convert `ty_in` into the widest interval for said type.
///
/// # Errors
/// If the type is not supported, then an error is returned.
impl From<AbcScalar> for IntervalKind {
    /// Attempt to convert `ty_in` into the widest interval for said type.
    ///
    /// # Errors
    /// If the type is not supported, then an error is returned.
    #[inline]
    fn from(ty_in: AbcScalar) -> Self {
        match ty_in {
            AbcScalar::Uint(4) => IntervalKind::U32(U32Interval::top()),
            AbcScalar::Sint(4) => IntervalKind::I32(I32Interval::top()),
            AbcScalar::Sint(8) => IntervalKind::I64(I64Interval::top()),
            AbcScalar::Uint(8) => IntervalKind::U64(U64Interval::top()),
            AbcScalar::Bool => IntervalKind::Bool(BoolInterval::Unknown),
            _ => IntervalKind::Top,
        }
    }
}

impl TryFrom<&AbcType> for IntervalKind {
    type Error = &'static str;

    /// Attempt to convert `ty_in` into the widest interval for said type.
    ///
    /// # Errors
    /// If the type is not supported, then an error is returned.
    #[inline]
    fn try_from(ty_in: &AbcType) -> Result<Self, Self::Error> {
        use AbcType::Scalar;
        match *ty_in {
            Scalar(t) => Ok(IntervalKind::from(t)),
            _ => Err("Unsupported type"),
        }
    }
}

impl TryFrom<AbcType> for IntervalKind {
    type Error = &'static str;

    /// Attempt to convert `ty_in` into the widest interval for said type.
    ///
    /// # Errors
    /// If the type is not supported, then an error is returned.
    #[inline]
    fn try_from(ty_in: AbcType) -> Result<Self, Self::Error> {
        use AbcType::Scalar;
        match ty_in {
            Scalar(t) => Ok(IntervalKind::from(t)),
            _ => Err("Unsupported type"),
        }
    }
}

// Translator needs to go from constraints & assumptions to intervals.

fn ty_to_interval(
    term: &Term, term_ty: &Handle<AbcType>, term_map: &mut FastHashMap<Term, IntervalKind>,
) -> Result<(), SolverError> {
    // If the ty is a basic scalar, then we can convert it to an interval.
    let inner_ty = term_ty.as_ref();
    if let AbcType::Scalar(s) = *inner_ty {
        term_map.insert(term.clone(), IntervalKind::from(s));
        return Ok(());
    }

    match *term_ty.as_ref() {
        // For sized arrays, we just mark the type of the length.
        AbcType::SizedArray { size, .. } => {
            // We insert into the term map the term representing the length of this array.
            let as_array_length = Term::make_array_length(term);
            term_map.insert(
                as_array_length,
                IntervalKind::U32(U32Interval::new_concrete(0, u32::from(size))),
            );
        }
        AbcType::DynamicArray { .. } => {
            // We insert into the term map the term representing the length of this array.
            let as_array_length = Term::make_array_length(term);
            term_map.insert(as_array_length, IntervalKind::U32(U32Interval::TOP));
        }

        AbcType::NoneType => {
            // We don't need to do anything here. This term does not have a type.
        }

        AbcType::Struct { ref members } => {
            for (pos, StructField { ref name, ref ty }) in members.iter().enumerate() {
                let member_access = Term::new_struct_access(term, name.clone(), ty.clone(), pos);
                ty_to_interval(&member_access, ty, term_map)?;
            }
        }

        AbcType::Scalar(_) => unreachable!("AbcScalar has already been handled"),
    };

    Ok(())
}

/// Initializes the intervals for each term from the provided type map.
pub(crate) fn initialize_intervals(
    type_map: &FastHashMap<Term, Handle<AbcType>>,
) -> Result<FastHashMap<Term, IntervalKind>, SolverError> {
    let mut term_map: FastHashMap<Term, IntervalKind> =
        FastHashMap::with_capacity_and_hasher(type_map.len(), Default::default());

    for (term, ty) in type_map {
        ty_to_interval(term, ty, &mut term_map)?;
    }
    Ok(term_map)
}

/// Each term maintains a set of dependencies
/// These are terms that are used by said term.
///
/// Is computing this even worth it??
struct TermDependency<'term> {
    predicates: FastHashSet<&'term Predicate>,
    terms: FastHashSet<&'term Term>,
}

impl<'term> TermDependency<'term> {
    fn new() -> Self {
        Self {
            predicates: FastHashSet::default(),
            terms: FastHashSet::default(),
        }
    }

    fn insert_predicate(&mut self, pred: &'term Predicate) {
        self.predicates.insert(pred);
    }

    fn insert_term(&mut self, term: &'term Term) {
        self.terms.insert(term);
    }

    fn contains_predicate(&self, pred: &'term Predicate) -> bool {
        self.predicates.contains(pred)
    }

    fn contains_term(&self, term: &'term Term) -> bool {
        self.terms.contains(term)
    }
}

impl std::ops::Index<&Term> for TermDependency<'_> {
    type Output = bool;
    /// Returns whether the term is a dependency.
    fn index(&self, index: &Term) -> &Self::Output {
        if self.terms.contains(index) {
            &true
        } else {
            &false
        }
    }
}

impl std::ops::Index<&Predicate> for TermDependency<'_> {
    type Output = bool;
    /// Returns whether the predicate is a dependency.
    fn index(&self, index: &Predicate) -> &Self::Output {
        if self.predicates.contains(index) {
            &true
        } else {
            &false
        }
    }
}

impl std::ops::Index<&Handle<Predicate>> for TermDependency<'_> {
    type Output = bool;
    /// Returns whether the predicate is a dependency.
    fn index(&self, index: &Handle<Predicate>) -> &Self::Output {
        if self.predicates.contains(index.as_ref()) {
            &true
        } else {
            &false
        }
    }
}

macro_rules! update_predicate(
    (@noret, $self:expr, $term:expr, $interval:expr) => {
        match $self.predicate_map.entry($term.clone()) {
            Entry::Occupied(mut entry) => {
                if entry.get() != &$interval {
                    $self.dirty_predicates.insert($term);
                    entry.insert(entry.get().intersection($interval));
                }
            }
            Entry::Vacant(mut entry) => {
                $self.dirty_predicates.insert($term);
                entry.insert($interval.clone().into());
            }
        }
    };
    ($self:expr, $term:expr, $interval:expr) => {
        match $self.predicate_map.entry($term.clone()) {
            Entry::Occupied(mut entry) => {
                if entry.get() != &$interval {
                    let new = entry.get().intersection($interval);
                    $self.dirty_predicates.insert($term);
                    entry.insert(new);
                    Ok(new)
                } else {
                    Ok($interval)
                }


            }
            Entry::Vacant(mut entry) => {
                let res = Ok($interval);
                $self.dirty_predicates.insert($term);
                entry.insert($interval.into());
                res
            }
        }
    }
);

/// Macro that updates the term map with the interval.
macro_rules! update_term(
    (@bool $self:expr, $term:expr, $interval:expr) => {
        match $self.term_map.entry($term.clone()) {
            Entry::Occupied(mut entry) => {
                let as_kind = IntervalKind::Bool($interval);
                if entry.get() != &as_kind {
                    let new = entry.get().intersection(&$interval.into())?;
                    let res = *new.try_as_bool().unwrap();
                    $self.dirty.insert($term);
                    entry.insert(new);
                    Ok(res)
                } else {
                    Ok($interval)
                }
            }
            Entry::Vacant(mut entry) => {
                entry.insert($interval.into());
                $self.dirty.insert($term);
                Ok($interval)
            }
        }
    };
    ($self:expr, $term:expr, $interval:expr) => {
        match $self.term_map.entry($term.clone()) {
            Entry::Occupied(mut entry) => {
                let new = entry.get().intersection(&$interval.into())?;
                let res = Ok(new.is_empty());
                entry.insert(new);
                $self.dirty.insert($term);
                res
            }
            Entry::Vacant(mut entry) => {
                let res = Ok($interval.is_empty());
                entry.insert($interval.into());
                $self.dirty.insert($term);
                res
            }
        }
    }
);

fn get_subsuming_assumptions_impl(
    predicate: Handle<Predicate>,
    subsumed_map: &mut FastHashMap<Handle<Predicate>, FastHashSet<Handle<Predicate>>>,
) {
    // If we already calculated the subsuming assumptions for this predicate, then we're done.
    if subsumed_map.contains_key(&predicate) {
        return;
    }
    let mut res = FastHashSet::default();

    // We always subsume `true`, as well as `self`.
    res.insert(Predicate::mk_true());

    res.insert(predicate.clone());

    if let Predicate::And(a, b) = predicate.as_ref() {
        get_subsuming_assumptions_impl(a.clone(), subsumed_map);
        get_subsuming_assumptions_impl(b.clone(), subsumed_map);
        // We only check for subsumption from `And` predicates
        get_subsuming_assumptions_impl(a.clone(), subsumed_map);
        get_subsuming_assumptions_impl(b.clone(), subsumed_map);
        // Safety: `get_subsuming_assumptions_impl` always inserts the predicate into the map, regardless.
        let other = unsafe { subsumed_map.get(a).unwrap_unchecked() };
        res.extend(unsafe { subsumed_map.get(a).unwrap_unchecked() }.clone());
        res.extend(unsafe { subsumed_map.get(b).unwrap_unchecked() }.clone());
    } else if predicate.as_ref().is_or() {
        // If this is an `or` predicate, then we insert ourselves into each of our `or` children's subsumed map.
        for child in predicate.get_children_set_handles() {
            get_subsuming_assumptions_impl(child.clone(), subsumed_map);
            unsafe { subsumed_map.get_mut(&child).unwrap_unchecked() }.insert(predicate.clone());
        }
    }

    subsumed_map.insert(predicate, res);
}

/// For each predicate in the set of keys, get the assumptions that it subsumes.
/// This maps each predicate to the predicates that must be true if it is true.
pub(crate) fn get_subsuming_assumptions<'a, T, K>(
    keys: &K, assumptions: &T,
    subsumed_map: &mut FastHashMap<Handle<Predicate>, FastHashSet<Handle<Predicate>>>,
) -> FastHashMap<Handle<Predicate>, FastHashSet<&'a Assumption>>
where
    K: Iterator<Item = Handle<Predicate>> + Clone,
    T: Iterator<Item = &'a Assumption> + Clone,
{
    // Have to clone because we want to use the iterator again a second time.
    for predicate in keys.clone() {
        get_subsuming_assumptions_impl(predicate, subsumed_map);
    }

    // Keys should be each predicate, values should be each assumption.
    // So, given a predicate, we find all assumptions that are subsumed by it..?
    let mut assumption_map: FastHashMap<Handle<Predicate>, FastHashSet<&'a Assumption>> =
        FastHashMap::default();

    // Now, we go through every predicate,
    // and for each predicate, we go through the list of assumptions.
    // If that assumption is subsumed by the predicate, then we add it.
    //
    // This loop can be very expensive in practice.
    for predicate in keys.clone() {
        use std::collections::hash_map::Entry;
        assumption_map.entry(predicate).or_insert_with_key(|k| {
            // If we have a guard, we must exist in the subsumed map.
            let subsumed = unsafe { subsumed_map.get(k).unwrap_unchecked() };
            assumptions
                .clone()
                .filter(|c| {
                    c.get_guard()
                        .is_some_and(|ref inner| subsumed.contains(inner))
                })
                .collect()
        });
    }

    assumption_map
}

#[allow(clippy::inline_always)] // It is OKAY to always inline this function, as its inline will just be a singular function call.
impl From<Literal> for IntervalKind {
    #[inline(always)]
    fn from(lit: Literal) -> Self {
        lit.as_interval()
    }
}

impl Literal {
    /// Convert the literal into an interval.
    pub fn as_interval(&self) -> IntervalKind {
        match *self {
            Literal::U32(val) => IntervalKind::U32(U32Interval::new_unit(val)),
            Literal::I32(val) => IntervalKind::I32(I32Interval::new_unit(val)),
            Literal::I64(val) => IntervalKind::I64(I64Interval::new_unit(val)),
            Literal::U64(val) => IntervalKind::U64(U64Interval::new_unit(val)),
            // Intervals that aren't supported are represented as an untyped `Top`.
            // Abstract types shouldn't ever make it here anyway..
            _ => IntervalKind::Top,
        }
    }
}

/// Convert the provided assumption into an interval.
///
/// `Dirty_units` is a flag that indicates whether or not we should recompute intervals
/// for terms that are already in the term map.
///
pub fn interval_from_assumption(
    assumption: &crate::Assumption, assumption_set: &FastHashSet<crate::Assumption>,
    term_map: &mut FastHashMap<Term, IntervalKind>, dirty_units: &mut bool,
) -> IntervalKind {
    let new_term_map: FastHashMap<Term, IntervalKind> = Default::default();
    let lhs = assumption.get_lhs();

    // If dirty units is false, then we don't recompute.
    if !*dirty_units {
        if let Some(interval) = term_map.get(lhs) {
            return interval.clone();
        }
    }

    if lhs.is_var() {
        *dirty_units = true;
    }

    // Okay, so we need to update the unit var with the rhs.

    // Now, we have to refine the lhs by evaluating the `rhs`

    todo!()
}

pub fn initialize_array_lengths(
    type_map: &FastHashMap<Term, Handle<AbcType>>,
) -> FastHashMap<Term, IntervalKind> {
    let mut array_length_map = FastHashMap::default();
    #[allow(clippy::explicit_iter_loop)] // clippy is incorrect here.
    for (term, ty) in type_map.iter() {
        // If this is a sized array, then we know the array length from it.
        match ty.as_ref() {
            AbcType::SizedArray { size, .. } => {
                let array_length_term = Term::make_array_length(term);
                array_length_map.insert(
                    term.clone(),
                    IntervalKind::U32(U32Interval::new_unit(size.get())),
                );
            }
            AbcType::Struct { members } => {
                for StructField { name, ty } in members {
                    if let AbcType::SizedArray { size, .. } = ty.as_ref() {
                        let array_length_term = Term::make_array_length(term);
                        array_length_map.insert(
                            array_length_term,
                            IntervalKind::U32(U32Interval::new_unit(size.get())),
                        );
                    }
                }
            }

            _ => (),
        }
    }

    array_length_map
}

impl Assumption {
    fn to_predicate(&self) -> Result<Handle<Predicate>, SolverError> {
        Ok(match self {
            Self::Assign { lhs, rhs, .. } => Predicate::new_comparison(CmpOp::Eq, lhs, rhs).into(),
            Self::Inequality { lhs, lower, upper } => match (lower, upper) {
                (None, None) => return Ok(Predicate::mk_true()),
                (Some((lower, inclusive)), None) => {
                    let op = if *inclusive { CmpOp::Geq } else { CmpOp::Gt };
                    Predicate::new_comparison(op, lhs, lower).into()
                }
                (None, Some((upper, inclusive))) => {
                    let op = if *inclusive { CmpOp::Leq } else { CmpOp::Lt };
                    Predicate::new_comparison(op, lhs, upper).into()
                }
                (Some((lower, left_inclusive)), Some((upper, right_inclusive))) => {
                    let op = if *left_inclusive {
                        CmpOp::Geq
                    } else {
                        CmpOp::Gt
                    };
                    let op2 = if *right_inclusive {
                        CmpOp::Leq
                    } else {
                        CmpOp::Lt
                    };
                    let lower = Predicate::new_comparison(op, lhs, lower);
                    let upper = Predicate::new_comparison(op2, lhs, upper);
                    Predicate::new_and(lower, upper)
                }
            },
        })
    }
}

/// Return a resolver that has refined all terms in the term map based on the guard,
/// and the assumptions that are subsumed by the guard.
fn resolve_from_assumption<'a>(
    guard: &Handle<Predicate>, core_resolver: &Resolver<'a>,
    assumption_map: &'a FastHashMap<Handle<Predicate>, FastHashSet<&Assumption>>,
    resolver_map: &mut FastHashMap<Handle<Predicate>, Resolver<'a>>,
) -> Result<Resolver<'a>, SolverError> {
    if let Some(resolver) = resolver_map.get(guard) {
        return Ok(resolver.clone());
    }
    /// Resolution logic for guards that subsume no other assumptions.
    /// Stuffed in a macro to avoid code duplication.
    macro_rules! no_subsuming_impl {
        ($guard:expr) => {{
            let mut new_resolver = core_resolver.clone();
            new_resolver.set_recompute(true);
            let result = new_resolver.refine_predicate(guard.as_ref())?;
            if !result.contains(BoolInterval::True) {
                return Err(SolverError::DeadCode);
            }
            new_resolver
        }};
    }
    let resolver = match assumption_map.get(guard) {
        // This subsumes no intervals, just resolve the guard.
        None => no_subsuming_impl!(guard),
        Some(assumption_set) if assumption_set.is_empty() => no_subsuming_impl!(guard),
        Some(assumption_set) => {
            let mut assumptions = assumption_set.iter();
            // Safety: We are guarded by the `is_empty` check above.
            let first = unsafe { assumptions.next().unwrap_unchecked() }.to_predicate()?;
            let mut new_resolver =
                resolve_from_assumption(&first, core_resolver, assumption_map, resolver_map)?
                    .clone();
            for assumption in assumptions {
                let existing_resolver = resolve_from_assumption(
                    &assumption.to_predicate()?,
                    core_resolver,
                    assumption_map,
                    resolver_map,
                )?;
                if !new_resolver.intersect(&existing_resolver) {
                    return Err(SolverError::DeadCode);
                }
            }

            new_resolver.set_recompute(true);
            new_resolver.refine_predicate(guard.as_ref())?;
            new_resolver
        }
    };

    Ok(resolver_map
        .entry(guard.clone())
        .or_insert(resolver)
        .clone())
}

/// Return an iterator over every guard predicate in the module and target.
///
/// These are the guards for each `assumption` and `constraint`.
fn get_all_guard_predicates<'a>(
    module: &'a ConstraintModule, target: &'a Handle<crate::Summary>,
) -> impl Iterator<Item = Handle<Predicate>> + 'a + Clone {
    module
        .global_constraints()
        .iter()
        .filter_map(|c| c.0.get_guard())
        .chain(
            module
                .global_assumptions()
                .values()
                .chain(target.assumptions.values())
                .filter_map(crate::Assumption::get_guard),
        )
}

/// Return an iterator over every assumption in the module and target.
#[inline]
fn get_all_assumptions<'a>(
    module: &'a ConstraintModule, target: &'a Handle<crate::Summary>,
) -> impl Iterator<Item = &'a Assumption> + 'a + Clone {
    module
        .global_assumptions()
        .values()
        .chain(target.assumptions.values())
}

/// Return an iterator over every constraint in the module and target.
#[inline]
fn get_all_constraints<'a>(
    module: &'a ConstraintModule, target: &'a Handle<crate::Summary>,
) -> impl Iterator<Item = &'a Constraint> + 'a + Clone {
    module
        .global_constraints()
        .iter()
        .map(|x| &x.0)
        .chain(target.constraints.iter().map(|x| &x.0))
}

fn refine_unguarded_assumptions<'a>(
    module: &'a ConstraintModule, target: &'a Handle<crate::Summary>,
    core_resolver: &mut Resolver<'a>,
) -> Result<(), SolverError> {
    for assumption in module
        .global_assumptions
        .values()
        .chain(target.assumptions.values())
        .filter(|x| x.get_guard().is_none())
    {
        // I just need a container to put these predicates in.

        let result = core_resolver.refine_predicate(assumption.to_predicate()?.as_ref())?;
        if result.is_empty() {
            return Err(SolverError::DeadCode);
        }
    }

    Ok(())
}

/// For every single term in the type map, mark the known sizes of the lengths of statically sized arrays.
fn mk_arraylen_types(
    type_map: &FastHashMap<Term, Handle<AbcType>>, term_map: &mut FastHashMap<Term, IntervalKind>,
) {
    for (term, ty) in type_map {
        if let AbcType::SizedArray { size, ty } = ty.as_ref() {
            let array_length_term = Term::make_array_length(term);
            term_map.insert(
                array_length_term,
                IntervalKind::U32(U32Interval::new_concrete(0, size.get())),
            );
        }
    }
}

/// Enumerate the constraints in the map, yielding the index of the constraint (in the form of `(is_target, idx)`) and the constraint itself.
#[allow(clippy::cast_possible_truncation)]
#[rustfmt::skip]  // rustfmt makes this look terrible.
pub fn enumerate_constraints<'a>(
    module: &'a ConstraintModule, target: &'a Handle<crate::Summary>, fn_idx: u32,
) -> impl Iterator<Item = &'a (Constraint, u32)> + 'a {

    module
        .global_constraints()
        .iter()       
        .chain(
            target
                .constraints
                .iter()
        )
}
/// Translates the constraints in the module, for the given key.
/// The key corresponds to an index in the constraint's `summaries` field.
///
/// # Errors
/// If the index does not exist in the summaries, then an
/// `InvalidSummary` is returned.
///
/// Also propagates errors that occur during the resolution of the constraints.
pub(crate) fn check_constraints(
    module: &ConstraintModule, idx: u32,
) -> Result<FastHashMap<u32, Vec<IntervalKind>>, SolverError> {
    // The target is the function that we are trying to prove bounds checks for.
    let target = module
        .summaries
        .get(idx as usize)
        .ok_or(SolverError::InvalidSummary)?;

    let mut term_map = initialize_intervals(&module.type_map)?;

    mk_arraylen_types(&module.type_map, &mut term_map);

    let all_constraints = get_all_constraints(module, target);
    let all_assumptions = get_all_assumptions(module, target);
    let all_guards = get_all_guard_predicates(module, target);

    let assumption_map =
        get_subsuming_assumptions(&all_guards, &all_assumptions, &mut FastHashMap::default());

    // Get the base assumptions, those without guards that are always true, and refine the intervals from them.

    // Go through the type map for types that are arrays, and add them into the array length map...

    // Now, for every single constraint, we will make a new resolver.

    let mut new_predicates = FastHashMap::<Term, Predicate>::default();
    let mut core_resolver = Resolver::new(
        Cow::Borrowed(&term_map),
        Cow::Owned(Default::default()),
        &module.type_map,
    );

    refine_unguarded_assumptions(module, target, &mut core_resolver)?;

    // Now, we will go through the assumptions with no guards and refine their intervals.

    // Each predicate gets its own resolver.
    let mut predicate_to_resolver: FastHashMap<Handle<Predicate>, Resolver> =
        FastHashMap::default();
    let mut results = FastHashMap::<u32, Vec<IntervalKind>>::default();

    for (constraint, idx) in enumerate_constraints(module, target, idx) {
        log::trace!("Checking constraint {} (id: {})", constraint, idx);
        if let Some(guard) = constraint.get_guard_ref() {
            let constraint_resolution = if let Some(resolver) = predicate_to_resolver.get(guard) {
                // If we already fully resolved the guard, just check the constraint
                resolver.check_constraint(constraint)?
            } else {
                let resolver = resolve_from_assumption(
                    guard,
                    &core_resolver,
                    &assumption_map,
                    &mut predicate_to_resolver,
                );
                match resolver {
                    Ok(resolver) => {
                        for (term, interval) in resolver.term_map.iter() {
                            log::trace!("Resolution for {term}: {interval}");
                        }
                        resolver.check_constraint(constraint)?
                    }
                    Err(SolverError::DeadCode) => BoolInterval::Empty,
                    Err(_) => BoolInterval::Unknown,
                }
            };

            log::trace!("Resolved constraint to {:}", constraint_resolution);
            results
                .entry(*idx)
                .or_default()
                .push(IntervalKind::Bool(constraint_resolution));
        } else {
            results.entry(*idx).or_default().push(IntervalKind::Bool(
                core_resolver.check_constraint(constraint)?,
            ));
        }
    }

    Ok(results)
}

#[allow(clippy::ref_option)]
impl Literal {
    /// Convert the literal into an `IntervalKind` of `[self, upper]`
    fn interval_from_optional_bounds(
        lower: &Option<(Self, bool)>, upper: &Option<(Self, bool)>,
    ) -> IntervalKind {
        let Some((low, lower_inclusive)) = lower else {
            return match upper {
                None => IntervalKind::Top,
                Some((upper, true)) => upper.as_interval().as_le_interval(),
                Some((upper, false)) => upper.as_interval().as_lt_interval(),
            };
        };

        let Some((upper, upper_inclusive)) = upper else {
            unreachable!("We should have already returned if `upper` is `None`");
        };

        macro_rules! with_offset {
            ($x:ident, $y:ident, $kind:ty) => {
                <$kind>::new_concrete(
                    if *lower_inclusive {
                        $x.saturating_add(1)
                    } else {
                        $x
                    },
                    if *upper_inclusive {
                        $y.saturating_sub(1)
                    } else {
                        $y
                    },
                )
                .into()
            };
        };

        match (*low, *upper) {
            (Literal::U32(a), Literal::U32(b)) => with_offset!(a, b, U32Interval),
            (Literal::I32(a), Literal::I32(b)) => with_offset!(a, b, I32Interval),
            (Literal::U64(a), Literal::U64(b)) => with_offset!(a, b, U64Interval),
            (Literal::I64(a), Literal::I64(b)) => with_offset!(a, b, I64Interval),
            _ => IntervalKind::Top,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports)]
    use super::*;
    use rstest::{fixture, rstest};

    /// Test that `initialize_intervals` is properly handled.
    ///
    /// We test the following things
    ///
    /// - If a term is a scalar, then it is converted to an interval.
    #[rstest]
    fn test_initialize_intervals() {
        let mut test_module = ConstraintModule::default();

        // Make a single type for a scalar.
        let scalar_ty = AbcType::Scalar(AbcScalar::Uint(4));

        // make a term for the scalar, and put it in the type map.

        let term = Term::new_var("x");

        test_module
            .type_map
            .insert(term.clone(), Handle::new(scalar_ty));

        let result = initialize_intervals(&test_module.type_map).unwrap();

        assert_eq!(
            result.get(&term).unwrap(),
            &IntervalKind::U32(U32Interval::TOP)
        );
    }
}

#[cfg(test)]
mod test_initialize_intervals {
    #![allow(unused_imports)]
    use std::num::NonZeroU32;

    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn u32_handle() -> Handle<AbcType> {
        AbcType::mk_u32()
    }

    #[fixture]
    fn i32_handle() -> Handle<AbcType> {
        AbcType::mk_i32()
    }

    /// Test that `initialize_intervals` is properly handled.
    ///
    /// We test the following things
    ///
    /// - If a term is a scalar, then it is converted to an interval.
    #[rstest]
    fn test_initialize_intervals(u32_handle: Handle<AbcType>) {
        let mut test_module = ConstraintModule::default();

        // make a term for the scalar, and put it in the type map.
        let term = Term::new_var("x");

        test_module
            .type_map
            .insert(term.clone(), u32_handle.clone());

        let result = initialize_intervals(&test_module.type_map).unwrap();

        assert_eq!(
            result.get(&term).unwrap(),
            &IntervalKind::U32(U32Interval::TOP)
        );
    }

    /// Test that `initialize_intervals` handles sized arrays correctly.
    #[rstest]
    fn test_initialize_intervals_sized_array(u32_handle: Handle<AbcType>) {
        let mut test_module = ConstraintModule::default();

        let array_ty = Handle::new(AbcType::SizedArray {
            size: unsafe { NonZeroU32::new_unchecked(10) },
            ty: u32_handle,
        });

        // make a term for the array, and put it in the type map.
        let term = Term::new_var("arr");

        test_module.type_map.insert(term.clone(), array_ty);

        let result = initialize_intervals(&test_module.type_map).unwrap();

        let array_length_term = Term::make_array_length(&term);
        let array_length_term2 = Term::make_array_length(&term);
        assert_eq!(array_length_term, array_length_term2);
        println!("{result:?}");

        // assert_eq!(
        //     result.get(&array_length_term).unwrap(),
        //     &IntervalKind::U32(U32Interval::new_concrete(0, 10))
        // )
    }

    /// Test that `initialize_intervals` handles dynamic arrays correctly.
    ///
    /// When encountering a dynamic array, we should mark
    /// the type of accesses to the array as `top`.
    #[rstest]
    fn test_initialize_intervals_dynamic_array() {
        let mut test_module = ConstraintModule::default();

        // Make a type for a dynamic array.
        let array_ty = AbcType::DynamicArray {
            ty: AbcType::mk_u32(),
        };

        // make a term for the array, and put it in the type map.
        let term = Term::new_var("arr");

        test_module
            .type_map
            .insert(term.clone(), Handle::new(array_ty));

        let result = initialize_intervals(&test_module.type_map).unwrap();

        let array_length_term = Term::make_array_length(&term);

        assert_eq!(
            result.get(&array_length_term).unwrap(),
            &IntervalKind::U32(U32Interval::TOP)
        );
    }

    /// Test that `initialize_intervals` handles structs correctly.
    #[rstest]
    fn test_initialize_intervals_struct(u32_handle: Handle<AbcType>, i32_handle: Handle<AbcType>) {
        let mut test_module = ConstraintModule::default();

        // Make a type for a struct.
        let struct_ty = AbcType::Struct {
            members: vec![
                StructField {
                    name: "field1".to_string(),
                    ty: AbcType::mk_u32(),
                },
                StructField {
                    name: "field2".to_string(),
                    ty: AbcType::mk_i32(),
                },
            ],
        };

        // make a term for the struct, and put it in the type map.
        let term = Term::new_var("s");

        test_module
            .type_map
            .insert(term.clone(), Handle::new(struct_ty));

        let result = initialize_intervals(&test_module.type_map).unwrap();

        let field1_term =
            Term::new_struct_access(&term, "field1".to_string(), u32_handle.clone(), 0);
        let field2_term =
            Term::new_struct_access(&term, "field2".to_string(), i32_handle.clone(), 1);

        assert_eq!(
            result.get(&field1_term).unwrap(),
            &IntervalKind::U32(U32Interval::TOP)
        );

        assert_eq!(
            result.get(&field2_term).unwrap(),
            &IntervalKind::I32(I32Interval::TOP)
        );
    }
}

#[cfg(test)]
mod test_refine {
    #![allow(unused_imports)]
    use std::num::NonZeroU32;

    use super::*;
    use rstest::{fixture, rstest};

    #[fixture]
    fn u32_handle() -> Handle<AbcType> {
        AbcType::mk_u32()
    }

    #[fixture]
    fn i32_handle() -> Handle<AbcType> {
        AbcType::mk_i32()
    }

    #[rstest]
    fn test_full(u32_handle: Handle<AbcType>) {
        let mut test_module = ConstraintModule::default();

        let term_x = Term::new_var("x");
        let term_y = Term::new_var("y");

        // So what we will do is define an array access. within an `if` loop that says if `i < 5`,
        // then we index into some array whose length we have defined to be 10.

        // And in here, we can successfully remove the check

        test_module.type_map.insert(
            term_x.clone(),
            Handle::new(AbcType::SizedArray {
                ty: u32_handle.clone(),
                size: unsafe { NonZeroU32::new_unchecked(10u32) },
            }),
        );

        let less_than_5 =
            Predicate::new_comparison(crate::CmpOp::Lt, &term_y, &Term::Literal(Literal::U32(5)));
        test_module.type_map.insert(term_y.clone(), u32_handle);

        test_module.global_constraints.push((
            Constraint::Cmp {
                guard: Some(Handle::new(less_than_5)),
                lhs: term_y,
                rhs: Term::Literal(Literal::U32(10)),
                op: crate::CmpOp::Lt,
            },
            0,
        ));

        // refine_intervals_with_assumptions(term_map, assumptions);

        // translate(&test_module).unwrap();

        // Now we do it
    }
}
