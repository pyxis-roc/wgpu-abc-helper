// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT
/*!
Interval analysis for solving the array bounds checking problem.

We expect constraints in the form of \[a, b\] where a and b are integers.
OR in the form of \[start:end:update\]

Given this, we identify the range that each value can hold for any constraint
Predicates filter the possible domain that variables can hold for a statement.
*/
#![allow(clippy::module_name_repetitions)]
pub(crate) mod translator;

use serde::Deserialize;
use serde::Serialize;

mod compound;
pub(super) use compound::CompoundInterval;

use core::ops::{Range, RangeInclusive};
use std::ops::{Bound, RangeBounds};

pub mod ops;
use ops::{Intersect, IntersectAssign, IntervalMax, IntervalMin, Union};

pub use translator::SolverError;

use crate::FastHashSet;

/// A supertrait allowing the type to be used as an element of `Interval`
pub trait IntervalBoundary:
    std::fmt::Debug
    + num::Integer
    + Copy
    + num::Bounded
    + std::hash::Hash
    + num::traits::SaturatingAdd
    + num::traits::SaturatingSub
    + num::traits::SaturatingMul
    + num::traits::WrappingAdd
    + num::traits::WrappingMul
    + num::traits::WrappingSub
    + num::traits::CheckedAdd
    + num::traits::CheckedMul
    + num::traits::CheckedSub
{
}

/// This is a blanket implementation of `IntervalBoundary` for any type that implements the necessary traits.
///
/// The trait has no functionality, sort of like how `Eq` has no functionality.
impl<T> IntervalBoundary for T where
    T: std::fmt::Debug
        + num::Integer
        + Copy
        + num::Bounded
        + std::hash::Hash
        + num::traits::SaturatingAdd
        + num::traits::SaturatingSub
        + num::traits::SaturatingMul
        + num::traits::WrappingAdd
        + num::traits::WrappingMul
        + num::traits::WrappingSub
        + num::traits::CheckedAdd
        + num::traits::CheckedMul
        + num::traits::CheckedSub
{
}

impl crate::Literal {
    #[inline]
    #[must_use]
    pub const fn is_min(&self) -> bool {
        match *self {
            Self::U32(v) => v == u32::MIN,
            Self::I32(v) => v == i32::MIN,
            Self::U64(v) => v == u64::MIN,
            Self::I64(v) | Self::AbstractInt(v) => v == i64::MIN,
            // We don't check min for float types...
            _ => false,
        }
    }

    #[inline]
    #[must_use]
    pub const fn is_max(&self) -> bool {
        match *self {
            Self::U32(v) => v == u32::MAX,
            Self::I32(v) => v == i32::MAX,
            Self::U64(v) => v == u64::MAX,
            Self::I64(v) | Self::AbstractInt(v) => v == i64::MAX,
            // We don't check max for float types...
            _ => false,
        }
    }
}

/// A trait indicating that the type operates like an interval.
pub trait Interval {
    type Inner: IntervalBoundary;
    /// Return whether the interval contains any values.
    fn is_empty_interval(&self) -> bool;
    /// If this interval is unit over a single value, return `Some(value)`.  Otherwise, returns `None`
    fn as_literal(&self) -> Option<Self::Inner>;

    /// Return a tuple of the form (value, `is_empty`).
    ///
    /// If `is_empty` is true, then `value` is meaningless.
    /// Otherwise, this represents the lowest value in the interval.
    fn get_lower(&self) -> (Self::Inner, bool);

    /// Return a tuple of the form (value, `is_empty`).
    ///
    /// If `is_empty` is true, then `value` is meaningless.
    /// Otherwise, this represents the greatest value in the interval.
    fn get_upper(&self) -> (Self::Inner, bool);

    /// Return whether the other interval is a subset of this interval.
    fn subsumes(&self, other: &Self) -> bool;

    /// Return whether this interval is the widest interval possible
    fn is_top(&self) -> bool;

    /// Return whether the interval contains `value`
    fn has_value(&self, value: Self::Inner) -> bool;

    /// Return the `top` interval for this type.
    fn top() -> Self;

    /// Create a new interval from the two values.
    fn from_literals(lower: Self::Inner, upper: Self::Inner) -> Self;
}

impl<T: IntervalBoundary> RangeBounds<T> for BasicInterval<T> {
    fn contains<U>(&self, item: &U) -> bool
    where
        T: PartialOrd<U>,
        U: ?Sized + PartialOrd<T>,
    {
        self.lower <= *item && *item <= self.upper
    }

    fn start_bound(&self) -> Bound<&T> {
        Bound::Included(&self.lower)
    }

    fn end_bound(&self) -> Bound<&T> {
        Bound::Included(&self.upper)
    }
}

#[derive(Debug, thiserror::Error, Clone, Serialize, Deserialize)]
pub enum IntervalError {
    #[error("Incompatible types")]
    IncompatibleTypes,
    #[error("{0} is not supported for intervals of type {1}")]
    InvalidOp(&'static str, &'static str),
    #[error("{0} is not supported between {1} and {2}")]
    InvalidBinOp(&'static str, &'static str, &'static str),
}

/// An interval that represents all values between two bounds, inclusive.
///
/// The interval is considered empty if the lower bound is greater than the upper bound.
#[derive(Clone, Copy, Debug)]
pub struct BasicInterval<T: IntervalBoundary> {
    lower: T,
    upper: T,
}
impl<T: IntervalBoundary> BasicInterval<T> {
    pub const fn new(lower: T, upper: T) -> Self {
        Self { lower, upper }
    }

    pub const fn new_unit(value: T) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub fn is_unit(&self) -> bool {
        self.lower == self.upper
    }
}

// We want empty intervals to hash the same, so we hash them as 0.
impl<T: IntervalBoundary> std::hash::Hash for BasicInterval<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // if this is empty, we write it as a 0.
        if self.is_empty_interval() {
            return 0u8.hash(state);
        }
        self.lower.hash(state);
        self.upper.hash(state);
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq for BasicInterval<T> {
    /// Two basic intervals are equal to each other if they are both empty, or if their bounds are identical.
    ///
    /// ```
    /// # use abc_helper::solvers::interval::BasicInterval;
    /// let a = BasicInterval::new(10, 20);
    /// let b = BasicInterval::new(10, 20);
    /// assert_eq!(a, b);
    ///
    /// let a = BasicInterval::new(15, 10); // An empty interval
    /// let b = BasicInterval::new(40, 20); // Also an empty interval,
    /// assert_eq!(a, b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.is_empty_interval() && other.is_empty_interval() {
            other.is_empty_interval()
        } else {
            self.lower == other.lower && self.upper == other.upper
        }
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq<T> for BasicInterval<T> {
    fn eq(&self, other: &T) -> bool {
        self.is_unit() && self.lower == *other
    }
}

impl<T: IntervalBoundary> std::cmp::Eq for BasicInterval<T> {}

impl<T: IntervalBoundary> Ord for BasicInterval<T> {
    /// Compare two intervals.
    ///
    /// The criteria for comparison is the same as the derived implementation (that is, compare `lower` and then `upper`),
    /// with special behavior for `empty` intervals:
    /// - If both intervals are empty, they compare equal.
    /// - Otherwise, `empty` always compares less than other.
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.is_empty_interval() {
            if other.is_empty_interval() {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Less
            }
        } else {
            self.lower
                .cmp(&other.lower)
                .then(self.upper.cmp(&other.upper))
        }
    }
}

impl<T: IntervalBoundary> PartialOrd for BasicInterval<T> {
    /// Compare two intervals.
    ///
    /// The criteria for comparison is the same as the derived implementation (that is, compare `lower` and then `upper`),
    /// with special behavior for `empty` intervals:
    /// - If both intervals are empty, they compare equal.
    /// - Otherwise, `empty` always compares less than other.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: IntervalBoundary> PartialOrd<T> for BasicInterval<T> {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        if self.is_empty_interval() {
            Some(std::cmp::Ordering::Less)
        } else if self.upper.lt(other) {
            Some(Ordering::Less)
        } else if self.lower.gt(other) {
            Some(Ordering::Greater)
        } else if self.is_unit() && self.lower.eq(other) {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
}

impl<T: IntervalBoundary + std::fmt::Display> std::fmt::Display for BasicInterval<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty_interval() {
            write!(f, "[]")
        } else {
            write!(f, "[{} : {}]", self.lower, self.upper)
        }
    }
}
// Implementations of From<Range> for `BasicInterval`. Meant to be as helpful as possible.

impl<T: IntervalBoundary> From<num::iter::Range<T>> for BasicInterval<T> {
    fn from(range: num::iter::Range<T>) -> Self {
        let Bound::Included(&lower) = range.start_bound() else {
            unreachable!("`num::iter::range` should always return `Bound::Included` variant.")
        };
        let Bound::Excluded(&upper) = range.end_bound() else {
            unreachable!("`num::iter::range` should always return `Bound::Excluded` variant.")
        };
        Self {
            lower,
            upper: upper.saturating_sub(&T::one()),
        }
    }
}

impl<T: IntervalBoundary> From<Range<T>> for BasicInterval<T> {
    fn from(range: Range<T>) -> Self {
        Self {
            lower: range.start,
            upper: range.end - T::one(),
        }
    }
}

impl<T: IntervalBoundary> From<RangeInclusive<T>> for BasicInterval<T> {
    fn from(range: RangeInclusive<T>) -> Self {
        Self {
            lower: *range.start(),
            upper: *range.end(),
        }
    }
}

impl<T: IntervalBoundary> From<BasicInterval<T>> for Range<T> {
    fn from(interval: BasicInterval<T>) -> Self {
        interval.lower..interval.upper
    }
}

impl<T: IntervalBoundary> From<BasicInterval<T>> for RangeInclusive<T> {
    fn from(interval: BasicInterval<T>) -> Self {
        interval.lower..=interval.upper
    }
}

impl<T: IntervalBoundary> From<Range<T>> for WrappedInterval<T> {
    fn from(range: Range<T>) -> Self {
        WrappedInterval::Basic(range.into())
    }
}

impl<T: IntervalBoundary> From<RangeInclusive<T>> for WrappedInterval<T> {
    fn from(range: RangeInclusive<T>) -> Self {
        WrappedInterval::Basic(range.into())
    }
}

impl<T: IntervalBoundary> std::default::Default for BasicInterval<T> {
    /// Return the interval of `[min_value, max_value]`
    fn default() -> Self {
        Self {
            lower: num::Bounded::min_value(),
            upper: num::Bounded::max_value(),
        }
    }
}

impl<T: IntervalBoundary> Interval for BasicInterval<T> {
    type Inner = T;
    #[inline]
    fn from_literals(lower: T, upper: T) -> Self {
        Self { lower, upper }
    }

    #[inline]
    fn top() -> Self {
        Self {
            lower: num::Bounded::min_value(),
            upper: num::Bounded::max_value(),
        }
    }

    /// Return whether this interval is the widest interval possible.
    #[inline]
    fn is_top(&self) -> bool {
        self.lower == num::Bounded::min_value() && self.upper == num::Bounded::max_value()
    }

    /// If this interval is unit over a single value, return `Some(value)`.  Otherwise, returns `None`
    #[inline]
    fn as_literal(&self) -> Option<T> {
        if self.lower == self.upper {
            Some(self.lower)
        } else {
            None
        }
    }

    /// Return whether the provdied value lies within the interval
    #[inline]
    fn has_value(&self, value: T) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Return whether the provided interval is contained entirely within `self`
    #[inline]
    fn subsumes(&self, other: &Self) -> bool {
        if self.is_empty_interval() {
            return other.is_empty_interval();
        } else if other.is_empty_interval() {
            return true;
        }
        self.lower <= other.lower && self.upper >= other.upper
    }

    /// Return the lower bound of the interval
    ///
    /// If the interval is empty, then the lower bound is meaningless. In this case, the second value will be `true`.
    #[inline]
    fn get_lower(&self) -> (T, bool) {
        (self.lower, self.is_empty_interval())
    }

    /// Return the upper bound of the interval.
    ///
    /// If the interval is empty, then the upper bound is meaningless. In this case, the second value will be `true`.
    #[inline]
    fn get_upper(&self) -> (T, bool) {
        (self.upper, self.is_empty_interval())
    }

    /// Return whether the interval is empty.
    ///
    /// An interval is empty if the lower bound is greater than the upper bound.
    #[inline]
    fn is_empty_interval(&self) -> bool {
        self.lower > self.upper
    }
}
/// A wrapper around an interval. The interval is either a unit interval or a union of intervals.
#[derive(Clone, Debug, Eq)]
pub enum WrappedInterval<T: IntervalBoundary> {
    Empty,
    Basic(BasicInterval<T>),
    Compound(CompoundInterval<T>),
    Top,
}

impl<T: IntervalBoundary> WrappedInterval<T> {
    pub fn is_unit(&self) -> bool {
        match *self {
            Self::Basic(ref t) => t.is_unit(),
            Self::Empty | Self::Top => false,
            Self::Compound(ref t) => t.is_unit(),
        }
    }
}

impl<T: IntervalBoundary> WrappedInterval<T> {
    pub const TOP: Self = Self::Top;
    pub const EMPTY: Self = Self::Empty;
}

impl<A: IntervalBoundary> FromIterator<BasicInterval<A>> for WrappedInterval<A> {
    fn from_iter<T: IntoIterator<Item = BasicInterval<A>>>(iter: T) -> Self {
        let compound = CompoundInterval::from_iter(iter);
        if compound.is_top() {
            Self::TOP
        } else if compound.is_empty_interval() {
            Self::EMPTY
        } else if compound.len() == 1 {
            Self::Basic(*compound.iter().next().unwrap())
        } else {
            Self::Compound(compound)
        }
    }
}

impl<T: IntervalBoundary + std::fmt::Display> std::fmt::Display for WrappedInterval<T> {
    #[cfg_attr(not(test), cold)] // isn't called frequently outside of tests.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "\u{2205}"),
            Self::Top => write!(f, "[{}, {}]", T::min_value(), T::max_value()),
            Self::Basic(interval) => write!(f, "{interval}"),
            Self::Compound(intervals) => {
                let mut iter = intervals.iter();
                if let Some(first) = iter.next() {
                    write!(f, "{first}")?;
                    for interval in iter {
                        write!(f, " \u{222A} {interval}")?;
                    }
                }
                Ok(())
            }
        }
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq<WrappedInterval<T>> for BasicInterval<T> {
    /// Compare a unit interval to a wrapped interval for equality.
    ///
    /// A [`BasicInterval`] `a` is equal to a [`WrappedInterval`] `b` if any of the following are true:
    /// - `is_empty()` is true for both `a` and `b`
    /// - `is_top()` is true for both a and b
    /// - b is [`WrappedInterval::Unit`] whose contained interval is equal to a
    /// - b is [`WrappedInterval::Union`] with a single interval that is equal to a
    ///
    /// [`BasicInterval`]: self::BasicInterval
    /// [`WrappedInterval`]: self::WrappedInterval
    fn eq(&self, other: &WrappedInterval<T>) -> bool {
        if self.is_empty_interval() {
            other.is_empty_interval()
        } else if self.is_top() {
            other.is_top()
        } else {
            match *other {
                WrappedInterval::Basic(ref interval) => self == interval,
                WrappedInterval::Compound(ref intervals) if intervals.len() == 1 => {
                    intervals.get_inner_set().contains(self)
                }
                _ => false,
            }
        }
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq for WrappedInterval<T> {
    /// Compare two wrapped intervals for equality.
    ///
    /// Two wrapped intervals are equal if they are both empty, both top, or if their contained intervals are equal.
    ///
    /// That is, even if `self` and `other` are different variants, they may still compare equal.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, other) => other.is_empty_interval(),
            (Self::Top, other) => other.is_top(),
            (Self::Basic(a), b) | (b, Self::Basic(a)) => a == b,
            (Self::Compound(a), b) => a == b,
        }
    }
}

impl<T: IntervalBoundary> IntersectAssign for WrappedInterval<T> {
    fn interval_intersect(&mut self, other: &Self) {
        match (self, other) {
            // Do nothing if a is empty, b is top, or a == b.
            (a, b) if b.is_top() || a == b || matches!(a, Self::Empty) => (),
            // If b is empty, then intersection is empty.
            (a, b) if b.is_empty_interval() => {
                *a = Self::Empty;
            }
            // If a is top, then its itersection becomes b.
            (a, b) if a.is_top() => {
                *a = b.clone();
            }
            // If both a and b are unit, then we just intersect them.
            (a @ Self::Basic(_), Self::Basic(ref other_interval)) => {
                let Self::Basic(ref mut interval) = a else {
                    unreachable!()
                };
                interval.interval_intersect(other_interval);
                // If their intersection was empty, then we prefer to set this to WrappedInterval::Empty.
                if interval.is_empty_interval() {
                    *a = Self::Empty;
                }
            }
            (a @ Self::Compound(_), Self::Basic(ref other)) => {
                let Self::Compound(ref mut this) = a else {
                    unreachable!()
                };
                this.interval_intersect(other);
                match this.len() {
                    0 => *a = Self::Empty,
                    1 => {
                        let inner = this.iter().next().unwrap();
                        *a = Self::Basic(*inner);
                    }
                    _ => (),
                }
            }
            (a @ Self::Compound(_), Self::Compound(ref other)) => {
                let Self::Compound(ref mut this) = a else {
                    unreachable!()
                };
                this.interval_intersect(other);
                match this.len() {
                    0 => *a = Self::Empty,
                    1 => {
                        let inner = this.drain().next().unwrap();
                        *a = Self::Basic(inner);
                    }
                    _ => (),
                }
            }
            (a @ Self::Basic(_), Self::Compound(ref other)) => {
                let Self::Basic(interval) = a else {
                    unreachable!()
                };
                let new = other.interval_intersection(interval);
                match new.len() {
                    0 => *a = Self::Empty,
                    1 => *a = Self::Basic(*new.iter().next().unwrap()),
                    _ => *a = Self::Compound(new),
                }
            }
            _ => unreachable!(),
        }
    }
}

impl<T: IntervalBoundary> Intersect for WrappedInterval<T> {
    type Output = Self;

    // Same code as above, just returns a new value instead of modifying in place.
    fn interval_intersection(&self, other: &Self) -> Self {
        macro_rules! match_len {
            ($a: expr, $new:expr) => {
                match $a.len() {
                    0 => Self::Empty,
                    1 => Self::Basic($new.drain().next().unwrap()),
                    _ => Self::Compound($new),
                }
            };
        }
        match (self, other) {
            (a, b) if b.is_top() || a == b || matches!(*a, Self::Empty) => a.clone(),
            (_, b) if b.is_empty_interval() => Self::Empty,
            (a, b) if a.is_top() => b.clone(),
            // If both a and b are unit, then we just intersect them.
            (WrappedInterval::Basic(interval), Self::Basic(other_interval)) => {
                let new = interval.interval_intersection(other_interval);
                // If their intersection was empty, then we prefer to set this to WrappedInterval::Empty.
                if new.is_empty_interval() {
                    Self::Empty
                } else {
                    Self::Basic(new)
                }
            }
            (Self::Compound(this), Self::Basic(other)) => {
                let mut new = this.interval_intersection(other);
                match_len!(this, new)
            }
            (Self::Compound(ref this), Self::Compound(ref other)) => {
                let mut new = this.interval_intersection(other);
                match_len!(this, new)
            }
            (WrappedInterval::Basic(ref this), Self::Compound(ref other)) => {
                let mut new = other.interval_intersection(this);
                match_len!(other, new)
            }
            _ => unreachable!(),
        }
    }
}

impl<T: IntervalBoundary> WrappedInterval<T> {
    /// Constructor to make a new interval from a lower and upper bound.
    pub fn new_concrete(lower: T, upper: T) -> Self {
        if lower > upper {
            Self::Empty
        } else if lower == num::Bounded::min_value() && upper == num::Bounded::max_value() {
            Self::Top
        } else {
            Self::Basic(BasicInterval::new(lower, upper))
        }
    }

    pub fn new_unit(value: T) -> Self {
        Self::Basic(BasicInterval::new_unit(value))
    }
}

impl<T: IntervalBoundary> Interval for WrappedInterval<T> {
    type Inner = T;
    #[inline]
    fn from_literals(lower: T, upper: T) -> Self {
        Self::new_concrete(lower, upper)
    }

    fn top() -> Self {
        Self::Top
    }

    /// Return whether this interval is the widest interval possible.
    fn is_top(&self) -> bool {
        match *self {
            Self::Top => true,
            Self::Basic(ref t) => t.is_top(),
            Self::Compound(ref this) => this.is_top(),
            Self::Empty => false,
        }
    }

    fn as_literal(&self) -> Option<T> {
        match self {
            Self::Basic(interval) => interval.as_literal(),
            Self::Compound(ref intervals) => intervals.as_literal(),
            _ => None,
        }
    }

    fn has_value(&self, value: T) -> bool {
        match *self {
            Self::Empty => false,
            Self::Top => true,
            Self::Basic(ref interval) => interval.has_value(value),
            Self::Compound(ref intervals) => {
                intervals.iter().any(|interval| interval.has_value(value))
            }
        }
    }

    /// Get the lower bound of the interval
    ///
    /// If `self` is an empty interval, return (`T::zero()`, `true`).
    fn get_lower(&self) -> (T, bool) {
        if self.is_empty_interval() {
            return (T::zero(), true);
        }
        match *self {
            Self::Top => (T::min_value(), false),
            Self::Basic(BasicInterval { lower, .. }) => (lower, false),
            Self::Compound(ref intervals) => (
                intervals
                    .iter()
                    .fold(num::Bounded::max_value(), |a, &b| a.min(b.lower)),
                false,
            ),
            Self::Empty => unreachable!("`self.is_empty()` should return true for Empty variant"),
        }
    }

    fn get_upper(&self) -> (T, bool) {
        if self.is_empty_interval() {
            return (T::zero(), true);
        }
        match *self {
            Self::Top => (T::max_value(), false),
            Self::Basic(BasicInterval { lower, .. }) => (lower, false),
            Self::Compound(ref intervals) => (
                intervals
                    .iter()
                    .fold(num::Bounded::min_value(), |a, &b| a.max(b.lower)),
                false,
            ),
            Self::Empty => unreachable!("`self.is_empty()` should return true for Empty variant"),
        }
    }

    fn is_empty_interval(&self) -> bool {
        match *self {
            Self::Top => false,
            Self::Basic(ref t) => t.is_empty_interval(),
            Self::Compound(ref this) => this.is_empty_interval(),
            Self::Empty => true,
        }
    }

    fn subsumes(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Empty, _) | (_, Self::Top) => false,
            (_, Self::Empty) | (Self::Top, _) => true,
            (Self::Basic(interval_a), Self::Basic(interval_b)) => interval_a.subsumes(interval_b),
            (Self::Compound(intervals_a), Self::Compound(intervals_b)) => {
                intervals_a.iter().all(|interval_a| {
                    intervals_b
                        .iter()
                        .any(|interval_b| interval_a.subsumes(interval_b))
                })
            }
            (Self::Compound(intervals), Self::Basic(interval)) => {
                intervals.iter().all(|i| i.subsumes(interval))
            }
            _ => false,
        }
    }
}

pub type U32Interval = WrappedInterval<u32>;
pub type I32Interval = WrappedInterval<i32>;
pub type I64Interval = WrappedInterval<i64>;
pub type U64Interval = WrappedInterval<u64>;

impl<T: IntervalBoundary + std::fmt::Display> WrappedInterval<T> {
    pub fn pretty_print(&self) -> impl std::fmt::Display {
        match *self {
            Self::Empty => "empty".to_string(),
            Self::Top => "unknown".to_string(),
            _ => format!("[{}, {}]", self.get_lower().0, self.get_upper().0),
        }
    }
}

impl<T: IntervalBoundary> From<BasicInterval<T>> for WrappedInterval<T> {
    fn from(unit: BasicInterval<T>) -> Self {
        if unit.is_empty_interval() {
            Self::EMPTY
        } else if unit.is_top() {
            Self::TOP
        } else {
            Self::Basic(unit)
        }
    }
}

impl<T: IntervalBoundary> From<&BasicInterval<T>> for WrappedInterval<T> {
    fn from(unit: &BasicInterval<T>) -> Self {
        if unit.is_empty_interval() {
            WrappedInterval::Empty
        } else if unit.is_top() {
            WrappedInterval::Top
        } else {
            WrappedInterval::Basic(*unit)
        }
    }
}

use bitflags::bitflags;

bitflags! {
    /// A bool interval is a bitflag of 2 bits. bit 0 represents `true`, bit 1 represents `false`
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
    pub struct BoolInterval: u8 {
        /// The empty interval.
        const Empty = 0;
        /// The interval that contains only `true`.
        const True = 1u8;
        /// The interval that contains only `false`.
        const False = 2u8;
        /// The interval that contains both `true` and `false`.
        const Unknown = BoolInterval::True.bits() | BoolInterval::False.bits();
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, strum_macros::Display)]
pub enum SolverResult {
    Yes,
    Maybe,
    No,
}

impl From<BoolInterval> for SolverResult {
    #[inline]
    fn from(interval: BoolInterval) -> Self {
        if interval == BoolInterval::True {
            Self::Yes
        } else {
            Self::No
        }
    }
}

impl BoolInterval {
    pub const TOP: Self = Self::Unknown;
    /// Return whether the two intervals are unit and equivalent. If either is empty, this returns empty.
    /// # Examples
    /// ```
    /// # use abc_helper::solvers::interval::BoolInterval;
    /// let a = BoolInterval::True;
    /// let b = BoolInterval::True;
    /// assert_eq!(a.interval_eq(b), BoolInterval::True);
    ///
    /// let c = BoolInterval::False;
    /// assert_eq!(a.interval_eq(c), BoolInterval::False);
    /// assert_eq!(a.interval_eq(BoolInterval::Unknown), BoolInterval::Unknown);
    ///
    /// assert_eq!(a.interval_eq(BoolInterval::Empty), BoolInterval::Empty);
    /// ```
    #[must_use]
    pub fn interval_eq(self, other: Self) -> Self {
        match (self, other) {
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::True, Self::True) | (Self::False, Self::False) => Self::True,
            (Self::True, Self::False) | (Self::False, Self::True) => Self::False,
            _ => Self::Unknown,
        }
    }

    #[must_use]
    pub fn interval_neq(self, other: Self) -> Self {
        match (self, other) {
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::True, Self::True) | (Self::False, Self::False) => Self::False,
            (Self::True, Self::False) | (Self::False, Self::True) => Self::True,
            _ => Self::Unknown,
        }
    }

    pub fn pretty_print(&self) -> impl std::fmt::Display {
        match *self {
            Self::Empty => "empty",
            Self::True => "true",
            Self::False => "false",
            _ => "unknown",
        }
    }
}
impl std::fmt::Display for BoolInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Empty => write!(f, "[]"),
            Self::True => write!(f, "[true]"),
            Self::False => write!(f, "[false]"),
            Self::Unknown => write!(f, "[true, false]"),
            _ => unreachable!(),
        }
    }
}

impl From<bool> for BoolInterval {
    #[inline]
    fn from(value: bool) -> Self {
        if value {
            Self::True
        } else {
            Self::False
        }
    }
}

impl BoolInterval {
    /// Synonym for `self == BoolInterval::Unknown`
    #[inline]
    pub fn is_top(&self) -> bool {
        *self == BoolInterval::Unknown
    }

    /// Swaps the `true` and `false` flag values.
    ///
    /// This is very different from `!self`, which would convert `Unknown` to `Empty` and vice versa.
    #[must_use]
    pub fn logical_not(self) -> Self {
        match self.intersection(Self::Unknown) {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Empty => Self::Empty,
            Self::Unknown => Self::Unknown,
            _ => unreachable!("Unknown flags should not be set after `& Self::all()`"),
        }
    }
}

pub trait IntervalComparison {
    /// Return a new interval of the form [min(self, other), max(self, other)]
    #[must_use]
    fn max(&self, other: Self) -> Self;
    #[must_use]
    fn min(&self, other: Self) -> Self;
}

#[cfg(test)]
mod tests {
    use std::hash::Hasher;

    use super::*;
    use rstest::rstest;

    /// Tests interval union behavior for various types.
    #[rstest]
    fn test_union_with_top() {
        // Unionin
        let interval_a = WrappedInterval::Top;

        for interval in [
            WrappedInterval::Empty,
            WrappedInterval::Top,
            // Union with an interval that is a unit interval.
            WrappedInterval::Compound(CompoundInterval::from_iter_unchecked([
                BasicInterval::from(10..20),
                BasicInterval::from(22..30),
            ])),
            // Union interval that is top
            WrappedInterval::Compound(CompoundInterval::top()),
            WrappedInterval::Basic(BasicInterval::from(10..20)),
        ] {
            assert_eq!(interval_a.interval_union(&interval), WrappedInterval::Top);
        }
    }

    /// Ensure that the intersection of any variant with `top` compares equal with top.
    #[rstest]
    fn test_intersection_with_top() {
        let interval_a = WrappedInterval::Top;

        for interval in [
            WrappedInterval::Empty,
            WrappedInterval::Top,
            // Union with an interval that is a unit interval.
            WrappedInterval::Compound(CompoundInterval::from_iter_unchecked([
                BasicInterval::from(10..20),
                BasicInterval::from(22..30),
            ])),
            WrappedInterval::Compound(CompoundInterval::top()),
            WrappedInterval::Basic(BasicInterval::from(10..20)),
        ] {
            let intersection = interval_a.interval_intersection(&interval);
            assert_eq!(intersection, interval, "{interval_a} \u{222A} {interval}");
        }
    }

    /// Ensure two empty UnitIntervals hash to the same value.
    #[rstest]
    fn test_hash_empty() {
        use std::hash::Hash;
        macro_rules! get_hash {
            ($interval:expr) => {{
                let mut hasher = rustc_hash::FxHasher::default();
                $interval.hash(&mut hasher);
                hasher.finish()
            }};
        }
        let top_hash = get_hash!(BasicInterval::new(12u32, 10u32));
        let other_hash = get_hash!(BasicInterval::new(15u32, 10u32));
        assert_eq!(top_hash, other_hash);
    }

    /// Ensures that two intervals that are adjacent to each other are unioned correctly.
    #[rstest]
    fn test_union_adjacent() {
        // Adjacent `Wrapped` intervals
        let interval_a = WrappedInterval::Basic(BasicInterval::from(10..20));
        let interval_b = WrappedInterval::Basic(BasicInterval::from(20..25));
        let union_ = interval_a.interval_union(&interval_b);
        assert_eq!(union_, WrappedInterval::Basic(BasicInterval::from(10..25)));

        // Adjacent `Unit` intervals
        let interval_a = BasicInterval::new(10, 20);
        let interval_b = BasicInterval::new(21, 25);
        assert_eq!(
            interval_b.interval_union(&interval_a),
            WrappedInterval::Basic(BasicInterval::new(10, 25))
        );

        // Adjacent `Unit` and `Wrapped` intervals
        let interval_a = BasicInterval::new(10, 20);
        let interval_b = WrappedInterval::Basic(BasicInterval::new(21, 25));
        assert_eq!(
            interval_b.interval_union(&interval_a),
            WrappedInterval::Basic(BasicInterval::new(10, 25))
        );
    }
}
