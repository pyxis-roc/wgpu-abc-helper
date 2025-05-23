//! Traits for operations on intervals.
use crate::solvers::interval::CompoundInterval;

use super::{BasicInterval, BoolInterval, Interval, IntervalBoundary, WrappedInterval};
use std::ops::BitAnd;
/********
 * TRAITS
 ***********/
#[doc = "Implementing this trait for `T` defines `T \u{222A} Rhs`"]
pub trait Union<Rhs = Self>: Interval {
    /// Return a new interval that is the union of `self` and `rhs`.
    /// The new interval will contain all values in `self` and `rhs`.
    ///
    /// [`WrappedInterval::Compound`]: super::WrappedInterval::Compound
    fn interval_union(&self, rhs: &Rhs) -> WrappedInterval<<Self as Interval>::Inner>;
}

/// The `UnionAssign` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait UnionAssign<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    /// Union `self` with `other` and assign the result to `self`.
    fn union_assign(&mut self, rhs: &Rhs);
}

/// The `Intersect` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait Intersect<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    /// Return `self` intersected with `other`
    fn interval_intersection(&self, rhs: &Rhs) -> Self::Output;
}

/// The `IntersectAssign` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait IntersectAssign<Rhs = Self>: Interval {
    /// Intersect `self` with `other` and assign the result to `self`.
    fn interval_intersect(&mut self, other: &Rhs);
}

/// The addition operation for intervals.
pub trait IntervalAdd<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_add(&self, rhs: &Rhs) -> Self::Output;
}

/// The subtraction operator for intervals.
pub trait IntervalSub<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_sub(&self, rhs: &Rhs) -> Self::Output;
}

/// The `Mul` operator for intervals
pub trait IntervalMul<Rhs = Self>: Interval
where
    Rhs: Interval<Inner = <Self as Interval>::Inner>,
{
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_mul(&self, rhs: &Rhs) -> Self::Output;
}

/// The division operator for intervals.
pub trait IntervalDiv<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_div(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalShr<Rhs>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_shr(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalMod<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_mod(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalMin<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_min(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalAbs<'a>: Interval {
    type Output;
    fn interval_abs(&'a self) -> Self::Output;
}

pub trait IntervalCast<Output> {
    /// Cast the interval to the output type,
    /// e.g., like in wgsl's [i32](https://www.w3.org/TR/WGSL/#i32-builtin) builtin.
    fn interval_cast(&self) -> Output;
}

impl<T: Interval + Clone> IntervalCast<Self> for T {
    fn interval_cast(&self) -> Self {
        self.clone()
    }
}

impl IntervalCast<WrappedInterval<u32>> for BasicInterval<i32> {
    /// Return a new interval that represents the range of the interval if its i32 values were cast to u32.
    ///
    /// Empty intervals are converted to empty
    #[allow(clippy::cast_sign_loss)] // Sign loss is intentional
    fn interval_cast(&self) -> WrappedInterval<u32> {
        if self.is_empty_interval() {
            return WrappedInterval::Empty;
        }

        // Top is always the whole domain...
        if self.is_top() {
            return WrappedInterval::Top;
        }

        let lower = self.get_lower().0;
        let upper = self.get_upper().0;

        let mut intervals = Vec::with_capacity(3);

        // If the interval contains 0, then we add it as a unit interval.
        if self.has_value(0) {
            intervals.push(BasicInterval::new_unit(0));
        }

        // If it is only positive or only negative, then [lower, upper] works normally.
        if upper < 0 || lower > 0 {
            // This cast is intentional.

            intervals.push(BasicInterval::from_literals(lower as u32, upper as u32));
        } else {
            // however, if we cross 0, then [lower] may be greater than [upper]
            intervals.push(BasicInterval::from_literals(upper as u32, u32::MAX));

            // Positive wraps...
            intervals.push(BasicInterval::from_literals(1, lower as u32));
        }
        WrappedInterval::from_iter(intervals)
    }
}

impl IntervalCast<WrappedInterval<i32>> for BasicInterval<u32> {
    /// Convert an interval from `u32` to `i32`, handling the wrapping semantics.
    #[allow(clippy::cast_possible_wrap)] // Wrapping is intentional
    fn interval_cast(&self) -> WrappedInterval<i32> {
        if self.is_empty_interval() {
            return WrappedInterval::Empty;
        }
        if self.is_top() {
            return WrappedInterval::Top;
        }

        let lower = self.get_lower().0;
        let upper = self.get_upper().0;

        let cutoff = i32::MAX as u32;

        // Aka, the maximum value of i32.
        if upper <= cutoff || lower > cutoff {
            WrappedInterval::from_literals(lower as i32, upper as i32)
        } else {
            WrappedInterval::from_iter([
                BasicInterval::from_literals(lower as i32, i32::MAX),
                BasicInterval::from_literals(i32::MIN, upper as i32),
            ])
        }
    }
}

/// Auto-implementation of `IntervalCast` to for `CompoundInterval<T>` if `BasicInterval<T>` implements it.
impl<In, Out> IntervalCast<WrappedInterval<Out>> for CompoundInterval<In>
where
    In: IntervalBoundary,
    Out: IntervalBoundary,
    BasicInterval<In>: IntervalCast<WrappedInterval<Out>>,
{
    fn interval_cast(&self) -> WrappedInterval<Out> {
        if self.is_empty_interval() {
            return WrappedInterval::Empty;
        }
        if self.is_top() {
            return WrappedInterval::Top;
        }
        let mut new = CompoundInterval::<Out>::new();
        for interval in self.iter() {
            new.union_with_wrapped(&interval.interval_cast());
        }
        WrappedInterval::<Out>::from(new)
    }
}

/// Auto-implementation of `IntervalCast` for `WrappedInterval<T>` if `BasicInterval<T> and CompoundInterval<T>` both implement it.
impl<T: IntervalBoundary, K: IntervalBoundary> IntervalCast<WrappedInterval<K>>
    for WrappedInterval<T>
where
    BasicInterval<T>: IntervalCast<WrappedInterval<K>>,
    CompoundInterval<T>: IntervalCast<WrappedInterval<K>>,
{
    fn interval_cast(&self) -> WrappedInterval<K> {
        match self {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => WrappedInterval::Top,
            WrappedInterval::Basic(interval) => interval.interval_cast(),
            WrappedInterval::Compound(intervals) => intervals.interval_cast(),
        }
    }
}

impl<T: Interval> IntervalCast<T> for BoolInterval {
    #[inline]
    fn interval_cast(&self) -> T {
        match *self {
            BoolInterval::True => T::from_literals(num::one(), num::one()),
            BoolInterval::False => T::from_literals(num::zero(), num::zero()),
            BoolInterval::Unknown => T::from_literals(num::zero(), num::one()),
            _ => T::from_literals(num::one(), num::zero()),
        }
    }
}

impl<T: Interval> IntervalCast<BoolInterval> for T {
    /// Convert the interval to a `bool`. Zeros are mapped to `false`. Everything else maps to `true`.
    fn interval_cast(&self) -> BoolInterval {
        if self.is_empty_interval() {
            return BoolInterval::Empty;
        }
        if self.has_value(num::zero()) {
            if self.get_lower().0 == self.get_upper().0 {
                return BoolInterval::False;
            }
            return BoolInterval::Unknown;
        }
        BoolInterval::True
    }
}

// /// Classify how two intervals are related to one another.
// ///
// /// Given two intervals, A, and B, then the following diagrams describe A's ordering with respect to `B`
// ///
// /// ```none
// /// Subsumes
// /// A: |------------------|
// /// B:     |---------|
// ///
// /// Subsumed:
// /// A:    |---------|
// /// B: |------------------|
// ///
// /// OverlapsRight:
// /// A:     |------|
// /// B: |------|
// /// OverlapsLeft:
// /// A: |------|
// /// B:     |------|
// ///
// /// JustBefore:
// /// A: |------|
// /// B:         |------|
// ///
// /// JustAfter:
// /// A:         |------|
// /// B: |------|
// ///
// /// DisjointRight:
// /// A:            |------|
// /// B: |------|
// ///
// /// DisjointLeft:
// /// A: |------|
// /// B:            |------|
// ///
// /// Equal:
// /// A: |------|
// /// B: |------|
// ///
// /// PartialOverlap:
// /// (CompoundIntervals only)
// /// A: |-----|     |-----|
// /// B:   |------|
// /// ```
// pub trait IntervalOrdering {}

// Implementation of `IntervalAbs` for unsigned integers. For all intervals, just returns what was provided.
macro_rules! unsigned_abs_impl {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'a> IntervalAbs<'a> for WrappedInterval<$ty> {
                type Output = &'a Self;
                #[inline(always)]
                #[doc = "Returns the passed argument."]
                fn interval_abs(&'a self) -> Self::Output {
                    self
                }
            }

            impl<'a> IntervalAbs<'a> for BasicInterval<$ty> {
                type Output = &'a Self;
                #[inline(always)]
                #[doc = "Returns the passed argument."]
                fn interval_abs(&'a self) -> Self::Output {
                    self
                }
            }

            impl<'a> IntervalAbs<'a> for CompoundInterval<$ty> {
                type Output = &'a Self;
                #[inline(always)]
                #[doc = "Returns the passed argument."]
                fn interval_abs(&'a self) -> Self::Output {
                    self
                }
            }
        )+

    }
}

unsigned_abs_impl! {
    u8, u16, u32, u64
}

/// Implements `IntervalAbs` for the signed integer types, for `WrappedInterval`, `BasicInterval`, and `CompoundInterval`.
macro_rules! signed_abs_impl {
    ($($ty:ty),+ $(,)?) => {
        $(
        impl<'a> IntervalAbs<'a> for BasicInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            #[inline(always)]
            fn interval_abs(&'a self) -> Self::Output {
                macro_rules! ordered(
                    ($variant:ty, $lower:expr, $upper:expr) => {
                        if $lower < $upper {
                            <$variant>::from_literals($lower, $upper)
                        } else {
                            <$variant>::from_literals($upper, $lower)
                        }
                    }
                );
                if self.is_empty_interval() {
                    return WrappedInterval::EMPTY;
                }
                if self.lower >= 0 {
                    return WrappedInterval::Basic(self.clone());
                };

                if self.lower == <$ty>::MIN && self.is_unit() {
                    return WrappedInterval::new_unit(<$ty>::MIN);
                }

                if self.lower == <$ty>::MIN {
                    let first = BasicInterval::new_unit(<$ty>::MIN);
                    let second = ordered!(BasicInterval<$ty>, (self.lower + 1).abs(), self.upper.abs());

                    return WrappedInterval::Compound(CompoundInterval::from_iter_unchecked([
                        first, second,
                    ]));
                }

                return ordered!(WrappedInterval<$ty>, self.lower.abs(), self.upper.abs());
            }
        }

        impl<'a> IntervalAbs<'a> for CompoundInterval<$ty> {
            type Output = std::borrow::Cow<'a, CompoundInterval<$ty>>;
            #[inline(always)]
            fn interval_abs(&'a self) -> Self::Output {
                use std::borrow::Cow::{Borrowed, Owned};
                if self.is_empty_interval() || self.get_lower().0 >= 0 {
                    return Borrowed(&self);
                };

                let mut new = CompoundInterval::new();

                // Keep track of whether we've inserted `min` already so that we avoid attempting to insert it multiple times.
                let mut min_inserted = false;
                for interval in self.iter() {
                    let mut lower = interval.get_lower().0;
                    if lower >= 0 {
                        new.insert(interval.clone());
                        continue;
                    }
                    if lower == <$ty>::MIN {
                        if !min_inserted {
                            min_inserted = true;
                            new.insert(BasicInterval::new_unit(<$ty>::MIN));
                        }
                        if self.is_unit() {
                            continue;
                        }
                        lower += 1;
                    }
                    let upper = interval.get_upper().0.abs();
                    let lower = lower.abs();
                    if upper > lower {
                        new.insert(BasicInterval::from_literals(lower, upper));
                    } else {
                        new.insert(BasicInterval::from_literals(upper, lower));
                    }
                }

                Owned(new)
            }
        }
        impl<'a> IntervalAbs<'a> for WrappedInterval<$ty> {
            type Output = std::borrow::Cow<'a, WrappedInterval<$ty>>;
            #[inline(always)]
            fn interval_abs(&'a self) -> Self::Output {
                use std::borrow::Cow::{Borrowed, Owned};
                match *self {
                    WrappedInterval::Top => Owned(WrappedInterval::new_concrete(0, <$ty>::MAX)),
                    ref any if any.is_empty_interval() => Borrowed(&WrappedInterval::EMPTY),
                    ref any if any.get_lower().0 >= 0 => Borrowed(any),
                    WrappedInterval::Basic(interval) => Owned(interval.interval_abs()),
                    WrappedInterval::Compound(ref interval) => {
                        Owned(WrappedInterval::Compound(interval.interval_abs().into_owned()))
                    }
                    // Safety: Unreachable_unchecked is safe here because the first arm checks whether the interval is empty.
                    WrappedInterval::Empty => unreachable!("Already matched against empty"),
                }
            }
        }
    )+
    }
}

signed_abs_impl! {
    i8, i16, i32, i64
}

macro_rules! interval_arith_impl {
    ($trait:ty, $output:ty, $trait_fn:ident, $checked_fn:ident, $wrapping_fn:ident, $lower_counterpart:ident, $upper_counterpart:ident) => {
        impl<T: IntervalBoundary> $trait for BasicInterval<T> {
            type Output = WrappedInterval<T>;
            fn $trait_fn(&self, rhs: &Self) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    WrappedInterval::Empty
                } else if self.is_top() || rhs.is_top() {
                    WrappedInterval::Top
                } else if let (Some(a), Some(ref b)) = (self.as_literal(), rhs.as_literal()) {
                    // optimize for the case where we are adding literals.
                    WrappedInterval::Basic(BasicInterval::new_unit(a.$wrapping_fn(b)))
                } else {
                    let (lower, lower_wrapped) =
                        match self.lower.$checked_fn(&rhs.$lower_counterpart) {
                            Some(val) => (val, false),
                            None => (self.lower.$wrapping_fn(&rhs.$lower_counterpart), true),
                        };
                    let (upper, upper_wrapped) =
                        match self.upper.$checked_fn(&rhs.$upper_counterpart) {
                            Some(val) => (val, false),
                            None => (self.upper.$wrapping_fn(&rhs.$upper_counterpart), true),
                        };

                    match (lower_wrapped, upper_wrapped) {
                        (true, true) => {
                            let res = WrappedInterval::Basic(BasicInterval::new(lower, upper));
                            // If this number is signed, and the lower bound is greater than the upper bound, then
                            // this means that overflow occured in both directions.
                            // In this case, the result should be `top` (and NOT empty!)
                            // compiler will be able to optimize this out.
                            if T::min_value() < T::zero() && lower > upper {
                                return WrappedInterval::TOP;
                            }
                            res
                        }
                        (false, false) => WrappedInterval::Basic(BasicInterval::new(lower, upper)),
                        (true, false) => {
                            // Lower wrapped, but upper did not. We have a union now.
                            let wrapped = BasicInterval::new(lower, T::max_value());
                            let no_wrap = BasicInterval::new(T::min_value(), upper);
                            WrappedInterval::Compound(CompoundInterval::from_iter([
                                wrapped, no_wrap,
                            ]))
                        }
                        (false, true) => {
                            let wrapped = BasicInterval::new(lower, T::max_value());
                            let no_wrap = BasicInterval::new(T::min_value(), upper);
                            WrappedInterval::Compound(CompoundInterval::from_iter([
                                wrapped, no_wrap,
                            ]))
                        }
                    }
                }
            }
        }
    };
}

// Implement Add and Sub here.
interval_arith_impl!(
    IntervalAdd,
    WrappedInterval<T>,
    interval_add,
    checked_add,
    wrapping_add,
    lower,
    upper
);

interval_arith_impl!(
    IntervalSub,
    WrappedInterval<T>,
    interval_sub,
    checked_sub,
    wrapping_sub,
    upper,
    lower
);

macro_rules! commutative_op_impl {
    ($generic:ident, $trait:ident, $trait_fn:ident $(,)?) => {
        impl<T: IntervalBoundary> $trait for WrappedInterval<T>
        where
            BasicInterval<T>: $trait,
            <BasicInterval<T> as $trait>::Output: Into<WrappedInterval<T>>,
            BasicInterval<T>: Interval<Inner = T>,
        {
            type Output = Self;
            fn $trait_fn(&self, rhs: &Self) -> Self::Output {
                match (self, rhs) {
                    (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
                    (Self::Top, _) | (_, Self::Top) => Self::Top,
                    (Self::Basic(a), Self::Basic(b)) => a.$trait_fn(b).into(),
                    (Self::Basic(interval), Self::Compound(intervals))
                    | (Self::Compound(intervals), Self::Basic(interval)) => {
                        intervals.$trait_fn(interval).into()
                    }
                    (Self::Compound(a), Self::Compound(b)) => a.$trait_fn(b).into(),
                }
            }
        }
    };
}

macro_rules! noncommutative_op_impl {
    ($generic:ident, $trait:ident, $trait_fn:ident $(,)?) => {
        impl<T: IntervalBoundary> $trait for WrappedInterval<T> {
            type Output = Self;
            fn $trait_fn(&self, rhs: &Self) -> Self::Output {
                match (self, rhs) {
                    (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
                    (Self::Top, _) | (_, Self::Top) => Self::Top,
                    (Self::Basic(a), Self::Basic(b)) => a.$trait_fn(b).into(),
                    (Self::Basic(interval), Self::Compound(intervals)) => {
                        interval.$trait_fn(intervals).into()
                    }
                    (Self::Compound(a), Self::Basic(b)) => a.$trait_fn(b).into(),
                    (Self::Compound(a), Self::Compound(b)) => a.$trait_fn(b).into(),
                }
            }
        }
    };
}

commutative_op_impl!(T, IntervalAdd, interval_add);
commutative_op_impl!(T, IntervalMul, interval_mul);
commutative_op_impl!(T, IntervalMax, interval_max);
commutative_op_impl!(T, IntervalMin, interval_min);
noncommutative_op_impl!(T, IntervalSub, interval_sub);

macro_rules! basic_interval_shr_impl {
    ($ty:ty $(,)?) => {
        impl IntervalShr<BasicInterval<u32>> for BasicInterval<$ty> {
            type Output = BasicInterval<$ty>;
            fn interval_shr(&self, rhs: &BasicInterval<u32>) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return BasicInterval::new(1, 0);
                }
                if rhs.is_top() || ((rhs.get_upper().0 - rhs.get_lower().0) >= <$ty>::BITS as _) {
                    return BasicInterval::new(0, self.upper);
                }
                // The number of bits to shift is the value of the right-hand side modulo the bit width.
                // source: https://www.w3.org/TR/WGSL/#bit-expr
                // When the right hand side is an interval, then shift according to the modulo of said interval.
                let lower_mod_class = rhs.get_lower().0 / <$ty>::BITS;
                let upper_mod_class = rhs.get_upper().0 / <$ty>::BITS;
                let lower_mod = rhs.get_lower().0 % <$ty>::BITS;
                let upper_mod = rhs.get_upper().0 % <$ty>::BITS;

                if lower_mod_class != upper_mod_class {
                    BasicInterval::new(self.lower >> (lower_mod.max(upper_mod)), self.upper)
                } else {
                    let lower = self.lower >> lower_mod;
                    let upper = self.upper >> upper_mod;
                    BasicInterval::new(lower, upper)
                }
            }
        }
    };
}

basic_interval_shr_impl!(u32);
basic_interval_shr_impl!(u64);
basic_interval_shr_impl!(i32);
basic_interval_shr_impl!(i64);

macro_rules! compound_interval_shr_impl {
    ($ty:ty $(,)?) => {
        impl IntervalShr<BasicInterval<u32>> for CompoundInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            fn interval_shr(&self, rhs: &BasicInterval<u32>) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                let mut new = CompoundInterval::new();
                for interval in self.iter() {
                    new.insert(interval.interval_shr(rhs));
                }
                new.into()
            }
        }
        impl IntervalShr<CompoundInterval<u32>> for BasicInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            fn interval_shr(&self, rhs: &CompoundInterval<u32>) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                let mut new = CompoundInterval::new();
                for interval in rhs.iter() {
                    new.insert(self.interval_shr(interval));
                }
                new.into()
            }
        }
        impl IntervalShr<CompoundInterval<u32>> for CompoundInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            fn interval_shr(&self, rhs: &CompoundInterval<u32>) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                // get the lower and upper for rhs's compound interval, we are
                // going to overapproximate the rhs
                let mut new = CompoundInterval::new();
                // compound interval will just use the max...

                for interval in self.iter() {
                    for rhs_interval in rhs.iter() {
                        new.insert(interval.interval_shr(rhs_interval));
                    }
                }
                new.into()
            }
        }
    };
}
compound_interval_shr_impl!(u32);
compound_interval_shr_impl!(i32);
compound_interval_shr_impl!(u64);
compound_interval_shr_impl!(i64);

macro_rules! wrapped_interval_shr_impl {
    ($ty:ty $(,)?) => {
        impl IntervalShr<BasicInterval<u32>> for WrappedInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            fn interval_shr(&self, rhs: &BasicInterval<u32>) -> Self::Output {
                match self {
                    WrappedInterval::Empty => WrappedInterval::Empty,
                    WrappedInterval::Top => WrappedInterval::Basic(
                        BasicInterval::new(self.get_lower().0, self.get_upper().0)
                            .interval_shr(rhs),
                    ),
                    WrappedInterval::Basic(interval) => {
                        WrappedInterval::Basic(interval.interval_shr(rhs))
                    }
                    WrappedInterval::Compound(intervals) => intervals.interval_shr(rhs).into(),
                }
            }
        }

        impl IntervalShr<WrappedInterval<u32>> for WrappedInterval<$ty> {
            type Output = WrappedInterval<$ty>;
            fn interval_shr(&self, rhs: &WrappedInterval<u32>) -> Self::Output {
                match (self, rhs) {
                    (WrappedInterval::Empty, _) | (_, WrappedInterval::Empty) => {
                        WrappedInterval::Empty
                    }
                    (WrappedInterval::Top, WrappedInterval::Top) => WrappedInterval::Top,
                    (WrappedInterval::Basic(a), WrappedInterval::Top) => {
                        WrappedInterval::Basic(BasicInterval::new(0, a.get_upper().0))
                    }
                    (WrappedInterval::Top, WrappedInterval::Basic(b)) => WrappedInterval::Basic(
                        BasicInterval::new(self.get_lower().0, self.get_upper().0).interval_shr(b),
                    ),
                    (WrappedInterval::Top, WrappedInterval::Compound(intervals)) => {
                        BasicInterval::new(self.get_lower().0, self.get_upper().0)
                            .interval_shr(intervals)
                            .into()
                    }
                    (WrappedInterval::Compound(intervals), WrappedInterval::Top) => intervals
                        .interval_shr(&BasicInterval::new(0, rhs.get_upper().0))
                        .into(),
                    (WrappedInterval::Basic(interval), WrappedInterval::Basic(rhs)) => {
                        interval.interval_shr(rhs).into()
                    }
                    (WrappedInterval::Basic(interval), WrappedInterval::Compound(intervals)) => {
                        interval.interval_shr(intervals).into()
                    }
                    (WrappedInterval::Compound(intervals), WrappedInterval::Basic(interval)) => {
                        intervals.interval_shr(interval).into()
                    }
                    (WrappedInterval::Compound(a), WrappedInterval::Compound(b)) => {
                        a.interval_shr(b).into()
                    }
                }
            }
        }
    };
}

wrapped_interval_shr_impl!(u32);
wrapped_interval_shr_impl!(i32);
wrapped_interval_shr_impl!(u64);
wrapped_interval_shr_impl!(i64);

// impl IntervalShr<WrappedInterval<u32>> for WrappedInterval<i32> {
//     type Output = WrappedInterval<i32>;
//     fn interval_shr(&self, rhs: &WrappedInterval<u32>) -> Self::Output {
//         match (self, rhs) {
//             (WrappedInterval::Empty, _) | (_, WrappedInterval::Empty) => WrappedInterval::Empty,
//             (WrappedInterval::Top, _) | (_, WrappedInterval::Top) => WrappedInterval::Top,
//             (WrappedInterval::Basic(a), WrappedInterval::Basic(b)) => a.interval_shr(b),
//             (WrappedInterval::Basic(interval), WrappedInterval::Compound(intervals))
//             | (WrappedInterval::Compound(intervals), WrappedInterval::Basic(interval)) => {
//                 intervals.interval_shr(interval)
//             }
//             (WrappedInterval::Compound(a), WrappedInterval::Compound(b)) => a.interval_shr(b),
//         }
//     }
// }

// impl<T> IntervalShr<WrappedInterval<u32>> for WrappedInterval<T>
// where
//     T: IntervalBoundary,
//     BasicInterval<T>: IntervalShr<BasicInterval<u32>, Output = WrappedInterval<T>>,
//     CompoundInterval<T>: IntervalShr<BasicInterval<u32>, Output = WrappedInterval<T>>,
//     CompoundInterval<T>: IntervalShr<CompoundInterval<u32>, Output = WrappedInterval<T>>,
// {
//     type Output = WrappedInterval<T>;
//     fn interval_shr(&self, rhs: &WrappedInterval<u32>) -> Self::Output {
//         match (self, rhs) {
//             (WrappedInterval::Empty, _) | (_, WrappedInterval::Empty) => WrappedInterval::Empty,
//             (WrappedInterval::Top, _) | (_, WrappedInterval::Top) => WrappedInterval::Top,
//             (WrappedInterval::Basic(a), WrappedInterval::Basic(b)) => a.interval_shr(b),
//             (WrappedInterval::Basic(interval), WrappedInterval::Compound(intervals))
//             | (WrappedInterval::Compound(intervals), WrappedInterval::Basic(interval)) => {
//                 intervals.interval_shr(interval)
//             }
//             (WrappedInterval::Compound(a), WrappedInterval::Compound(b)) => a.interval_shr(b),
//         }
//     }
// }

impl<T: IntervalBoundary + num::Signed> IntervalNeg for BasicInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_neg(&self) -> WrappedInterval<T> {
        if self.is_empty_interval() {
            WrappedInterval::Empty
        } else if self.is_top() {
            WrappedInterval::Top
        } else if let Some(val) = self.as_literal() {
            WrappedInterval::Basic(BasicInterval::new_unit(-val))
        } else if self.lower == T::min_value() {
            WrappedInterval::from_iter([
                BasicInterval::new_unit(T::min_value()),
                BasicInterval::new(-self.upper, T::max_value()),
            ])
        } else {
            WrappedInterval::Basic(BasicInterval::new(-self.upper, -self.lower))
        }
    }
}

impl<T: IntervalBoundary + num::Signed> IntervalNeg for CompoundInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_neg(&self) -> WrappedInterval<T> {
        if self.is_empty_interval() {
            return WrappedInterval::Empty;
        }
        if self.is_top() {
            return WrappedInterval::Top;
        }
        let mut new = CompoundInterval::new();
        for interval in self.iter() {
            new.union_with_wrapped(&interval.interval_neg());
        }
        new.into()
    }
}

impl<T: IntervalBoundary + num::Signed> IntervalNeg for WrappedInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_neg(&self) -> WrappedInterval<T> {
        match self {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => WrappedInterval::Top,
            WrappedInterval::Basic(interval) => interval.interval_neg(),
            WrappedInterval::Compound(intervals) => intervals.interval_neg(),
        }
    }
}

impl<T: IntervalBoundary> IntervalMax<BasicInterval<T>> for BasicInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_max(&self, rhs: &BasicInterval<T>) -> Self::Output {
        if self.is_empty_interval() || rhs.is_empty_interval() {
            return WrappedInterval::Empty;
        }
        if self.is_top() {
            rhs.into()
        } else if rhs.is_top() {
            self.into()
        } else {
            WrappedInterval::Basic(BasicInterval::new(
                self.lower.max(rhs.lower),
                self.upper.max(rhs.upper),
            ))
        }
    }
}

impl<T: IntervalBoundary> IntervalMin<BasicInterval<T>> for BasicInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_min(&self, rhs: &BasicInterval<T>) -> Self::Output {
        if self.is_empty_interval() || rhs.is_empty_interval() {
            return WrappedInterval::Empty;
        }
        if self.is_top() {
            rhs.into()
        } else if rhs.is_top() {
            self.into()
        } else {
            WrappedInterval::Basic(BasicInterval::new(
                self.lower.min(rhs.lower),
                self.upper.min(rhs.upper),
            ))
        }
    }
}

impl<T: IntervalBoundary> IntervalMax<BasicInterval<T>> for WrappedInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_max(&self, rhs: &BasicInterval<T>) -> Self::Output {
        match self {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => rhs.into(),
            WrappedInterval::Basic(interval) => interval.interval_max(rhs),
            WrappedInterval::Compound(intervals) => intervals.interval_max(rhs),
        }
    }
}

impl<T: IntervalBoundary> IntervalMin<BasicInterval<T>> for WrappedInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_min(&self, rhs: &BasicInterval<T>) -> Self::Output {
        match self {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => rhs.into(),
            WrappedInterval::Basic(interval) => interval.interval_min(rhs),
            WrappedInterval::Compound(intervals) => intervals.interval_min(rhs),
        }
    }
}

/// Implementation of `IntervalMul` for `BasicInterval` on unsigned integers.
macro_rules! unsigned_mul_impl {
    ($format:ty, $intermediate:ty, $width:literal) => {
        impl IntervalMul<BasicInterval<$format>> for BasicInterval<$format> {
            type Output = WrappedInterval<$format>;
            fn interval_mul(&self, rhs: &BasicInterval<$format>) -> Self::Output {
                let max: $format = num::Bounded::max_value();
                let zero: $format = num::Zero::zero();
                let one: $intermediate = num::One::one();
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                if self.is_top() || rhs.is_top() {
                    return WrappedInterval::Top;
                }
                if let (Some(a), Some(b)) = (self.as_literal(), rhs.as_literal()) {
                    return WrappedInterval::Basic(BasicInterval::new_unit(a.wrapping_mul(b)));
                }
                if let (Some(a), other) = match (self.as_literal(), rhs.as_literal()) {
                    (Some(val), None) => (Some(val), rhs),
                    (None, Some(val)) => (Some(val), self),
                    (None, None) => (None, rhs),
                    _ => unreachable!("Already matched against (Some(_), Some(_))")
                    } {
                        if a == 0 {
                            return WrappedInterval::new_unit(zero);
                        } else if a == 1 {
                            return WrappedInterval::Basic(*other);
                        }
                }
                // One of them is a literal, and the other is not.
                if self.as_literal().is_some_and(|v| v == zero) || rhs.as_literal().is_some_and(|v| v == zero) {
                    return WrappedInterval::new_unit(zero);
                }

                let lower = <$intermediate>::from(self.lower) * <$intermediate>::from(rhs.lower);
                let upper = <$intermediate>::from(self.upper) * <$intermediate>::from(rhs.upper);
                let upper_wrap_count = upper >> <$format>::BITS;
                let lower_wrap_count = lower >> <$format>::BITS;

                // Safety: we bitand with the max value of the type, so we know the value is within the right range.
                let lower_bound = unsafe {
                    <$format>::try_from(lower.bitand(<$intermediate>::from(max))).unwrap_unchecked()
                };
                let upper_bound = unsafe {
                    <$format>::try_from(upper.bitand(<$intermediate>::from(max))).unwrap_unchecked()
                };

                if upper_wrap_count == lower_wrap_count {
                    // If they wrapped the same number of times, then it must be the case that lower < upper.
                    WrappedInterval::new_concrete(lower_bound, upper_bound)
                } else if (lower_wrap_count + one) == upper_wrap_count
                    // If upper wrapped one more time than lower and lower_bound > upper_bound...
                    // With unsigned, we also know that lower cannot wrap when upper does not.
                    && lower_bound > upper_bound
                {
                    WrappedInterval::Compound(CompoundInterval::from_iter([
                        BasicInterval::new(upper_bound, <$format as num::traits::Bounded>::max_value()),
                        BasicInterval::new(<$format as num::traits::Bounded>::min_value(), lower_bound),
                    ]))
                } else {
                    WrappedInterval::Top
                }
            }
        }
    }
}

unsigned_mul_impl!(u8, u16, 8u8);
unsigned_mul_impl!(u16, u32, 16u8);
unsigned_mul_impl!(u32, u64, 32u8);
unsigned_mul_impl!(u64, u128, 64u8);

macro_rules! signed_mul_impl {
    ($format:ty, $intermediate:ty, $width:literal) => {
        impl IntervalMul<BasicInterval<$format>> for BasicInterval<$format> {
        type Output = WrappedInterval<$format>;
        fn interval_mul(&self, rhs: &BasicInterval<$format>) -> Self::Output {
            use num::traits::ConstOne;
            use num::traits::ConstZero;
            let max = <$format>::MAX;
            let min = <$format>::MIN;
            let zero: $format = ConstZero::ZERO;
            let one: $format = ConstOne::ONE;
            if self.is_empty_interval() || rhs.is_empty_interval() {
                return WrappedInterval::Empty;
            }
            if self.is_top() || rhs.is_top() {
                return WrappedInterval::Top;
            }
            if let (Some(a), Some(b)) = (self.as_literal(), rhs.as_literal()) {
                return WrappedInterval::Basic(BasicInterval::new_unit(a.wrapping_mul(b)));
            }
            // if one of them is a literal..
            if let (Some(a), other) = match (self.as_literal(), rhs.as_literal()) {
                (Some(val), None) => (Some(val), rhs),
                (None, Some(val)) => (Some(val), self),
                (None, None) => (None, rhs),
                _ => unreachable!("Already matched against (Some(_), Some(_))"),
            } {
                if a == zero {
                    return WrappedInterval::new_unit(zero);
                }
                if a == one {
                    return WrappedInterval::Basic(*other);
                }
                if a == -one {
                    // if this is negative one and we are multiplying by
                    if other.lower >= zero {
                        return WrappedInterval::Basic(BasicInterval::new(
                            other.upper * a,
                            other.lower * a,
                        ));
                    }
                    if other.upper <= zero {
                        let lower = other.upper * a;
                        if other.lower == min {
                            return WrappedInterval::from_iter([
                                BasicInterval::new(lower, max),
                                BasicInterval::new_unit(min),
                            ]);
                        }
                    }
                }
            }

            let bounds = [
                self.lower.checked_mul(rhs.lower),
                self.lower.checked_mul(rhs.upper),
                self.upper.checked_mul(rhs.lower),
                self.upper.checked_mul(rhs.upper),
            ];
            // If any bounds are `None`, then there was an overflow/underflow,
            // in which case we return `Top`

            if bounds.iter().any(|&x| x.is_none()) {
                return WrappedInterval::Top;
            }
            // We know that all of these are Some, so we can unwrap them.
            let bounds = [
                bounds[0].unwrap(),
                bounds[1].unwrap(),
                bounds[2].unwrap(),
                bounds[3].unwrap(),
            ];
            // now get the lower and upper bounds
            let lower = bounds.iter().min().unwrap();
            let upper = bounds.iter().max().unwrap();
            return WrappedInterval::Basic(BasicInterval::new(*lower, *upper));
        }
    }
}
}

signed_mul_impl!(i8, i16, 8u8);
signed_mul_impl!(i16, i32, 16u8);
signed_mul_impl!(i32, i64, 32u8);
signed_mul_impl!(i64, i128, 64u8);

impl<T: IntervalBoundary> Union<BasicInterval<T>> for BasicInterval<T> {
    /// Return a new `WrappedInterval` that is the union of self and other
    #[allow(clippy::similar_names)] // subsumer and subsumed are fine.
    fn interval_union(&self, other: &Self) -> WrappedInterval<T> {
        match (self, other) {
            (a, b) if a.is_top() || b.is_top() => WrappedInterval::Top,
            (a, b) if a.is_empty_interval() && b.is_empty_interval() => WrappedInterval::Empty,
            (a, b) if a == b => WrappedInterval::Basic(*a),
            // If just one is empty, then Unit of other
            (empty, other) | (other, empty) if empty.is_empty_interval() => {
                WrappedInterval::Basic(*other)
            }
            (subsumer, subsumed) | (subsumed, subsumer) if subsumer.subsumes(subsumed) => {
                WrappedInterval::Basic(*subsumer)
            }
            // Now, If the upper bound of one of our intervals lies within the other interval,
            // Then we can extend the interval.
            (greater, smaller) | (smaller, greater) if greater.has_value(smaller.upper) => {
                // We already know they're not subsumed, as we tested for this.
                WrappedInterval::Basic(BasicInterval {
                    lower: smaller.lower,
                    upper: greater.upper,
                })
            }
            // The lower bound of one of the intervals lies within the other interval.
            (greater, smaller) | (smaller, greater) if smaller.has_value(greater.lower) => {
                WrappedInterval::Basic(BasicInterval {
                    lower: greater.lower,
                    upper: smaller.upper,
                })
            }
            // The two intervals are adjacent, so we can make a new one that extends them both.
            // This occurs when the lower bound of one interval is one more than the upper bound of the other.
            (greater, smaller) | (smaller, greater)
                if smaller.upper.saturating_add(&T::one()) == greater.lower =>
            {
                WrappedInterval::Basic(BasicInterval {
                    lower: smaller.lower,
                    upper: greater.upper,
                })
            }
            // Otherwise, the resulting interval is a union of the two intervals.
            (a, b) => WrappedInterval::Compound(CompoundInterval::from_iter_unchecked([*a, *b])),
        }
    }
}
impl<T: IntervalBoundary> Union<BasicInterval<T>> for WrappedInterval<T> {
    fn interval_union(&self, other: &BasicInterval<T>) -> Self {
        match self {
            Self::Empty => Self::Basic(*other),
            Self::Top => Self::Top,
            Self::Basic(interval) => interval.interval_union(other),
            Self::Compound(intervals) => {
                let mut new = intervals.clone();
                new.insert(*other);
                Self::Compound(new)
            }
        }
    }
}

impl<T: IntervalBoundary> Union<WrappedInterval<T>> for WrappedInterval<T> {
    fn interval_union(&self, other: &Self) -> WrappedInterval<T> {
        match (self, other) {
            // This covers (Empty, Empty)
            (other, empty) | (empty, other) if empty.is_empty_interval() => other.clone(),
            // This covers (_, Top) and (Top, _)
            (_, top) | (top, _) if top.is_top() => WrappedInterval::Top,

            (a, b) if a == b => a.clone(),
            //
            (&Self::Basic(interval), &Self::Basic(other)) => interval.interval_union(&other),
            (&Self::Basic(ref unit), &Self::Compound(ref union_))
            | (&Self::Compound(ref union_), &Self::Basic(ref unit)) => union_.interval_union(unit),
            (Self::Compound(a), other) => a.interval_union(other),
            _ => unreachable!(), // (Empty, _) covered by first arm, (Top, _) covered by second arm.
        }
    }
}

impl<T: IntervalBoundary> Intersect<BasicInterval<T>> for BasicInterval<T> {
    type Output = BasicInterval<T>;
    /// Return a new interval that is the intersection of this interval and `other`.
    fn interval_intersection(&self, other: &Self) -> Self::Output {
        Self {
            lower: std::cmp::max(self.lower, other.lower),
            upper: std::cmp::min(self.upper, other.upper),
        }
    }
}

impl<T: IntervalBoundary> IntersectAssign<BasicInterval<T>> for BasicInterval<T> {
    fn interval_intersect(&mut self, other: &Self) {
        self.lower = std::cmp::max(self.lower, other.lower);
        self.upper = std::cmp::min(self.upper, other.upper);
    }
}

/// Implementing this trait for types provides `a.interval_max(b) -> Output`.
pub trait IntervalMax<Rhs = Self>: Interval {
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_max(&self, rhs: &Rhs) -> Self::Output;
}

/// Implementing this trait allows for an interval to be negated.
pub trait IntervalNeg: Interval
where
    <Self as Interval>::Inner: num::traits::Signed,
{
    type Output: Interval<Inner = <Self as Interval>::Inner>;
    fn interval_neg(&self) -> Self::Output;
}

pub trait IntervalLt<Rhs = Self>: Interval {
    fn interval_lt(&self, rhs: &Rhs) -> BoolInterval;
}

pub trait IntervalLe<Rhs = Self>: Interval {
    fn interval_le(&self, rhs: &Rhs) -> BoolInterval;
}

pub trait IntervalGt<Rhs = Self>: Interval {
    fn interval_gt(&self, rhs: &Rhs) -> BoolInterval;
}

/// Implementing this trait allows for an interval to be compared for greater than or equal to.
pub trait IntervalGe<Rhs = Self>: Interval {
    fn interval_ge(&self, rhs: &Rhs) -> BoolInterval;
}

// Checks if two values can be equal. This is unknown unless both are scalars, or both are empty.
pub trait IntervalEq<Rhs = Self>: Interval
where
    Rhs: Interval<Inner = <Self as Interval>::Inner>,
{
    fn interval_eq(&self, rhs: &Rhs) -> BoolInterval;
}

/// `IntervalEq`
impl<K: Interval, Rhs: Interval<Inner = <K as Interval>::Inner>> IntervalEq<Rhs> for K
where
    K: Intersect<Rhs>,
{
    /// Return whether the two intervals are always equal.
    fn interval_eq(&self, rhs: &Rhs) -> BoolInterval {
        if self.is_empty_interval() || rhs.is_empty_interval() {
            BoolInterval::Empty
        } else if let (Some(lhs), Some(rhs)) = (self.as_literal(), rhs.as_literal()) {
            BoolInterval::from(lhs == rhs)
        } else if self.interval_intersection(rhs).is_empty_interval() {
            BoolInterval::False
        } else {
            BoolInterval::Unknown
        }
    }
}

pub trait IntervalNe<Rhs = Self> {
    fn interval_ne(&self, rhs: &Rhs) -> BoolInterval;
}

/// `IntervalEq`
impl<K: Interval, Rhs: Interval<Inner = <K as Interval>::Inner>> IntervalNe<Rhs> for K
where
    K: Intersect<Rhs>,
{
    /// Return whether the two intervals are always not equal.
    fn interval_ne(&self, rhs: &Rhs) -> BoolInterval {
        if self.is_empty_interval() || rhs.is_empty_interval() {
            BoolInterval::Empty
        } else if let (Some(lhs), Some(rhs)) = (self.as_literal(), rhs.as_literal()) {
            BoolInterval::from(lhs == rhs)
        } else if self.interval_intersection(rhs).is_empty_interval() {
            BoolInterval::False
        } else {
            BoolInterval::Unknown
        }
    }
}

macro_rules! interval_cmp_impl {
    (@body, $rhs:ty, $method:ident, ($low_a:ident, $high_a:ident, $low_b:ident, $high_b:ident), $true_cond:expr, $false_cond:expr) => {
        fn $method(&self, rhs: &$rhs) -> BoolInterval {
            let ($low_a, false) = self.get_lower() else {
                return BoolInterval::Empty;
            };
            let ($high_a, false) = self.get_upper() else {
                return BoolInterval::Empty;
            };
            let ($low_b, false) = rhs.get_lower() else {
                return BoolInterval::Empty;
            };
            let ($high_b, false) = rhs.get_upper() else {
                return BoolInterval::Empty;
            };

            if $true_cond {
                BoolInterval::True
            } else if $false_cond {
                BoolInterval::False
            } else {
                BoolInterval::Unknown
            }
        }
    };
    ($trait:ident, $method:ident, ($low_a:ident, $high_a:ident, $low_b:ident, $high_b:ident), $true_cond:expr, $false_cond:expr) => {
        impl<T: IntervalBoundary> $trait for BasicInterval<T> {
            interval_cmp_impl!(@body, Self, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait<WrappedInterval<T>> for BasicInterval<T> {
            interval_cmp_impl!(@body, WrappedInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait<CompoundInterval<T>> for BasicInterval<T> {
            interval_cmp_impl!(@body, CompoundInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait<BasicInterval<T>> for WrappedInterval<T> {
            interval_cmp_impl!(@body, BasicInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait for WrappedInterval<T> {
            interval_cmp_impl!(@body, Self, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait<CompoundInterval<T>> for WrappedInterval<T> {
            interval_cmp_impl!(@body, CompoundInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }


        impl<T: IntervalBoundary> $trait<BasicInterval<T>> for CompoundInterval<T> {
            interval_cmp_impl!(@body, BasicInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait<WrappedInterval<T>> for CompoundInterval<T> {
            interval_cmp_impl!(@body, WrappedInterval<T>, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }

        impl<T: IntervalBoundary> $trait for CompoundInterval<T> {
            interval_cmp_impl!(@body, Self, $method, ($low_a, $high_a, $low_b, $high_b), $true_cond, $false_cond);
        }
    };
}

// Can always compare an i32 against a u32.  We need this because wgsl allows for indices to be i32,
// but we always assume the size of array is u32
macro_rules! signed_unsigned_cmp_impl {
    ($self_ty:ty) => {
        impl IntervalGt<WrappedInterval<u32>> for WrappedInterval<$self_ty> {
            fn interval_gt(&self, rhs: &WrappedInterval<u32>) -> BoolInterval {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return BoolInterval::Empty;
                }
                let Ok(lower_a) = u32::try_from(self.get_lower().0) else {
                    return BoolInterval::False;
                };
                let Ok(upper_a) = u32::try_from(self.get_lower().0) else {
                    return BoolInterval::Empty;
                };
                BasicInterval::new(lower_a, upper_a).interval_gt(rhs)
            }
        }

        impl IntervalLt<WrappedInterval<u32>> for WrappedInterval<$self_ty> {
            fn interval_lt(&self, rhs: &WrappedInterval<u32>) -> BoolInterval {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return BoolInterval::Empty;
                }
                let Ok(upper_a) = u32::try_from(self.get_upper().0) else {
                    return BoolInterval::True;
                };
                let Ok(lower_a) = u32::try_from(self.get_lower().0) else {
                    return BoolInterval::Empty;
                };
                BasicInterval::new(lower_a, upper_a).interval_lt(rhs)
            }
        }

        impl IntervalLe<WrappedInterval<u32>> for WrappedInterval<$self_ty> {
            fn interval_le(&self, rhs: &WrappedInterval<u32>) -> BoolInterval {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return BoolInterval::Empty;
                }
                let Ok(upper_a) = u32::try_from(self.get_upper().0) else {
                    return BoolInterval::True;
                };
                let Ok(lower_a) = u32::try_from(self.get_lower().0) else {
                    return BoolInterval::Empty;
                };
                BasicInterval::new(lower_a, upper_a).interval_le(rhs)
            }
        }

        impl IntervalGt<WrappedInterval<$self_ty>> for WrappedInterval<u32> {
            #[inline]
            fn interval_gt(&self, rhs: &WrappedInterval<$self_ty>) -> BoolInterval {
                rhs.interval_lt(self)
            }
        }

        impl IntervalGe<WrappedInterval<$self_ty>> for WrappedInterval<u32> {
            #[inline]
            fn interval_ge(&self, rhs: &WrappedInterval<$self_ty>) -> BoolInterval {
                rhs.interval_le(self)
            }
        }
    };
}

signed_unsigned_cmp_impl!(i8);
signed_unsigned_cmp_impl!(i16);
signed_unsigned_cmp_impl!(i32);

interval_cmp_impl!(
    IntervalLt,
    interval_lt,
    (low_a, high_a, low_b, high_b),
    high_a < low_b,
    low_a < high_b
);
interval_cmp_impl!(
    IntervalLe,
    interval_le,
    (low_a, high_a, low_b, high_b),
    high_a <= low_b,
    low_a < high_b
);
interval_cmp_impl!(
    IntervalGt,
    interval_gt,
    (low_a, high_a, low_b, high_b),
    low_a > high_b,
    high_a > low_b
);
interval_cmp_impl!(
    IntervalGe,
    interval_ge,
    (low_a, high_a, low_b, high_b),
    low_a >= high_b,
    high_a > low_b
);

macro_rules! unsigned_div_impl {
    (@impl $format:ty, $kind:ty) => {
        impl IntervalDiv for $kind {
            type Output = WrappedInterval<<Self as Interval>::Inner>;
            fn interval_div(&self, rhs: &Self) -> Self::Output {
                // If either is empty, then return empty.
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::EMPTY;
                }

                // Fast path for when both are literals.
                if let (Some(a), Some(b)) = (self.as_literal(), rhs.as_literal()) {
                    return if b == 0 {
                        WrappedInterval::new_unit(a)
                    } else if a == 0 {
                        WrappedInterval::new_unit(0)
                    } else {
                        WrappedInterval::new_unit(a / b)
                    };
                }
                // Otherwise, we get the lowest and highest values from the interval.
                // We aren't going to do div with union shenanigans..
                let (low_a, false) = self.get_lower() else {
                    return WrappedInterval::EMPTY;
                };
                let (high_a, false) = self.get_upper() else {
                    return WrappedInterval::EMPTY;
                };
                let (low_b, false) = rhs.get_lower() else {
                    return WrappedInterval::EMPTY;
                };
                let (high_b, false) = rhs.get_upper() else {
                    return WrappedInterval::EMPTY;
                };

                // If the lowest that `b` can be is zero, then the highest result is always
                // going to be the highest value of `a`.

                // This is because, in wgsl, `a` / 0 is always `a`
                if low_b == 0 {
                    return WrappedInterval::from_literals(low_a, high_a);
                }

                // Otherwise, the result becomes:
                let low_bound = low_a / high_b;
                if high_b == 0 {
                    unreachable!("high_b should not be zero if low_b is not zero");
                }
                let high_bound = high_a / low_b;

                WrappedInterval::from_literals(low_bound, high_bound)
            }
        }
    };

    ($($format:ty),+ $(,)?) => {
        $(
            unsigned_div_impl!(@impl $format, WrappedInterval<$format>);
            unsigned_div_impl!(@impl $format, BasicInterval<$format>);
            unsigned_div_impl!(@impl $format, CompoundInterval<$format>);
        )+
    };
}

unsigned_div_impl!(u8, u16, u32, u64);

macro_rules! unsigned_mod_impl {
    ($format:ty, $kind:ty) => {
        impl IntervalMod for $kind {
            type Output = WrappedInterval<$format>;
            fn interval_mod(&self, rhs: &Self) -> Self::Output {
                // If either is empty, then return empty.
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::EMPTY;
                }
                if rhs.is_top() {
                    return self.clone().into();
                }

                // Fast path for when both are literals.
                if let (Some(a), Some(b)) = (self.as_literal(), rhs.as_literal()) {
                    return if a == 0 || b == 0 {
                        WrappedInterval::new_unit(0)
                    } else {
                        WrappedInterval::new_unit(a % b)
                    };
                }

                let rhs_bound = rhs.get_upper().0.saturating_sub(1);
                return WrappedInterval::new_concrete(0, rhs_bound.min(self.get_upper().0));
            }
        }

    };
    ($($format:ty),+) => {
        $(
            unsigned_mod_impl!($format, WrappedInterval<$format>);
            unsigned_mod_impl!($format, BasicInterval<$format>);
        )+
    };
}

macro_rules! signed_mod_impl {
    (@impl $format:ty, $kind:ty) => {
        impl IntervalMod for $kind {
            type Output = WrappedInterval<$format>;
            fn interval_mod(&self, rhs: &Self) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::EMPTY;
                }
                if rhs.is_top() {
                    return self.clone().into();
                }

                if let (Some(a), Some(b)) = (self.as_literal(), rhs.as_literal()) {
                    return if b == 0 || a == 0 || (a == <$format>::MIN && b == -1) {
                        WrappedInterval::new_unit(0)
                    } else {
                        WrappedInterval::new_unit(a % b)
                    };
                }

                let a_neg = self.get_lower().0 < 0;
                let a_pos = self.get_upper().0 > 0;

                let lower = rhs.get_lower().0;
                let upper = rhs.get_upper().0;

                let max_mag = if lower < 0 {
                    (lower + 1).abs()
                } else {
                    lower - 1
                }
                .max(if upper < 0 {
                    (upper + 1).abs()
                } else {
                    upper - 1
                });

                // Signed modulo goes like so:
                // it is either [-max_mag, 0], [0, max_mag], or [-max_mag, max_mag], depending on the possible signedness of the left.
                // Except the bounds are also clamped to the bounds of the right operand.
                // In `wgsl`, the modulo operator's sign is always the same sign as the left operand.
                match (a_neg, a_pos) {
                    (true, true) => WrappedInterval::new_concrete(
                        (-max_mag).max(self.get_lower().0),
                        max_mag.min(self.get_upper().0),
                    ),
                    (true, false) => {
                        WrappedInterval::new_concrete((-max_mag).max(self.get_lower().0), 0)
                    }
                    (false, true) => WrappedInterval::new_concrete(
                        0.max(self.get_lower().0),
                        max_mag.min(self.get_upper().0),
                    ),
                    (false, false) => WrappedInterval::new_unit(0),
                }
            }
        }
    };
    ($format:ty) => {
        signed_mod_impl!(@impl $format, BasicInterval<$format>);
        signed_mod_impl!(@impl $format, WrappedInterval<$format>);
    };
}

unsigned_mod_impl!(u8);
unsigned_mod_impl!(u16);
unsigned_mod_impl!(u32);
unsigned_mod_impl!(u64);

signed_mod_impl!(i8);
signed_mod_impl!(i16);
signed_mod_impl!(i32);
signed_mod_impl!(i64);

/// Fast path for division when both are literals. This will short circuit the division in the case that both terms are literals.
///
/// This just computes `a / b`, and returns the result as [`WrappedInterval::Basic`].
///
/// [`WrappedInterval::Basic`]: super::WrappedInterval::Basic
macro_rules! lit_div_fastpath {
    ($lhs:expr, $rhs:expr) => {
        match ($lhs.as_literal(), $rhs.as_literal()) {
            (Some(a), Some(b)) => {
                return if b == 0 {
                    WrappedInterval::new_unit(a)
                } else if a == 0 {
                    WrappedInterval::new_unit(0)
                } else {
                    WrappedInterval::new_unit(a / b)
                };
            }
            (Some(a), None) => {
                if a == 0 {
                    return WrappedInterval::new_unit(0);
                }
            }
            _ => {
                // handled outside of match
            }
        };
    };
}

/// Evaluates to a tuple containing options for the minimum magnitude (or maximum magnitude)
/// of the (positive, negative) values in the interval. If an interval has no positive or negative,
/// then evaluates to `None`.
///
// Note for maintainers: This just extracts out the boilerplate logic into a macro in order to
// force inlining, but keep the definition of signed interval division nice and clean.
macro_rules! get_magnitude_extrema {
    (@min_mag $val:expr) => {{
        let min_mag_neg = if $val.get_lower().0 >= 0 {
            None
        } else {
            Some($val.get_upper().0.min(-1))
        };

        let min_mag_pos = if $val.get_upper().0 <= 0 {
            None
        } else {
            Some($val.get_lower().0.max(1))
        };

        (min_mag_pos, min_mag_neg)
    }};

    (@max_mag $val:expr) => {{
        let max_mag_pos = if $val.get_upper().0 > 0 {
            Some($val.get_upper().0)
        } else {
            None
        };

        let max_mag_neg = if $val.get_lower().0 < 0 {
            Some($val.get_lower().0)
        } else {
            None
        };

        (max_mag_pos, max_mag_neg)
    }};
}

/// Implementation of signed division.
///
/// Note that this is results in an overapproximation for [`CompoundIntervals`].
/// It is deemed too expensive to be precise for `CompoundIntervals`, so we treat them
/// as intervals with a single range, of (min, max).
///
/// [`CompoundIntervals`]: super::CompoundInterval
// !IMPORTANT!
// This implementation has a correctness proof at `proofs/signed_div.smt2`.
// Any changes to this implementation must be verified!
macro_rules! signed_div_impl {
    (@impl $format:ty, $kind:ty) => {
        impl IntervalDiv for $kind {
            type Output = WrappedInterval<<Self as Interval>::Inner>;
            fn interval_div(&self, rhs: &Self) -> Self::Output {
                // If either is empty, then return empty.
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return WrappedInterval::EMPTY;
                }
                // Fast path division when both are literals.
                lit_div_fastpath!(self, rhs);

                let (low_a, _) = self.get_lower();
                let (high_a, _) = self.get_upper();
                let (low_b, _) = rhs.get_lower();
                let (high_b, _) = rhs.get_upper();

                let mut compound = CompoundInterval::new();

                // Special case: if `min`/-1 is always `min`. https://www.w3.org/TR/WGSL/#arithmetic-expr
                if (low_a == <$format>::MIN) && rhs.has_value(-1) {
                    compound.insert(BasicInterval::new_unit(<$format>::MIN));
                }

                let mut low_bound: $format = <$format>::MAX;
                let mut high_bound: $format = <$format>::MIN;

                if rhs.has_value(0) {
                    low_bound = low_a;
                    high_bound = high_a;
                }

                if self.has_value(0) {
                    low_bound = low_bound.min(0);
                    high_bound = high_bound.max(0);
                }

                let (max_mag_pos_a, max_mag_neg_a) = get_magnitude_extrema!(@max_mag self);
                let (min_mag_pos_b, min_mag_neg_b) = get_magnitude_extrema!(@min_mag rhs);
                let (max_mag_pos_b, max_mag_neg_b) = get_magnitude_extrema!(@max_mag rhs);
                let (min_mag_pos_a, min_mag_neg_a) = get_magnitude_extrema!(@min_mag self);

                let a_only_neg = high_a < 0;
                let b_only_neg = high_b < 0;
                let a_only_pos = low_a > 0;
                let b_only_pos = low_b > 0;

                // Set the high bound from positives.
                if let (Some(max_mag_pos_a), Some(min_mag_pos_b)) = (max_mag_pos_a, min_mag_pos_b) {
                    high_bound = high_bound.max(max_mag_pos_a / min_mag_pos_b);
                }
                // Set the high bound from negatives.
                if let (Some(max_mag_neg_a), Some(min_mag_neg_b)) = (max_mag_neg_a, min_mag_neg_b) {
                    if max_mag_neg_a == <$format>::MIN && min_mag_neg_b == -1 {
                        // Special case here. MIN / -1 is always MIN (from wgsl)
                        // So our upper bound can't be captured by the above.
                        // If `high_a` is not also `MIN`, then that means that `MIN + 1` can appear in the interval, and thus the max is MIN + 1 / -1, which would be MAX.
                        if high_a != <$format>::MIN {
                            high_bound = <$format>::MAX;
                        } else if low_b != -1 {
                            // In the case that `a` is unit but `b` is not unit, then the high bound can be computed from doing `MIN / -2`.
                            // That is, we know that `-2` lies in the interval.
                            high_bound = <$format>::MIN / -2;
                        } else if high_b > 0 && a_only_neg {
                            high_bound = <$format>::MIN / high_b;
                        }
                    } else {
                        high_bound = high_bound.max(max_mag_neg_a / min_mag_neg_b);
                    }
                }

                if a_only_neg && b_only_pos {
                    // If `a` is always negative while `b` is always positive...
                    // Then `max` is negative, and as such must be be the smallest magnitude `a` divided by the largest magnitude `b`.

                    high_bound = high_bound.max(min_mag_neg_a.unwrap() / max_mag_pos_b.unwrap());
                } else if a_only_pos && b_only_neg {
                    // or vice versa...
                    high_bound = high_bound.max(min_mag_pos_a.unwrap() / max_mag_neg_b.unwrap());
                }

                // Set the low bound as the largest magnitude after division.
                // That is, we want the largest a divided by the smallest `b` of opposite signs.
                // If `a` and `b` are only ever the same sign, then this is...
                if a_only_neg && b_only_neg {
                    low_bound = low_bound.min(min_mag_neg_a.unwrap() / max_mag_neg_b.unwrap());
                } else if a_only_pos && b_only_pos {
                    low_bound = low_bound.min(min_mag_pos_a.unwrap() / max_mag_pos_b.unwrap());
                } else {
                    if let (Some(max_mag_pos_a), Some(min_mag_neg_b)) = (max_mag_pos_a, min_mag_neg_b) {
                        low_bound = low_bound.min(max_mag_pos_a / min_mag_neg_b);
                    }
                    if let (Some(max_mag_neg_a), Some(min_mag_pos_b)) = (max_mag_neg_a, min_mag_pos_b) {
                        low_bound = low_bound.min(max_mag_neg_a / min_mag_pos_b);
                    }
                }

                compound.insert(BasicInterval::from_literals(low_bound, high_bound));
                compound.into()
            }
        }
    };
    ($format:ty) => {
        signed_div_impl!(@impl $format, BasicInterval<$format>);
        signed_div_impl!(@impl $format, WrappedInterval<$format>);
        signed_div_impl!(@impl $format, CompoundInterval<$format>);
    };

}

signed_div_impl!(i8);
signed_div_impl!(i16);
signed_div_impl!(i32);
signed_div_impl!(i64);

/// Implementation for the interval traits for `CompoundInterval`
macro_rules! interval_trait_impl {
    ($output:ty, $trait:ident, $method:ident $(,)?) => {
        interval_trait_impl!(@basic $output, $trait, $method);
        interval_trait_impl!(@compound $output, $trait, $method);
        interval_trait_impl!(@wrapped $output, $trait, $method);
    };

    (@compound $output:ty, $trait:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait for CompoundInterval<T> {
            type Output = $output;
            fn $method(&self, rhs: &CompoundInterval<T>) -> Self::Output {
                if self.is_empty_interval() || rhs.is_empty_interval() {
                    return Self::Output::Empty
                }
                if rhs.is_top() || self.is_top() {
                    return Self::Output::Top
                }

                let mut new =  Self::new();

                for other in rhs.iter() {
                    match self.$method(other) {
                        WrappedInterval::Basic(interval) => {
                            new.insert(interval);
                        }
                        WrappedInterval::Compound(union) => {
                            new.union_with(&union);
                        }
                        WrappedInterval::Empty => {
                            return WrappedInterval::Empty
                        }
                        WrappedInterval::Top => {
                            return WrappedInterval::Top
                        }

                    }
                }
                new.into()

            }
        }
    };
    (@basic $output:ty, $trait:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait<BasicInterval<T>> for CompoundInterval<T> {
            type Output = $output;
            fn $method(&self, rhs: &BasicInterval<T>) -> Self::Output {
                if rhs.is_empty_interval() || self.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                if rhs.is_top() || self.is_top() {
                    return WrappedInterval::Top;
                }

                let mut new = Self::new();

                for interval in self.iter() {
                    match interval.interval_add(rhs) {
                        WrappedInterval::Basic(interval) => {
                            new.insert(interval);
                        }
                        WrappedInterval::Compound(union) => {
                            new.union_with(&union);
                        }
                        WrappedInterval::Empty => return WrappedInterval::Empty,
                        WrappedInterval::Top => return WrappedInterval::Top,
                    }
                }
                new.into()
            }
        }
    };
    (@wrapped $output:ty, $trait:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait<WrappedInterval<T>> for CompoundInterval<T> {
            type Output = $output;
            fn $method(&self, rhs: &WrappedInterval<T>) -> Self::Output {
                match rhs {
                    WrappedInterval::Empty => WrappedInterval::Empty,
                    WrappedInterval::Top => WrappedInterval::Top,
                    WrappedInterval::Basic(unit) => self.$method(unit),
                    WrappedInterval::Compound(union_) => self.$method(union_),
                }
            }
        }
    };
}

interval_trait_impl!(WrappedInterval<T>, IntervalAdd, interval_add);
interval_trait_impl!(WrappedInterval<T>, IntervalSub, interval_sub);
interval_trait_impl!(WrappedInterval<T>, IntervalMul, interval_mul);
interval_trait_impl!(WrappedInterval<T>, IntervalMod, interval_mod);

/// Implements `BasicInterval op CompoundInterval` for non-commutative operations.
///
/// These operations are assumed to return a `WrappedInterval` type.
///
/// These operations require a different implementation for the `BasicInterval` and `CompoundInterval` types than their commutative counterparts.
///
macro_rules! non_commutative_trait_impl {
    ($trait:ident, $trait_fn:ident) => {
        impl<T: IntervalBoundary> $trait<CompoundInterval<T>> for BasicInterval<T>
        where
            BasicInterval<T>: $trait,
            BasicInterval<T>: Interval<Inner = T>,
            <BasicInterval<T> as $trait>::Output: Into<WrappedInterval<T>>,
        {
            type Output = WrappedInterval<T>;
            fn $trait_fn(&self, rhs: &CompoundInterval<T>) -> Self::Output {
                if rhs.is_empty_interval() {
                    return WrappedInterval::Empty;
                }
                if rhs.is_top() {
                    return WrappedInterval::Top;
                }

                let mut new = CompoundInterval::new();

                for interval in rhs.iter() {
                    match self.$trait_fn(interval).into() {
                        WrappedInterval::Basic(interval) => {
                            new.insert(interval);
                        }
                        WrappedInterval::Compound(union) => {
                            new.union_with(&union);
                        }
                        WrappedInterval::Empty => return WrappedInterval::Empty,
                        WrappedInterval::Top => return WrappedInterval::Top,
                    }
                }
                new.into()
            }
        }
    };
}

non_commutative_trait_impl!(IntervalSub, interval_sub);
non_commutative_trait_impl!(IntervalDiv, interval_div);
non_commutative_trait_impl!(IntervalMod, interval_mod);

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Generate test cases for division operation on different signed integer types.
    macro_rules! test_cases {
        ($name:ident, $ty:ty) => {
            #[rstest]
            // [-10, -5] / [2, 5] = [-5, -1]
            #[case((-10, -5), (2, 5), WrappedInterval::Basic(BasicInterval::from_literals(-5, -1)))]
            // [10, 5] / [-5, -2] = [-5, -1]
            #[case((5, 10), (-5, -2), WrappedInterval::Basic(BasicInterval::from_literals(-5, -1)))]
            // [ -10, -5] / [ -5, 5] = [-2, 2]
            #[case((-10, -5), (-5, -2), WrappedInterval::Basic(BasicInterval::from_literals(1, 5)))]
            // [5, 10] / [2, 5] = [1, 5]
            #[case((5, 10), (2, 5), WrappedInterval::Basic(BasicInterval::from_literals(1, 5)))]
            // [-10, -5] / [0, 0] = [-10, -5]  (dividing by 0 is the identity.)
            #[case((-10, -5), (0, 0), WrappedInterval::Basic(BasicInterval::from_literals(-10, -5)))]
            // [5, 10] / [0, 0] = [5, 10]  (dividing by 0 is the identity.)
            #[case((5, 10), (0, 0), WrappedInterval::Basic(BasicInterval::from_literals(5, 10)))]
            // [-10, -5] / [-5, 5] = [-10, 10]
            #[case((-10, -5), (-5, 5), WrappedInterval::Basic(BasicInterval::from_literals(-10, 10)))]
            // [MIN, MIN] / [-15, 8] = [MIN, MIN] U [MIN, MIN / -2]
            #[case((<$ty>::MIN, <$ty>::MIN), (-15, 8), WrappedInterval::from_iter(vec![BasicInterval::new_unit(<$ty>::MIN), BasicInterval::from_literals(<$ty>::MIN, <$ty>::MIN / -2)]))]
            // [MIN, -1] / [-15, 8] = [MIN, MAX] (we have both -1 and 1 in the divisor)
            #[case((<$ty>::MIN, -1), (-15, 8), WrappedInterval::Basic(BasicInterval::from_literals(<$ty>::MIN, <$ty>::MAX)))]
            // [MIN, MIN] / [-15, -4] = [MIN / -15, MIN / -4]
            #[case((<$ty>::MIN, <$ty>::MIN), (-15, -4), WrappedInterval::Basic(BasicInterval::from_literals(<$ty>::MIN / -15, <$ty>::MIN / -4)))]
            // [MIN, MIN] / [-1, 15] = [MIN, <$ty>::MIN / 15]  (test boundary when the largest result is still negative)
            #[case((<$ty>::MIN, <$ty>::MIN), (-1, 15), WrappedInterval::Basic(BasicInterval::from_literals(<$ty>::MIN, <$ty>::MIN / 15)))]
            fn $name(
                #[case] a: ($ty, $ty),
                #[case] b: ($ty, $ty),
                #[case] expected: WrappedInterval<$ty>,
            ) {
                let interval_1 = BasicInterval::from_literals(a.0, a.1);
                let interval_2 = BasicInterval::from_literals(b.0, b.1);
                let result = interval_1.interval_div(&interval_2);
                assert_eq!(
                    result, expected,
                    "[{}; {}] / [{}; {}] should be: {expected}, got {result}",
                    a.0, a.1, b.0, b.1
                );
            }
        }
    }

    test_cases!(signed_div_i8, i8);
    test_cases!(signed_div_i16, i16);
    test_cases!(signed_div_i32, i32);
    test_cases!(signed_div_i64, i64);

    /// Constructs a wrapped interval.
    /// If just a single element, then it will be a `BasicInterval`
    /// Otherwise, it will be a `CompoundInterval`
    macro_rules! make_intervals {
        ($(($lower:expr, $upper:expr)),+ $(,)?) => {
            WrappedInterval::from_iter([
                $(BasicInterval::new($lower, $upper),)+
            ]
            )
        }
    }

    #[rstest]
    // [10, 30] - [5, 7] = [5, 25]
    #[case((10u32, 30u32), (5u32, 7u32), make_intervals!((3u32, 25u32)))]
    // [4, 4] - [0, 15] = [MAX - 10, MAX] U [0, 4]
    #[case::partial_underflow((4u32, 4u32), (0u32, 15u32), make_intervals!((u32::MAX - 10, u32::MAX), (0, 4)))]
    // [4, 4] - [8, 15] = [MAX - 10, MAX - 3]
    #[case::full_underflow((4u32, 4u32), (8u32, 15u32), make_intervals!((u32::MAX - 10, u32::MAX - 3)))]
    fn test_interval_sub_u32<T, F>(#[case] x: (T, T), #[case] y: (T, T), #[case] expected: F)
    where
        T: IntervalBoundary,
        <BasicInterval<T> as IntervalSub>::Output: Into<F>,
        F: Eq + std::fmt::Debug,
    {
        let x = BasicInterval::new(x.0, x.1);
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_sub(&y);
        assert_eq!(result.into(), expected);
    }

    #[rstest]
    // [4, 4] + [0, 15] = [4, 19]
    #[case((4u32, 4u32), (0u32, 15u32), make_intervals!((4u32, 19u32)))]
    // [4, 4] + [MAX - 8, MAX] = [MAX - 4, MAX] U [0, 3]
    #[case::partial_overflow((4u32, 4u32), (u32::MAX-8, u32::MAX), make_intervals!((u32::MAX - 4, u32::MAX), (0, 3)))]
    // [4, 4] + [MAX - 3, MAX] = [0, 3]
    #[case::full_overflow((4u32, 4u32), (u32::MAX - 3, u32::MAX), make_intervals!((0u32, 3u32)))]
    fn test_interval_add_u32<T, F>(#[case] x: (T, T), #[case] y: (T, T), #[case] expected: F)
    where
        T: IntervalBoundary,
        <BasicInterval<T> as IntervalAdd>::Output: Into<F>,
        F: Eq + std::fmt::Debug,
    {
        let x = BasicInterval::new(x.0, x.1);
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_add(&y);
        assert_eq!(result.into(), expected);
    }

    #[rstest]
    // [4, 4] + [-4, -4] = [0, 0]
    #[case::unit((4i32, 4i32), (-4i32, -4i32), make_intervals!((0, 0)))]
    // [MIN, MIN + 18] + [-4, -4] = [MAX - 3, MAX] U [MIN, MIN + 14]
    #[case::partial_underflow((i32::MIN, i32::MIN + 18i32), (-4i32, -4i32), make_intervals!((i32::MAX - 3, i32::MAX), (i32::MIN, i32::MIN + 14)))]
    // [MIN, MIN + 18] + [-24, -20] = [MAX - 24, MAX - 1]
    #[case::full_underflow((i32::MIN, i32::MIN + 18), (-24i32, -20i32), make_intervals!((i32::MAX - 23, i32::MAX - 1)))]
    // [MAX - 14, MAX] + [10, 400] = [MAX - 4, MAX] U [MIN, MIN + 399]
    #[case::partial_overflow((i32::MAX - 14, i32::MAX), (10i32, 400i32), make_intervals!((i32::MAX - 4, i32::MAX), (i32::MIN, i32::MIN + 399)))]
    // [MAX - 14, MAX] + [15, 400] = [MIN, MIN + 399]
    #[case::full_overflow((i32::MAX - 14, i32::MAX), (15i32, 400i32), make_intervals!((i32::MIN, i32::MIN + 399)))]
    // [MIN + 500, MAX - 500] + [-501, 501] = Top
    #[case::underflow_overflow((i32::MIN + 500, i32::MAX - 500), (-501i32, 501i32), WrappedInterval::Top)]
    fn test_interval_add_i32<T, F>(#[case] x: (T, T), #[case] y: (T, T), #[case] expected: F)
    where
        T: IntervalBoundary,
        <BasicInterval<T> as IntervalAdd>::Output: Into<F>,
        F: Eq + std::fmt::Debug,
    {
        let x = BasicInterval::new(x.0, x.1);
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_add(&y);
        assert_eq!(result.into(), expected);
    }

    #[rstest]
    // [4, 4] - [-4, -4] = [4, 4]
    #[case::both_units((4i32, 4i32), (-4i32, -4i32), make_intervals!((8, 8)))]
    // [4, 4] - [-4, -2] = [6, 8]
    #[case::one_non_unit((4i32, 4i32), (-4i32, -2i32), make_intervals!((6, 8)))]
    // [0, 10] - [-40, 50] = [-50, 50]
    #[case::both_non_units((0i32, 10i32), (-40i32, 50i32), make_intervals!((-50, 50)))]
    // [MIN, MIN + 4] - [2, 4] = [MIN, MIN + 2] U [MAX - 3, MAX]
    #[case::partial_underflow((i32::MIN, i32::MIN + 4), (2i32, 4i32), make_intervals!((i32::MIN, i32::MIN + 2), (i32::MAX -3, i32::MAX)))]
    // [MIN, MIN + 200] [-201, 500] = [MAX - 499, MAX]
    #[case::full_underflow((i32::MIN, i32::MIN + 200), (201i32, 500i32), make_intervals!((i32::MAX - 499, i32::MAX)))]
    fn test_interval_sub_i32<T, F>(#[case] x: (T, T), #[case] y: (T, T), #[case] expected: F)
    where
        T: IntervalBoundary,
        <BasicInterval<T> as IntervalAdd>::Output: Into<F>,
        F: Eq + std::fmt::Debug,
    {
        let x = BasicInterval::new(x.0, x.1);
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_sub(&y);
        assert_eq!(result.into(), expected);
    }

    #[rstest]
    #[case::both_units((4u32, 4u32), (2u32, 2u32), WrappedInterval::Basic(BasicInterval::new(1u32, 1u32)))]
    #[case::signed_basic((-4i32, 8), (2u32, 2u32), WrappedInterval::Basic(BasicInterval::from_literals(-1i32, 2)))]
    #[case::rhs_needs_mod((100u32, 200u32), (34u32, 34u32), WrappedInterval::Basic(BasicInterval::new(25u32, 50u32)))]
    #[case::rhs_zero((4u32, 8u32), (0u32, 0u32), WrappedInterval::Basic(BasicInterval::from_literals(4u32, 8u32)))]
    #[case::rhs_wraps_mod_boundary((4u32, 8u32), (2u32, 34u32), WrappedInterval::Basic(BasicInterval::from_literals(0, 8u32)))]
    fn test_interval_shr_basic<T>(
        #[case] x: (T, T), #[case] y: (u32, u32), #[case] expected: WrappedInterval<T>,
    ) where
        T: IntervalBoundary,
        BasicInterval<T>: IntervalShr<BasicInterval<u32>>,
        <BasicInterval<T> as IntervalShr<BasicInterval<u32>>>::Output: Into<WrappedInterval<T>>,
    {
        let x = BasicInterval::new(x.0, x.1);
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_shr(&y);
        assert_eq!(result.into(), expected);
    }

    #[rstest]
    #[case::wrapped_basic(make_intervals!((4u32, 8u32)), (2u32, 2u32), WrappedInterval::Basic(BasicInterval::new(1u32, 2u32)))]
    #[case::wrapped_compound(make_intervals!((4u32, 8u32), (200u32, 250u32)), (2u32, 2u32), make_intervals!((1u32, 2u32), (50u32, 62u32)))]
    fn test_interval_shr_wrapped<T>(
        #[case] x: WrappedInterval<T>, #[case] y: (u32, u32), #[case] expected: WrappedInterval<T>,
    ) where
        T: IntervalBoundary,
        WrappedInterval<T>: IntervalShr<BasicInterval<u32>>,
        <WrappedInterval<T> as IntervalShr<BasicInterval<u32>>>::Output: Into<WrappedInterval<T>>,
    {
        let y = BasicInterval::new(y.0, y.1);

        let result = x.interval_shr(&y);
        assert_eq!(result.into(), expected);
    }

    // We want to test interval cast
    #[rstest]
    #[case::cast_positive((4i32, 8i32), WrappedInterval::Basic(BasicInterval::from_literals(4u32, 8u32)))]
    #[case::cast_negative((-4i32, 8i32), WrappedInterval::Basic(BasicInterval::from_literals(0u32, u32::MAX)))]
    fn test_cast_i32_u32(#[case] x: (i32, i32), #[case] expected: WrappedInterval<u32>) {
        let x = BasicInterval::new(x.0, x.1);
        let result: WrappedInterval<u32> = x.interval_cast();
        assert_eq!(result, expected);
    }

    #[rstest]
    #[case::cast_positive((0u32, 15u32), WrappedInterval::Basic(BasicInterval::from_literals(0i32, 15i32)))]
    fn test_cast_u32_i32(#[case] x: (u32, u32), #[case] expected: WrappedInterval<i32>) {
        let x = BasicInterval::new(x.0, x.1);
        let result: WrappedInterval<i32> = x.interval_cast();
        assert_eq!(result, expected);
    }
}
