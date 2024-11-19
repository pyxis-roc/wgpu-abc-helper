//! Traits for operations on intervals.

use crate::solvers::interval::CompoundInterval;

use super::{BasicInterval, BoolInterval, Interval, IntervalBoundary, WrappedInterval};

#[doc = "Implementing this trait for `T` defines `T \u{222A} Rhs`"]
pub trait Union<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    /// The kind of interval that is output by the union operation.
    ///
    /// This is generally going to be one of the interval wrapper kinds.
    /// Return the new interval that is the union of this interval and another interval.
    #[must_use]
    fn union(&self, rhs: &Rhs) -> WrappedInterval<T>;
}

/// The `UnionAssign` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait UnionAssign<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    /// Union `self` with `other` and assign the result to `self`.
    fn union(&mut self, rhs: &Rhs);
}

/// The `Intersect` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait Intersect<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output: Interval<T>;
    /// Return `self` intersected with `other`
    #[must_use]
    fn intersection(&self, rhs: &Rhs) -> Self::Output;
}

/// The `IntersectAssign` operation.
///
/// Note that `Rhs` is `Self` by default.
pub trait IntersectAssign<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    /// Intersect `self` with `other` and assign the result to `self`.
    fn intersect(&mut self, other: &Rhs);
}

/// The addition operation for intervals.
pub trait IntervalAdd<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_add(&self, rhs: &Rhs) -> Self::Output;
}

/// The subtraction operator for intervals.
pub trait IntervalSub<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_sub(&self, rhs: &Rhs) -> Self::Output;
}

/// The `Mul` operator for intervals
pub trait IntervalMul<T: IntervalBoundary, Rhs>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_mul(&self, rhs: &Rhs) -> Self::Output;
}

/// The division operator for intervals.
pub trait IntervalDiv<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_div(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalMod<T: IntervalBoundary, Rhs>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_mod(&self, rhs: &Rhs) -> Self::Output;
}

pub trait IntervalMin<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_min(&self, rhs: &Rhs) -> Self::Output;
}

/// Trait for the `max` operator on a pair of intervals.
///
/// Implementing this trait for types provides `a.interval_max(b) -> Output`.
pub trait IntervalMax<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    type Output;
    #[must_use]
    fn interval_max(&self, rhs: &Rhs) -> Self::Output;
}

/// Implementing this trait allows for an interval to be negated.
pub trait IntervalNeg<T: IntervalBoundary + num::traits::Signed, Output = Self>:
    Interval<T>
{
    #[must_use]
    fn interval_neg(&self) -> Output;
}

pub trait IntervalLt<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    fn interval_lt(&self, rhs: &Self) -> BoolInterval;
}

pub trait IntervalLe<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    fn interval_le(&self, rhs: &Self) -> BoolInterval;
}

pub trait IntervalGt<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    fn interval_gt(&self, rhs: &Self) -> BoolInterval;
}

/// Implementing this trait allows for an interval to be compared for greater than or equal to.
pub trait IntervalGe<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    fn interval_ge(&self, rhs: &Self) -> BoolInterval;
}

// Checks if two values can be equal. This is unknown unless both are scalars, or both are empty.
pub trait IntervalEq<T: IntervalBoundary, Rhs = Self>: Interval<T> {
    fn interval_eq(&self, rhs: &Self) -> BoolInterval {
        if self.is_empty() || rhs.is_empty() {
            BoolInterval::False
        } else if let (Some(lhs), Some(rhs)) = (self.as_literal(), rhs.as_literal()) {
            BoolInterval::from(lhs == rhs)
        } else {
            BoolInterval::Unknown
        }
    }
}

pub trait IntervalNe<T: IntervalBoundary, Rhs = Self> {
    fn interval_ne(&self, rhs: &Self) -> BoolInterval;
}

macro_rules! interval_cmp_impl {
    ($trait:ident, $method:ident, ($low_a:ident, $high_a:ident, $low_b:ident, $high_b:ident), $true_cond:expr, $false_cond:expr) => {
        impl<T: IntervalBoundary, I: Interval<T>, Rhs: Interval<T>> $trait<T, Rhs> for I {
            fn $method(&self, rhs: &Self) -> BoolInterval {
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
        }
    };
}

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

interval_cmp_impl!(
    IntervalEq,
    interval_eq,
    (low_a, high_a, low_b, high_b),
    low_a == low_b && high_a == high_b && low_a == high_b,
    (high_a < low_b) || (low_a > high_b)
);

interval_cmp_impl!(
    IntervalNe,
    interval_ne,
    (low_a, high_a, low_b, high_b),
    (high_a < low_b) || (low_a > high_b),
    low_a == low_b && high_a == high_b && low_a == high_b
);

macro_rules! unsigned_div_impl {
    ($format:ty) => {
        impl<I: Interval<$format>> IntervalDiv<$format> for I {
            type Output = WrappedInterval<$format>;
            fn interval_div(&self, rhs: &Self) -> Self::Output {
                // If either is empty, then return empty.
                if self.is_empty() || rhs.is_empty() {
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
}

unsigned_div_impl!(u8);
unsigned_div_impl!(u16);
unsigned_div_impl!(u32);
unsigned_div_impl!(u64);

/// Fast path for division when both are literals. This will short circuit the division in the case that both terms are literals.
///
/// This just computes `a / b`, and returns the result as [`WrappedInterval::Basic`].
///
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
    ($format:ty) => {
    impl<I: Interval<$format>> IntervalDiv<$format> for I {
        type Output = WrappedInterval<$format>;
        fn interval_div(&self, rhs: &Self) -> Self::Output {
            // If either is empty, then return empty.
            if self.is_empty() || rhs.is_empty() {
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
}
}

signed_div_impl!(i8);
signed_div_impl!(i16);
signed_div_impl!(i32);
signed_div_impl!(i64);

#[cfg(test)]
mod tests {
    use super::BasicInterval;
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
}
