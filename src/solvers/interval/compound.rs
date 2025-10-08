use super::{
    BasicInterval, FastHashSet, Intersect, IntersectAssign, Interval, IntervalBoundary,
    IntervalMax, IntervalMin, Union, WrappedInterval,
};
#[cfg(feature = "logging")]
use log::warn as log_warn;
use std::collections::BTreeSet;

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundInterval<T: IntervalBoundary> {
    inner: BTreeSet<BasicInterval<T>>,
}

#[allow(unused)]
impl<T: IntervalBoundary> CompoundInterval<T> {
    /// Returns true if the this compound interval contains exactly one interval that has only one value.
    pub fn is_unit(&self) -> bool {
        // SafetY: We are guarded on set being non empty, so the iterator is guaranteed to have at least one element.
        self.inner.len() == 1 && unsafe { self.inner.iter().next().unwrap_unchecked() }.is_unit()
    }
    #[inline]
    pub fn is_top(&self) -> bool {
        self.inner.contains(&BasicInterval::top())
    }
    /// Create a new `CompoundInterval` from a set of intervals.
    ///
    /// # Requirements
    /// This method requires that the set of intervals is already in simplified form.
    /// A set of intervals is in simplified form if no two intervals in the set *overlap* or are *adjacent* to each other.
    ///
    /// Intervals *overlap* if their intersection is non-empty; they are adjacent if one of the lower bounds is one less than the other's upper bound.
    /// (e.g. \[1, 2\] and \[3, 4\] are adjacent, but \[1, 2\] and \[4, 5\] are not)
    ///
    /// Formally, a set of intervals is in simplified form if, for all pairs of intervals `a` and `b`,
    /// min(a.lower, b.lower) + 1 < max(a.lower, b.lower) and max(a.upper, b.upper) - 1 > min(a.upper, b.upper)
    #[inline]
    pub(super) fn from_iter_unchecked<A: IntoIterator<Item = BasicInterval<T>>>(other: A) -> Self {
        Self {
            inner: BTreeSet::from_iter(other),
        }
    }

    #[inline]
    pub const fn get_inner_set(&self) -> &BTreeSet<BasicInterval<T>> {
        &self.inner
    }

    #[inline]
    pub(crate) fn get_inner_set_mut(&mut self) -> &mut BTreeSet<BasicInterval<T>> {
        &mut self.inner
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Same as [`BTreeSet::retain`]
    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&BasicInterval<T>) -> bool) {
        self.inner.retain(f);
    }

    /// Same as [`BTreeSet::iter`]
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &BasicInterval<T>> {
        self.inner.iter()
    }

    /// Remove the elements from the union and return them as an iterator..
    #[inline]
    pub(super) fn drain(&mut self) -> impl Iterator<Item = BasicInterval<T>> + '_ {
        std::mem::take(&mut self.inner).into_iter()
    }

    /// Same as [`BTreeSet::get`]
    #[inline]
    pub fn get(&self, value: &BasicInterval<T>) -> Option<&BasicInterval<T>> {
        self.inner.get(value)
    }

    /// Determine if the provided value is contained by any of the intervals in this union.
    #[inline]
    pub fn contains(&self, value: T) -> bool {
        self.inner.iter().any(|interval| interval.has_value(value))
    }

    /// Check if the union subsumes the provided unit interval.
    #[inline]
    pub fn subsumes_unit(&self, interval: &BasicInterval<T>) -> bool {
        if interval.is_empty_interval() {
            return true;
        }
        let mut iter = self.inner.iter();
        {
            // Safety: We checked against the interval having elements in the call to `self.is_empty()`
            let first = unsafe { iter.next().unwrap_unchecked() };
            if first.lower > interval.upper {
                return false;
            } else if first.subsumes(interval) {
                return true;
            }
        }

        iter.any(|i| i.subsumes(interval))
    }

    /// Same as [`FastHashSet::new`]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            inner: BTreeSet::new(),
        }
    }

    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    ///
    /// This is a O(n*m) operation (n being the number of intervals in `self`, and m being the number of intervals in `other`).
    /// Use sparingly.
    pub fn union_with(&mut self, other: &Self) {
        for item in &other.inner {
            self.insert(*item);
        }
    }

    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    pub fn union_with_wrapped(&mut self, other: &WrappedInterval<T>) {
        match *other {
            WrappedInterval::Empty => {}
            WrappedInterval::Top => {
                self.inner.clear();
                self.inner.insert(BasicInterval::top());
            }
            WrappedInterval::Basic(unit) => {
                self.insert(unit);
            }
            WrappedInterval::Compound(ref compound) => {
                self.union_with(compound);
            }
        }
    }

    /// Attempt to insert an interval into the union. If this interval is subsumed by another interval, or if it is empty,
    /// then it will not be inserted, and `false` will be returned.
    ///
    /// This is a O(n) operation, as `IntervalUnionVariant` aggressively keeps itself in simplified form.
    pub fn insert(&mut self, interval: BasicInterval<T>) -> bool {
        let mut interval = interval;
        if interval.is_empty_interval() {
            return false;
        }
        // Before we insert into the IntervalUnion, we need to check if the interval is subsumed or subsumes another interaval.
        // We want to do this in as few passes as possible.
        // First, check if we are in the set. If we are, do nothing
        if self.inner.contains(&interval) {
            return false;
        }

        // So, we have a few things to check: 1, if we are a subset of another.
        // If we are, then we know that we don't have anything to do at all, so we don't need to add the interval,
        // and we can return early.

        // If any of the other intervals subsumes us, then we return early.
        for member in &self.inner {
            if member.subsumes(&interval) {
                return false;
            }
        }

        // Otherwise, we are going to remove elements as we build up the interval we are inserting.
        self.inner.retain(|i| {
            // If we subsume the interval, just remove it without doing anything.
            if interval.subsumes(i) {
                false
            } else if let WrappedInterval::Basic(new) = i.interval_union(&interval) {
                // If the intervals merge into one, then we do so.
                interval = new;
                false
            } else {
                true
            }
        });

        self.inner.insert(interval)
    }
}

#[allow(clippy::implicit_hasher)] // We specifically use FastHashSet...
impl<T: IntervalBoundary> From<CompoundInterval<T>> for FastHashSet<BasicInterval<T>> {
    fn from(variant: CompoundInterval<T>) -> Self {
        let mut new = FastHashSet::with_capacity_and_hasher(variant.len(), Default::default());
        new.extend(variant.inner);
        new
    }
}

impl<A: IntervalBoundary> FromIterator<BasicInterval<A>> for CompoundInterval<A> {
    fn from_iter<T: IntoIterator<Item = BasicInterval<A>>>(iter: T) -> Self {
        let mut new = Self::new();
        for item in iter {
            new.insert(item);
        }
        new
    }
}

impl<T: IntervalBoundary> Interval for CompoundInterval<T> {
    type Inner = T;
    #[allow(clippy::wrong_self_convention)] // Really don't care about that here.
    fn as_literal(&self) -> Option<T> {
        if self.inner.len() == 1 {
            self.inner.iter().next().unwrap().as_literal()
        } else {
            None
        }
    }

    fn from_literals(lower: T, upper: T) -> Self {
        let mut new = Self::new();
        new.insert(BasicInterval::from_literals(lower, upper));
        new
    }

    /// Get the lowest bound of the union.
    #[inline]
    fn get_lower(&self) -> (T, bool) {
        match self.inner.first() {
            Some(BasicInterval { lower, .. }) => (*lower, false),
            None => (T::zero(), true),
        }
    }

    /// Get the highest bound of the union.
    #[inline]
    fn get_upper(&self) -> (T, bool) {
        match self.inner.last() {
            Some(BasicInterval { upper, .. }) => (*upper, false),
            None => (T::zero(), true),
        }
    }

    fn has_value(&self, value: T) -> bool {
        if self.is_empty_interval() {
            return false;
        }
        // Fast path: if we know the value is outside the bounds of the union...
        if self.get_lower().0 > value || self.get_upper().0 < value {
            return false;
        }

        // Maybe look to doing this in parallel later on..
        self.inner.iter().any(|interval| interval.has_value(value))
    }

    /// Check if the `CompoundInterval` is empty.
    ///
    /// A `CompoundInterval` is empty if and only if it contains no intervals.
    #[inline]
    fn is_empty_interval(&self) -> bool {
        self.inner.is_empty()
    }

    fn is_top(&self) -> bool {
        self.inner.last().is_some_and(super::Interval::is_top)
    }

    /// A `CompountInterval` subsumes another interval if and only if one of its contained intervals subsumes it.
    /// This is a O(n) operation, where `n` is the number of intervals in the union.
    #[inline]
    fn subsumes(&self, other: &Self) -> bool {
        // fast path to check if empty...
        if self.is_empty_interval() {
            return false;
        }

        if self.is_top() || other.is_empty_interval() {
            return true;
        }

        other.iter().all(|interval| self.subsumes_unit(interval))
    }

    /// This method should generally not be used, as it is not in simplified form.
    ///
    /// Prefer to use [`WrappedInterval::top`] instead.
    ///
    /// [`WrappedInterval::top`](super::WrappedInterval::top)
    #[inline]
    fn top() -> Self {
        #[cfg(feature = "logging")]
        log_warn!("Call to `CompoundInterval::Top`");
        let mut inner = BTreeSet::new();
        inner.insert(BasicInterval::top());
        Self { inner }
    }
}
impl<T: IntervalBoundary> IntersectAssign<BasicInterval<T>> for CompoundInterval<T> {
    fn interval_intersect(&mut self, other: &BasicInterval<T>) {
        let old = std::mem::take(&mut self.inner);

        for mut interval in old {
            interval.interval_intersect(other);
            self.insert(interval);
        }
    }
}

impl<T: IntervalBoundary> IntersectAssign<CompoundInterval<T>> for CompoundInterval<T> {
    fn interval_intersect(&mut self, other: &CompoundInterval<T>) {
        let mut new = Self::new();

        for interval in &self.inner {
            for other_interval in &other.inner {
                new.insert(interval.interval_intersection(other_interval));
            }
        }

        self.inner = new.inner;
    }
}

impl<T: IntervalBoundary> IntersectAssign<WrappedInterval<T>> for CompoundInterval<T> {
    fn interval_intersect(&mut self, other: &WrappedInterval<T>) {
        match other {
            WrappedInterval::Empty => {
                self.inner.clear();
            }
            WrappedInterval::Top => {}
            WrappedInterval::Basic(unit) => {
                self.interval_intersect(unit);
            }
            WrappedInterval::Compound(union_) => {
                self.interval_intersect(union_);
            }
        }
    }
}

impl<T: IntervalBoundary> Intersect<BasicInterval<T>> for CompoundInterval<T> {
    type Output = Self;
    /// Intersection
    fn interval_intersection(&self, other: &BasicInterval<T>) -> Self {
        let mut new = Self::new();
        for interval in &self.inner {
            new.insert(interval.interval_intersection(other));
        }

        new
    }
}

impl<T: IntervalBoundary> Intersect<CompoundInterval<T>> for CompoundInterval<T> {
    type Output = Self;
    fn interval_intersection(&self, other: &Self) -> Self {
        let mut new = Self::new();

        for interval in &self.inner {
            for other_interval in &other.inner {
                new.insert(interval.interval_intersection(other_interval));
            }
        }

        new
    }
}

macro_rules! union_with_empty {
    ($interval:expr) => {
        if $interval.is_empty_interval() {
            WrappedInterval::Empty
        } else if $interval.is_top() {
            WrappedInterval::top()
        } else if $interval.inner.len() == 1 {
            WrappedInterval::Basic($interval.inner.iter().next().unwrap().clone())
        } else {
            WrappedInterval::Compound($interval.clone())
        }
    };
}

impl<T: IntervalBoundary> Union<WrappedInterval<T>> for CompoundInterval<T> {
    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    fn interval_union(&self, other: &WrappedInterval<T>) -> WrappedInterval<T> {
        match *other {
            WrappedInterval::Empty => union_with_empty!(self),
            _ if other.is_empty_interval() => union_with_empty!(self),
            WrappedInterval::Top => WrappedInterval::Top,
            WrappedInterval::Basic(ref unit) => self.interval_union(unit),
            WrappedInterval::Compound(ref other) => self.interval_union(other),
        }
    }
}

impl<T: IntervalBoundary> Union<BasicInterval<T>> for CompoundInterval<T> {
    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    fn interval_union(&self, other: &BasicInterval<T>) -> WrappedInterval<T> {
        // if `other` is empty, then we do this:
        if other.is_top() {
            WrappedInterval::Top
        } else if other.is_empty_interval() {
            if self.is_empty_interval() {
                WrappedInterval::Empty
            } else if self.is_top() {
                WrappedInterval::top()
            } else if self.inner.len() == 1 {
                WrappedInterval::Basic(*self.inner.iter().next().unwrap())
            } else {
                WrappedInterval::Compound(self.clone())
            }
        } else {
            // We have to clone ourselves, because we are going to modify it
            let mut new = self.clone();
            new.insert(*other);
            match new.len() {
                0 => WrappedInterval::Empty,
                1 => WrappedInterval::Basic(new.drain().next().unwrap()),
                _ => WrappedInterval::Compound(new),
            }
        }
    }
}

impl<T: IntervalBoundary> Union<CompoundInterval<T>> for CompoundInterval<T> {
    fn interval_union(&self, other: &CompoundInterval<T>) -> WrappedInterval<T> {
        // fastest way to union is probably to clone and then union with other.
        // Though, we should probably clone the one that is smaller.
        let mut res = self.clone();
        res.union_with(other);
        match res.len() {
            0 => WrappedInterval::Empty,
            1 => WrappedInterval::Basic(res.drain().next().unwrap()),
            _ => WrappedInterval::Compound(res),
        }
    }
}

impl<T: IntervalBoundary> IntoIterator for CompoundInterval<T> {
    type Item = BasicInterval<T>;
    type IntoIter = std::collections::btree_set::IntoIter<BasicInterval<T>>;
    /// Gets an owning iterator over the contained intervals.
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T: IntervalBoundary> IntoIterator for &'a CompoundInterval<T> {
    type Item = &'a BasicInterval<T>;
    type IntoIter = std::collections::btree_set::Iter<'a, BasicInterval<T>>;
    // Gets a borrowing iterator over the contained intervals.
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<T: IntervalBoundary + std::fmt::Display> std::fmt::Display for CompoundInterval<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.inner.iter();
        if let Some(first) = iter.next() {
            write!(f, "{first}")?;
            for item in iter {
                write!(f, " \u{222A} {item}")?;
            }
        } else {
            write!(f, "\u{2205}")?;
        }
        Ok(())
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq<WrappedInterval<T>> for CompoundInterval<T> {
    fn eq(&self, other: &WrappedInterval<T>) -> bool {
        match other {
            t if t.is_top() => self.is_top(),
            t if t.is_empty_interval() => self.is_empty_interval(),
            WrappedInterval::Compound(other) => self.inner == other.inner,
            WrappedInterval::Basic(unit) => self.inner.len() == 1 && self.inner.contains(unit),
            _ => unreachable!(),
        }
    }
}

impl<T: IntervalBoundary> std::cmp::PartialEq<BasicInterval<T>> for CompoundInterval<T> {
    fn eq(&self, other: &BasicInterval<T>) -> bool {
        match other {
            t if t.is_top() => self.is_top(),
            t if t.is_empty_interval() => self.is_empty_interval(),
            _ => self.inner.len() == 1 && self.inner.contains(other),
        }
    }
}

macro_rules! interval_cmp_impl {
    ($trait:ident, $trait_method:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait<BasicInterval<T>> for CompoundInterval<T> {
            type Output = WrappedInterval<T>;
            fn $trait_method(&self, rhs: &BasicInterval<T>) -> WrappedInterval<T> {
                if self.is_empty_interval() {
                    WrappedInterval::Empty
                } else {
                    let (low, high) =
                        self.iter()
                            .fold((rhs.lower, rhs.upper), |(low, high), interval| {
                                (
                                    std::cmp::$method(low, interval.lower),
                                    std::cmp::$method(high, interval.upper),
                                )
                            });
                    WrappedInterval::new_concrete(low, high)
                }
            }
        }
        impl<T: IntervalBoundary> $trait<CompoundInterval<T>> for CompoundInterval<T> {
            type Output = WrappedInterval<T>;
            fn $trait_method(&self, rhs: &CompoundInterval<T>) -> Self::Output {
                if self.is_empty_interval() {
                    WrappedInterval::Empty
                } else {
                    let (low, high) = self.iter().chain(rhs.iter()).fold(
                        (T::min_value(), T::max_value()),
                        |(low, high), interval| {
                            (
                                std::cmp::$method(low, interval.lower),
                                std::cmp::$method(high, interval.upper),
                            )
                        },
                    );

                    WrappedInterval::new_concrete(low, high)
                }
            }
        }
    };
}

interval_cmp_impl!(IntervalMin, interval_min, min);
interval_cmp_impl!(IntervalMax, interval_max, max);

// these can't use the macro because of the handling with WrappedInterval::Top
impl<T: IntervalBoundary> IntervalMin<WrappedInterval<T>> for CompoundInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_min(&self, rhs: &WrappedInterval<T>) -> Self::Output {
        match rhs {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => {
                let (upper, false) = self.get_upper() else {
                    return WrappedInterval::Empty;
                };
                WrappedInterval::from_literals(T::min_value(), upper)
            }
            WrappedInterval::Basic(unit) => self.interval_min(unit),
            WrappedInterval::Compound(union_) => self.interval_min(union_),
        }
    }
}

impl<T: IntervalBoundary> IntervalMax<WrappedInterval<T>> for CompoundInterval<T> {
    type Output = WrappedInterval<T>;
    fn interval_max(&self, rhs: &WrappedInterval<T>) -> Self::Output {
        match rhs {
            WrappedInterval::Empty => WrappedInterval::Empty,
            WrappedInterval::Top => {
                let (lower, false) = self.get_lower() else {
                    return WrappedInterval::Empty;
                };
                WrappedInterval::from_literals(lower, T::max_value())
            }
            WrappedInterval::Basic(unit) => self.interval_max(unit),
            WrappedInterval::Compound(union_) => self.interval_max(union_),
        }
    }
}

impl<T: IntervalBoundary> From<CompoundInterval<T>> for WrappedInterval<T> {
    fn from(variant: CompoundInterval<T>) -> Self {
        if variant.is_empty_interval() {
            WrappedInterval::Empty
        } else if variant.is_top() {
            WrappedInterval::Top
        } else if variant.inner.len() == 1 {
            WrappedInterval::Basic(*variant.inner.iter().next().unwrap())
        } else {
            WrappedInterval::Compound(variant)
        }
    }
}
