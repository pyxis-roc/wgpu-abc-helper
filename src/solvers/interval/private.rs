use super::*;

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundInterval<T: IntervalBoundary> {
    inner: FastHashSet<BasicInterval<T>>,
}

impl<T: IntervalBoundary> CompoundInterval<T> {
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
            inner: FastHashSet::from_iter(other),
        }
    }

    #[inline]
    pub const fn get_inner_set(&self) -> &FastHashSet<BasicInterval<T>> {
        &self.inner
    }

    #[inline]
    pub(crate) fn get_inner_set_mut(&mut self) -> &mut FastHashSet<BasicInterval<T>> {
        &mut self.inner
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Same as [`FastHashSet::retain`]
    #[inline]
    pub fn retain(&mut self, f: impl FnMut(&BasicInterval<T>) -> bool) {
        self.inner.retain(f);
    }

    /// Same as [`FastHashSet::iter`]
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &BasicInterval<T>> {
        self.inner.iter()
    }

    #[inline]
    pub(super) fn drain(&mut self) -> impl Iterator<Item = BasicInterval<T>> + '_ {
        self.inner.drain()
    }

    /// Same as [`FastHashSet::get`]
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
        if interval.is_empty() {
            return true;
        }
        self.inner.iter().any(|i| i.subsumes(interval))
    }

    /// Same as [`FastHashSet::new`]
    pub fn new() -> Self {
        Self {
            inner: FastHashSet::default(),
        }
    }

    /// Akin to [`HashSet::with_capacity`](std::collections::HashSet::with_capacity)
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: FastHashSet::with_capacity_and_hasher(capacity, Default::default()),
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

    /// Attempt to insert an interval into the union. If this interval is subsumed by another interval, or if it is empty,
    /// then it will not be inserted, and `false` will be returned.
    ///
    /// This is a O(n) operation, as `IntervalUnionVariant` aggressively keeps itself in simplified form.
    pub fn insert(&mut self, mut interval: BasicInterval<T>) -> bool {
        if interval.is_empty() {
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
            } else if let WrappedInterval::Basic(new) = i.union(&interval) {
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
        variant.inner
    }
}

impl<A: IntervalBoundary> FromIterator<BasicInterval<A>> for CompoundInterval<A> {
    fn from_iter<T: IntoIterator<Item = BasicInterval<A>>>(iter: T) -> Self {
        let mut new = Self {
            inner: FastHashSet::default(),
        };
        for item in iter {
            new.insert(item);
        }

        new
    }
}

impl<T: IntervalBoundary> Interval<T> for CompoundInterval<T> {
    #[allow(clippy::wrong_self_convention)] // Really don't care about that here.
    fn as_literal(&self) -> Option<T> {
        if self.inner.len() == 1 {
            self.inner.iter().next().unwrap().as_literal()
        } else {
            None
        }
    }

    fn from_literals(lower: T, upper: T) -> Self {
        let mut new = Self {
            inner: FastHashSet::with_capacity_and_hasher(1, Default::default()),
        };
        new.insert(BasicInterval::from_literals(lower, upper));
        new
    }

    /// Get the lowest bound of the union.
    fn get_lower(&self) -> (T, bool) {
        let mut self_as_iter = self.inner.iter();
        let Some(BasicInterval { lower: mut res, .. }) = self_as_iter.next() else {
            return (T::zero(), true);
        };
        for interval in self_as_iter {
            res = res.min(interval.lower);
        }
        (res, false)
    }

    /// Get the highest bound of the union.
    fn get_upper(&self) -> (T, bool) {
        let mut self_as_iter = self.inner.iter();
        let Some(BasicInterval { upper: mut res, .. }) = self_as_iter.next() else {
            return (T::zero(), true);
        };
        for interval in self_as_iter {
            res = res.max(interval.upper);
        }
        (res, false)
    }

    fn has_value(&self, value: T) -> bool {
        self.inner.iter().any(|interval| interval.has_value(value))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn is_top(&self) -> bool {
        self.inner.contains(&BasicInterval::top())
    }

    /// An `IntervalUnionVariant` subsumes another interval if and only if one of its contained intervals subsumes it.
    /// This is a O(n) operation, where `n` is the number of intervals in the union.
    #[inline]
    fn subsumes(&self, other: &Self) -> bool {
        other.iter().all(|interval| self.subsumes_unit(interval))
    }

    /// This method should generally not be used, as it is not in simplified form.
    ///
    /// Prefer to use [`WrappedInterval::top`] instead.
    ///
    /// [`WrappedInterval::top`](super::WrappedInterval::top)
    #[inline]
    fn top() -> Self {
        Self {
            inner: FastHashSet::from_iter(vec![BasicInterval::top()]),
        }
    }
}
impl<T: IntervalBoundary> IntersectAssign<T, BasicInterval<T>> for CompoundInterval<T> {
    fn intersect(&mut self, other: &BasicInterval<T>) {
        let mut new = Self {
            inner: FastHashSet::default(),
        };

        self.inner.drain().for_each(|mut interval| {
            interval.intersect(other);
            new.insert(interval);
        });
        self.inner = new.inner;
    }
}

impl<T: IntervalBoundary> IntersectAssign<T, CompoundInterval<T>> for CompoundInterval<T> {
    fn intersect(&mut self, other: &CompoundInterval<T>) {
        let mut new = Self {
            inner: FastHashSet::default(),
        };

        for interval in &self.inner {
            for other_interval in &other.inner {
                new.insert(interval.intersection(other_interval));
            }
        }

        self.inner = new.inner;
    }
}

impl<T: IntervalBoundary> IntersectAssign<T, WrappedInterval<T>> for CompoundInterval<T> {
    fn intersect(&mut self, other: &WrappedInterval<T>) {
        match other {
            WrappedInterval::Empty => {
                self.inner.clear();
            }
            WrappedInterval::Top => {}
            WrappedInterval::Basic(unit) => {
                self.intersect(unit);
            }
            WrappedInterval::Compound(union_) => {
                self.intersect(union_);
            }
        }
    }
}

impl<T: IntervalBoundary> Intersect<T, BasicInterval<T>> for CompoundInterval<T> {
    type Output = Self;
    /// Intersection
    fn intersection(&self, other: &BasicInterval<T>) -> Self {
        let mut new = Self {
            inner: FastHashSet::default(),
        };
        for interval in &self.inner {
            new.insert(interval.intersection(other));
        }

        new
    }
}

impl<T: IntervalBoundary> Intersect<T, CompoundInterval<T>> for CompoundInterval<T> {
    type Output = Self;
    fn intersection(&self, other: &Self) -> Self {
        let mut new = Self {
            inner: FastHashSet::default(),
        };

        for interval in &self.inner {
            for other_interval in &other.inner {
                new.insert(interval.intersection(other_interval));
            }
        }

        new
    }
}

macro_rules! union_with_empty {
    ($interval:expr) => {
        if $interval.is_empty() {
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

impl<T: IntervalBoundary> Union<T, WrappedInterval<T>> for CompoundInterval<T> {
    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    fn union(&self, other: &WrappedInterval<T>) -> WrappedInterval<T> {
        match *other {
            WrappedInterval::Empty => union_with_empty!(self),
            _ if other.is_empty() => union_with_empty!(self),
            WrappedInterval::Top => WrappedInterval::Top,
            WrappedInterval::Basic(ref unit) => self.union(unit),
            WrappedInterval::Compound(ref other) => self.union(other),
        }
    }
}

impl<T: IntervalBoundary> Union<T, BasicInterval<T>> for CompoundInterval<T> {
    /// Union `self` with `other` by inserting all elements of `other` into `self`.
    fn union(&self, other: &BasicInterval<T>) -> WrappedInterval<T> {
        // if `other` is empty, then we do this:
        if other.is_top() {
            WrappedInterval::Top
        } else if other.is_empty() {
            if self.is_empty() {
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
                1 => WrappedInterval::Basic(new.inner.drain().next().unwrap()),
                _ => WrappedInterval::Compound(new),
            }
        }
    }
}

impl<T: IntervalBoundary> Union<T, CompoundInterval<T>> for CompoundInterval<T> {
    fn union(&self, other: &CompoundInterval<T>) -> WrappedInterval<T> {
        // fastest way to union is probably to clone and then union with other.
        // Though, we should probably clone the one that is smaller.
        let mut res = self.clone();
        res.union_with(other);
        match res.len() {
            0 => WrappedInterval::Empty,
            1 => WrappedInterval::Basic(res.inner.drain().next().unwrap()),
            _ => WrappedInterval::Compound(res),
        }
    }
}

impl<'a, T: IntervalBoundary> IntoIterator for &'a CompoundInterval<T> {
    type Item = &'a BasicInterval<T>;
    type IntoIter = std::collections::hash_set::Iter<'a, BasicInterval<T>>;
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
            t if t.is_empty() => self.is_empty(),
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
            t if t.is_empty() => self.is_empty(),
            _ => self.inner.len() == 1 && self.inner.contains(other),
        }
    }
}

macro_rules! interval_cmp_impl {
    ($trait:ident, $trait_method:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait<T, BasicInterval<T>> for CompoundInterval<T> {
            type Output = WrappedInterval<T>;
            fn $trait_method(&self, rhs: &BasicInterval<T>) -> WrappedInterval<T> {
                if self.is_empty() {
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
        impl<T: IntervalBoundary> $trait<T, CompoundInterval<T>> for CompoundInterval<T> {
            type Output = WrappedInterval<T>;
            fn $trait_method(&self, rhs: &CompoundInterval<T>) -> Self::Output {
                if self.is_empty() {
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
impl<T: IntervalBoundary> IntervalMin<T, WrappedInterval<T>> for CompoundInterval<T> {
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

impl<T: IntervalBoundary> IntervalMax<T, WrappedInterval<T>> for CompoundInterval<T> {
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
        if variant.is_empty() {
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

/// Implementation for the interval traits for `CompoundInterval`
macro_rules! interval_trait_impl {
    ($output:ty, $trait:ident, $method:ident $(,)?) => {
        interval_trait_impl!(@basic $output, $trait, $method);
        interval_trait_impl!(@compound $output, $trait, $method);
        interval_trait_impl!(@wrapped $output, $trait, $method);
    };

    (@compound $output:ty, $trait:ident, $method:ident) => {
        impl<T: IntervalBoundary> $trait<T, CompoundInterval<T>> for CompoundInterval<T> {
            type Output = $output;
            fn $method(&self, rhs: &CompoundInterval<T>) -> Self::Output {
                if self.is_empty() || rhs.is_empty() {
                    return Self::Output::Empty
                }
                if rhs.is_top() || self.is_top() {
                    return Self::Output::Top
                }

                let mut new =  Self {
                    inner: FastHashSet::with_capacity_and_hasher(self.len(), Default::default())
                };

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
        impl<T: IntervalBoundary> $trait<T, BasicInterval<T>> for CompoundInterval<T> {
            type Output = $output;
            fn $method(&self, rhs: &BasicInterval<T>) -> Self::Output {
                if rhs.is_empty() || self.is_empty() {
                    return Self::Output::Empty;
                }
                if rhs.is_top() || self.is_top() {
                    return WrappedInterval::Top;
                }

                let mut new = Self {
                    inner: FastHashSet::with_capacity_and_hasher(self.len(), Default::default()),
                };

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
        impl<T: IntervalBoundary> $trait<T, WrappedInterval<T>> for CompoundInterval<T> {
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
