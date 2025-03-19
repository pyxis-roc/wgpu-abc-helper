use crate::{
    solvers::interval::SolverResult, AbcExpression, AbcScalar, AbcType, Assumption, BinaryOp,
    CmpOp, ConstraintModule, FastHashMap, FastHashSet, Handle, Literal, Predicate, StructField,
    Term,
};
use std::borrow::Cow;

use super::{
    warn, BoolInterval, Borrow, Constraint, Entry, Interval, IntervalAbs, IntervalAdd, IntervalEq,
    IntervalError, IntervalGe, IntervalGt, IntervalKind, IntervalLe, IntervalLt, IntervalMax,
    IntervalMin, IntervalMul, IntervalNe, IntervalSub, SolverError, U32Interval, Union,
};
use crate::macros::error_if_different_variants;

/// Map the operation to an interval that can be used to refine the term boundaries.
/// `literal` must be an expression that resolves to a
///
/// For `Neq`, this interval maps to:
/// []
///
/// This macro is designed to be used by [`Resolver::cmp_refinement`].
///
/// [`Resolver::cmp_refinement`]: Resolver::cmp_refinement
macro_rules! cmp_to_interval(
    ($self:expr, $term:expr, $literal:expr, $op:expr) => {
        match $op {
            CmpOp::Eq => $literal.as_interval(),
            CmpOp::Lt => $literal.as_interval().as_lt_interval(),
            CmpOp::Geq => $literal.as_interval().as_ge_interval(),
            CmpOp::Leq => $literal.as_interval().as_le_interval(),
            CmpOp::Gt => $literal.as_interval().as_gt_interval(),
            CmpOp::Neq => {$literal.as_interval().as_lt_interval().interval_union(&$literal.as_interval().as_gt_interval())?}
        }
    }
);

/****************************************************************
    Macro rules
    One-off macros with descriptive names used to keep logic short and sweet.
******************************************************************* */

macro_rules! get_cmp_interval {
    (@lt) => {(IntervalKind::as_lt_interval, IntervalKind::as_gt_interval)};
    (@gt) => {(IntervalKind::as_gt_interval, IntervalKind::as_lt_interval)};
    (@le) => {(IntervalKind::as_le_interval, IntervalKind::as_ge_interval)};
    (@ge) => {(IntervalKind::as_ge_interval, IntervalKind::as_le_interval)};
    ($method:path, $interval:expr) => {
        if $interval.matches_top() { None } else {Some($method($interval))}
    };

    (@$method:ident, $lhs:expr, $rhs:expr) => {
        {
            let (rhs_method, lhs_method) = get_cmp_interval!(@$method);
            (get_cmp_interval!(rhs_method, &$rhs), get_cmp_interval!(lhs_method, &$lhs))
        }
    }
}

/// Sugar that expands to `Return Ok(BoolInterval::Empty)` if the term is empty,
/// or `Ok(BoolInterval::False)` if this is `literal` > MAX or `literal` < MIN.
/// or `Ok(BoolInterval::True)` if this is `literal` >= MIN or `literal` <= MAX.
///
/// Otherwise, this expands to the interval for the term.
macro_rules! short_circuit_literal_comparison {
    ($term:expr, $self:expr, $literal:expr, $op:expr, $ifnone:expr) => {
        match $self.term_map.get($term) {
            None => $ifnone,
            Some(interval) if interval.is_empty() => return Ok(BoolInterval::False),
            Some(interval) => match $op {
                CmpOp::Lt if $literal.is_min() => {
                    return Ok(BoolInterval::False);
                }
                CmpOp::Geq if $literal.is_min() => {
                    return Ok(BoolInterval::True);
                }
                CmpOp::Leq if $literal.is_max() => {
                    return Ok(BoolInterval::True);
                }
                CmpOp::Gt if $literal.is_max() => {
                    return Ok(BoolInterval::False);
                }
                CmpOp::Gt if $literal.is_max() => {
                    return Ok(BoolInterval::False);
                }
                _ => interval,
            },
        }
    };
}

/// Sugar that resovles to `Return Ok(BoolInterval::Empty)` if either term is empty,
/// or calls the  resolve/refine method when at least one of the terms is `Term::Literal`
///
/// Otherwise, resolves to the empty expression
macro_rules! handle_literals_or_empty {
    (@resolve, $self:expr, $lhs:expr, $rhs:expr, $op:expr $(,)?) => {
        match ($lhs, $rhs) {
            (Term::Empty, _) | (_, Term::Empty) => {
                return Ok(BoolInterval::Empty);
            }
            (Term::Literal(l), Term::Literal(r)) => {
                return Ok(resolve_literal_cmp_literal(l, $op, r));
            }
            (Term::Literal(l), a) | (a, Term::Literal(l)) => {
                return $self.resolve_literal_comparison(a, l, $op)
            }
            _ => (),
        };
    };
    (@refine, $self:expr, $lhs:expr, $rhs:expr, $op:expr $(,)?) => {
        match ($lhs, $rhs) {
            (Term::Empty, _) | (_, Term::Empty) => {
                return Ok(BoolInterval::Empty);
            }
            (Term::Literal(ref l), Term::Literal(ref r)) => {
                return Ok(resolve_literal_cmp_literal(l, $op, r));
            }
            (Term::Literal(ref l), a) | (a, Term::Literal(ref l)) => {
                return $self.refine_literal_comparison(a, l, $op)
            }
            _ => (),
        };
    };
}

/****************************************************************
    Helper methods
    Methods with descriptive names used to keep logic short and sweet.
******************************************************************* */
/// Resolve the comparison between a literal and a literal.
#[inline]
fn resolve_literal_cmp_literal(l: &Literal, op: CmpOp, literal: &Literal) -> BoolInterval {
    let res = l.partial_cmp(literal);
    match (res, op) {
        (Some(std::cmp::Ordering::Equal), CmpOp::Lt | CmpOp::Gt | CmpOp::Neq)
        | (Some(std::cmp::Ordering::Less), CmpOp::Eq | CmpOp::Geq | CmpOp::Gt)
        | (Some(std::cmp::Ordering::Greater), CmpOp::Lt | CmpOp::Leq | CmpOp::Eq) => {
            BoolInterval::False
        }
        _ => BoolInterval::True,
    }
}

/// A resolver resolves terms to intervals.
/// It needs an interval map.
pub(super) struct Resolver<'a> {
    /// The map of terms for the resolver
    pub term_map: Cow<'a, FastHashMap<Term, IntervalKind>>,
    pub predicate_map: Cow<'a, FastHashMap<Predicate, BoolInterval>>,
    uniform_map: Cow<'a, FastHashSet<Term>>,
    /// Flag determining whether the interval for expressions should be recomputed.
    recompute: bool,
    type_map: &'a FastHashMap<Term, Handle<AbcType>>,
}

impl Clone for Resolver<'_> {
    fn clone_from(&mut self, source: &Self) {
        self.term_map.clone_from(&source.term_map);
        self.predicate_map.clone_from(&source.predicate_map);
        self.type_map.clone_from(&source.type_map);
        self.uniform_map.clone_from(&source.uniform_map);
        self.recompute = source.recompute;
    }
    fn clone(&self) -> Self {
        Self {
            term_map: self.term_map.clone(),
            predicate_map: self.predicate_map.clone(),
            uniform_map: self.uniform_map.clone(),
            recompute: self.recompute,
            type_map: self.type_map,
        }
    }
}

impl Resolver<'_> {
    /// intersect this resolver with the other resolver.
    pub(super) fn intersect(&mut self, other: &Self) -> bool {
        if self.term_map.is_empty() && self.predicate_map.is_empty() {
            self.clone_from(other);
            return true;
        }
        if other.term_map.is_empty() && other.predicate_map.is_empty() {
            return true;
        }

        let terms = self.term_map.to_mut();
        self.uniform_map
            .to_mut()
            .extend(other.uniform_map.iter().cloned());

        for (term, interval) in other.term_map.iter() {
            match self.term_map.to_mut().entry(term.clone()) {
                Entry::Occupied(mut entry) => {
                    // Update the interval by intersecting it in place with the interval in other.
                    // if this raised an error, then we return false.
                    if entry.get_mut().intersect(interval).is_err() {
                        return false;
                    }
                    // if ANY of the intervals are empty atter intersection, that means that these two resolvers lead to dead code.
                    if entry.get().is_empty() {
                        return false;
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(interval.clone());
                }
            }
        }

        let preds = self.predicate_map.to_mut();

        for (predicate, interval) in other.predicate_map.iter() {
            match preds.entry(predicate.clone()) {
                Entry::Occupied(mut entry) => {
                    let res = entry.get().intersection(*interval);
                    if res == BoolInterval::Empty {
                        return false;
                    }
                    entry.insert(res);
                }
                Entry::Vacant(entry) => {
                    entry.insert(*interval);
                }
            }
        }

        true
    }
}

impl<'a> Resolver<'a> {
    pub fn new(
        term_map: Cow<'a, FastHashMap<Term, IntervalKind>>,
        predicate_map: Cow<'a, FastHashMap<Predicate, BoolInterval>>,
        type_map: &'a FastHashMap<Term, Handle<AbcType>>, uniform_map: Cow<'a, FastHashSet<Term>>,
    ) -> Self {
        Self {
            term_map,
            predicate_map,
            recompute: true,
            type_map,
            uniform_map,
        }
    }
}

impl<'resolver> Resolver<'resolver> {
    ///  Given an op, return a new interval that represents the comparison between `term` and `literal`.
    /// E.g., if `op` is `CmpOp::Lt`, then this will return as [Min, literal-1])
    #[inline]
    fn op_to_interval(op: CmpOp, interval: &Literal) -> IntervalKind {
        match op {
            CmpOp::Eq => interval.as_interval(),
            CmpOp::Lt => interval.as_interval().as_lt_interval(),
            CmpOp::Geq => interval.as_interval().as_ge_interval(),
            CmpOp::Leq => interval.as_interval().as_le_interval(),
            CmpOp::Gt => interval.as_interval().as_gt_interval(),
            CmpOp::Neq => {
                // Unwrap is fine here, as the two intervals will always be the same variant, and thus `interval_hnion`
                interval
                    .as_interval()
                    .as_lt_interval()
                    .interval_union(&interval.as_interval().as_gt_interval())
                    .unwrap()
            }
        }
    }

    /// Refine the intervals for the comparison between `term` and `literal` with the given operator.
    ///
    /// This updates term map.
    pub(super) fn refine_literal_comparison(
        &mut self, term: &Term, literal: &Literal, op: CmpOp,
    ) -> Result<BoolInterval, SolverError> {
        match term {
            Term::Empty => Ok(BoolInterval::Empty),
            Term::Predicate(_) => Err(SolverError::TypeMismatch {
                expected: "Boolean",
                have: "Literal",
                file: file!(),
                line: line!(),
            }),
            Term::Literal(l) => Ok(resolve_literal_cmp_literal(l, op, literal)),
            Term::Var(_) | Term::Expr(_) => {
                let intersect_with = cmp_to_interval!(self, term, literal, op);
                let existing = short_circuit_literal_comparison!(term, self, literal, op, {
                    self.term_map.to_mut().insert(term.clone(), intersect_with);
                    return Ok(BoolInterval::Unknown);
                });
                let intersection = existing.intersection(&intersect_with)?;
                if intersection.is_empty() {
                    Ok(BoolInterval::False)
                } else if *existing == intersection {
                    Ok(BoolInterval::True)
                } else {
                    self.term_map.to_mut().insert(term.clone(), intersection);
                    Ok(BoolInterval::Unknown)
                }
            }
        }
    }

    /// Special case for resolving a comparison against a literal.
    ///
    /// If `update` is true, then this also updates the intervals for the term map.
    pub(super) fn resolve_literal_comparison(
        &self, term: &'resolver Term, literal: &Literal, op: CmpOp,
    ) -> Result<BoolInterval, SolverError> {
        match term {
            Term::Empty => Ok(BoolInterval::Empty),
            Term::Predicate(_) => Err(SolverError::TypeMismatch {
                expected: "Bool",
                have: literal.variant_name(),
                file: file!(),
                line: line!(),
            }),
            Term::Literal(l) => Ok(resolve_literal_cmp_literal(l, op, literal)),
            Term::Var(_) | Term::Expr(_) => {
                let term_interval = short_circuit_literal_comparison!(term, self, literal, op, {
                    return Ok(BoolInterval::Unknown);
                });
                let intersect_with = cmp_to_interval!(self, term, literal, op);

                let intersection = term_interval.intersection(&intersect_with)?;

                // If it is empty, then we return false.
                if intersection.is_empty() {
                    Ok(BoolInterval::False)
                } else if *term_interval == intersection {
                    Ok(BoolInterval::True)
                } else {
                    Ok(BoolInterval::Unknown)
                }
            }
        }
    }

    /// Resolve the comparison between `lhs` and `rhs` with the given operator.
    pub(super) fn resolve_comparison(
        &self, lhs: &Term, rhs: &Term, op: CmpOp,
    ) -> Result<BoolInterval, SolverError> {
        handle_literals_or_empty!(@resolve, self, lhs, rhs, op);

        let lhs_intrvl = self.term_map.get(lhs).unwrap_or(&IntervalKind::Top).clone();
        let rhs_intrvl = self.term_map.get(rhs).unwrap_or(&IntervalKind::Top).clone();

        let lhs_is_top = lhs_intrvl.matches_top();
        let rhs_is_top = rhs_intrvl.matches_top();

        // if either are empty, this is empty.
        if lhs_intrvl.is_empty() || rhs_intrvl.is_empty() {
            return Ok(BoolInterval::Empty);
        }

        let result = match op {
            CmpOp::Eq => Ok(lhs_intrvl.interval_eq(&rhs_intrvl)),
            CmpOp::Geq => Ok(lhs_intrvl.interval_ge(&rhs_intrvl)),
            CmpOp::Gt => Ok(lhs_intrvl.interval_gt(&rhs_intrvl)),
            CmpOp::Leq => Ok(lhs_intrvl.interval_le(&rhs_intrvl)),
            CmpOp::Lt => Ok(lhs_intrvl.interval_lt(&rhs_intrvl)),
            CmpOp::Neq => Ok(lhs_intrvl.interval_ne(&rhs_intrvl)),
        };

        if let Ok(BoolInterval::Unknown) = result {
            error_if_different_variants!(lhs_intrvl, rhs_intrvl);
        }
        result
    }
    /// Resolve the comparison between two terms into an interval, and optionally update the
    /// intervals being compared.
    ///
    /// The return value for this function will be a reference to a `BooleanInterval` that
    ///
    /// If `update` is true, then this will refine the intervals in the term map
    pub(super) fn refine_comparison(
        &mut self, lhs: &Term, rhs: &Term, op: CmpOp,
    ) -> Result<BoolInterval, SolverError> {
        log::debug!("[TEST] Refining intervals for comparison: {lhs} {op} {rhs}");
        handle_literals_or_empty!(@refine, self, lhs, rhs, op);

        // If the term does not exist in the map, then we set its type...

        let (lhs_intrvl, rhs_intrvl);
        if self.recompute {
            lhs_intrvl = self.resolve_term(lhs)?.into_owned();
            rhs_intrvl = self.resolve_term(rhs)?.into_owned();
        } else {
            lhs_intrvl = self.term_map.get(lhs).unwrap_or(&IntervalKind::Top).clone();
            rhs_intrvl = self.term_map.get(rhs).unwrap_or(&IntervalKind::Top).clone();
        }

        // if either are empty, this is empty.
        if lhs_intrvl.is_empty() || rhs_intrvl.is_empty() {
            log::info!("Found empty interval during resolution. Assuming dead code...");
            return Ok(BoolInterval::Empty);
        }

        log::debug!("lhs is: {lhs_intrvl}, rhs is: {rhs_intrvl}");

        /// Given the mode, resolve to a pair of Option<IntervalKind> that can be intersected with to refine `lhs` and `rhs` re
        if lhs_intrvl.is_top() && rhs_intrvl.is_top() {
            log::trace!("Both values are top. No refinement possible.");
            // We can do no refinement here, other than for inequalities.
            // This is not likely to be useful.
            return Ok(BoolInterval::True);
        }

        let (lhs_intersect, rhs_intersect) = match op {
            // For `Eq`, we intersect the two intervals.
            CmpOp::Eq => {
                // If they are both unit intervals that are equal.
                if lhs_intrvl.is_unit() && rhs_intrvl.is_unit() && lhs_intrvl == rhs_intrvl {
                    return Ok(BoolInterval::True);
                }
                let new_intrvl = lhs_intrvl.intersection(&rhs_intrvl)?;
                let result = if new_intrvl.is_empty() {
                    BoolInterval::False
                } else {
                    BoolInterval::Unknown
                };

                if new_intrvl != lhs_intrvl {
                    log::trace!("Inserting {lhs} with {new_intrvl}");
                    self.term_map
                        .to_mut()
                        .insert(lhs.clone(), new_intrvl.clone());
                }
                if new_intrvl != rhs_intrvl {
                    self.term_map.to_mut().insert(rhs.clone(), new_intrvl);
                }
                return Ok(result);
            }
            CmpOp::Neq => {
                // Check if the two intervals overlap...
                // If they are both unit intervals, then we can solve this easily.
                if lhs_intrvl.is_unit() && rhs_intrvl.is_unit() {
                    if lhs_intrvl != rhs_intrvl {
                        return Ok(BoolInterval::True);
                    }
                    return Ok(BoolInterval::False);
                }
                let overlap = lhs_intrvl.intersection(&rhs_intrvl)?;
                if overlap.is_empty() {
                    return Ok(BoolInterval::True);
                }
                return Ok(BoolInterval::False);

                // If they are not both unit intervals, then we need to check if they overlap.
            }
            CmpOp::Lt => get_cmp_interval!(@lt, lhs_intrvl, rhs_intrvl),
            CmpOp::Gt => get_cmp_interval!(@gt, lhs_intrvl, rhs_intrvl),
            CmpOp::Leq => get_cmp_interval!(@le, lhs_intrvl, rhs_intrvl),
            CmpOp::Geq => get_cmp_interval!(@ge, lhs_intrvl, rhs_intrvl),
        };

        if !lhs_intrvl.is_same_variant(&rhs_intrvl) {
            return Err(SolverError::TypeMismatch {
                expected: lhs_intrvl.variant_name(),
                have: rhs_intrvl.variant_name(),
                file: file!(),
                line: line!(),
            });
        }

        let mut truthy = false;

        /// DRY macro for intersecting and updating the term map.
        macro_rules! intersect_and_update {
            ($term:expr, $interval:expr, $intersect:expr) => {
                if let Some(intersect) = $intersect {
                    let new_intrvl = $interval.intersection(&intersect)?;
                    if new_intrvl.is_empty() {
                        return Ok(BoolInterval::False);
                    }
                    if new_intrvl == $interval {
                        truthy = true;
                    } else {
                        self.term_map.to_mut().insert($term.clone(), new_intrvl);
                    }
                }
            };
        }

        intersect_and_update!(lhs, lhs_intrvl, lhs_intersect);
        intersect_and_update!(rhs, rhs_intrvl, rhs_intersect);

        if truthy {
            Ok(BoolInterval::True)
        } else {
            Ok(BoolInterval::Unknown)
        }
    }

    /// Body of `predicate_and` match arm for `refine_predicate`
    pub(super) fn refine_predicate_and(
        &'resolver mut self, lhs: &'resolver Handle<Predicate>, rhs: &'resolver Handle<Predicate>,
        update: bool,
    ) -> Result<BoolInterval, SolverError> {
        let (a, b);
        if update {
            a = self.refine_predicate(lhs)?;
            b = self.refine_predicate(rhs)?;
        } else {
            a = self.resolve_predicate(lhs)?;
            b = self.resolve_predicate(rhs)?;
        }
        if a == BoolInterval::Empty {
            return Ok(BoolInterval::Empty);
            if a == BoolInterval::False {
                return Ok(BoolInterval::False);
            }
        }

        if b == BoolInterval::Empty {
            return Ok(BoolInterval::Empty);
        }
        if b == BoolInterval::False {
            return Ok(BoolInterval::False);
        }

        if a == BoolInterval::True && b == BoolInterval::True {
            return Ok(BoolInterval::True);
        }

        Ok(BoolInterval::Unknown)
    }

    pub(super) fn refine_predicate_or(
        &self, or_term: &'resolver Predicate, update: bool,
    ) -> Result<(Option<&'resolver Predicate>, BoolInterval), SolverError> {
        // keep track of the singular truthy prediate.
        let mut truthy: Option<&Predicate> = None;

        for (&term, truthiness) in or_term
            .get_children_set()
            .iter()
            .map(|x| (x, self.resolve_predicate(x)))
        {
            match truthiness? {
                BoolInterval::True => {
                    if truthy.is_some() {
                        return Ok((None, BoolInterval::True));
                    }
                    truthy = Some(term);
                }
                // If `resolve_predicate` returns unknown, something strange has happened...
                BoolInterval::Empty => return { Ok((None, BoolInterval::Empty)) },
                BoolInterval::Unknown => return { Ok((None, BoolInterval::Unknown)) },
                _ => {}
            }
        }

        match truthy {
            None => Ok((None, BoolInterval::False)),
            other => Ok((other, BoolInterval::True)),
        }
    }
}

impl<'resolver> Resolver<'resolver> {
    /// Set the recompute flag.
    ///
    /// This should be set whenever a new context is entered.
    pub(super) fn set_recompute(&mut self, other: bool) {
        self.recompute = other;
    }

    /// Resolve a term to an interval.
    ///
    /// This returns `Cow` to avoid unnecessary cloning.
    pub(crate) fn resolve_term<'a>(
        &'a self, term: &'resolver Term,
    ) -> Result<Cow<'a, IntervalKind>, SolverError> {
        use Cow;

        if !self.recompute {
            return Ok(Cow::Borrowed(
                self.term_map.get(term).unwrap_or(&IntervalKind::TOP),
            ));
        }

        match *term {
            Term::Empty => Err(SolverError::Unexpected("Empty term when resolving")),
            Term::Expr(ref expr) => Ok(Cow::Owned(self.resolve_expression(expr)?)),
            Term::Var(_) => Ok(self
                .term_map
                .get(term)
                .map_or_else(|| Cow::Owned(IntervalKind::Top), Cow::Borrowed)),
            Term::Predicate(ref pred) => Ok(Cow::Owned(IntervalKind::Bool(
                self.resolve_predicate(pred)?,
            ))),
            Term::Literal(l) => Ok(Cow::Owned(l.as_interval())),
        }

        // Get the type of the term, if we can...
    }
}

/// Methods that resolve expressions
///
#[allow(non_snake_case)]
impl<'resolver> Resolver<'resolver> {
    pub(super) fn resolve_binop(
        &self, op: BinaryOp, lhs: &'resolver Term, rhs: &'resolver Term,
    ) -> Result<IntervalKind, SolverError> {
        let lhs_interval = self.resolve_term(lhs)?.into_owned();
        let rhs_interval = self.resolve_term(rhs)?.into_owned();

        let my_fn = match op {
            BinaryOp::Plus => IntervalKind::interval_add,
            BinaryOp::Minus => IntervalKind::interval_sub,
            BinaryOp::Times => IntervalKind::interval_mul,
            BinaryOp::Div => IntervalKind::interval_div,
            BinaryOp::Mod => IntervalKind::interval_mod,
            BinaryOp::Shr => IntervalKind::interval_shr,
            _ => {
                return Err(SolverError::Unsupported(
                    op.variant_name(),
                    file!(),
                    line!(),
                ));
            }
        };

        my_fn(&lhs_interval, &rhs_interval).map_err(IntervalError::into)
    }

    fn resolve_cast(&self, from: &Term, dest_type: AbcScalar) -> IntervalKind {
        let Some(interval) = self.term_map.get(from) else {
            return IntervalKind::Top;
        };
        match dest_type {
            AbcScalar::Bool => interval.interval_cast_bool(),
            AbcScalar::Uint(4) => interval.interval_cast_u32(),
            AbcScalar::Sint(4) => interval.interval_cast_i32(),
            _ => IntervalKind::Top,
        }
    }

    fn resolve_select(
        &self, cond: &Term, iftrue: &Term, iffalse: &Term,
    ) -> Result<IntervalKind, SolverError> {
        // Resolve the `cond` to a boolean interval..
        let cond_interval = self.resolve_term(cond)?;
        match cond_interval.borrow() {
            IntervalKind::Top | IntervalKind::Bool(BoolInterval::Unknown) => {
                let iftrue_interval = self.resolve_term(iftrue)?;
                let iffalse_interval = self.resolve_term(iffalse)?;
                iftrue_interval
                    .interval_union(&iffalse_interval)
                    .map_err(|_| SolverError::IntervalError(IntervalError::IncompatibleTypes))
            }
            IntervalKind::Bool(BoolInterval::True) => {
                self.resolve_term(iftrue).map(Cow::into_owned)
            }
            IntervalKind::Bool(BoolInterval::False) => {
                self.resolve_term(iffalse).map(Cow::into_owned)
            }
            _ => Err(SolverError::IntervalError(IntervalError::IncompatibleTypes)),
        }
    }

    #[allow(clippy::unused_self)]
    fn resolve_field_access(
        &self, base: &Term, field_idx: usize, struct_ty: &AbcType,
    ) -> IntervalKind {
        // Figure out what the type of the field is

        let AbcType::Struct { ref members } = struct_ty else {
            // field access on a non-struct type somehow..
            return IntervalKind::Top;
        };

        let Some(StructField { ref ty, .. }) = members.get(field_idx) else {
            return IntervalKind::Top;
        };

        // Check to see if we can even estimate the interval for the base..
        match ty.as_ref() {
            AbcType::Scalar(scalar) => IntervalKind::from(*scalar),
            _ => IntervalKind::Top,
        }
    }

    #[allow(clippy::unused_self)] // Self will be used later when we implement array value ranges.
    fn get_array_value_interval(&self, base: &Term, ty: &Handle<AbcType>) -> IntervalKind {
        // When we see a write to a vector, we need to update the domain of the array values.
        match ty.as_ref() {
            AbcType::DynamicArray { ty, .. } | AbcType::SizedArray { ty, .. } => {
                log::trace!("Array found to contain type {ty}");
                match ty.as_ref() {
                    AbcType::Scalar(scalar) => IntervalKind::from(*scalar),
                    _ => IntervalKind::Top,
                }
            }
            _ => IntervalKind::Top,
        }
    }

    /// Resolve the interval for the index access.
    #[allow(clippy::cast_sign_loss)]
    fn resolve_index_access(&self, base: &Term, index: &Term) -> Result<IntervalKind, SolverError> {
        let Some(ty) = self.type_map.get(base) else {
            for (term, ty) in self.type_map {
                log::trace!("===Term: {term} has type {ty}===");
            }
            log::trace!("No type found for {base}. Resolving to top...");

            return Ok(IntervalKind::Top);
        };

        let index = self.resolve_term(index)?;

        // The index is always a u32 (required in wgsl).
        let idx: u32;

        // If this is a unit index, then we get the domain of the term at this index.
        if index.is_unit() {
            // Great, now we need the singular unit value...
            match index.as_ref() {
                IntervalKind::I32(interval) => {
                    let lower = interval.get_lower().0;
                    if lower >= 0 {
                        idx = lower as u32;
                    } else {
                        return Ok(IntervalKind::Top);
                    }
                }
                IntervalKind::U32(interval) => {
                    idx = interval.get_lower().0;
                }
                _ => {
                    return Ok(IntervalKind::Top);
                }
            }

            Ok(self
                .term_map
                .get(&Term::new_index_access(base, &Term::new_literal(idx)))
                .unwrap_or(&self.get_array_value_interval(base, ty))
                .clone())
        } else {
            Ok(self.get_array_value_interval(base, ty))
        }
    }

    /// Resolve an expression to an interval.
    ///
    /// If the expression does not resolve to a scalar type, this will return `Top`.
    ///
    /// This does not refine the interval. That can only be done with `refine_term`.
    pub(super) fn resolve_expression(
        &self, expr: &'resolver AbcExpression,
    ) -> Result<IntervalKind, SolverError> {
        use AbcExpression as Expr;
        log::trace!(
            "Resolving expression: {expr} (variant is: {})",
            expr.variant_name()
        );
        let res = match expr {
            Expr::ArrayLength(t) => Ok(self
                .term_map
                .get(&Term::make_array_length(t))
                .unwrap_or(&IntervalKind::U32(U32Interval::TOP))
                .clone()),
            Expr::ArrayLengthDim(inner, dim) => todo!("ArrayLengthDim"),
            Expr::Abs(inner) => {
                let inner_interval = self.resolve_term(inner)?;
                Ok(inner_interval.interval_abs().into_owned())
            }
            Expr::Select(cond, iftrue, iffalse) => self.resolve_select(cond, iftrue, iffalse),
            Expr::BinaryOp(op, lhs, rhs) => self.resolve_binop(*op, lhs, rhs),

            Expr::Cast(term, dest_ty) => Ok(self.resolve_cast(term, *dest_ty)),
            Expr::Dot(_, _) => {
                warn!(
                    "{} expression not supported. Resolving interval to top...",
                    expr.variant_name()
                );
                Ok(IntervalKind::Top)
            }
            Expr::UnaryOp(op, Term) => todo!("UnaryOp"),

            // Matrix, Vector, and Splat resolve to vector types, for which we don't have intervals.
            Expr::Matrix { .. } | Expr::Vector { .. } | Expr::Splat(_, _)
            | Expr::StructStore { .. } // These both resolve to struct types
            | Expr::Store { .. } // We don't represent intervals for store statements.
            => Ok(IntervalKind::Top),

            Expr::FieldAccess { base, field_idx, ty, ..} => Ok(self.resolve_field_access(base, *field_idx, ty)),
            Expr::IndexAccess { base, index } => self.resolve_index_access(base, index),


            Expr::Max(a, b) => {
                let a = self.resolve_term(a)?;
                let b = self.resolve_term(b)?;
                a.interval_max(&b).map_err(IntervalError::into)
            }
            Expr::Min(a, b) => {
                let a = self.resolve_term(a)?;
                let b = self.resolve_term(b)?;
                a.interval_min(&b).map_err(IntervalError::into)
            }
            Expr::Pow { .. } => {
                todo!("Pow")
            }
        };

        if let Ok(ref res) = res {
            log::trace!("Resolved {expr} to {res}");
        }
        res
    }
}
impl Resolver<'_> {
    /// Evaluate the constraint in the context of the current term map.
    pub(super) fn check_constraint(
        &self, constraint: &Constraint,
    ) -> Result<SolverResult, SolverError> {
        // If this is equality...
        match *constraint {
            Constraint::Cmp {
                ref op,
                ref lhs,
                ref rhs,
                ..
            } => {
                let res = self.resolve_comparison(lhs, rhs, *op)?;
                if res == BoolInterval::Unknown
                    && lhs.only_uniforms(&self.uniform_map)
                    && rhs.only_uniforms(&self.uniform_map)
                {
                    return Ok(SolverResult::Maybe);
                }

                Ok(res.into())
            }
            Constraint::Identity { ref term, .. } => match self.resolve_term(term)?.as_ref() {
                IntervalKind::Bool(interval) => Ok((*interval).into()),
                other => Err(SolverError::TypeMismatch {
                    expected: "Bool",
                    have: other.variant_name(),
                    file: file!(),
                    line: line!(),
                }),
            },
        }
    }
    ///  Refine the intervals for the terms in the predicate.
    ///
    /// This is similar to `resolve`, except that it refines the intervals for the terms in the predicate.
    pub(super) fn refine_predicate(
        &mut self, pred: &Predicate,
    ) -> Result<BoolInterval, SolverError> {
        log::trace!("Refining predicate: {pred}");
        match pred {
            Predicate::True => Ok(BoolInterval::True),
            Predicate::False => Ok(BoolInterval::False),
            Predicate::And(ref lhs, ref rhs) => {
                // Get the terms in the conjunction of this predicate, and resolve them.
                // If any of the predicates end up being `false`, then we're done.
                // If any of them are `unknown`, then we respond with unknown...
                let mut result = BoolInterval::True;
                for predicate in pred.get_children_set() {
                    // We _will_ update!
                    // If this ends up being `false`, then we can return `false` and short circuit..
                    let resolved = self.refine_predicate(predicate)?;
                    if resolved == BoolInterval::False {
                        return Ok(BoolInterval::False);
                    }
                    if resolved == BoolInterval::Empty {
                        return Ok(BoolInterval::Empty);
                    }
                    if resolved == BoolInterval::Unknown {
                        result = BoolInterval::Unknown;
                    }
                }
                Ok(result)
            }
            Predicate::Or(ref lhs, ref rhs) => {
                // Get the lhs predicate and the rhs predicate.
                let mut curr = BoolInterval::Empty;
                for term in Predicate::get_children_set(pred) {
                    match self.resolve_predicate(term)? {
                        BoolInterval::True => return Ok(BoolInterval::True),
                        BoolInterval::Empty => return Ok(BoolInterval::Empty),
                        e => {
                            curr |= e;
                        }
                    }
                }
                Ok(curr)
            }
            Predicate::Unit(ref term) => {
                // Get the interval for the term.
                // If the term is not in the map, then we return top.
                match self.term_map.get(term) {
                    None | Some(IntervalKind::Top) => Ok(BoolInterval::Unknown),
                    Some(&IntervalKind::Bool(interval)) => Ok(interval),
                    Some(IntervalKind::I32(_)) => Err(SolverError::TypeMismatch {
                        file: file!(),
                        line: line!(),
                        expected: "Bool",
                        have: "I32",
                    }),
                    Some(IntervalKind::U32(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "U32",
                        file: file!(),
                        line: line!(),
                    }),
                    Some(IntervalKind::I64(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "I64",
                        file: file!(),
                        line: line!(),
                    }),
                    Some(IntervalKind::U64(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "U64",
                        file: file!(),
                        line: line!(),
                    }),
                }
            }
            Predicate::Not(ref pred) => {
                match self.predicate_map.to_mut().entry(pred.as_ref().clone()) {
                    Entry::Occupied(mut entry) => {
                        if entry.get() == &BoolInterval::True {
                            return Ok(BoolInterval::Empty);
                        }
                    }
                    Entry::Vacant(mut entry) => {
                        let res = Ok(BoolInterval::Unknown);
                        entry.insert(BoolInterval::Unknown);
                        return res;
                    }
                }
                // When we negate a predicate,
                // we swap the bits in two positions.
                // Negating a predicate that is `True` results in `False`.
                // Negating a predicate that is `False` results in `True`.
                Ok(match self.resolve_predicate(pred)? {
                    BoolInterval::True => BoolInterval::False,
                    BoolInterval::False => BoolInterval::True,
                    BoolInterval::Unknown => BoolInterval::Unknown,
                    BoolInterval::Empty => BoolInterval::Empty,
                    _ => unreachable!("Unknown flags should never be set in a BoolInterval"),
                })
            }
            Predicate::Comparison(op, lhs, rhs) => self.refine_comparison(lhs, rhs, *op),
        }
    }
    /// Obtain the domain of the predicate.
    ///
    /// This uses the current term map to see if we can resolve the
    /// predicate to a singular value, `true`, or `false`.
    ///
    /// This is used for guards and for select.
    ///
    /// Assumptions are used for assignment and limiting.
    ///
    /// An empty predicate indicates dead code.
    pub(super) fn resolve_predicate(&self, pred: &Predicate) -> Result<BoolInterval, SolverError> {
        if let Some(existing) = self.predicate_map.get(pred) {
            return Ok(*existing);
        }
        match pred {
            Predicate::True => Ok(BoolInterval::True),
            Predicate::False => Ok(BoolInterval::False),
            Predicate::And(ref lhs, ref rhs) => {
                // Get the terms in the conjunction of this predicate.
                let conjunction = Predicate::get_children_set(pred);
                let lhs_interval = self.resolve_predicate(lhs)?;
                let rhs_interval = self.resolve_predicate(rhs)?;
                Ok(lhs_interval.intersection(rhs_interval))
            }
            Predicate::Or(ref lhs, ref rhs) => {
                // Get the lhs predicate and the rhs predicate.
                let mut curr = BoolInterval::Empty;
                for term in Predicate::get_children_set(pred) {
                    match self.resolve_predicate(term)? {
                        BoolInterval::True => return Ok(BoolInterval::True),
                        BoolInterval::Empty => return Ok(BoolInterval::Empty),
                        e => {
                            curr |= e;
                        }
                    }
                }
                Ok(curr)
            }
            Predicate::Unit(ref term) => {
                // Get the interval for the term.
                // If the term is not in the map, then we return top.
                match self.term_map.get(term) {
                    None | Some(IntervalKind::Top) => Ok(BoolInterval::Unknown),
                    Some(&IntervalKind::Bool(interval)) => Ok(interval),
                    Some(IntervalKind::I32(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "I32",
                        file: file!(),
                        line: line!(),
                    }),
                    Some(IntervalKind::U32(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "U32",
                        file: file!(),
                        line: line!(),
                    }),
                    Some(IntervalKind::I64(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "I64",
                        file: file!(),
                        line: line!(),
                    }),
                    Some(IntervalKind::U64(_)) => Err(SolverError::TypeMismatch {
                        expected: "Bool",
                        have: "U64",
                        file: file!(),
                        line: line!(),
                    }),
                }
            }
            Predicate::Not(ref pred) => self
                .resolve_predicate(pred.as_ref())
                .map(BoolInterval::logical_not),
            Predicate::Comparison(op, lhs, rhs) => {
                let lhs_interval = self.resolve_term(lhs)?;
                let rhs_interval = self.resolve_term(rhs)?;
                self.resolve_comparison(lhs, rhs, *op)
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use std::num::NonZero;

    use super::*;
    use rstest::rstest;

    extern crate env_logger;
    use log;

    fn init_logging() {
        env_logger::builder()
            .is_test(true)
            .filter_level(log::LevelFilter::Trace)
            .try_init()
            .unwrap();
    }

    #[rstest]
    fn test_array_value_refinement() {
        init_logging();
        let x_var = Term::Var(Handle::new(crate::Var {
            name: "x".to_string(),
        }));
        let mut type_map = FastHashMap::default();

        let x_0 = Term::new_index_access(&x_var, &Term::new_literal(0u32));

        let y = Term::Var(Handle::new(crate::Var {
            name: "y".to_string(),
        }));

        type_map.insert(
            x_var.clone(),
            Handle::new(AbcType::SizedArray {
                ty: AbcType::mk_u32(),
                size: unsafe { NonZero::new_unchecked(14) },
            }),
        );

        let mut resolver = Resolver::new(
            Cow::Owned(Default::default()),
            Cow::Owned(FastHashMap::default()),
            &type_map,
            Cow::Owned(FastHashSet::default()),
        );
        resolver.recompute = true;

        let my_assumption = Assumption::Assign {
            guard: None,
            lhs: y.clone(),
            rhs: x_0,
        };
        let assumption_to_predicate = my_assumption.to_predicate();

        resolver.refine_predicate(assumption_to_predicate.as_ref().expect("Refinement failed"));

        for term in resolver.term_map.iter() {
            println!("{term:?}");
        }

        // Now, check the interval for y.
        assert_eq!(
            *resolver.term_map.get(&y).unwrap(),
            IntervalKind::U32(U32Interval::TOP)
        );

        // Get the resolution for the term...
    }
    #[rstest]
    fn test_predicate_refinement() {
        // Create two variables, x, and y, both of type u32.
        let x_var = Term::Var(Handle::new(crate::Var {
            name: "x".to_string(),
        }));
        let y_var = Term::Var(Handle::new(crate::Var {
            name: "y".to_string(),
        }));
        let mut type_map = FastHashMap::default();
        type_map.insert(y_var.clone(), AbcType::mk_u32());
        type_map.insert(x_var.clone(), AbcType::mk_u32());

        let mut resolver = Resolver::new(
            Cow::Owned(Default::default()),
            Cow::Owned(FastHashMap::default()),
            &type_map,
            Cow::Owned(FastHashSet::default()),
        );

        // Refine `x` with the predicate x < 5
        let literal = Literal::U32(5);
        let op = CmpOp::Lt;
        let result = resolver
            .refine_literal_comparison(&x_var, &literal, op)
            .unwrap();

        // x should now be [0, 4]
        {
            let res = resolver.term_map.get(&x_var).unwrap();
            assert_eq!(res, &IntervalKind::U32(U32Interval::new_concrete(0, 4)));
        }

        // Now, refine x with x > 2
        resolver
            .refine_literal_comparison(&x_var, &Literal::U32(2), CmpOp::Gt)
            .unwrap();
        // x sxhould now be [3, 4]
        {
            let res = resolver.term_map.get(&x_var).unwrap();
            assert_eq!(res, &IntervalKind::U32(U32Interval::new_concrete(3, 4)));
        }

        // Refine `y` with y > x
        resolver
            .refine_comparison(&y_var, &x_var, CmpOp::Gt)
            .unwrap();
        // y should now be [0, u32::MAX]
        {
            let res = resolver.term_map.get(&y_var).unwrap();
            assert_eq!(
                res,
                &IntervalKind::U32(U32Interval::new_concrete(4, u32::MAX))
            );
        }
    }
}
