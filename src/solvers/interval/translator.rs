#![allow(unused)]
use std::io::Empty;

///! The order of constraints in the system must be agnostic.
///! To properly account for this, intervals are captured with `RefCell` in order to allow for
///! terms that are assigned to be equal to each other to have the same interval.
///! x = a
///! a = x
///!
///! Then the interval for `x` should be [4, 4]
///! And the interval for `a` should be [4, 4].
///!
///! That means that what we *really* want are references to the interval.
///! That is, our intervals will always correspond with SSA names.
///!
///! If they don't correspond to SSA names, then we can't do anything.
///!
///! This is because the order of the constraints is supposed to be agnostic.
///! So when we see `x = a`, how are we supposed to know what `a` is?
///! That means that we HAVE to use Rc<RefCell> for this.
///!
///! That allows us to basically have smart pointers to the interval.
use crate::{
    helper::ConstraintModule, AbcExpression, AbcScalar, AbcType, FastHashMap, Handle, Term,
};
use crate::{CmpOp, Constraint, ConstraintOp, Literal, Predicate, StructField};

use super::{BoolInterval, I32Interval, Interval, IntervalError, U32Interval};

use thiserror;

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
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
}

/// Wrapper for intervals of specific kinds.
///
/// Primarily used to allow intervals corresponding to different value types to be stored in the same map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntervalKind {
    I32(I32Interval),
    U32(U32Interval),
    Bool(BoolInterval),
}

impl From<I32Interval> for IntervalKind {
    fn from(value: I32Interval) -> Self {
        IntervalKind::I32(value)
    }
}

impl From<U32Interval> for IntervalKind {
    fn from(interval: U32Interval) -> Self {
        IntervalKind::U32(interval)
    }
}

impl From<BoolInterval> for IntervalKind {
    fn from(interval: BoolInterval) -> Self {
        IntervalKind::Bool(interval)
    }
}

impl IntervalKind {
    fn is_top(&self) -> bool {
        match *self {
            IntervalKind::U32(ref interval) => interval.is_top(),
            IntervalKind::I32(ref interval) => interval.is_top(),
            IntervalKind::Bool(ref interval) => interval.is_top(),
        }
    }

    fn is_empty(&self) -> bool {
        match *self {
            IntervalKind::U32(ref interval) => interval.is_empty(),
            IntervalKind::I32(ref interval) => interval.is_empty(),
            IntervalKind::Bool(ref interval) => interval.is_empty(),
        }
    }

    /// Return the intersection `self` with `other`
    fn intersection(&self, other: &Self) -> Result<Self, IntervalError> {
        if other.is_empty() {
            Ok(self.clone());
        }

        match (*self, *other) {
            (IntervalKind::Bool(interval_a), IntervalKind::Bool(interval_b)) => {
                Ok(IntervalKind::Bool(interval_a.intersection(interval_b)))
            }
            (IntervalKind::I32(ref interval_a), IntervalKind::I32(ref interval_b)) => {
                Ok(IntervalKind::I32(interval_a.intersection(interval_b)))
            }
            (IntervalKind::U32(interval_a), IntervalKind::U32(ref interval_b)) => {
                Ok(IntervalKind::U32(interval_a.intersection(interval_b)))
            }

            _ => Err(IntervalError::IncompatibleTypes),
        }
    }

    /// In-place update `self` to be the intersection with `other.`
    fn intersect(&mut self, other: &Self) -> Result<(), IntervalError> {
        match (*self, *other) {
            (IntervalKind::Bool(ref mut interval_a), IntervalKind::Bool(interval_b)) => {
                *interval_a = interval_a.intersection(interval_b);
            }
            (IntervalKind::I32(ref mut interval_a), IntervalKind::I32(ref interval_b)) => {
                interval_a.intersect(interval_b)
            }
            (IntervalKind::U32(ref mut interval_a), IntervalKind::U32(ref interval_b)) => {
                interval_a.intersect(interval_b)
            }
            _ => Err(IntervalError::IncompatibleTypes),
        }

        Ok(())

        // Otherwise, these must be the same kind
    }
}

impl IntervalKind {
    pub fn as_u32(self) -> Option<U32Interval> {
        match self {
            IntervalKind::U32(interval) => Some(interval),
            _ => None,
        }
    }

    pub fn as_i32(self) -> Option<I32Interval> {
        match self {
            IntervalKind::I32(interval) => Some(interval),
            _ => None,
        }
    }

    pub fn as_bool(self) -> Option<BoolInterval> {
        match self {
            IntervalKind::Bool(interval) => Some(interval),
            _ => None,
        }
    }
}

type Error = &'static str;

/// Attempt to convert `ty_in` into the widest interval for said type.
///
/// # Errors
/// If the type is not supported, then an error is returned.
impl TryFrom<AbcScalar> for IntervalKind {
    type Error = &'static str;

    /// Attempt to convert `ty_in` into the widest interval for said type.
    ///
    /// # Errors
    /// If the type is not supported, then an error is returned.
    #[inline]
    fn try_from(ty_in: AbcScalar) -> Result<Self, Self::Error> {
        match ty_in {
            AbcScalar::Uint(4) => Ok(IntervalKind::U32(U32Interval::top())),
            AbcScalar::Sint(4) => Ok(IntervalKind::I32(I32Interval::top())),
            AbcScalar::Bool => Ok(IntervalKind::Bool(BoolInterval::Unknown)),
            _ => Err("Unsupported type"),
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
            Scalar(t) => IntervalKind::try_from(t),
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
            Scalar(t) => IntervalKind::try_from(t),
            _ => Err("Unsupported type"),
        }
    }
}

// Translator needs to go from constraints & assumptions to intervals.

fn ty_to_interval<'a>(
    term: &Term,
    term_ty: Handle<AbcType>,
    term_map: &'a mut FastHashMap<Term, IntervalKind>,
) -> Result<(), TranslateError> {
    // If the ty is a basic scalar, then we can convert it to an interval.
    let inner_ty = term_ty.as_ref();
    if let AbcType::Scalar(s) = *inner_ty {
        term_map.insert(
            term.clone(),
            IntervalKind::try_from(s).map_err(|_| TranslateError::UnsupportedType)?,
        );
        return Ok(());
    }

    match *term_ty.as_ref() {
        // For sized arrays, we just mark the type of the length.
        AbcType::SizedArray { size, .. } => {
            // We insert into the term map the term representing the length of this array.
            let as_array_length = Term::make_array_length(&term);
            term_map.insert(
                as_array_length,
                IntervalKind::U32(U32Interval::new_concrete(0, u32::from(size))),
            );
        }
        AbcType::DynamicArray { .. } => {
            // We insert into the term map the term representing the length of this array.
            let as_array_length = Term::make_array_length(&term);
            term_map.insert(as_array_length, IntervalKind::U32(U32Interval::new_top()));
        }

        AbcType::NoneType => {
            // We don't need to do anything here. This term does not have a type.
        }

        AbcType::Struct { ref members } => {
            for &StructField { ref name, ref ty } in members {
                let member_access = Term::new_struct_access(term, name.clone(), ty.clone());
                ty_to_interval(&member_access, ty.clone(), term_map)?;
            }
        }

        _ => todo!(),
    };

    Ok(())
}

/// Initializes the intervals for each term from the provided type map.
pub(crate) fn initialize_intervals(
    type_map: &FastHashMap<Term, Handle<AbcType>>,
) -> Result<FastHashMap<Term, IntervalKind>, TranslateError> {
    let mut term_map: FastHashMap<Term, IntervalKind> =
        FastHashMap::with_capacity_and_hasher(type_map.len(), Default::default());

    for (term, ty) in type_map {
        ty_to_interval(&term, ty.clone(), &mut term_map).unwrap();
    }
    Ok(term_map)
}

pub(crate) fn resolve_term_to_interval(
    term: &Term,
    term_map: &FastHashMap<Term, IntervalKind>,
) -> Result<IntervalKind, TranslateError> {
    todo!("Resolve term to interval not yet implemented");
    match term_map.get(term) {
        None => Err(TranslateError::SsaViolation(term.clone())),
        Some(interval) => Ok(interval.clone()),
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
pub(crate) fn resolve_predicate(
    pred: &Predicate,
    term_map: &FastHashMap<Term, IntervalKind>,
) -> Result<BoolInterval, TranslateError> {
    match pred {
        Predicate::True => Ok(BoolInterval::True),
        Predicate::False => Ok(BoolInterval::False),
        Predicate::And(ref lhs, ref rhs) => {
            // Get the lhs predicate and the rhs predicate.
            let lhs_interval = resolve_predicate(lhs, term_map)?;
            let rhs_interval = resolve_predicate(rhs, term_map)?;
            Ok(lhs_interval.intersection(rhs_interval))
        }
        Predicate::Or(ref lhs, ref rhs) => {
            // Get the lhs predicate and the rhs predicate.
            let lhs_interval = resolve_predicate(lhs, term_map)?;
            let rhs_interval = resolve_predicate(rhs, term_map)?;
            Ok(lhs_interval.union(rhs_interval))
        }
        Predicate::Unit(ref term) => {
            // Get the interval for the term.
            // If the term is not in the map, then we return top.
            match term_map.get(term) {
                None => Ok(BoolInterval::Unknown),
                Some(&IntervalKind::Bool(interval)) => Ok(interval),
                Some(&IntervalKind::Empty) => Ok(BoolInterval::Empty),
                Some(IntervalKind::I32(_)) => Err(TranslateError::TypeMismatch {
                    expected: "Bool",
                    have: "I32",
                }),
                Some(IntervalKind::U32(_)) => Err(TranslateError::TypeMismatch {
                    expected: "Bool",
                    have: "U32",
                }),
            }
        }
        Predicate::Not(ref pred) => {
            // When we negate a predicate,
            // we swap the bits in two positions.
            // Negating a predicate that is `True` results in `False`.
            // Negating a predicate that is `False` results in `True`.
            Ok(match resolve_predicate(pred, term_map)? {
                BoolInterval::True => BoolInterval::False,
                BoolInterval::False => BoolInterval::True,
                BoolInterval::Unknown => BoolInterval::Unknown,
                BoolInterval::Empty => BoolInterval::Empty,
                _ => unreachable!("Unknown flags should never be set in a BoolInterval"),
            })
        }
        Predicate::Comparison(op, lhs, rhs) => {
            let lhs_interval = resolve_term_to_interval(lhs, term_map)?;
            let rhs_interval = resolve_term_to_interval(rhs, term_map)?;

            Ok((BoolInterval::Unknown))
        }
    }
}

// A predicate could be maybe true, in which case we would need to compute the refinement of the intervals within...
// How would we do this?

// E.g., if we saw {x > 10} x == 4,
// then we need to resolve the guard......
// That is, we would have a context for this guard.
// When we see something like `x > y`,
// then that means that our interval for `x` becomes [y_min + 1, x_max]
// and our interval for `y` becomes [y_min, x_max - 1]
// What if we had something like `{a + b < 10}`?
//
// Here, we need to "solve" for each variable,
// like so:
// a < 10 - b
// That means that the interval for `a` becomes [a_min, 10 - b_max]
// The interval for `b` becomes [b_min, min(10 - a_min, b_max)] U [TYPE_MAX - a_max, TYPE_MAX] INTERSECT B
// However, the problem now involves the potential for wrapping.
pub(crate) fn resolve_interval_in_context<'terms>(
    expr: &AbcExpression,
    term_map: &'terms FastHashMap<Term, IntervalKind>,
) -> Result<FastHashMap<&'terms Term, IntervalKind>, TranslateError> {
    // The keys for the terms in the new map are just references to the old one.
    // The intervals are the refined intervals based on the predicate.
    // Basically, we refine the intervals based on what we see in the predicate.

    // If we see, e.g., (x >= y), (y >= 15),
    // Then we resolve `x` based on `y`...
    match expr {
        _ => {
            todo!();
        }
    }
    todo!();
}

/// Given an expression, resolves it to an interval based on the term map.
pub(crate) fn resolve_expression_to_interval(
    expr: &AbcExpression,
    term_map: &FastHashMap<Term, IntervalKind>,
) -> Result<IntervalKind, TranslateError> {
    match expr {
        _ => {
            todo!();
        }
    }
    todo!();
}

pub(crate) fn refine_intervals_with_assumptions(
    term_map: &mut FastHashMap<Term, IntervalKind>,
    mut assumptions: Vec<Constraint>,
) -> Result<(), TranslateError> {
    use std::collections::hash_map::Entry;
    // Assume that about half of the assumptions are static assignments.
    let mut unhandled: Vec<Constraint> = Vec::with_capacity(assumptions.len() / 2);
    // We need the LHS and RHS.
    // Step 1: Go through the assumptions that assign to literals.

    for assumption in assumptions.drain(..) {
        // snapshot the operation for this assumption.
        let op: ConstraintOp = match assumption {
            Constraint::Assign { .. } => ConstraintOp::Assign,
            Constraint::Cmp { op, .. } => ConstraintOp::Cmp(op),
            Constraint::Identity { guard, term } => todo!(),
        };
        let (Constraint::Assign {
            ref guard,
            ref lhs,
            ref rhs,
        }
        | Constraint::Cmp {
            ref guard,
            ref lhs,
            ref rhs,
            ..
        }) = assumption
        else {
            unhandled.push(assumption);
            continue;
        };

        // Here, we have `lhs` and `rhs`
        // First, see if we can get a range for `rhs`.
        let rhs_as_interval = match *rhs {
            Term::Literal(l) => match l {
                Literal::U32(l) => IntervalKind::U32(U32Interval::new_concrete(l, l)),
                Literal::I32(l) => IntervalKind::I32(I32Interval::new_concrete(l, l)),
                _ => {
                    unhandled.push(assumption);
                    continue;
                }
            },
            Term::Predicate(ref p) => match p.as_ref() {
                Predicate::True => IntervalKind::Bool(BoolInterval::True),
                Predicate::False => IntervalKind::Bool(BoolInterval::False),
                _ => {
                    unhandled.push(assumption);
                    continue;
                }
            },
            Term::Empty => IntervalKind::Empty,
            _ => {
                unhandled.push(assumption);
                continue;
            }
        };

        // Every single term is assigned to at most ONCE.
        // If we see a second update, then we report a violation.
        //
        // Here, we assign the interval of `x` to be the intersection of what it was before and
        // what it is now.
        match op {
            ConstraintOp::Assign => |e: &mut std::collections::hash_map::OccupiedEntry<
                Term,
                IntervalKind,
            >| e.get_mut().intersect(&rhs_as_interval),
        }
        match term_map.entry(lhs.clone()) {
            Entry::Vacant(mut e) => {
                e.insert(rhs_as_interval.clone());
            }
            Entry::Occupied(ref mut e) => e.get_mut().intersect(&rhs_as_interval)?,
        };

        // Okay, so our goal here is to try to resolve all of the constraints
        // from the
        // fr
    }

    // At this point, we will have done the insertions on everything that is a literal.

    // After we've got done with assigning to all of these literals...
    // Wait, what if we do a ref cell?

    // That is, our intervals end up being ref cells. This allows us to

    Ok(())
}

/// Translates the constraints in the module.
pub fn translate(module: ConstraintModule) -> Result<(), TranslateError> {
    // Step 1: Convert each term in the module into an interval. We refine the interval as we see more
    // constraints.

    // Each term is given an Interval. These begin at `top`.

    // First, we need the type of each term so we can assign them default intervals.

    // Every `value` gets narrowed.

    let mut term_map = initialize_intervals(&module.type_map)?;

    // We need to remove the assumptions from the map as we complete them.

    // Now, we need to iterate through the module's constraints.

    let unhandled_globals =
        refine_intervals_with_assumptions(&mut term_map, module.global_assumptions.clone());

    Ok(())
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
            &IntervalKind::U32(U32Interval::new_top())
        )
    }
}

#[cfg(test)]
mod test_initialize_intervals {
    #![allow(unused_imports)]
    use std::num::NonZeroU32;

    use super::*;
    use rstest::{fixture, rstest};

    use crate::{ABC_I32_TY, ABC_U32_TY};

    #[fixture]
    fn u32_handle() -> Handle<AbcType> {
        Handle::new(ABC_U32_TY)
    }

    #[fixture]
    fn i32_handle() -> Handle<AbcType> {
        Handle::new(ABC_I32_TY)
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
            &IntervalKind::U32(U32Interval::new_top())
        )
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
        println!("{:?}", result);

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
            ty: Handle::new(ABC_U32_TY),
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
            &IntervalKind::U32(U32Interval::new_top())
        )
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
                    ty: Handle::new(crate::ABC_U32_TY),
                },
                StructField {
                    name: "field2".to_string(),
                    ty: Handle::new(crate::ABC_I32_TY),
                },
            ],
        };

        // make a term for the struct, and put it in the type map.
        let term = Term::new_var("s");

        test_module
            .type_map
            .insert(term.clone(), Handle::new(struct_ty));

        let result = initialize_intervals(&test_module.type_map).unwrap();

        let field1_term = Term::new_struct_access(&term, "field1".to_string(), u32_handle.clone());
        let field2_term = Term::new_struct_access(&term, "field2".to_string(), i32_handle.clone());

        assert_eq!(
            result.get(&field1_term).unwrap(),
            &IntervalKind::U32(U32Interval::new_top())
        );

        assert_eq!(
            result.get(&field2_term).unwrap(),
            &IntervalKind::I32(I32Interval::new_top())
        );
    }
}

#[cfg(test)]
mod test_refine {
    #![allow(unused_imports)]
    use std::num::NonZeroU32;

    use super::*;
    use crate::{ABC_I32_TY, ABC_U32_TY};
    use rstest::{fixture, rstest};

    #[fixture]
    fn u32_handle() -> Handle<AbcType> {
        Handle::new(ABC_U32_TY)
    }

    #[fixture]
    fn i32_handle() -> Handle<AbcType> {
        Handle::new(ABC_I32_TY)
    }

    #[rstest]
    fn test_simple_refinement() {
        let mut term_map = FastHashMap::default();

        let term = Term::new_var("x");

        term_map.insert(term.clone(), IntervalKind::U32(U32Interval::top()));

        let assumptions = vec![Constraint::Assign {
            guard: None,
            lhs: term.clone(),
            rhs: Term::Literal(Literal::U32(5)),
        }];

        refine_intervals_with_assumptions(&mut term_map, assumptions).unwrap();

        assert_eq!(
            term_map.get(&term).unwrap(),
            &IntervalKind::U32(U32Interval::new_concrete(5, 5))
        );

        // After adding the constraint, our term map for `x` should be narrowed to `5`.
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

        test_module.global_constraints.push(Constraint::Cmp {
            guard: Some(Handle::new(less_than_5)),
            lhs: term_y,
            rhs: Term::Literal(Literal::U32(10)),
            op: crate::CmpOp::Lt,
        });

        refine_intervals_with_assumptions(term_map, assumptions);

        translate(test_module).unwrap();

        // Now we do it
    }
}
