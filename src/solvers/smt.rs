/*! Solves the array bounds checking constraints using smt as a backend

*/
// turn off during development..
#![allow(unused, dead_code)]

use core::prelude::v1;
use std::collections::hash_map::Entry;
use std::default;

use z3::ast::{Ast, BV};
use z3::{Solver, SortKind};

use crate::{helper, ConstraintHelper, FastHashMap, Literal, Predicate};

use crate::{
    AbcExpression, AbcScalar, AbcType, BinaryOp, CmpOp, Constraint, Handle, Summary, Term, Var,
};

#[derive(Clone, Debug)]
pub(crate) enum Z3SortWrapper<'ctx> {
    Bool {
        value: z3::ast::Bool<'ctx>,
    },
    // All integer types are represented as bitvectors.
    // We need to know the size and whether it is signed
    // so that it can be used in expressions.
    BV {
        value: z3::ast::BV<'ctx>,
        size: u32,
        signed: bool,
    },
    // We probably don't have any integers leftover....
    Int {
        value: z3::ast::Int<'ctx>,
    },
}

impl<'ctx> Z3SortWrapper<'ctx> {
    pub(crate) fn get_bool(&self) -> Option<&z3::ast::Bool<'ctx>> {
        match self {
            Z3SortWrapper::Bool { value } => Some(value),
            _ => None,
        }
    }

    pub(crate) fn get_bv(&self) -> Option<&z3::ast::BV<'ctx>> {
        match self {
            Z3SortWrapper::BV { value, .. } => Some(value),
            _ => None,
        }
    }

    pub(crate) fn get_int(&self) -> Option<&z3::ast::Int<'ctx>> {
        match self {
            Z3SortWrapper::Int { value } => Some(value),
            _ => None,
        }
    }
}

impl<'ctx> Z3SortWrapper<'ctx> {
    /// Get the context of the underlying z3 term.
    fn get_ctx(&self) -> &'ctx z3::Context {
        match self {
            Z3SortWrapper::Bool { value } => value.get_ctx(),
            Z3SortWrapper::BV { value, .. } => value.get_ctx(),
            Z3SortWrapper::Int { value } => value.get_ctx(),
        }
    }

    fn get_sort_name(&self) -> String {
        match self {
            Z3SortWrapper::Bool { .. } => "Bool".to_string(),
            Z3SortWrapper::BV { signed: true, .. } => "signed BV".to_string(),
            Z3SortWrapper::BV { signed: false, .. } => "unsigned BV".to_string(),
            Z3SortWrapper::Int { .. } => "Int".to_string(),
        }
    }
}

// Creates binary comparison functions, which have similar patterns and differ only in the name of the comparison method.
macro_rules! impl_binary_comparison {
    // this form is for versions where the operation can be applied to bitvectors and integers, but not bools.
    ($self:expr, $other:expr, $sop:ident, $uop:ident, $iop:ident, $opname:literal) => {{
        match ($self, $other) {
            (
                Z3SortWrapper::BV {
                    value: ref v1,
                    size: ref s1,
                    signed: ref sign1,
                },
                Z3SortWrapper::BV {
                    value: ref v2,
                    size: ref s2,
                    signed: ref sign2,
                },
            ) if sign1 == sign2 => Ok(if *sign1 { v1.$sop(&v2) } else { v1.$uop(&v2) }),
            (Z3SortWrapper::Int { value: v1 }, Z3SortWrapper::Int { value: v2 }) => {
                Ok(v1.$iop(&v2))
            }

            _ => Err(SolverError::BinaryOpError(
                $opname.to_string(),
                $self.get_sort_name(),
                $other.get_sort_name(),
            )),
        }
    }};
    // This form is for versions where the operation can be applied to booleans, bitvectors, and integers.
    ($self:expr, $other:expr, $boolop:ident, $sop:ident, $uop:ident, $iop:ident, $opname:literal) => {{
        match ($self, $other) {
            (Z3SortWrapper::Bool { value: v1 }, Z3SortWrapper::Bool { value: v2 }) => {
                Ok(Z3SortWrapper::Bool {
                    value: v1.$sop(&v2),
                })
            }
            (
                Z3SortWrapper::BV {
                    value: ref v1,
                    size: ref s1,
                    signed: ref sign1,
                },
                Z3SortWrapper::BV {
                    value: ref v2,
                    size: ref s2,
                    signed: ref sign2,
                },
            ) if sign1 == sign2 => Ok(if *sign1 { v1.$sop(&v2) } else { v1.$uop(&v2) }),
            (Z3SortWrapper::Int { value: v1 }, Z3SortWrapper::Int { value: v2 }) => {
                Ok(v1.$iop(&v2))
            }
            _ => Err(SolverError::BinaryOpError(
                $opname.to_string(),
                $self.get_sort_name(),
                $other.get_sort_name(),
            )),
        }
    }};
}

macro_rules! impl_binary_operation {
    // This version is for ops that have the same method call for both signed and unsigned bitvectors.
    ($self:expr, $other:expr, $bvop:ident, $iop:tt, $opname:literal) => {
        match ($self, $other) {
            (
                Z3SortWrapper::BV {
                    value: ref v1,
                    size: ref s1,
                    signed: ref sign1,
                },
                Z3SortWrapper::BV {
                    value: ref v2,
                    size: ref s2,
                    signed: ref sign2,
                },
            ) if sign1 == sign2 => Ok(Z3SortWrapper::BV {
                value: v1.$bvop(&v2),
                size: *s1,
                signed: *sign1,
            }),
            (Z3SortWrapper::Int { value: v1 }, Z3SortWrapper::Int { value: v2 }) => {
                Ok(Z3SortWrapper::Int {
                    value: v1 $iop v2,
                })
            }
            _ => Err(SolverError::BinaryOpError(
                $opname.to_string(),
                $self.get_sort_name(),
                $other.get_sort_name(),
            )),
        }
    };
}

impl<'ctx> Z3SortWrapper<'ctx> {
    // Implement the comparison functions...
    /// Return a new `SortWrapper` holding the result of applying the `lt` operation to
    /// `self` and `other`.
    ///
    /// # Errors
    /// [`SolverError::BinaryOpError`] if `self` and `other` are either different variants
    /// or if they are both bitvector variants of different signedness.
    ///
    fn lt(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        impl_binary_comparison!(self, other, bvslt, bvult, gt, "<")
    }
    fn le(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        impl_binary_comparison!(self, other, bvsle, bvule, le, "<=")
    }
    fn gt(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        impl_binary_comparison!(self, other, bvsgt, bvugt, gt, ">")
    }
    fn ge(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        impl_binary_comparison!(self, other, bvsge, bvuge, ge, ">=")
    }
    fn eq_(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        impl_binary_comparison!(self, other, _eq, _eq, _eq, "==")
    }
    fn neq(&self, other: &Self) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        match (self, other) {
            (Z3SortWrapper::Bool { value: v1 }, Z3SortWrapper::Bool { value: v2 }) => {
                Ok(v1._eq(v2).not())
            }
            (
                Z3SortWrapper::BV {
                    value: ref v1,
                    size: ref s1,
                    signed: ref sign1,
                },
                Z3SortWrapper::BV {
                    value: ref v2,
                    size: ref s2,
                    signed: ref sign2,
                },
            ) if sign1 == sign2 => Ok(if *sign1 { v1._eq(v2) } else { v1._eq(v2) }.not()),
            (Z3SortWrapper::Int { value: v1 }, Z3SortWrapper::Int { value: v2 }) => {
                Ok(v1._eq(v2).not())
            }
            _ => Err(SolverError::BinaryOpError(
                "!=".to_string(),
                self.get_sort_name(),
                other.get_sort_name(),
            )),
        }
    }
    fn mul(&self, other: &Self) -> Result<Self, SolverError> {
        impl_binary_operation!(self, other, bvmul, *, "multiplication")
    }

    /// Return a new `SortWrapper` holding the result of applying the `add` operation to
    /// `self` and `other`.
    ///
    /// # Errors
    /// [`SolverError::BinaryOpError`] if `self` and `other` are either different variants
    /// or if they are both bitvector variants of different signedness.
    fn add(&self, other: &Self) -> Result<Self, SolverError> {
        impl_binary_operation!(self, other, bvadd, +, "addition")
    }

    /// Return a new `SortWrapper` holding the result of applying the `sub` operation to
    /// `self` and `other`.
    ///
    /// # Errors
    /// [`SolverError::BinaryOpError`] if `self` and `other` are either different variants
    /// or if they are both bitvector variants of different signedness.
    fn sub(&self, other: &Self) -> Result<Self, SolverError> {
        impl_binary_operation!(self, other, bvsub, -, "subtraction")
    }

    /// Return a new `SortWrapper` holding the result of applying the `div` operation to
    /// `self` and `other`.
    ///
    /// # Errors
    /// [`SolverError::BinaryOpError`] if `self` and `other` are either different variants
    /// or if they are both bitvector variants of different signedness.
    fn floordiv(&self, other: &Self) -> Result<Self, SolverError> {
        impl_binary_operation!(self, other, bvsdiv, /, "division")
    }
    /// Return a new `SortWrapper` holding the result of applying the `mod` operation to
    /// `self` and `other`.
    ///
    /// # Errors
    /// [`SolverError::BinaryOpError`] if `self` and `other` are either different variants
    /// or if they are both bitvector variants of different signedness.
    fn r#mod(&self, other: &Self) -> Result<Self, SolverError> {
        impl_binary_operation!(self, other, bvsrem, %, "modulus")
    }
}

// Quick type alias...
type Z3TermType<'ctx> = std::rc::Rc<Z3SortWrapper<'ctx>>;

impl<'ctx> From<z3::ast::Bool<'ctx>> for Z3SortWrapper<'ctx> {
    fn from(value: z3::ast::Bool<'ctx>) -> Self {
        Self::Bool { value }
    }
}

#[derive(Debug, Clone)]
struct AbcSolver<'ctx> {
    pub(crate) solver: z3::Solver<'ctx>,
    helper: &'ctx ConstraintHelper,
    summary: Handle<Summary>,

    pub(self) term_map: FastHashMap<Term, Z3TermType<'ctx>>,

    pub(self) expression_map: FastHashMap<Handle<AbcExpression>, Z3TermType<'ctx>>,
    pub(self) predicate_map: FastHashMap<Handle<Predicate>, std::rc::Rc<z3::ast::Bool<'ctx>>>,
    pub(self) var_map: FastHashMap<Handle<Var>, Z3TermType<'ctx>>,
}

impl<'ctx> AbcSolver<'ctx> {
    fn new(
        solver: z3::Solver<'ctx>,
        helper: &'ctx ConstraintHelper,
        summary: Handle<Summary>,
    ) -> Self {
        // make a new, empty hash map for resolved terms
        Self {
            solver,
            helper,
            summary,
            term_map: FastHashMap::default(),
            expression_map: FastHashMap::default(),
            predicate_map: FastHashMap::default(),
            var_map: FastHashMap::default(),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum SolverError {
    #[error("Empty expression used in predicate.")]
    EmptyExpression,
    #[error("Dynamic expression couldn't cast...")]
    CastError,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Operation {0} can't work with {1} and {2}")]
    BinaryOpError(String, String, String),
    #[error("Missing type for term.")]
    MissingType,
}

impl<'ctx> AbcSolver<'ctx> {
    fn comparison_to_bool(&self, op: CmpOp, lhs: &Term, rhs: &Term) -> z3::ast::Bool<'ctx> {
        todo!()
    }
    fn predicate_to_z3_bool(&self, p: &Predicate) -> z3::ast::Bool<'ctx> {
        match p {
            Predicate::True => z3::ast::Bool::from_bool(self.solver.get_context(), true),
            Predicate::False => z3::ast::Bool::from_bool(self.solver.get_context(), false),
            Predicate::Not(p) => self.predicate_to_z3_bool(p.as_ref()).not(),
            Predicate::Comparison(op, lhs, rhs) => self.comparison_to_bool(*op, lhs, rhs),
            Predicate::And(a, b) => z3::ast::Bool::and(
                self.solver.get_context(),
                &[
                    &self.predicate_to_z3_bool(a.as_ref()),
                    &self.predicate_to_z3_bool(b.as_ref()),
                ],
            ),
            Predicate::Or(a, b) => z3::ast::Bool::or(
                self.solver.get_context(),
                &[
                    &self.predicate_to_z3_bool(a.as_ref()),
                    &self.predicate_to_z3_bool(b.as_ref()),
                ],
            ),
            Predicate::Unit(p) => unimplemented!("Unit predicate"),
        }
    }

    fn create_equality(&mut self, lhs: &Term, rhs: &Term) -> z3::ast::Bool {
        // Check if we already have Terms in our Term mapping.
        todo!()
    }

    // Creates a fresh term of the specified type.
    fn fresh_term_of_type(&self, ty: Handle<AbcType>, prefix: &str) -> Z3SortWrapper<'ctx> {
        match ty.as_ref() {
            AbcType::Scalar(AbcScalar::Uint(size)) => Z3SortWrapper::BV {
                value: z3::ast::BV::fresh_const(
                    self.solver.get_context(),
                    prefix,
                    u32::from(*size) * 8u32,
                ),
                size: u32::from(*size) * 8u32,
                signed: false,
            },
            AbcType::Scalar(AbcScalar::Sint(size)) => Z3SortWrapper::BV {
                value: z3::ast::BV::fresh_const(
                    self.solver.get_context(),
                    prefix,
                    u32::from(*size) * 8u32,
                ),
                size: u32::from(*size) * 8u32,
                signed: true,
            },
            AbcType::Scalar(AbcScalar::Bool) => Z3SortWrapper::Bool {
                value: z3::ast::Bool::fresh_const(self.solver.get_context(), prefix),
            },
            _ => todo!(),
        }
    }

    // To handle structs, we probably need a datatype.
    // This is going to be pretty annoying, though...
    fn get_or_create_term(
        &mut self,
        term: &Term,
        term_type_map: &FastHashMap<Term, Handle<AbcType>>,
    ) -> Result<Z3TermType<'ctx>, SolverError> {
        // We need to know the type that the term resolves to...
        // That way, we know whether we can really add two terms..
        // If we start with constraints, then we only need to worry about
        // the terms that appear in the constraints.

        // If this term is a literal...
        if self.term_map.get(term).is_some() {
            return Ok(self.term_map[term].clone());
        }

        let newterm: Z3TermType<'ctx> = match term {
            Term::Literal(Literal::I32(i)) => Ok(Z3SortWrapper::BV {
                value: z3::ast::BV::from_i64(self.solver.get_context(), i64::from(*i), 32),
                size: 32,
                signed: true,
            }
            .into()),
            Term::Literal(Literal::I64(i)) => Ok(Z3SortWrapper::BV {
                value: z3::ast::BV::from_i64(self.solver.get_context(), *i, 64),
                size: 64,
                signed: true,
            }
            .into()),
            Term::Literal(Literal::U64(u)) => Ok(Z3SortWrapper::BV {
                value: z3::ast::BV::from_u64(self.solver.get_context(), *u, 64),
                size: 64,
                signed: false,
            }
            .into()),
            Term::Literal(Literal::U32(u)) => Ok(Z3SortWrapper::BV {
                value: z3::ast::BV::from_u64(self.solver.get_context(), u64::from(*u), 32),
                size: 32,
                signed: false,
            }
            .into()),
            Term::Predicate(p) => {
                // We know that the term *is* a bool...
                let pred = self.predicate_to_z3_bool(p.as_ref());
                Ok(Z3SortWrapper::Bool { value: pred }.into())
            }
            Term::Empty => Err(SolverError::EmptyExpression),
            Term::Var(v) => {
                // The var *better* have a type!
                match term_type_map.get(term) {
                    None => Err(SolverError::MissingType),
                    Some(ty) => Ok(self.fresh_term_of_type(ty.clone(), &v.name).into()),
                }
            }
            _ => Err(SolverError::NotImplemented(
                "get_or_create_term".to_string(),
            )),
        }?;

        self.term_map.insert(term.clone(), newterm.clone());
        Ok(newterm)
    }

    fn get_or_create_expression(
        &mut self,
        expression: &AbcExpression,
    ) -> Result<z3::ast::Dynamic<'ctx>, SolverError> {
        todo!()
    }

    fn get_or_create_predicate(
        &mut self,
        expression: &Predicate,
    ) -> Result<z3::ast::Bool<'ctx>, SolverError> {
        todo!()
    }

    fn get_or_create_var(&mut self, var: &Var) -> Result<z3::ast::Dynamic<'ctx>, SolverError> {
        // We *need* the type of the var
        todo!()
    }
}

// A wrapper with a term and its inner sort...

// For now, we don't support structs.
// but, they will probably use datatypes..

impl helper::ConstraintHelper {
    pub fn solve_using(
        &self,
        backend: super::SolverBackends,
        entry: crate::Handle<Summary>,
    ) -> Result<(), SolverError> {
        match backend {
            super::SolverBackends::Z3 => self.solve_z3(&entry),
        }
    }

    fn solve_z3(&self, entry: &crate::Handle<Summary>) -> Result<(), SolverError> {
        // make a new context
        let cfg = z3::Config::new();
        let context = z3::Context::new(&cfg);

        let mut solver = AbcSolver::new(z3::Solver::new(&context), self, entry.clone());

        // We need to go through every single variable and mark its type here so that we know
        // how to make expressions in z3.....

        // Start with the global terms...

        // Start with constraints.

        // We need to check the satisfiability of each constraint.

        // Begin with the term type map...

        for (term, ty) in &self.term_type_map {
            // We need to resolve these terms to an expression...
            println!("Term: {term:?} Type: {ty:?}");
        }

        for constraint in self
            .global_constraints
            .iter()
            .chain(entry.constraints.iter())
        {
            // Push a context
            solver.solver.push();
            let (guard, constraint) = match constraint {
                Constraint::Assign { guard, lhs, rhs } => {
                    let lhs = solver.get_or_create_term(lhs, &self.term_type_map)?;
                    let rhs = solver.get_or_create_term(rhs, &self.term_type_map)?;
                    (guard, lhs.eq_(&rhs)?)
                }
                Constraint::Cmp {
                    guard,
                    lhs,
                    op,
                    rhs,
                } => {
                    let lhs = solver.get_or_create_term(lhs, &self.term_type_map)?;
                    let rhs = solver.get_or_create_term(rhs, &self.term_type_map)?;
                    (
                        guard,
                        match op {
                            CmpOp::Lt => lhs.lt(&rhs),
                            CmpOp::Leq => lhs.le(&rhs),
                            CmpOp::Gt => lhs.gt(&rhs),
                            CmpOp::Geq => lhs.ge(&rhs),
                            CmpOp::Eq => lhs.eq_(&rhs),
                            CmpOp::Neq => lhs.neq(&rhs),
                        }?,
                    )
                    // Figure out lhs...
                }
                #[allow(clippy::match_wildcard_for_single_variants)]
                _ => unimplemented!("Only support comparison constraints right now."),
            };
            if let Some(guard) = guard {
                // We visit the predicate
                let guard = solver.get_or_create_predicate(guard)?;
            } else {
                solver.solver.assert(&constraint);
                // We know this is always going to be a boolean...
            }
            solver.solver.pop(1);

            // We need to assert the constraint..
            // Visit the
        }

        // Let's go through the assumptions first...

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::ConstraintInterface;
    use rstest::{fixture, rstest};

    use super::*;

    /// Test the solver with these constraints:
    /// x = 5
    /// x < 10
    #[rstest]
    fn test_solver_simple() {
        // First, create a constraint
        let mut helper = ConstraintHelper::default();
        // Now, declare 'x'
        let x = helper
            .declare_var(Var {
                name: "x".to_string(),
            })
            .unwrap();

        let myu32 = helper
            .declare_type(AbcType::Scalar(AbcScalar::Uint(4)))
            .unwrap();
        helper.mark_type(x.clone(), myu32).unwrap();

        helper.begin_summary("test".to_string(), 0).unwrap();

        // Now that we have marked the type of x, assert that it is equal to 5
        helper
            .add_assumption(x.clone(), crate::ConstraintOp::Assign, 5u32.into())
            .unwrap();

        // Now, assert that x is less than 10.
        helper
            .add_constraint(x.clone(), crate::ConstraintOp::Cmp(CmpOp::Lt), 10u32.into())
            .unwrap();

        let summary = helper.end_summary().unwrap();

        // Now, solve it with the smt backend.

        helper
            .solve_using(crate::solvers::SolverBackends::Z3, summary)
            .unwrap();
    }

    // Generate tests for each

    macro_rules! comparison_test {
        ($name:ident, $op:ident, $lhs:expr, $rhs:expr, $expected:path) => {
            #[rstest]
            fn $name() {
                let ctx = z3::Context::new(&z3::Config::default());
                let a = Z3SortWrapper::BV {
                    value: z3::ast::BV::from_i64(&ctx, $lhs, 32),
                    size: 32,
                    signed: true,
                };
                let b = Z3SortWrapper::BV {
                    value: z3::ast::BV::from_i64(&ctx, $rhs, 32),
                    size: 32,
                    signed: true,
                };
                let result = a.$op(&b).unwrap();
                // Assert they are ctually true
                let solver = z3::Solver::new(&ctx);
                solver.assert(&result);
                assert_eq!(solver.check(), $expected);
            }
        };
    }

    comparison_test!(test_eq_pass, eq_, 5, 5, z3::SatResult::Sat);

    comparison_test!(test_eq_fail, eq_, 5, 6, z3::SatResult::Unsat);

    comparison_test!(test_lt_pass, lt, 5, 6, z3::SatResult::Sat);

    comparison_test!(test_lt_fail, lt, 6, 5, z3::SatResult::Unsat);

    comparison_test!(test_gt_pass, gt, 6, 5, z3::SatResult::Sat);

    comparison_test!(test_gt_fail, gt, 5, 6, z3::SatResult::Unsat);

    comparison_test!(test_ge_pass, ge, 6, 5, z3::SatResult::Sat);

    comparison_test!(test_ge_fail, ge, 5, 6, z3::SatResult::Unsat);

    comparison_test!(test_le_pass, le, 5, 6, z3::SatResult::Sat);

    comparison_test!(test_le_fail, le, 6, 5, z3::SatResult::Unsat);

    // comparison_test!(test_le, le, 5, 6);

    // comparison_test!(test_neq, neq, 5, 6);
}
