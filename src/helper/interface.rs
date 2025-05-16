// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

// Module uses the "helper" name.
#![allow(clippy::module_name_repetitions)]

use serde::{Deserialize, Serialize};

use log::{info as log_info, trace as log_trace, warn as log_warning};

use crate::{
    add_assumption_impl,
    solvers::interval::{translator::IntervalKind, BoolInterval, SolverResult},
    AbcExpression, AbcScalar, AbcType, Assumption, AssumptionOp, BinaryOp, CmpOp, Constraint,
    ConstraintId, ConstraintOp, FastHashMap, FastHashSet, Handle, Predicate, SubstituteTerm,
    Summary, Term, Var, CONSTRAINT_LIMIT, NONETYPE, RET, SUMMARY_LIMIT,
};

#[derive(Clone, Debug, thiserror::Error, Serialize, Deserialize)]
pub enum ConstraintError {
    #[error("Predicate stack is empty")]
    PredicateStackEmpty,
    #[error("Active summary is not set")]
    SummaryError,
    #[error("Attempt to declare the type of a Term a second time.")]
    DuplicateType,
    #[error("{0} is not yet implemented [Line: {1}, file: {2}]")]
    NotImplemented(String, u32, &'static str),
    #[error("Attempt to end a loop when no loop is active")]
    NotInLoopContext,
    #[error("Maximum loop depth exceeded")]
    MaxLoopDepthExceeded,
    #[error("Empty Term in disallowed context.")]
    EmptyTerm,
    #[error("Invalid number of arguments passed to call")]
    InvalidArguments,
    #[error("Attempt to assign a return value to a function with no return value")]
    NoReturnValue,
    #[error("Unsupported loop operation")]
    UnsupportedLoopOperation,
    #[error("Unsupported term in loop condition")]
    UnsupportedLoopCondition(Term),
    #[error("Unsupported type: {0}")]
    UnsupportedType(&'static str),
    #[error("Terms must have at most one assumption.")]
    DuplicateAssumption,
    #[error("Error solving constraints: {0}")]
    SolverError(#[from] crate::solvers::interval::translator::SolverError),
    #[error("Maximum constraint count exceeded.")]
    ConstraintLimitExceeded,
    #[error("Maximum summary count exceeded")]
    SummaryLimitExceeded,
    #[error("No summary with that id exists.")]
    InvalidSummary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct SummaryId(pub(crate) usize);

impl std::fmt::Display for SummaryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl From<u32> for SummaryId {
    fn from(id: u32) -> Self {
        SummaryId(id as usize)
    }
}
impl From<usize> for SummaryId {
    fn from(id: usize) -> Self {
        SummaryId(id)
    }
}

impl std::ops::Deref for SummaryId {
    type Target = usize;
    fn deref(&self) -> &usize {
        &self.0
    }
}

#[allow(clippy::missing_errors_doc)]
pub trait ConstraintInterface<Marker = ()>
where
    Marker: Clone + std::hash::Hash + Eq + std::fmt::Debug + Copy,
{
    /// Can be used to reference an element of the constraint system.
    type Handle<T>: Clone;
    /// The error type for the constraint interface
    type E;

    /// Get the handle to an empty expression.
    ///
    /// This is provided since empty expression is meant to be a singleton.
    fn empty_expression(&self) -> Term {
        Term::Empty
    }

    /// Get the handle for the `NoneType`
    fn none_type(&self) -> Self::Handle<AbcType>;

    /// Add an argument to the function summary
    ///
    /// Name is the name of the argument.
    /// Returns a var handle.
    fn add_argument(&mut self, name: String, ty: &Self::Handle<AbcType>) -> Result<Term, Self::E>;

    /// Makes a call expression.
    ///
    /// # Arguments
    /// * `func` - A handle to the function that is being invoked
    /// * `args` - The arguments to the function
    /// * `into` - If the result of the function is used, this is the variable that holds it.
    ///
    /// This returns a handle to the expression that allows it to be used in other expressions
    ///
    /// # Errors
    /// The implementation should return an error if the function called does not reside in the arena.
    fn make_call(
        &mut self, func: SummaryId, args: Vec<Term>, into: Option<&Term>,
    ) -> Result<Term, Self::E>;

    /// Add a new constraint to the constraint system. Any active predicates are applied to the constraint to "filter" the domain of its expression.
    ///
    /// The provided `id` is used to identify the constraint, and acts as a bridge between the user of the constraint system and the solver.
    /// Solutions to constriants reference this ID.  It is important to note that when invoking a summary via `make_call`, each of the constraints from
    /// that summary are inserted into the constraint system using the same ID.
    fn add_constraint(
        &mut self, lhs: &Term, op: ConstraintOp, rhs: &Term, id: u32,
    ) -> Result<(), Self::E>;

    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E>;

    /// Declare the variable in the constraint system.
    ///
    /// This passes by value since the variable is replaced with a handle to it.
    fn declare_var(&mut self, name: Var) -> Result<Term, Self::E>;

    /// Mark the type of a variable.
    /// If the type is an array, this will also mark the number of dimensions.
    /// If the type is an array with a fixed size, this will also mark the size of the array.
    fn mark_type(&mut self, term: &Term, ty: &self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the length of a variable. The type of the variable must be a dynamic array.
    fn mark_length(&mut self, var: &Term, dim: u8, size: u64) -> Result<(), Self::E>;

    /// Mark an assumption.
    ///
    /// An assumption is akin to a constraint, except it is not considered a goal.
    /// Anything marked as an assumption is considered to be a "soft" constraint.
    ///
    /// # Errors
    /// Constraints must take on `SSA` form, meaning each term can have at most one assumption associated with it.
    /// However, assumptions that are inequalities may have one of each type (a lower bound and an upper bound), permitting
    /// this method to be used on the same term multiple times.
    ///
    /// To instead `replace` an assumption, use [`ConstraintInterface::replace_assumption`].
    ///
    /// If `Op` is not `Assign`, then `rhs` must be a literal.
    ///
    /// # Examples
    /// ```rs,none
    /// // Assume helper is some object with `ConstraintInterface`
    /// let x = Term::new_var("x");
    /// helper.add_assumption(x, AssumptionOp::Geq, Term::new_literal(0u32)); // Allowed, no assumptions on `x` yet
    /// helper.add_assumption(x, AssumptionOp::Leq, Term::new_literal(14u32)); // Allowed, no upper bound on `x` yet.
    /// helper.add_assumption(x, AssumptionOp::Geq, Term::new_literal(1u32)); // Not permitted, lower bound has already been marked.
    ///
    /// let y = Term::new_var("y");
    /// helper.add_assumption(y, AssumptionOp::Assign, Term::new_literal(0u32)); // Allowed, no assumptions on `y` yet.
    /// helper.add_assumption(y, AssumptionOp::Assign, Term::new_literal(1u32)); // Not permitted, y has already been assigned.
    ///
    /// let z = Term::new_var("z");
    /// let w = Term::new_var("w");
    /// let v = Term::new_var("v");
    ///
    /// helper.add_assumption(z, AssumptionOp::Assign, w); // Allowed, no assumptions on `z` yet.
    /// helper.add_assumption(w, AssumptionOp::Leq, v); // Not permitted, `v` is not a Literal.
    /// ```
    ///
    /// [`ConstraintInterface::replace_assumption`]: ConstraintInterface::replace_assumption
    fn add_assumption(&mut self, lhs: &Term, op: AssumptionOp, rhs: &Term) -> Result<(), Self::E>;

    /// Replace an existing assumption for a Term.
    ///
    /// If there is no assumption for the term, then this is equivalent to [`ConstraintInterface::add_assumption`].
    fn replace_assumption(
        &mut self, lhs: &Term, op: AssumptionOp, rhs: &Term,
    ) -> Result<(), Self::E>;

    /// Begin a predicate block. This indicates to the solver
    /// that all expressions that follow can be filtered by the predicate.
    /// Any constraint that falls within a predicate becomes a soft constraint
    ///
    /// In other words, it would be as if all constraints were of the form ``p -> c``
    /// Nested predicate blocks end up composing the predicates. E.g.,
    /// ``begin_predicate_block(p1)`` followed by ``begin_predicate_block(p2)`` would
    /// mark all constraints as ``p1 && p2 -> c``
    /// That is, when determining if the constraint is violated, the solver
    /// will essentially check p -> c.
    fn begin_predicate_block(&mut self, p: &Term) -> Result<(), Self::E>;

    /// End the active predicate block.
    ///
    /// If there was a return statement within the block, then all future constraints are marked as ``!p -> c``
    ///
    /// # Errors
    /// Returns an error if there is no active predicate block.
    fn end_predicate_block(&mut self) -> Result<(), Self::E>;

    /// A summary block corresponds to a function.
    /// This will return a handle that can be used to access the summary for a function.
    ///
    /// # Errors
    /// Returns an error if there is already an active summary.
    fn begin_summary(&mut self, name: String, nargs: u8) -> Result<(), Self::E>;

    /// Ends the current summary, returning an identifer that can be used to reference it.
    ///
    /// # Errors
    /// Returns an error if there is no active summary.
    fn end_summary(&mut self) -> Result<SummaryId, Self::E>;

    /// Marks a return statement
    ///
    ///
    /// All future constraints that are added will be marked as `[!p] -> [c]`
    ///
    /// When an IR system sees a return statement, the
    /// Proper handling of control flow is difficult.
    /// Rather than require the user of this interface to precisely
    /// model control flow and its impact on the constraint system,
    /// the interface provides a way to mark a point at which the return occurs.
    /// When this occurs within a block, the IR system must end the current block immediately after
    /// the return. If this occured within a predicate block (e.g., an If statement), then
    /// [`ConstraintInterface::end_predicate_block`] must be called immediately after the return.
    ///
    /// At a low-level, what this function does is add a
    /// "global" predicate block to all future statements that appear
    /// The current predicate block is immediately popped.
    /// From then on, any time a predicate block is closed, it is immediately enclosed
    /// within a new predicate block that inverts the conditions that were popped.
    ///
    fn mark_return(&mut self, retval: Option<Term>) -> Result<(), Self::E>;

    fn mark_return_type(&mut self, ty: &Self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the beginning of a loop.
    fn begin_loop(&mut self, condition: &Term) -> Result<(), Self::E>;

    /// Mark the end of a loop.
    fn end_loop(&mut self) -> Result<(), Self::E>;

    /// This is sugar for marking a predicate block with a single entry...
    fn mark_break(&mut self) -> Result<(), Self::E>;

    /// Marks a continue statement
    fn mark_continue(&mut self) -> Result<(), Self::E>;

    /// Mark a variable that is used as a loop counter
    fn mark_loop_variable(
        &mut self,
        var: &Term,  // The variable that is incremented
        init: &Term, // What it is initialized to.
        // What this term is compared to

        // How the term is incremented
        inc_term: &Term,

        inc_op: BinaryOp,
    ) -> Result<(), Self::E>;

    /// Mark a range of values that a term can hold (inclusive).
    ///
    /// # Uses
    /// This method is meant to be used to mark the range of runtine constants, such as
    /// the range of invocation IDs, etc., that a value may have.  
    /// It is not meant to mark induction variables for loops. For that, see
    /// [`ConstraintInterface::mark_loop_variable`].
    ///
    ///
    fn mark_range<T: Into<crate::Literal>>(
        &mut self, var: &Term, min: T, max: T,
    ) -> Result<(), Self::E>;
}

/// Serialize an assumption map. This serializes them as a vector of pairs. Deserialization will work as expected.
#[doc(hidden)]
#[allow(non_snake_case)]
pub(crate) mod AssumptionSerializer {
    use super::FastHashMap;
    use super::{Assumption, AssumptionOp, Term};
    use serde::Serialize;
    use serde::{
        de::{self, Visitor},
        Deserialize,
    };

    #[derive(serde::Serialize, serde::Deserialize)]
    struct TermAssumption {
        pub(crate) term: Term,
        pub(crate) assumption: Assumption,
    }

    pub(crate) fn serialize<S: serde::Serializer>(
        item: &FastHashMap<super::Term, super::Assumption>, serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(item.len()))?;
        for (term, assumption) in item {
            seq.serialize_element(&TermAssumption {
                term: term.clone(),
                assumption: assumption.clone(),
            })?;
        }
        seq.end()
    }

    #[allow(clippy::default_trait_access)]
    pub(crate) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<FastHashMap<Term, Assumption>, D::Error> {
        let vec: Vec<TermAssumption> = Vec::deserialize(deserializer)?;

        let mut built = FastHashMap::with_capacity_and_hasher(vec.len(), Default::default());

        for TermAssumption { term, assumption } in vec {
            built.insert(term, assumption);
        }
        Ok(built)
    }
}

#[doc(hidden)]
#[allow(non_snake_case)]
mod ConstraintModuleDeserializer {
    use super::FastHashMap;
    use super::{AbcType, Handle, Term};

    use serde::Serialize;
    use serde::{
        de::{self, Visitor},
        Deserialize,
    };

    #[derive(serde::Serialize, serde::Deserialize)]
    struct TypeAssumption {
        pub(crate) term: Term,
        pub(crate) ty: AbcType,
    }

    // The serializer for FastHashMap(Term, Vec) will actually serialize this as though
    // it were a vector of tuples.
    pub(super) fn serialize<S: serde::Serializer>(
        item: &FastHashMap<super::Term, super::Handle<AbcType>>, serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(item.len()))?;
        for (term, ty) in item {
            seq.serialize_element(&TypeAssumption {
                term: term.clone(),
                // We serialize a clone of the inner type, not the Handle to it.
                // When deserializing, we maintain a hash of the types to their handles.
                ty: (**ty).clone(),
            })?;
        }
        seq.end()
    }

    #[allow(clippy::default_trait_access)]
    pub(super) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<FastHashMap<Term, Handle<AbcType>>, D::Error> {
        // Deserialize AbcType into the Term.
        let mut term_map: FastHashMap<AbcType, Handle<AbcType>> = FastHashMap::default();

        let vec: Vec<TypeAssumption> = Vec::deserialize(deserializer)?;

        let mut built = FastHashMap::with_capacity_and_hasher(vec.len(), Default::default());

        for term in vec {
            let arc_term = term_map
                .entry(term.ty)
                .or_insert_with_key(|k| Handle::new(k.clone()));
            built.insert(term.term, arc_term.clone());
        }
        Ok(built)
    }
}

#[doc(hidden)]
#[allow(non_snake_case)]
mod TermSetSerializer {
    use super::FastHashSet;
    use super::Term;
    use serde::Serialize;
    use serde::{
        de::{self, Visitor},
        Deserialize,
    };

    pub(crate) fn serialize<S: serde::Serializer>(
        item: &FastHashSet<super::Term>, serializer: S,
    ) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;

        let mut seq = serializer.serialize_seq(Some(item.len()))?;
        for term in item {
            seq.serialize_element(term)?;
        }
        seq.end()
    }

    #[allow(clippy::default_trait_access)]
    pub(crate) fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<FastHashSet<Term>, D::Error> {
        let vec: Vec<Term> = Vec::deserialize(deserializer)?;

        let mut built = FastHashSet::with_capacity_and_hasher(vec.len(), Default::default());

        for term in vec {
            built.insert(term);
        }
        Ok(built)
    }
}

/// A constraint module contains just the information about the constraints generated by the helper.
#[derive(Debug, Serialize, Deserialize)]
pub struct ConstraintModule {
    #[serde(rename = "version")]
    _version: String,
    /// The types of specific terms.
    // N.B. This uses a custom serializer so that we can keep the hash map, while serialize/deserialize as a vector.
    #[serde(with = "ConstraintModuleDeserializer")]
    pub(crate) type_map: FastHashMap<Term, Handle<AbcType>>,
    /// Global assumptions not tied to a function
    #[serde(with = "AssumptionSerializer")]
    pub(crate) global_assumptions: FastHashMap<Term, Assumption>,

    /// Types that bave been declared in the helper.
    pub(crate) types: Vec<Handle<AbcType>>,
    /// Global constraints not tied to a function
    pub(crate) global_constraints: Vec<(Constraint, u32)>,
    /// Summaries contain the set of summaries that have previously been parsed.
    pub(crate) summaries: Vec<Handle<Summary>>,
    #[serde(with = "TermSetSerializer")]
    pub(crate) uniform_vars: FastHashSet<Term>,
}

impl Default for ConstraintModule {
    fn default() -> Self {
        ConstraintModule {
            _version: include_str!("../../version.txt").trim().to_string(),
            type_map: FastHashMap::default(),
            global_assumptions: FastHashMap::default(),
            types: Vec::new(),
            global_constraints: Vec::new(),
            summaries: Vec::new(),
            uniform_vars: FastHashSet::default(),
        }
    }
}

impl ConstraintModule {
    /// Get the number of summaries in the module.
    pub fn get_num_summaries(&self) -> usize {
        self.summaries.len()
    }

    /// Solve the constraints in the module.
    /// This will return a map from the constraint ID to the result of the constraint.
    /// The result is a vector of `IntervalKind` that represents the possible values of the constraint.
    ///
    /// # Errors
    /// Propagates any errors encountered when solving the constraints.
    pub fn solve(
        &self, idx: SummaryId,
    ) -> Result<
        FastHashMap<u32, Vec<SolverResult>>,
        crate::solvers::interval::translator::SolverError,
    > {
        log_info!("Solving constraints for summary {idx}");
        crate::solvers::interval::translator::check_constraints(self, idx)
    }
    #[inline]
    pub fn global_constraints(&self) -> &[(Constraint, u32)] {
        &self.global_constraints
    }

    #[inline]
    pub fn global_assumptions(&self) -> &FastHashMap<Term, Assumption> {
        &self.global_assumptions
    }

    #[inline]
    pub fn summaries(&self) -> &[Handle<Summary>] {
        &self.summaries
    }

    #[inline]
    pub fn types(&self) -> &[Handle<AbcType>] {
        &self.types
    }
}

/// The helper interface
/// The main helper class for the constraint system.
///
/// Tracks a module and each of the constraints, variables, and types it contains.
#[derive(Default)]
pub struct ConstraintHelper {
    /// The stack of the conditions in the current predicate block..
    predicate_stack: Vec<Handle<Predicate>>,

    /// The stack of conditions in the current loop block.
    loop_predicate_stack: Vec<Handle<Predicate>>,

    /// When popping a predicate stack, if we had a return, then we add the inverse of the condition to the global predicate stack's expression list
    ///
    /// For example, if we had
    /// ```wgsl
    /// if (x > 0) {
    ///    y = 1;
    ///    if (x < 10) {
    ///      y_1 = 2;
    ///      return;
    ///    }
    ///    w_1 = 14;
    ///    if (z <= 11 ) {
    ///      w_2 = 9;
    ///      return;
    ///    }
    ///    y_2 = 3;
    /// }
    /// y_3 = 4;
    /// ```
    /// Then we would have the following constraints:
    ///
    /// ```wgsl
    /// {x > 0} y = 1
    /// { x < 10 } y_1 = 2
    /// {!( x < 10 ) } w_1 = 14
    /// {!( x < 10 ) && z <= 11 } w_2 = 9
    /// {!(x < 10 || z <= 11)} y_2 = 3
    /// {!(x < 10 || z <= 11)} y_3 = 4
    /// ```
    /// E.g., if we hit a return, then the new return predicate becomes the `or` of the existing one.
    return_predicate: Option<Handle<Predicate>>,

    /// Contains all of the information needed to interface with the other layers of the constraint system.
    /// This includes summaries, types, and global constraints.
    module: ConstraintModule,

    /// The set of predicates that are appended to all future constraints.
    ///
    /// This is used in conjunction with the `had_return` field.
    /// As soon as a return is marked, its predicate is appended to the `permanent_predicate` field.
    permanent_predicate: Option<Handle<Predicate>>,

    /// Summaries can't be nested, so we only need to track what the active one is.
    /// When we pop a summary, we clear out the predicate stack.
    active_summary: Option<Summary>,

    /// Map from varnames to an ssa counter..
    ssa_map: FastHashMap<String, u32>,

    /// The number of loop layers that are active.
    ///
    /// Incremented when a loop begins, decremented when a loop ends.
    loop_depth: u8,

    statements: Vec<String>,
}

// For some reason, we need to use this instead of derive, otherwise it requires that `T` be `Default`
// which is not actually needed.

impl ConstraintHelper {
    /// Mark a variable as being uniform.
    ///
    /// This is not yet fully implemented.
    ///
    /// # Errors
    /// [`ConstraintError::UnsupportedType`] if the term is not a variable.
    ///
    /// [`ConstraintError::UnsupportedType`]: crate::ConstraintError::UnsupportedType
    pub fn mark_uniform_var(&mut self, term: &Term) -> Result<(), ConstraintError> {
        self.write(format!("uniform {term}"));
        if !term.is_var() {
            return Err(ConstraintError::UnsupportedType(
                "non-variable term being marked as unif",
            ));
        }
        self.module.uniform_vars.insert(term.clone());
        Ok(())
    }
    /// Convert the solution to a vector containing the (id, result) pairs.
    pub fn solution_to_result(
        solution: &FastHashMap<u32, Vec<SolverResult>>,
    ) -> Vec<(u32, SolverResult)> {
        solution
            .iter()
            .map(|(k, v)| {
                let mut result = SolverResult::Yes;
                for kind in v {
                    match kind {
                        SolverResult::No => {
                            return (*k, SolverResult::No);
                        }
                        SolverResult::Maybe => result = SolverResult::Maybe,
                        SolverResult::Yes => {}
                    }
                }
                (*k, result)
            })
            .collect()
    }

    /// Solve the constraints for the function at the given index.
    ///
    /// # Errors
    /// If any part of the constraint system is not well formed, then this will return an error.
    /// If the provided index does not exist, this will also return an error.
    pub fn solve(
        &self, idx: SummaryId,
    ) -> Result<
        FastHashMap<u32, Vec<SolverResult>>,
        crate::solvers::interval::translator::SolverError,
    > {
        #[allow(clippy::useless_conversion)]
        crate::solvers::interval::translator::check_constraints(&self.module, idx)
        // Now, we map the results back to the constraints.
    }
    /// Write the constraints for the module to the provided stream
    ///
    /// # Errors
    /// Propagates any errors encountered when writing to the provided `stream`
    pub fn write_to_stream(
        &self, stream: &mut impl std::io::Write,
    ) -> Result<usize, std::io::Error> {
        stream.write_all(self.statements.join("\n").as_bytes())?;
        stream.write("\n".as_bytes())
    }

    /// Helper method used to denote the type of the term as having the same type as `other`.
    /// If `other` does not have a type, then this method does nothing.
    ///
    /// # Errors
    /// [`TypeMismatch`] if `term` already has a type and it is different from `other`'s type.
    ///
    /// [`TypeMismatch`]: crate::ConstraintError::TypeMismatch
    pub(crate) fn mark_type_as_other(
        &mut self, term: &Term, other: &Term,
    ) -> Result<(), ConstraintError> {
        // Get whatever type `other` is in the global type map.
        let ty = match self.module.type_map.get(other) {
            Some(ty) => ty.clone(),
            None => return Ok(()),
        };
        match self.module.type_map.entry(term.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) if entry.get() == &ty => Ok(()),
            std::collections::hash_map::Entry::Occupied(_) => Err(ConstraintError::DuplicateType),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(ty.clone());
                Ok(())
            }
        }
    }

    /// Access the inner constraint module.
    ///
    /// Meant only to be used for serialization.
    pub fn get_module(&self) -> &ConstraintModule {
        &self.module
    }

    /// Get a mutable reference to the active summary's assumptions, or the global assumptions if there is no active summary.
    fn get_cur_assumption_vec(&mut self) -> &mut FastHashMap<Term, Assumption> {
        if let Some(ref mut summary) = self.active_summary {
            &mut summary.assumptions
        } else {
            &mut self.module.global_assumptions
        }
    }

    /// Get a mutable reference to the active summary's constraints, or the global constraints if there is no active summary.
    fn get_cur_constraint_vec(&mut self) -> &mut Vec<(Constraint, u32)> {
        if let Some(ref mut summary) = self.active_summary {
            &mut summary.constraints
        } else {
            &mut self.module.global_constraints
        }
    }

    /// Get a fresh `@ret` variable
    ///
    /// RET variables are special, but they still need a unique suffix in order to be properly bound.
    /// Otherwise, we will have multiple `@ret`....
    ///
    /// # Panics
    /// If the counter for the variable name would overflow.
    fn fresh_ret_var<T: AsRef<str>>(&mut self, postfix: Option<T>) -> Var {
        let mut s = String::from("@ret");
        if let Some(postfix) = postfix {
            s.push('_');
            s.push_str(postfix.as_ref());
        }
        let counter = self.ssa_map.entry(s.clone()).or_insert(0);
        if *counter == u32::MAX {
            panic!("Variable counter overflow");
        } else if counter != &0 {
            *counter += 1;
            s.push('$');
            s.push_str(&counter.to_string());
        }

        Var { name: s }
    }

    fn fresh_var_from_map<T: AsRef<str>>(
        ssa_map: &mut FastHashMap<String, u32>, postfix: Option<T>,
    ) -> Var {
        let mut s = String::from("@ret");
        if let Some(postfix) = postfix {
            s.push('_');
            s.push_str(postfix.as_ref());
        }
        let counter = ssa_map.entry(s.clone()).or_insert(0);
        if *counter == u32::MAX {
            panic!("Variable counter overflow");
        } else if counter != &0 {
            *counter += 1;
            s.push('$');
            s.push_str(&counter.to_string());
        }

        Var { name: s }
    }

    /// Creates the predicate guard, if there should be one.
    ///
    /// # Returns
    /// `None` if there should be no predicate guard.
    fn make_guard(&self) -> Option<Handle<Predicate>> {
        let pred_block_pred = self.join_predicate_stack();
        let perm_pred = match (&self.permanent_predicate, pred_block_pred) {
            (Some(perm), Some(pred)) => Some(Predicate::new_and(perm.clone(), pred)),
            (Some(ref p), None) | (None, Some(ref p)) => Some(p.clone()),
            (None, None) => None,
        };
        // Conjunction with the loop predicate
        let loop_pred = self.join_loop_predicate_stack();
        match (perm_pred, loop_pred) {
            (Some(perm), Some(pred)) => Some(Predicate::new_and(perm.clone(), pred)),
            (Some(ref p), None) | (None, Some(ref p)) => Some(p.clone()),
            (None, None) => None,
        }
    }

    /// Mark an action taken by the constraint system...
    #[inline]
    fn write<T: AsRef<str>>(&mut self, s: T) {
        let s = s.as_ref();
        self.statements.push(s.to_string());
        log_trace!("{s}",);
    }

    /// Mark the length of a dimension
    ///
    /// This isn't really used. The length of a dimension is usually inferred from the type.
    #[allow(unused)]
    pub(crate) fn mark_ndim(&mut self, term: &Term, ndim: u8) {
        self.statements.push(format!("ndim({term}) = {ndim}"));
        log_trace!("ndim({term}) = {ndim}");
    }

    /// Join the predicate stack together, returning a predicate which is the conjunction of all predicates in the stack.
    fn join_predicate_stack(&self) -> Option<Handle<Predicate>> {
        self.predicate_stack
            .iter()
            .cloned()
            .reduce(Predicate::new_and)
    }

    /// Join the loop predicate stack together, returning a predicate which is the conjunction of all predicates in the stack.
    fn join_loop_predicate_stack(&self) -> Option<Handle<Predicate>> {
        self.loop_predicate_stack
            .iter()
            .cloned()
            .reduce(Predicate::new_and)
    }

    /// Creates a [`Constraint`] for use as a constraint
    ///
    /// If [`ConstraintOp::Unary`] is passed, then `rhs` must be [`Term::Empty`].
    ///
    /// # Errors
    /// [`EmptyExpression`] if `lhs` is [`Term::Empty`] or `rhs` is [`Term::Empty`] and `op` is not [`ConstraintOp::Unary`].
    ///
    /// [`EmptyExpression`]: crate::ConstraintError::EmptyExpression
    /// [`Constraint`]: crate::Constraint
    fn build_constraint(
        &mut self, term: &Term, op: ConstraintOp, rhs: &Term,
    ) -> Result<Constraint, ConstraintError> {
        let guard = self.make_guard();
        if term.is_empty() {
            return Err(ConstraintError::EmptyTerm);
        }
        match op {
            ConstraintOp::Unary => Ok(Constraint::Identity {
                guard,
                term: term.clone(),
            }),
            ConstraintOp::Cmp(op) => Ok(Constraint::Cmp {
                guard,
                lhs: term.clone(),
                op,
                rhs: rhs.clone(),
            }),
        }
    }

    /// Build the assumption here.
    fn build_assumption(
        &mut self, lhs: &Term, op: AssumptionOp, rhs: &Term,
    ) -> Result<Assumption, ConstraintError> {
        if lhs.is_empty() {
            return Err(ConstraintError::EmptyTerm);
        }

        if matches!(op, AssumptionOp::Assign)
            || !(lhs.is_array_length_like() || rhs.is_array_length_like())
        {
            // If neither of these are array-length like (which allows for comparison between U32 and I32),
            // then we mark the type of the rhs as the same as lhs.
            self.mark_type_as_other(rhs, lhs);
            self.mark_type_as_other(lhs, rhs);
        }

        if let Some(ty) = self.module.type_map.get(lhs) {
            if !rhs
                .try_get_expr()
                .is_some_and(|t| t.is_array_length() || t.is_array_length_dim())
            {}
        }

        // The common case.
        if let AssumptionOp::Assign = op {
            return Ok(Assumption::Assign {
                guard: self.make_guard(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
            });
        }
        // rhs must be a literal, var, or expression here
        if matches!(rhs, Term::Empty | Term::Predicate(_)) {
            return Err(ConstraintError::EmptyTerm);
        }

        let inclusive = matches!(op, AssumptionOp::Geq | AssumptionOp::Leq);
        let boundary = Some((rhs.clone(), inclusive));

        match op {
            AssumptionOp::Geq | AssumptionOp::Gt => Ok(Assumption::Inequality {
                lhs: lhs.clone(),
                lower: boundary,
                upper: None,
            }),
            AssumptionOp::Leq | AssumptionOp::Lt => Ok(Assumption::Inequality {
                lhs: lhs.clone(),
                lower: None,
                upper: boundary,
            }),
            AssumptionOp::Assign => unreachable!(), // handled above..
        }
    }
}

impl ConstraintInterface for ConstraintHelper {
    type Handle<T> = std::sync::Arc<T>;
    type E = ConstraintError;

    /// A reference to the empty expression in this constraint system.
    ///
    /// The empty expression is meant to be a singleton, and this provides a wrapper to reference it.
    fn empty_expression(&self) -> Term {
        Term::Empty
    }
    /// A reference to the `NoneType` in this constraint system.
    ///
    /// Nonetype is meant to be a singleton, and this provides a wrapper to reference it.
    fn none_type(&self) -> Self::Handle<AbcType> {
        NONETYPE.clone()
    }

    /// Define a type that is used in the summary.
    ///
    /// This returns a handle that can be used to refer to the type.
    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E> {
        let ty: Self::Handle<AbcType> = ty.into();
        self.module.types.push(ty.clone());
        Ok(ty)
    }

    // When we begin loop, then we get an induction variable
    // We can then mark anything that uses this induction variable..
    fn mark_loop_variable(
        &mut self,
        var: &Term, // The variable that is incremented
        // What it is initialized to.
        init: &Term,
        // How the term is incremented
        inc_term: &Term,

        inc_op: BinaryOp,
    ) -> Result<(), Self::E> {
        // If term is increasing, then add an assumption that it is greater than the initial value.
        // right now, we only support increment expressions on scalars.
        match (inc_term, inc_op) {
            // Adding a positive literal...
            (
                Term::Literal(crate::Literal::U32(1..) | crate::Literal::I32(1..)),
                BinaryOp::Plus | BinaryOp::Times | BinaryOp::Shl,
            )
            | (Term::Literal(crate::Literal::I32(i32::MIN..=-1)), BinaryOp::Minus) => {
                // We know that v is between 1 and 10...
                self.add_assumption(var, AssumptionOp::Geq, init)?;
            }
            // Decreases if we are subtracting a positive literal
            (
                Term::Literal(crate::Literal::U32(1..) | crate::Literal::I32(1..)),
                BinaryOp::Minus | BinaryOp::Shr,
            )
            | (Term::Literal(crate::Literal::I32(i32::MIN..=-1)), BinaryOp::Plus) => {
                self.add_assumption(var, AssumptionOp::Leq, init)?;
            }
            _ => {
                return Err(ConstraintError::UnsupportedLoopOperation);
            }
        }

        // Now, we check against the limit..
        Ok(())
    }

    /// Add the constraints of the summary by invoking the function call with the proper arguments.
    ///
    /// This will add the constraints of the summary to the constraint system, with
    /// the arguments substituted with `args`, and the return value substituted with `into`.
    /// If `into` is `None`, then a new term is created for the return value (for proper ssa handling)
    fn make_call(
        &mut self, func: SummaryId, args: Vec<Term>, into: Option<&Term>,
    ) -> Result<Term, Self::E> {
        let Some(func) = self.module.summaries.get(func.0).cloned() else {
            return Err(ConstraintError::InvalidSummary);
        };
        // We have to add the constraints from the summary.
        // For now, we don't do this, but we will have to on the desugaring pass.
        // First, we add assumptions that all arguments equal their call.
        if args.len() != func.args.len() {
            return Err(ConstraintError::InvalidArguments);
        }

        // We replace the arguments in the constraints...
        let mut term_vec: Vec<(&Term, &Term)> =
            Vec::with_capacity(args.len() + usize::from(matches!(func.ret_term, Term::Empty)));
        func.args
            .iter()
            .zip(args.iter())
            .for_each(|(left, right)| term_vec.push((left, right)));

        // Replace references to the ret term with this new term...
        let ret_term = if let Some(t) = into {
            t.clone()
        } else {
            let new_ret = self.fresh_ret_var(Some(&func.name));
            self.declare_var(new_ret)?
        };

        // We don't do replacements on an empty term...
        if func.ret_term.is_empty() {
            self.mark_type(&ret_term, &NONETYPE)?;
        } else {
            term_vec.push((&func.ret_term, &ret_term));
        }
        // Add constraints from the summary

        let added_constraints: Vec<(Constraint, u32)> = func
            .constraints
            .iter()
            .map(|c| (c.0.substitute_multi(&term_vec), c.1))
            .collect();

        // Echo the constraints to the output stream.
        for constraint in &added_constraints {
            self.write(constraint.0.to_string());
        }
        self.get_cur_constraint_vec().extend(added_constraints);

        let mut added_assumptions: FastHashMap<Term, Assumption> =
            FastHashMap::with_capacity_and_hasher(func.assumptions.len(), Default::default());
        for assumption in func.assumptions.values() {
            let new = assumption.substitute_multi(&term_vec);
            // problem: What if I see the same term multiple times in an assumption?
            self.write(new.to_string());
            if added_assumptions
                .insert(new.get_lhs().clone(), new)
                .is_some()
            {
                return Err(ConstraintError::DuplicateAssumption);
            }
        }
        self.get_cur_assumption_vec().extend(added_assumptions);

        Ok(ret_term)
    }

    fn add_argument(&mut self, name: String, ty: &Self::Handle<AbcType>) -> Result<Term, Self::E> {
        let mut argname = String::with_capacity(name.len() + 1);
        argname.push('@');
        argname.push_str(&name);
        log::trace!("Renamed {name} to {}", argname);
        let var = self.declare_var(Var { name: argname })?;
        self.mark_type(&var, ty)?;
        match self.active_summary {
            Some(ref mut summary) => {
                summary.add_argument(&var);
                Ok(var)
            }
            None => Err(ConstraintError::SummaryError),
        }
    }

    /// Create a new variable, returning a `Term` that wraps it.
    fn declare_var(&mut self, name: Var) -> Result<Term, ConstraintError> {
        // We need to track this variable...
        Ok(Term::Var(name.into()))
    }

    /// Mark a break statement. (Currently unimplemented)
    ///
    /// # Errors
    /// `ConstraintError::NotImplemented`
    fn mark_break(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented(
            "break".to_string(),
            line!(),
            file!(),
        ))
    }

    /// Mark a continue statement. (Currently unimplemented)
    ///
    /// # Errors
    /// Errors if there is no active loop context.
    fn mark_continue(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented(
            "continue".to_string(),
            line!(),
            file!(),
        ))
    }

    /// Mark the type of the provided term.
    ///
    /// # Errors
    /// [`DuplicateType`] if the type of the term has already been marked.
    ///
    /// [`DuplicateType`]: crate::ConstraintError::DuplicateType
    fn mark_type(&mut self, term: &Term, ty: &Self::Handle<AbcType>) -> Result<(), Self::E> {
        use std::collections::hash_map::Entry;
        self.write(format!("type({term}) = {ty}").as_str());
        match self.module.type_map.entry(term.clone()) {
            Entry::Occupied(entry) if entry.get() == ty => Ok(()),
            Entry::Occupied(_) => Err(ConstraintError::DuplicateType),
            Entry::Vacant(entry) => {
                entry.insert(ty.clone());
                Ok(())
            }
        }
    }

    /// Mark the return type for the active summary.
    ///
    /// # Errors
    /// - [`SummaryError`] if there is no active summary
    /// - [`DuplicateType`] if the return type has already been marked for the summary
    ///
    /// [`SummaryError`]: crate::ConstraintError::SummaryError
    /// [`DuplicateType`]: crate::ConstraintError::DuplicateType
    fn mark_return_type(&mut self, ty: &Self::Handle<AbcType>) -> Result<(), Self::E> {
        let Some(ref mut summary) = self.active_summary else {
            return Err(ConstraintError::SummaryError);
        };
        if !summary.ret_term.is_empty() {
            return Err(ConstraintError::DuplicateType);
        }
        let name = summary.name.clone();
        let new_ret = Term::Var(self.fresh_ret_var(Some(&name)).into());
        self.write(format!("type({new_ret}) = {ty}").as_str());
        self.module.type_map.insert(new_ret.clone(), ty.clone());
        match self.active_summary {
            Some(ref mut summary) if matches!(summary.return_type.as_ref(), AbcType::NoneType) => {
                summary.return_type = ty.clone();
                summary.ret_term = new_ret;
                Ok(())
            }
            _ => Err(ConstraintError::DuplicateType),
        }
    }

    /// Add a constraint to the system that narrows the domain of `term`
    ///
    /// Any active predicates are applied.
    ///
    /// # Errors
    /// - `TypeMismatch` if the type of `term` is different from the type of `rhs` and `op` is `ConstraintOp::Assign`
    /// - `MaxConstraintCountExceeded` if the maximum number of constraints has been exceeded.
    #[allow(clippy::cast_possible_truncation)] // summary len can't exceed u32 max.
    fn add_constraint(
        &mut self, term: &Term, op: ConstraintOp, rhs: &Term, id: u32,
    ) -> Result<(), Self::E> {
        // build the predicate. To start with, we have the permanent predicate
        // Then, we have the AND of the current predicate stack.

        let new_constraint = self.build_constraint(term, op, rhs)?;
        self.write(new_constraint.to_string());
        // Now we add the constraint.

        // This is the unique identifier for the constraint.
        if let Some(ref mut summary) = self.active_summary {
            &mut summary.constraints
        } else {
            &mut self.module.global_constraints
        }
        .push((new_constraint.clone(), id));
        // If this is an equality constraint, then try to mark the type of the term.
        if let Constraint::Cmp {
            op: CmpOp::Eq,
            ref lhs,
            ref rhs,
            ..
        } = new_constraint
        {
            match self.mark_type_as_other(lhs, rhs) {
                Ok(()) | Err(ConstraintError::DuplicateType) => Ok(()),
                Err(e) => Err(e),
            }
        } else {
            Ok(())
        }
    }

    /// When we encounter a return, push the current predicate on the current stack, if there is a current predicate stack...
    fn mark_return(&mut self, retval: Option<Term>) -> Result<(), ConstraintError> {
        let Some(ref mut summary) = self.active_summary else {
            return Err(ConstraintError::SummaryError);
        };

        let pred_ctx = self
            .predicate_stack
            .last()
            .cloned()
            .unwrap_or(Handle::new(Predicate::True));
        match &self.return_predicate {
            None => {
                self.return_predicate = Some(pred_ctx.clone());
            }
            Some(pred) => {
                self.return_predicate = Some(Predicate::new_or(pred.clone(), pred_ctx.clone()));
            }
        }

        let Some(retval) = retval else {
            return if summary.return_type.is_none_type() {
                Ok(())
            } else {
                Err(ConstraintError::NoReturnValue)
            };
        };

        // Get the current assumption for the return value, or create it if it does not exist.
        let Some(other) = summary.assumptions.get(&summary.ret_term) else {
            let other_term = summary.ret_term.clone();
            return self.add_assumption(&other_term, AssumptionOp::Assign, &retval);
        };

        // Get the guard and what it used to be assigned to.
        let Assumption::Assign {
            ref guard, ref rhs, ..
        } = *other
        else {
            return Err(ConstraintError::NotImplemented(
                "Assumptions on return values may only be equalities.".into(),
                line!(),
                file!(),
            ));
        };

        let Some(old_guard) = guard.clone() else {
            log_warning!("`mark_return` following an unguarded return. Ignoring...");
            return Ok(());
        };

        // Create the new return value and mark its assumption.
        let new_ret =
            Term::Var(Self::fresh_var_from_map(&mut self.ssa_map, Some(&summary.name)).into());

        // Replace old ret term with the new one.
        let old_ret = std::mem::replace(&mut summary.ret_term, new_ret.clone());

        // The return value will be a Select(old_guard, old_ret, retval).
        let sel = Term::new_select(&Term::Predicate(old_guard.clone()), &old_ret, &retval);

        // Mark the type of this term.
        self.module
            .type_map
            .insert(new_ret.clone(), summary.return_type.clone());
        // Okay, now we replace the return value.

        self.add_assumption(&new_ret, AssumptionOp::Assign, &sel)?;

        // It is likely an error if the return predicate already exists and is not None,
        // But just in case, we will make the return predicate the `or` of the current predicate stack...

        Ok(())
    }

    /// Mark an assumption.
    ///
    /// Assumptions are invariants that must hold at all times.
    /// At solving time, these differ from constraints in that they are not inverted
    /// to test for satisfiability.
    ///
    /// # Errors
    ///
    /// Errors if the assumption is not well formed or supported by the system.
    fn add_assumption(&mut self, lhs: &Term, op: AssumptionOp, rhs: &Term) -> Result<(), Self::E> {
        let assumption = self.build_assumption(lhs, op, rhs)?;
        self.write(assumption.to_string());
        let added = if let Some(ref mut s) = self.active_summary {
            add_assumption_impl!(s.assumptions, assumption)
        } else {
            add_assumption_impl!(self.module.global_assumptions, assumption)
        };

        if !added {
            return Err(ConstraintError::DuplicateAssumption);
        }

        Ok(())
    }

    fn replace_assumption(
        &mut self, lhs: &Term, op: AssumptionOp, rhs: &Term,
    ) -> Result<(), Self::E> {
        let assumption = self.build_assumption(lhs, op, rhs)?;
        self.write(assumption.to_string());
        if let Some(ref mut s) = self.active_summary {
            s.assumptions.insert(lhs.clone(), assumption);
        } else {
            self.module
                .global_assumptions
                .insert(lhs.clone(), assumption);
        }
        Ok(())
    }

    /// Mark the length of an array's dimension.
    ///
    /// Used for arrays with a fixed size.
    ///
    /// It is STRONGLY preferred to use the type system to mark the variable as a [`SizedArray`]
    ///
    /// [`SizedArray`]: crate::AbcType::SizedArray
    fn mark_length(&mut self, var: &Term, dim: u8, size: u64) -> Result<(), ConstraintError> {
        self.write(format!("length({var}, {dim}) = {size}").as_str());
        let term = if dim == 0u8 {
            Term::make_array_length(var)
        } else {
            Term::make_array_length_dim(var, dim.try_into().unwrap())
        };
        self.add_assumption(&term, AssumptionOp::Assign, &Term::new_literal(size))?;
        Ok(())
    }

    /// Mark the range of a runtime constant. Sugar for a pair of assumptions, (var >= min) and (var <= max).
    ///
    /// ## Notes
    /// - The current predicate block is ignored, though the constraints *are* added to the active summary (or global if no summary is active)
    /// - This is not meant to be used for loop variables. for those, see [`mark_loop_variable`]
    ///
    /// [`mark_loop_variable`]: `crate::helper::ConstraintHelper::mark_loop_variable`
    fn mark_range<T: Into<crate::Literal>>(
        &mut self, var: &Term, min: T, max: T,
    ) -> Result<(), ConstraintError> {
        let min = min.into();
        let max = max.into();
        self.write(format!("{var} \\in [{min}, {max}]").as_str());
        let assumption = if min == max {
            Assumption::Assign {
                guard: None,
                lhs: var.clone(),
                rhs: Term::new_literal(max),
            }
        } else {
            Assumption::Inequality {
                lhs: var.clone(),
                lower: Some((Term::Literal(min), true)),
                upper: Some((Term::Literal(max), true)),
            }
        };
        self.write(assumption.to_string());
        let added = if let Some(ref mut s) = self.active_summary {
            add_assumption_impl!(s.assumptions, assumption)
        } else {
            add_assumption_impl!(self.module.global_assumptions, assumption)
        };
        if !added {
            return Err(ConstraintError::DuplicateAssumption);
        }
        Ok(())
    }

    /// Push a predicate block onto the stack
    fn begin_predicate_block(&mut self, p: &Term) -> Result<(), ConstraintError> {
        // In this form, every subsequent constraint we add is guarded by the conjunction of each predicate...
        let pred: Self::Handle<Predicate> = match p {
            Term::Predicate(p) => p.clone(),
            _ => Predicate::new_unit(p),
        };
        log_info!("Beginning predicate block: {}", pred.as_ref());
        self.predicate_stack.push(pred);
        Ok(())
    }

    /// Ends the previous predicate block.
    ///
    /// Returns an error if the predicate stack is empty.
    fn end_predicate_block(&mut self) -> Result<(), ConstraintError> {
        // Predicate block is sugar. It disappears, so there should be no use for it...
        log_info!("Popping predicate");
        // Make the permanent predicate the negation of the current predicate.

        // If we currently have a return predicate, then the permanent predicate becomes the And of the current permanent predicate and the negation of the return predicate.
        // When we do this, the return predicate becomes the permanent predicate, right?
        // This is because we would do !(a || b) which would translate to !(a) && !(b)
        if let Some(p) = self.return_predicate.take() {
            let p = Predicate::new_not(p);
            self.permanent_predicate = self
                .permanent_predicate
                .clone()
                .map_or(Some(p.clone()), move |perm| {
                    Predicate::new_and(perm, p).into()
                });
        }

        self.predicate_stack
            .pop()
            .map_or(Err(ConstraintError::PredicateStackEmpty), |_| Ok(()))
    }

    /// Begins a loop context.
    ///
    /// When we are inside of a loop context, any update to a variable is marked as a range constraint.
    /// It also allows for special handling of break and continue statements.
    /// HOWEVER, for the time being, we can't do any of this fancy handling
    /// So we are just going to emit a "`begin_loop(condition)`" statement, and will assume this is a part of the constraint system.
    fn begin_loop(&mut self, condition: &Term) -> Result<(), ConstraintError> {
        // In begin_loop, we add a new predicate block corresponding to the loop predicate.
        // Here, we need to begin a new loop predicate block.
        match condition {
            Term::Predicate(p) => {
                self.loop_predicate_stack.push(p.clone());
            }
            t @ Term::Var(_) => {
                self.loop_predicate_stack
                    .push(Predicate::new_unit(t.clone()));
            }
            _ => {
                return Err(ConstraintError::UnsupportedLoopCondition(condition.clone()));
            }
        }

        // When we hit a break, we are overestimating the range of the loop variable.

        // We begin the predicate block here...
        // self.begin_predicate_block(condition);
        if self.loop_depth == u8::MAX {
            return Err(ConstraintError::MaxLoopDepthExceeded);
        }
        self.loop_depth += 1;
        Ok(())
    }

    /// End a loop context.
    fn end_loop(&mut self) -> Result<(), ConstraintError> {
        // At the end of a loop, we pop the loop predicate stack.
        self.loop_predicate_stack
            .pop()
            .ok_or(ConstraintError::PredicateStackEmpty)?;
        // check if it an assignment constraint,
        // and if it is, convert it into a range constraint.
        if self.loop_depth == 0 {
            return Err(ConstraintError::NotInLoopContext);
        }
        self.loop_depth -= 1;
        Ok(())
    }

    /// Begin a summary block.
    fn begin_summary(&mut self, name: String, nargs: u8) -> Result<(), ConstraintError> {
        if self.module.summaries.len() >= SUMMARY_LIMIT {
            return Err(ConstraintError::SummaryLimitExceeded);
        }
        // Mini optimization for allocating the perfect amount of characters needed for the string
        let str_len = 17
            + name.len()
            + match nargs {
                0..=9 => 1,
                10..=99 => 2,
                100..=255 => 3,
            };

        // The most efficient way to construct a string in rust.
        let mut param_list_str = String::with_capacity(str_len);

        param_list_str.push_str("begin_summary(");
        param_list_str.push_str(&name);
        param_list_str.push_str(", ");
        param_list_str.push_str(&nargs.to_string());
        param_list_str.push(')');
        self.write(param_list_str.as_str());

        // Create a new summary
        let new_summary = Summary {
            name,
            args: Vec::with_capacity(nargs as usize),
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
            assumptions: FastHashMap::default(),
            ret_term: Term::Empty,
        };
        self.active_summary = Some(new_summary);

        Ok(())
    }

    /// End a summary block.
    ///
    /// This returns a tuple of an identifier that can be used to refer to the summary, as well as a handle to the summary itself.
    fn end_summary(&mut self) -> Result<SummaryId, ConstraintError> {
        #[allow(clippy::cast_possible_truncation)] // summaries can't exceed u32 max.
        let id = SummaryId(self.module.summaries.len());
        let summary = self
            .active_summary
            .take()
            .ok_or(ConstraintError::SummaryError)?;
        let mut fmt_str = String::with_capacity(13 + summary.name.len());
        fmt_str.push_str("end_summary(");
        fmt_str.push_str(&summary.name);
        fmt_str.push_str(")\n");
        self.write(fmt_str.as_str());

        // Clear the state from the current summary.
        self.return_predicate = None;
        self.permanent_predicate = None;
        self.predicate_stack.clear();

        let summary = Handle::new(summary);
        self.module.summaries.push(summary.clone());
        Ok(id)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use std::num::NonZero;

    use super::*;
    use crate::{AbcScalar, Term};
    use rstest::{fixture, rstest};
    #[fixture]
    fn constraint_helper() -> ConstraintHelper {
        ConstraintHelper::default()
    }

    fn fresh_var(constraint_helper: &mut ConstraintHelper, name: String) -> Term {
        constraint_helper.declare_var(name.into()).unwrap()
    }

    fn check_constraint_output(helper: &ConstraintHelper, expected: &str) {
        let mut stream = Vec::new();
        helper.write_to_stream(&mut stream).unwrap();
        assert_eq!(
            String::from_utf8(stream).unwrap().trim(),
            expected.to_string().trim()
        );
    }

    #[rstest]
    fn test_mark_type(mut constraint_helper: ConstraintHelper) {
        let var = fresh_var(&mut constraint_helper, "x".to_string());
        // mark the type
        let ty = constraint_helper
            .declare_type(AbcType::Scalar(AbcScalar::Sint(4)))
            .unwrap();
        assert!(constraint_helper.mark_type(&var, &ty).is_ok());
        check_constraint_output(&constraint_helper, "type(x) = i32");
    }

    #[rstest]
    fn test_mark_ndim(mut constraint_helper: ConstraintHelper) {
        let x = fresh_var(&mut constraint_helper, "x".to_string());
        constraint_helper.mark_ndim(&x, 3);
        check_constraint_output(&constraint_helper, "ndim(x) = 3");
    }

    /// Test that functions called within a sumamry have their constraints inlined.
    #[rstest]
    fn test_func_inline(mut constraint_helper: ConstraintHelper) {
        constraint_helper
            .begin_summary("test_func".to_string(), 1)
            .unwrap();
        let my_ty = constraint_helper
            .declare_type(AbcType::Scalar(AbcScalar::Uint(4)))
            .unwrap();
        let arg = constraint_helper
            .add_argument("test_arg_1".to_string(), &my_ty)
            .unwrap();
        constraint_helper
            .add_assumption(&arg, AssumptionOp::Assign, &Term::new_literal(4u32))
            .unwrap();

        let summary_id = constraint_helper.end_summary().unwrap();

        // Now, begin a new summary.
        constraint_helper
            .begin_summary("actual_func".to_string(), 1)
            .unwrap();
        // make a new literal
        let my_x = constraint_helper
            .declare_var(Var {
                name: "x".to_string(),
            })
            .unwrap();

        let args = vec![my_x.clone()];
        // Now, push a call.
        constraint_helper.make_call(summary_id, args, None).unwrap();

        let expected_assumption = constraint_helper
            .build_assumption(&my_x, AssumptionOp::Assign, &Term::new_literal(4u32))
            .unwrap();

        let result = constraint_helper
            .active_summary
            .as_ref()
            .unwrap()
            .assumptions
            .iter()
            .any(|(t, c)| *c == expected_assumption);
        // println!("{}", String::from_utf8(output).unwrap());

        assert!(result);
    }

    #[rstest]
    fn test_serialize_module() {
        let mut constraint_helper = ConstraintHelper::default();
        let x = fresh_var(&mut constraint_helper, "x".to_string());
        let y = fresh_var(&mut constraint_helper, "y".to_string());
        constraint_helper.mark_uniform_var(&x).unwrap();
        let ty = constraint_helper
            .declare_type(AbcType::Scalar(AbcScalar::Sint(4)))
            .unwrap();
        constraint_helper.mark_type(&x, &ty).unwrap();
        constraint_helper.mark_type(&y, &ty).unwrap();
        constraint_helper
            .add_assumption(&x, AssumptionOp::Assign, &y)
            .unwrap();
        constraint_helper
            .add_constraint(&x, ConstraintOp::Cmp(CmpOp::Eq), &y, 0)
            .unwrap();
        let module = constraint_helper.get_module();

        let result = serde_json::to_string_pretty(module).unwrap();

        // now test deserialization
        let module: ConstraintModule = serde_json::from_str(&result).unwrap();

        // If we don't get any errors, then we've serialized / deserialized properly.

        // Make sure the length of the assumptions is what we expect.
        assert_eq!(module.global_assumptions.len(), 1);
        // Make sure the length of the constraints is what we expect.
        assert_eq!(module.global_constraints.len(), 1);
        // Make sure the type map is what we expect.
        assert_eq!(module.type_map.len(), 2);

        assert_eq!(module.uniform_vars.len(), 1);
        assert!(module.uniform_vars.contains(&x));

        // println!("{}", result);
    }
}
