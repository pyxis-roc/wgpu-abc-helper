// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

// Module uses the "helper" name.
#![allow(clippy::module_name_repetitions)]

use log::info as log_info;

use crate::{
    AbcExpression, AbcScalar, AbcType, BinaryOp, CmpOp, Constraint, ConstraintOp, FastHashMap,
    Handle, Literal, OpaqueMarker, Predicate, SubstituteTerm, Summary, Term, UnaryOp, Var,
    NONETYPE,
};

#[derive(Clone, Debug, thiserror::Error)]
pub enum ConstraintError {
    #[error("Predicate stack is empty")]
    PredicateStackEmpty,
    #[error("Active summary is not set")]
    SummaryError,
    #[error("This is not yet implemented")]
    NotImplemented(String),
    #[error("Attempt to end a loop when no loop is active")]
    NotInLoopContext,
    #[error("Maximum loop depth exceeded")]
    MaxLoopDepthExceeded,
    #[error("Empty expression in disallowed context.")]
    EmptyExpression,
    #[error("Invalid number of arguments passed to call")]
    InvalidArguments,
    #[error("Attempt to assign a return value to a function with no return value")]
    NoReturnValue,
    #[error("Attempt to assign two distinct types to a term.")]
    TypeMismatch,
}

#[allow(non_snake_case)]
/// Implementing `TermArena` means that Terms can be returned from the arena.
pub trait TermArena {
    // Expressions
    fn new_binary_op(&mut self, op: BinaryOp, lhs: Term, rhs: Term) -> Term;
    fn new_select(&mut self, cond: Term, if_true: Term, if_false: Term) -> Term;
    fn new_unary_op(&mut self, op: UnaryOp, rhs: Term) -> Term;
    fn new_var(&mut self, name: Var) -> Term;
    fn new_splat(&mut self, term: Term, count: u32) -> Term;
    fn new_array_length(&mut self, var: Term) -> Term;
    fn new_cast(&mut self, term: Term, ty: AbcScalar) -> Term;
    fn new_call(&mut self, func: Handle<Summary>, args: Vec<Term>) -> Term;
    fn new_empty(&mut self) -> Term;
    fn new_struct_access(&mut self, term: Term, field: String, ty: Handle<AbcType>) -> Term;
    fn new_index_access(&mut self, term: Term, index: Term) -> Term;

    // Predicates
    fn new_logical_and(&mut self, lhs: Term, rhs: Term) -> Term;
    fn new_logical_or(&mut self, lhs: Term, rhs: Term) -> Term;
    fn new_logical_not(&mut self, term: Term) -> Term;
    fn new_literal_false(&mut self) -> Term;
    fn new_literal_true(&mut self) -> Term;
    fn new_unit_pred(&mut self, term: Term) -> Term;
    fn new_comparison(&mut self, op: CmpOp, lhs: Term, rhs: Term) -> Term;

    // Literals

    fn new_literal_term<T: Into<Literal>>(&mut self, lit: T) -> Term;
}

#[allow(clippy::missing_errors_doc)]
pub trait ConstraintInterface {
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

    /// Add an expression into the constraint system.
    fn add_expression(&mut self, expr: AbcExpression) -> Result<Term, Self::E>;

    /// Add an argument to the function summary
    ///
    /// Name is the name of the argument.
    /// Returns a var handle.
    fn add_argument(&mut self, name: String, ty: Self::Handle<AbcType>) -> Result<Term, Self::E>;

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
        &mut self,
        func: Self::Handle<Summary>,
        args: Vec<Term>,
        into: Option<Term>,
    ) -> Result<Term, Self::E>;

    /// Add a new constraint to the constraint system. Any active predicates are applied to the constraint to "filter" the domain of its expression.
    ///
    /// Note: These will need to be desugared.
    ///
    /// # Errors
    ///
    fn add_constraint(&mut self, lhs: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E>;

    fn declare_type(&mut self, ty: AbcType) -> Result<Self::Handle<AbcType>, Self::E>;

    /// Add a new constraint to the constraint system, marked by `source`
    fn add_tracked_constraint<T: std::fmt::Debug + Clone>(
        &mut self,
        lhs: Term,
        op: ConstraintOp,
        rhs: Term,
        source: OpaqueMarker<T>,
    ) -> Result<(), Self::E>;

    /// Declare the variable in the constraint system.
    ///
    /// This passes by value since the variable is replaced with a handle to it.
    fn declare_var(&mut self, name: Var) -> Result<Term, Self::E>;

    /// Mark the type of a variable.
    /// If the type is an array, this will also mark the number of dimensions.
    /// If the type is an array with a fixed size, this will also mark the size of the array.
    fn mark_type(&mut self, term: Term, ty: self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the length of a variable. The type of the variable must be a dynamic array.
    fn mark_length(&mut self, var: Term, dim: u8, size: u64) -> Result<(), Self::E>;

    /// Mark an assumption.
    ///
    /// These are constraints that must be met.
    ///
    /// That is, these are constraints that are not inverted when proving the satisfiability of the system.
    fn add_assumption(&mut self, lhs: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E>;

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
    fn begin_predicate_block(&mut self, p: Term) -> Result<(), Self::E>;

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

    /// Ends the current summary, returning a handle that references it.
    ///
    /// # Errors
    /// Returns an error if there is no active summary.
    fn end_summary(&mut self) -> Result<Self::Handle<Summary>, Self::E>;

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

    fn mark_return_type(&mut self, ty: Self::Handle<AbcType>) -> Result<(), Self::E>;

    /// Mark the beginning of a loop.
    fn begin_loop(&mut self, condition: Term) -> Result<(), Self::E>;

    /// Mark the end of a loop.
    fn end_loop(&mut self) -> Result<(), Self::E>;

    /// This is sugar for marking a predicate block with a single entry...
    fn mark_break(&mut self) -> Result<(), Self::E>;

    /// Marks a continue statement
    fn mark_continue(&mut self) -> Result<(), Self::E>;

    /// Mark the range of a term.
    fn mark_range<T>(&mut self, var: Term, low: T, high: T) -> Result<(), Self::E>
    where
        T: ToString;

    // Marks a call to a function.
}

/// The helper interface
/// The main helper class for the constraint system.
///
/// Tracks a module and each of the constraints, variables, and types it contains.
#[derive(Default)]

pub struct ConstraintHelper {
    /// For right now, the handles are just Arcs.
    /// This is so that they have an easy time printing
    ///
    /// When we make the interface with the constraint solver itself, we will use a real handle, like naga's arena.
    pub(crate) predicate_stack: Vec<Handle<Predicate>>,

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
    pub(crate) return_predicate: Option<Handle<Predicate>>,

    /// The set of predicates that are appended to all future constraints.
    ///
    /// This is used in conjunction with the `had_return` field.
    /// As soon as a return is marked, its predicate is appended to the `permanent_predicate` field.
    pub(crate) permanent_predicate: Option<Handle<Predicate>>,

    /// Summaries can't be nested, so we only need to track what the active one is.
    /// When we pop a summary, we clear out the predicate stack.
    pub(crate) active_summary: Option<Summary>,

    // Statements are the set of statements that comprise the constraint system...
    pub(crate) statements: Vec<String>,

    /// Summaries contain the set of summaries that have previously been parsed.
    pub(crate) summaries: Vec<Handle<Summary>>,

    /// Types that have been declared in the helper.
    pub(crate) types: Vec<Handle<AbcType>>,

    /// Map of terms to their underlying type.
    pub(crate) term_type_map: FastHashMap<Term, Handle<AbcType>>,

    /// Global constraints not tied to a function
    pub(crate) global_constraints: Vec<Constraint>,

    /// Global assumptions not tied to a function
    pub(crate) global_assumptions: Vec<Constraint>,

    /// Map from varnames to an ssa counter..
    pub(crate) ssa_map: FastHashMap<String, u32>,

    /// The number of loop layers that are active.
    ///
    /// Incremented when a loop begins, decremented when a loop ends.
    pub(crate) loop_depth: u8,

    pub(crate) terms: Vec<Term>,
}

impl ConstraintHelper {
    /// Get a mutable reference to the active summary's assumptions, or the global assumptions if there is no active summary.
    fn get_cur_assumption_vec(&mut self) -> &mut Vec<Constraint> {
        if let Some(ref mut summary) = self.active_summary {
            &mut summary.assumptions
        } else {
            &mut self.global_assumptions
        }
    }

    /// Get a mutable reference to the active summary's constraints, or the global constraints if there is no active summary.
    fn get_cur_constraint_vec(&mut self) -> &mut Vec<Constraint> {
        if let Some(ref mut summary) = self.active_summary {
            &mut summary.constraints
        } else {
            &mut self.global_constraints
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

    /// Creates the predicate guard, if there should be one.
    ///
    /// # Returns
    /// `None` if there should be no predicate guard.
    fn make_guard(&self) -> Option<Handle<Predicate>> {
        let pred_block_pred = self.join_predicate_stack();
        match (&self.permanent_predicate, pred_block_pred) {
            (Some(perm), Some(pred)) => Some(Predicate::new_and(perm.clone(), pred)),
            (Some(ref p), None) | (None, Some(ref p)) => Some(p.clone()),
            (None, None) => None,
        }
    }

    fn write(&mut self, s: String) {
        self.statements.push(s);
    }

    /// Append a comment after the end of the last statement.
    fn append_last(&mut self, s: String) {
        if let Some(o) = self.statements.last_mut() {
            o.push(' ');
            o.push_str(&s);
        } else {
            self.write(s);
        }
    }

    #[allow(unused)]
    /// Mark the length of a dimension
    /// Not sure if this is even needed, honestly.
    pub(crate) fn mark_ndim(&mut self, term: &Term, ndim: u8) {
        self.write(format!("ndim({term}) = {ndim}"));
    }

    /// Write the constraints for the module to the provided stream
    ///
    /// # Errors
    /// Propagates any errors encountered when writing to the provided `stream`
    pub fn write_to_stream(
        &self,
        stream: &mut impl std::io::Write,
    ) -> Result<usize, std::io::Error> {
        stream.write(self.statements.join("\n").as_bytes())
    }

    /// Join the predicate stack together, returning a predicate which is the conjunction of all predicates in the stack.
    fn join_predicate_stack(&self) -> Option<Handle<Predicate>> {
        self.predicate_stack
            .iter()
            .cloned()
            .reduce(Predicate::new_and)
    }

    /// Creates a [`Constraint`] for use as an assumption or requirement.
    ///
    /// [`Constraint`]: crate::Constraint
    fn build_constraint(
        &mut self,
        term: Term,
        op: ConstraintOp,
        rhs: Term,
    ) -> Result<Constraint, ConstraintError> {
        let guard = self.make_guard();
        match op {
            ConstraintOp::Unary => Ok(Constraint::Expression { guard, term }),
            // An empty rhs is not allowed unless this is a unary constraint.
            _ if matches!(rhs, Term::Expr(ref e) if matches!(e.as_ref(), AbcExpression::Empty)) => {
                Err(ConstraintError::EmptyExpression)
            }
            ConstraintOp::Assign => Ok(Constraint::Assign {
                lhs: term,
                rhs,
                guard,
            }),
            ConstraintOp::Cmp(op) => Ok(Constraint::Cmp {
                guard,
                lhs: term,
                op,
                rhs,
            }),
            // Allow sus to add constraints in the future without having to implement them.
            #[allow(unreachable_patterns)]
            _ => Err(ConstraintError::NotImplemented(format!(
                "ConstraintOp::{op:?}"
            ))),
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
        self.types.push(ty.clone());
        Ok(ty)
    }

    /// Add the constraints of the summary by invoking the function call with the proper arguments.
    ///
    /// This will add the constraints of the summary to the constraint system, with
    /// the arguments substituted with `args`, and the return value substituted with `into`.
    /// If `into` is `None`, then a new term is created for the return value (for proper ssa handling)
    fn make_call(
        &mut self,
        func: Self::Handle<Summary>,
        args: Vec<Term>,
        into: Option<Term>,
    ) -> Result<Term, Self::E> {
        // We have to add the constraints from the summary.
        // For now, we don't do this, but we will have to on the desugaring pass.
        // First, we add assumptions that all arguments equal their call.
        if args.len() != func.args.len() {
            return Err(ConstraintError::SummaryError);
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
            t
        } else {
            let new_ret = self.fresh_ret_var(Some(&func.name));
            self.declare_var(new_ret)?
        };

        // We don't do replacements on an empty term...
        if func.ret_term.is_empty() {
            self.mark_type(ret_term.clone(), NONETYPE.clone())?;
        } else {
            term_vec.push((&func.ret_term, &ret_term));
        }
        // Add constraints from the summary

        let added_constraints: Vec<Constraint> = func
            .constraints
            .iter()
            .map(|c| c.substitute_multi(&term_vec))
            .collect();
        for constraint in &added_constraints {
            self.write(format!("{constraint}"));
        }
        self.get_cur_constraint_vec().extend(added_constraints);

        let added_assumptions: Vec<Constraint> = func
            .assumptions
            .iter()
            .map(|c| c.substitute_multi(&term_vec))
            .collect();
        for assumption in &added_assumptions {
            self.write(format!("assume({assumption})"));
        }
        self.get_cur_assumption_vec().extend(added_assumptions);

        Ok(ret_term)
    }

    fn add_argument(&mut self, name: String, ty: Self::Handle<AbcType>) -> Result<Term, Self::E> {
        let mut argname = String::with_capacity(name.len() + 1);
        argname.push('@');
        argname.push_str(&name);
        let var = self.declare_var(Var { name: argname })?;
        self.mark_type(var.clone(), ty)?;
        match self.active_summary {
            Some(ref mut summary) => {
                summary.add_argument(var.clone());
                Ok(var)
            }
            None => Err(ConstraintError::SummaryError),
        }
    }

    /// Add an expression to the constraint system, returning a handle to it.
    ///
    /// # Errors
    /// Never (always returns an Ok)
    fn add_expression(&mut self, expr: AbcExpression) -> Result<Term, Self::E> {
        let term = Term::Expr(expr.into());
        self.terms.push(term.clone());
        Ok(term)
    }

    /// Create a new variable, returning a `Term` that wraps it.
    fn declare_var(&mut self, var: Var) -> Result<Term, ConstraintError> {
        let term = Term::new_var(var);
        self.terms.push(term.clone());
        Ok(term)
    }

    /// Mark a break statement
    ///
    /// # Errors
    /// Errors if there is no active loop context.
    fn mark_break(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("break".to_string()))
    }

    /// Mark a continue statement
    ///6a
    /// # Errors
    /// Errors if there is no active loop context.
    fn mark_continue(&mut self) -> Result<(), Self::E> {
        Err(ConstraintError::NotImplemented("continue".to_string()))
    }

    /// Mark the type of the provided term
    fn mark_type(&mut self, term: Term, ty: Self::Handle<AbcType>) -> Result<(), Self::E> {
        self.write(format!("type({term}) = {ty})"));
        if let std::collections::hash_map::Entry::Vacant(e) = self.term_type_map.entry(term) {
            e.insert(ty.clone());
            Ok(())
        } else {
            Err(ConstraintError::TypeMismatch)
        }
    }

    /// Mark the return type for the active summary.
    /// Can only be called once per active summary.
    ///
    /// # Errors
    /// Returns [`ConstraintError::SummaryError`] if there is no active summary.
    fn mark_return_type(&mut self, ty: Self::Handle<AbcType>) -> Result<(), Self::E> {
        self.write(format!("type(@ret) = {ty}"));
        match self.active_summary {
            None => Err(ConstraintError::SummaryError),
            Some(ref mut summary) => {
                summary.return_type = ty;
                Ok(())
            }
        }
    }

    /// Add a constraint that is tracked by the provided source.
    fn add_tracked_constraint<T: std::fmt::Debug + Clone>(
        &mut self,
        term: Term,
        op: ConstraintOp,
        rhs: Term,
        source: OpaqueMarker<T>,
    ) -> Result<(), Self::E> {
        // self.write(format!("/* {:?} */", source));
        self.add_constraint(term, op, rhs)?;
        self.append_last(format!("{:?}", source.payload));
        // Then, just write the source
        Ok(())
    }

    /// Add a constraint to the system that narrows the domain of `term`
    ///
    /// Any active predicates are applied.
    fn add_constraint(&mut self, term: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E> {
        // build the predicate. To start with, we have the permanent predicate
        // Then, we have the AND of the current predicate stack.
        let new_constraint = self.build_constraint(term, op, rhs)?;
        self.write(new_constraint.to_string());
        // Now we add the constraint.
        match self.active_summary {
            Some(ref mut summary) => &mut summary.constraints,
            None => &mut self.global_constraints,
        }
        .push(new_constraint);
        Ok(())
    }

    /// When we encounter a return, push the current predicate on the current stack, if there is a current predicate stack...
    fn mark_return(&mut self, retval: Option<Term>) -> Result<(), ConstraintError> {
        match self.active_summary {
            None => {
                return Err(ConstraintError::SummaryError);
            }
            Some(Summary {
                ret_term: Term::Empty,
                ..
            }) => (),
            Some(Summary {
                ret_term: ref r, ..
            }) if retval.is_some() => {
                self.add_constraint(r.clone(), ConstraintOp::Assign, unsafe {
                    /* Unsafe unwrap allowed here due to is_some() check above. */
                    retval.unwrap_unchecked()
                })?;
            }
            _ => (),
        }

        // It is likely an error if the return predicate already exists and is not None,
        // But just in case, we will make the return predicate the `or` of the current predicate stack...
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
        };
        Ok(())
    }

    /// Mark an assumption.
    ///
    /// Assumptions are invariants that must hold at all times.
    /// At solving time, these differ from constraints in that they are not inverted
    /// to test for satisfiability.
    ///
    /// # Errors
    /// Errors if the assumption is not well formed or supported by the system.
    fn add_assumption(&mut self, lhs: Term, op: ConstraintOp, rhs: Term) -> Result<(), Self::E> {
        let constraint = self.build_constraint(lhs, op, rhs)?;
        self.write(format!("assume({constraint})"));
        if let Some(ref mut s) = self.active_summary {
            s.add_assumption(constraint);
        } else {
            self.global_assumptions.push(constraint);
        }
        Ok(())
    }

    /// Mark the length of an array's dimension.
    ///
    /// Used for arrays with a fixed size.
    fn mark_length(&mut self, var: Term, dim: u8, size: u64) -> Result<(), ConstraintError> {
        self.write(format!("length({var}, {dim}) = {size}"));
        Ok(())
    }

    /// Mark the range of a variable. This works as an assumption when the range of a variable is fixed.
    fn mark_range<T>(&mut self, var: Term, low: T, high: T) -> Result<(), ConstraintError>
    where
        T: ToString,
    {
        self.write(format!(
            "range({var}) \\in {{{}, {}}}",
            low.to_string(),
            high.to_string()
        ));
        Ok(())
    }

    /// Push a predicate block onto the stack
    fn begin_predicate_block(&mut self, p: Term) -> Result<(), ConstraintError> {
        // In this form, every subsequent constraint we add is guarded by the conjunction of each predicate...
        let pred: Self::Handle<Predicate> = match p {
            Term::Predicate(p) => p,
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
        };

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
    fn begin_loop(&mut self, condition: Term) -> Result<(), ConstraintError> {
        self.write(format!("begin_loop({condition})"));
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
        self.write("end_loop()".to_string());
        if self.loop_depth == 0 {
            return Err(ConstraintError::NotInLoopContext);
        }
        self.loop_depth -= 1;
        Ok(())
    }

    /// Begin a summary block.
    fn begin_summary(&mut self, name: String, nargs: u8) -> Result<(), ConstraintError> {
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
        self.write(param_list_str);

        // Create a new summary
        let new_summary = Summary {
            name,
            args: Vec::with_capacity(nargs as usize),
            constraints: Vec::new(),
            return_type: NONETYPE.clone(),
            assumptions: Vec::new(),
            ret_term: Term::Empty,
        };
        self.active_summary = Some(new_summary);

        Ok(())
    }

    /// End a summary block.
    fn end_summary(&mut self) -> Result<Self::Handle<Summary>, ConstraintError> {
        let summary = self
            .active_summary
            .take()
            .ok_or(ConstraintError::SummaryError)?;
        let mut fmt_str = String::with_capacity(13 + summary.name.len());
        fmt_str.push_str("end_summary(");
        fmt_str.push_str(&summary.name);
        fmt_str.push_str(")\n");
        self.write(fmt_str);

        // Clear the state from the current summary.
        self.return_predicate = None;
        self.permanent_predicate = None;
        self.predicate_stack.clear();

        let summary = Handle::new(summary);
        self.summaries.push(summary.clone());
        Ok(summary)
    }
}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use crate::{AbcScalar, Term};
    use rstest::{fixture, rstest};
    #[fixture]
    fn constraint_helper<'a>() -> ConstraintHelper {
        ConstraintHelper::default()
    }

    fn fresh_var(constraint_helper: &mut ConstraintHelper, name: String) -> Term {
        constraint_helper.declare_var(name.into()).unwrap()
    }

    fn check_constraint_output(helper: &ConstraintHelper, expected: &str) {
        let mut stream = Vec::new();
        helper.write_to_stream(&mut stream).unwrap();
        assert_eq!(String::from_utf8(stream).unwrap(), expected.to_string());
    }

    #[rstest]
    fn test_mark_type(mut constraint_helper: ConstraintHelper) {
        let var = fresh_var(&mut constraint_helper, "x".to_string());
        // mark the type
        let ty = constraint_helper
            .declare_type(AbcType::Scalar(AbcScalar::Sint(4)))
            .unwrap();
        assert!(constraint_helper.mark_type(var, ty).is_ok());
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
            .add_argument("test_arg_1".to_string(), my_ty)
            .unwrap();
        constraint_helper
            .add_assumption(
                arg,
                ConstraintOp::Cmp(crate::CmpOp::Eq),
                Term::new_literal(4u32),
            )
            .unwrap();

        let old_method = constraint_helper.end_summary().unwrap();

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
        constraint_helper.make_call(old_method, args, None).unwrap();

        let expected_constraint = constraint_helper
            .build_constraint(
                my_x.clone(),
                ConstraintOp::Cmp(crate::CmpOp::Eq),
                Term::new_literal(4u32),
            )
            .unwrap();

        let result = constraint_helper
            .active_summary
            .as_ref()
            .unwrap()
            .assumptions
            .iter()
            .any(|c| *c == expected_constraint);
        // println!("{}", String::from_utf8(output).unwrap());

        assert!(result);
    }
}
