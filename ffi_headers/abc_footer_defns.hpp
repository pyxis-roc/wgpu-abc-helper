/*
This file is used only for reference and synatx checking.

It adds each of the `abc_` methods that take `Context` as their first parameter, and just calls the abc version.

This makes it slightly more ergonomic to deal with contexts, as it enables the familiar
`ctx.method()` syntax, rather than `abc_method(ctx, ...)`.

This file is meant to be copy/pasted and inserted into the "trailer" field of `cbindgen.toml` so
that it is inserted into the generated header file.
*/
namespace abc_helper
{
    /// Create a new unit predicate. Must only be used on the `Variable` variant of `Term` or the following `Expression` variants
    /// - A `Select` where the `iftrue` and `iffalse` are booleans.
    /// - A `FieldAccess` where the accessed field is a `bool`
    /// - An `AccessIndex` on an array of `bool`s
    /// - A `cast` to a boolean type.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] if the term does not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] if the provided context does not exist in the library's collection.
    MaybeTerm Context::new_unit_pred(FfiTerm term) {
        return abc_new_unit_pred(*this, term);
    }

    /// Create a Term holding the `true` predicate
    MaybeTerm Context::new_literal_true() {
        return abc_new_literal_true(*this);
    }

    /// Create a Term holding the `false` predicate
    MaybeTerm Context::new_literal_false() {
        return abc_new_literal_false(*this);
    }

    /// Constructs the Predicate term `lhs && rhs`.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the logical and. Must be a predicate term.
    /// - `rhs`: The right-hand side of the logical and. Must be a predicate term.
    ///
    /// Use `new_unit_pred` to convert variables and expression terms to predicates that can be used in this function.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global context is poisoned.
    /// - `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the library's collection.
    /// - `ErrorCode::InvalidContext` if the provided context does not exist in the library's collection.
    MaybeTerm Context::new_logical_and(FfiTerm lhs, FfiTerm rhs) {
        return abc_new_logical_and(*this, lhs, rhs);
    }

    /// Constructs the Predicate Term `lhs || rhs`.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the logical or. Must be a predicate term.
    /// - `rhs`: The right-hand side of the logical or. Must be a predicate term.
    ///
    /// Use [`abc_new_unit_pred`] to convert variables and expression terms to predicates that can be used in this function.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidTerm`] if either `lhs` or `rhs` do not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] if the provided context does not exist in the library's collection.
    MaybeTerm Context::new_logical_or(FfiTerm lhs, FfiTerm rhs) {
        return abc_new_logical_or(*this, lhs, rhs);
    }

    /// Constructs the predicate term `lhs op rhs`
    ///
    /// # Arguments
    /// - `op`: The comparison operator to use.
    /// - `lhs`: The left-hand side of the comparison. Must be a predicate term.
    /// - `rhs`: The right-hand side of the comparison.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    MaybeTerm Context::new_comparison(CmpOp op, FfiTerm lhs, FfiTerm rhs) {
        return abc_new_comparison(*this, op, lhs, rhs);
    }

    /// Constructs the predicate term `!t`
    ///
    /// If `t` is already a [`Predicate::Not`], then it removes the `!`
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if `t` does not exist in the library's collection.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    ///
    /// [`Predicate::Not`]: crate::Predicate::Not
    MaybeTerm Context::new_not(FfiTerm t) {
        return abc_new_not(*this, t);
    }

    /// Create a new variable term, with the provided name.
    ///
    /// It is important to remember that terms must have unique names! All terms
    /// are required to use SSA naming within a context. This includes terms defined within their own summaries.
    ///
    /// # Errors
    /// - [`ErrorCode::NullPointer`] is returned if the string passed is null.
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global context is poisoned.
    /// - [`ErrorCode::InvalidContext`] is returned if the provided context does not exist in the library's collection.
    MaybeTerm Context::new_var(const char *s) {
        return abc_new_var(*this, s);
    }

    /// Create a new `cast` expression. This corresponds to the `as` operator in WGSL.
    ///
    /// `source_term` is the term to cast, and `ty` is the type to cast it to.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if the term was not valid.
    ///
    /// # Errors
    /// - [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms or Types is poisoned.
    /// - [`ErrorCode::InvalidType`] is returned if `ty` is not found in the context.
    /// - [`ErrorCode::InvalidTerm`] is returned if `source_term` is not found in the context.
    /// - [`ErrorCode::WrongType`] is returned if the type passed is not a scalar type.
    MaybeTerm Context::new_cast(FfiTerm source_term, FfiAbcType ty) {
        return abc_new_cast(*this, source_term, ty);
    }

    /// Create a new comparison term, e.g. `x > y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// - [`ErrorCode::InvalidTerm`] is returned if either `lhs` or `rhs` do not exist in the library's collection.
    MaybeTerm Context::new_cmp_term(CmpOp op, FfiTerm lhs, FfiTerm rhs) {
        return abc_new_cmp_term(*this, op, lhs, rhs);
    }

    /// Create a new index access term, e.g. `x[y]`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `base` or `index` do not exist in the context.
    MaybeTerm Context::new_index_access(FfiTerm base, FfiTerm index) {
        return abc_new_index_access(*this, base, index);
    }

    /// Create a new struct access term, e.g. `x.y`.
    ///
    /// # Arguments
    /// - `base`: The base term whose field is being accessed
    /// - `field`: The name of the field being accessed.
    /// - `ty`: The type of the struct being accessed. This is needed for term validation.
    /// - `field_idx`: The index of the field in the structure being accessed.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term or type could not be found.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `base` does not exist in the context.
    /// [`ErrorCode::PoisonedLock`] is returned if the lock on the global Terms or Types container is poisoned.
    MaybeTerm Context::new_struct_access(FfiTerm base, const char *field, FfiAbcType ty, size_t field_idx) {
        return abc_new_struct_access(*this, base, field, ty, field_idx);
    }

    /// Create a new splat term, e.g. `vec3(x)`.
    ///
    /// A `splat` is just shorthand for a vector of size `size` where each element is `term`.
    ///
    /// # Arguments
    /// - `term`: The term to splat.
    /// - `size`: The number of elements in the vector. Must be between 2 and 4  (inclusive).
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the library's collection.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::ValueError` if `size` is not between 2 and 4.
    MaybeTerm Context::new_splat(FfiTerm term, uint32_t size) {
        return abc_new_splat(*this, term, size);
    }

    /// Create a new literal term.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a new `Literal` variant of `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_literal(Literal lit) {
        return abc_new_literal(*this, lit);
    }

    /// Create a binary operation term, e.g. `x + y`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the library's collection.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_binary_op(BinaryOp op, FfiTerm lhs, FfiTerm rhs) {
        return abc_new_binary_op(*this, op, lhs, rhs);
    }

    /// Create a new unary operation term, e.g. `-x`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_unary_op(UnaryOp op, FfiTerm term) {
        return abc_new_unary_op(*this, op, term);
    }

    /// Create a new term corresponding to wgsl's [`max`](https://www.w3.org/TR/WGSL/#max-float-builtin) bulitin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_max(FfiTerm lhs, FfiTerm rhs) {
        return abc_new_max(*this, lhs, rhs);
    }

    /// Create a new term corresponding to wgsl's [`min`](https://www.w3.org/TR/WGSL/#min-float-builtin) builtin.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_min(FfiTerm lhs, FfiTerm rhs) {
        return abc_new_min(*this, lhs, rhs);
    }

    /// Create a new term akin to wgsl's [`select`](https://www.w3.org/TR/WGSL/#select-builtin) bulitin.
    ///
    /// ### Note: In this method, the condition is the first argument, while it is the last argument in the WGSL builtin.
    ///
    ///
    /// # Arguments
    /// - `iftrue`: The expression this term evaluates to if `predicate` is true
    /// - `iffalse`: The expression this term evaluates to if `predicate` is false
    /// - `Predicate`: The term that determines the resolution of `iftrue` or `iffalse`.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs`, `m`, or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_select(FfiTerm iftrue, FfiTerm iffalse, FfiTerm predicate) {
        return abc_new_select(*this, iftrue, iffalse, predicate);
    }

    /// Create a new term corresponding to wgsl's `vec2` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_vec2(FfiTerm term_0, FfiTerm term_1, FfiAbcType ty) {
        return abc_new_vec2(*this, term_0, term_1, ty);
    }

    /// Create a new term corresponding to wgsl's `vec3` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `term_3`: The third term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::InvalidType` if `ty` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_vec3(FfiTerm term_0, FfiTerm term_1, FfiTerm term_2, FfiAbcType ty) {
        return abc_new_vec3(*this, term_0, term_1, term_2, ty);
    }

    /// Create a new term corresponding to wgsl's `vec4` type. This should not be confused with array types.
    ///
    /// # Arguments
    /// - `term_0`: The first term in the vector
    /// - `term_1`: The second term in the vector
    /// - `term_2`: The third term in the vector
    /// - `term_3`: The fourth term in the vector
    /// - `ty`: The type of the terms in the vector
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the terms do not exist in the context.
    /// `ErrorCode::InvalidType` if `ty` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_vec4(FfiTerm term_0, FfiTerm term_1, FfiTerm term_2, FfiTerm term_3, FfiAbcType ty) {
        return abc_new_vec4(*this, term_0, term_1, term_2, term_3, ty);
    }

    /// Create a new `array_length` term corresponding to the `arrayLength` operator in WGSL.
    /// Note that `wgsl` only defines this method for dynamically sized arrays. This method is
    /// not valid for fixed-sized arrays, matrices, or vectors.
    ///
    /// ### Notes
    /// Do not use this term to add a constraint on the length of an array. Instead, invoke the solver's `mark_length` method.
    /// This method is only meant to be used in the case that `arrayLength` is spelled in wgsl source.
    ///
    /// # Arguments
    /// - `term`: The array term that this expression is being applied to.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_array_length(FfiTerm term) {
        return abc_new_array_length(*this, term);
    }

    /// Create a new `store` term.
    ///
    /// There is no explicit wgsl method that this corresponds to. However, it must be used when writing to array in
    /// order to respect SSA requirements. This method creates a new term where each array element is the same as the
    /// original array, except for the element at `index`, which is `value`.
    ///
    /// # Arguments
    /// - `term`: The array term that is being written to.
    /// - `index`: The index of the array that is being written to.
    /// - `value`: The value that is being written to the array.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_array_store(FfiTerm base, FfiTerm index, FfiTerm value) {
        return abc_new_store(*this, base, index, value);
    }

    /// Create a new struct store term.
    ///
    /// There is no explicit wgsl method that this corresponds to. However, it must be used when writing to a struct in
    /// order to respect SSA requirements. This method creates a new term where each field is the same as the
    /// original struct, except for the field at `field_idx`, which is `value`.
    ///
    /// # Arguments
    /// - `term`: The struct term that is being written to.
    /// - `field_idx`: The index of the field that is being written to.
    /// - `value`: The value that is being written to the struct.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_struct_store(FfiTerm base, size_t field_idx, FfiTerm value) {
        return abc_new_struct_store(*this, base, field_idx, value);
    }

    /// Create a new absolute value term corresponding to the [`abs`](https://www.w3.org/TR/WGSL/#abs-float-builtin) operator in WGSL.
    ///
    /// # Arguments
    /// - `term`: The term that is being passed to the `abs` operator.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if `term` does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_abs(FfiTerm term) {
        return abc_new_abs(*this, term);
    }

    /// Create a new `pow` term corresponding to the [`pow`](https://www.w3.org/TR/WGSL/#pow-builtin) builtin in WGSL.
    ///
    /// # Arguments
    /// - `base`: The base of the power operation.
    /// - `exponent`: The exponent of the power operation.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `base` or `exponent` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_pow(FfiTerm base, FfiTerm exponent) {
        return abc_new_pow(*this, base, exponent);
    }

    /// Create a new term corresponding to wgsl's [`dot`](https://www.w3.org/TR/WGSL/#dot-builtin) builtin.
    ///
    /// # Arguments
    /// - `lhs`: The left-hand side of the dot product.
    /// - `rhs`: The right-hand side of the dot product.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if either `lhs` or `rhs` do not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    MaybeTerm Context::new_dot(FfiTerm lhs, FfiTerm rhs) {
        return abc_new_dot(*this, lhs, rhs);
    }

    /// Create a new `mat2x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec2`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec2`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat2x2(FfiTerm row_0, FfiTerm row_1, FfiAbcType ty) {
        return abc_new_mat2x2(*this, row_0, row_1, ty);
    }

    /// Create a new `mat2x3` term
    ///
    /// The components of the matrix must correspond to `vec3` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat2x3(FfiTerm row_0, FfiTerm row_1, FfiAbcType ty) {
        return abc_new_mat2x3(*this, row_0, row_1, ty);
    }

    /// Create a new `mat2x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat2x4(FfiTerm row_0, FfiTerm row_1, FfiAbcType ty) {
        return abc_new_mat2x4(*this, row_0, row_1, ty);
    }

    /// Create a new `mat3x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec2`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec2`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat3x2(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiAbcType ty) {
        return abc_new_mat3x2(*this, row_0, row_1, row_2, ty);
    }

    /// Create a new `mat3x3` term
    ///
    /// The components of the matrix must correspond to `vec3` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat3x3(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiAbcType ty) {
        return abc_new_mat3x3(*this, row_0, row_1, row_2, ty);
    }

    /// Create a new `mat3x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat3x4(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiAbcType ty) {
        return abc_new_mat3x4(*this, row_0, row_1, row_2, ty);
    }

    /// Create a new `mat4x2` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat4x2(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiTerm row_3, FfiAbcType ty) {
        return abc_new_mat4x2(*this, row_0, row_1, row_2, row_3, ty);
    }

    /// Create a new `mat4x3` term
    ///
    /// The components of the matrix must correspond to `vec2` terms, which can be created via the `new_vec3`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec3`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat4x3(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiTerm row_3, FfiAbcType ty) {
        return abc_new_mat4x3(*this, row_0, row_1, row_2, row_3, ty);
    }

    /// Create a new `mat4x4` term
    ///
    /// The components of the matrix must correspond to `vec4` terms, which can be created via the `new_vec4`
    /// method. A component may also be a term corresponding to a variable or expression that resolves to a `vec4`.
    ///
    /// # Arguments
    /// - `row_0`: The first row of the matrix.
    /// - `row_1`: The second row of the matrix.
    /// - `row_2`: The third row of the matrix.
    /// - `row_3`: The fourth row of the matrix.
    /// - `ty`: The type of the terms in the matrix. This must be a scalar.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either a `Term` if the term was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// `ErrorCode::InvalidTerm` if any of the rows do not exist in the context.
    /// `ErrorCode::InvalidType` if the type does not exist in the context.
    /// `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// `ErrorCode::WrongType` if the type passed is not a scalar type.
    MaybeTerm Context::new_mat4x4(FfiTerm row_0, FfiTerm row_1, FfiTerm row_2, FfiTerm row_3, FfiAbcType ty) {
        return abc_new_mat4x4(*this, row_0, row_1, row_2, row_3, ty);
    }
    /// Get the string representation of the term.
    ///
    /// If the term is invalid, `<NotFound>` is returned.
    /// Note: The returned string *must* be freed by calling `abc_free_string` or this will lead to a memory leak.
    char * Context::term_to_cstr(FfiTerm term) {
        return abc_term_to_cstr(*this, term);
    }

    /// Create a new scalar type as an `FfiAbcType` from an `AbcScalar`.
    ///
    /// # Arguments
    /// - `scalar` The constructed scalar.
    ///
    /// # Safety
    /// The caller must ensure that the `fields` and `types` pointers are not null, and that the `len`
    /// parameter is at least as long as the number of elements and fields.
    ///
    /// This method does check that the pointers are properly aligned and not null.
    /// However, the caller must ensure that the pointers hold at least `len` elements.
    /// Otherwise, behavior is undefined.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if either `fields` or `types` is null.
    /// - `ErrorCode::Alignmenterror` if the pointers are not properly aligned.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidType` if any of the types passed do not exist the context.
    ///
    /// Example:
    /// ```cpp
    /// context.new_Scalar_type(AbcScalar::Sint(4));
    /// ```
    MaybeAbcType Context::new_Scalar_type(AbcScalar scalar) {
        return abc_new_Scalar_type(*this, scalar);
    }

    /// Create a new struct type. Takes a list of fields, each of which is a tuple of a string and an `AbcType`.
    ///
    /// `num_fields` specifies how many fields are passed. It MUST be at least as long as the number of fields and types
    /// passed.
    ///
    /// # Safety
    /// The caller must ensure that the `fields` and `types` pointers are not null, and that the `len`
    /// parameter is at least as long as the number of elements and fields.
    ///
    /// This method does check that the pointers are properly aligned and not null.
    /// However, the caller must ensure that the pointers hold at least `len` elements.
    /// Otherwise, behavior is undefined.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if either `fields` or `types` is null.
    /// - `ErrorCode::Alignmenterror` if the pointers are not properly aligned.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidType` if any of the types passed do not exist the context.
    MaybeAbcType Context::new_Struct_type(const char* *fields, const FfiAbcType *types, size_t num_fields) {
        return abc_new_Struct_type(*this, fields, types, num_fields);
    }
    /// Declare a new `SizedArray` type. `size` is the number of elements in the array. This cannot be 0.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidType` is returned if the type passed does not exist in the context.
    /// - `ErrorCode::PoisonedLock` is returned if the lock on the global contexts is poisoned.
    /// - `ErrorCode::ForbiddenZero` is returned if the size is 0.
    MaybeAbcType Context::new_SizedArray_type(FfiAbcType ty, uint32_t size) {
        return abc_new_SizedArray_type(*this, ty, size);
    }

    /// Declare a new Dynamic Array type of the elements of the type passed.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidType` is returned if the type passed does not exist in the context.
    /// - `ErrorCode::PoisonedLock` is returned if the lock on the global contexts is poisoned.
    /// - `ErrorCode::ForbiddenZero` is returned if the size is 0.
    MaybeAbcType Context::new_DynamicArray_type(FfiAbcType ty) {
        return abc_new_DynamicArray_type(*this, ty);
    }



    /// Solve the constraints for the provided summary.
    ///
    /// Note that the returned solution *must be freed* by calling `abc_free_solution` otherwise
    /// a memory leak will occur.
    ///
    /// This will return an `FfiSolution` which contains the results.
    /// The results are in the form (id, bool). `id` corresponds to the
    /// id of the constraint that was provided when the constraint was added.
    /// the `bool` corresponds to the result of the constraint. It is `true`
    /// if the constraint is always satisfied. In this case, the bounds check
    /// can be removed. `false` means that the constraint could not
    /// be proven to always be satisfied. In this case, the bounds check
    /// must be kept.
    ///
    /// In this current implementation, there is not a distinction
    /// between checks that are always violated and checks that may be violated.
    ///
    ///
    /// # Errors
    /// - `SolverErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `SolverErrorCode::InvalidContext` if the context does not exist.
    /// - `SolverErrorCode::InvalidSummary` if the summary does not exist in the context.
    /// - Many other errors may be returned that occur during the solving process.
    MaybeSolution Context::solve_constraints(FfiSummary summary) {
        return abc_solve_constraints(*this, summary);
    }

    /// Serializes the constraint system to json.
    /// The input should be a file path where the serialized json will be written.
    ErrorCode Context::serialize_constraints(const char* path) {
        return abc_serialize_constraints(*this, path);
    }

    /// Mark a variable as a loop variable.
    ///
    /// A loop variable is any variable that is updated within a loop. Supported operations
    /// are limited to binary operations. For best results, the increment / decrement term must be
    /// a constnat.
    ///
    /// # Arguments
    /// - `var`: The induction variable.
    /// - `init`: The expression the induction variable is initialized to.
    /// - `update_rhs`: The expression that the term is incremented / decremented by
    /// - `update_op`: The operation that is used to update the term.
    ///
    /// Note that `inc_term` and `inc_op` are parts of the expression with the induction term.
    /// That is, only expressions of the form `a = a op b` are allowed, where `a` is the induction variable.
    /// This method should come **before** the `begin_loop` call.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if any of the terms passed are invalid.
    /// - `ErrorCode::UnsupportedOperation` if the operation is not supported for loops.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    ErrorCode Context::mark_loop_variable(FfiTerm var, FfiTerm init, FfiTerm update_rhs, BinaryOp update_op) {
        return abc_mark_loop_variable(*this, var, init, update_rhs, update_op);
    }

    /// Add the constraints of the summary by invoking the function call with the proper arguments.
    ///
    ///
    /// # Arguments
    /// - `summary`: The summary for the function being invoked
    /// - `args`: An array containing the arguments that are passed to the function.
    /// - `num_args`: The size of the `args` array.
    /// - `return_dest`: A fresh variable term that will hold the return value. If the return value is not used,
    /// then this must be the `Empty` term.
    ///
    /// This will add the constraints of the summary to the constraint system, with
    /// the arguments substituted with `args`, and the return value substituted with `return_dest`.
    ///
    /// # Safety
    /// The caller must ensure that `args` holds at least `num_args` elements.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `args` is null and `num_args` is not 0.
    /// - `ErrorCode::AlignmentError` if `args` is not properly aligned.
    /// - `ErrorCode::InvalidTerm` if any of the terms passed do not exist in the context.
    /// - `ErrorCode::InvalidSummary` if the summary does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    MaybeTerm Context::new_call(FfiSummary summary, const FfiTerm *args, uint32_t num_args, FfiTerm return_dest) {
        return abc_new_call(*this, summary, args, num_args, return_dest);
    }

    /// Begins a loop context.
    ///
    /// A loop context is akin to a predicate context. It allows for the convenience methods for `break` and `continue`
    /// to be properly handled. Additionally, all updates to variables within are marked as range constraints.
    ///
    /// The `condition` should correspond to a boolean term where one of the operands has been marked as a loop variable.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::UnsupportedLoopCondition` if the condition is not a `Predicate` or `Var` term.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    ErrorCode Context::begin_loop(FfiTerm condition) {
        return abc_begin_loop(*this, condition);
    }

    /// End a loop context.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInLoopcontext` if there has been no matching call to `begin_loop`.
    ErrorCode Context::end_loop() {
        return abc_end_loop(*this);
    }

    /// Mark an assumption.
    ///
    /// Assumptions are invariants that must hold at all times.
    /// At solving time, these differ from constraints in that they are not inverted
    /// to test for satisfiability.
    ///
    /// # Arguments
    /// - `lhs`: The left hand side of the assumption.
    /// - `op`: The operation that is used to compare the terms.
    /// - `rhs`: The right hand side of the assumption.
    ///
    /// There can only be one assumption per direction for each term. That is,
    /// each term may *either* have one equality assumption (:=) *or* one
    ///  inequality assumption per direction (one less / less equal, one
    /// greater / greater equal). Violating this will result in a
    /// `DuplicateAssumption` error.
    ///
    /// # Errors
    /// - `ErrorCode::InvalidTerm` if any term does not exist in the context.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::DuplicateAssumption` if `lhs` has already had an assumption on the specified boundary.
    ErrorCode Context::add_assumption(FfiTerm lhs, AssumptionOp op, FfiTerm rhs) {
        return abc_add_assumption(*this, lhs, op, rhs);
    }

    /// Begin a summary block.
    ///
    /// A summary block corresponds to a function definition. All constraints
    /// that are added within the summary block are considered to be constraints
    /// of the function.
    ///
    /// A summary block must be ended with a call to `end_summary`, which will
    /// provide the `FfiSummary` that can be used to refer to it.
    ///
    /// # Arguments
    /// - `name`: The name of the function.
    /// - `num_args`: The number of arguments the function takes.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `name` is null.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::SummaryLimitExceeded` if the number of summaries exceeds the limit.
    ErrorCode Context::begin_summary(const char* name, uint8_t num_args) {
        return abc_begin_summary(*this, name, num_args);
    }

    /// End a summary block.
    ///
    /// This will return the `FfiSummary` that can be used to refer to the summary.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInSummary` if there has been no matching call to `begin_summary`.
    MaybeSummary Context::end_summary() {
        return abc_end_summary(*this);
    }

    /// Add an argument to the active summary.
    ///
    /// The `begin_summary` method must be called before this method.
    ///
    /// # Arguments
    /// - `name`: The name of the argument.
    /// - `ty`: The type of the argument.
    ///
    /// # Returns
    /// A `MaybeTerm` which is either an `FfiTerm` if the argument was successfully created, or an `ErrorCode` if a provided term was invalid.
    ///
    /// # Errors
    /// - `ErrorCode::NullPointer` if `name` is null.
    /// - `ErrorCode::InvalidType` if the type does not exist in the context.
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::NotInSummary` if there has been no matching call to `begin_summary`.
    MaybeTerm Context::add_argument(const char* name, FfiAbcType ty) {
        return abc_add_argument(*this, name, ty);
    }

    /// Mark a break statement. (Currently unimplemented)
    ///
    /// # Errors
    /// `ConstraintError::NotImplemented`
    ErrorCode Context::mark_break() {
        return abc_mark_break(*this);
    }

    /// Mark a continue statement. (Currently unimplemented)
    ///
    /// # Errors
    /// `ConstraintError::NotImplemented`
    ErrorCode Context::mark_continue() {
        return abc_mark_continue(*this);
    }

    /// Mark the type of the provided term.
    ///
    /// # Errors
    /// [`DuplicateType`] if the type of the term has already been marked.
    ///
    /// [`DuplicateType`]: crate::ConstraintError::DuplicateType
    ErrorCode Context::mark_type(FfiTerm term, FfiAbcType ty) {
        return abc_mark_type(*this, term, ty);
    }

    /// Begin a predicate block. This indicates to the solver
    /// that all expressions that follow can be filtered by the predicate.
    /// Any constraint that falls within a predicate becomes a soft constraint
    ///
    /// In other words, it would be as if all constraints were of the form
    /// ``p -> c``
    /// Nested predicate blocks end up composing the predicates. E.g.,
    /// ``begin_predicate_block(p1)`` followed by ``begin_predicate_block(p2)``
    /// would mark all constraints as ``p1 && p2 -> c``
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    ErrorCode Context::begin_predicate_block(FfiTerm predicate) {
        return abc_begin_predicate_block(*this, predicate);
    }

    /// End the active predicate block.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::PredicateStackEmpty` if there is no active predicate block.
    ErrorCode Context::end_predicate_block() {
        return abc_end_predicate_block(*this);
    }

    /// Add a constraint to the system that narrows the domain of `term`.
    ///
    /// Any active predicates are automatically applied.
    ///
    /// There can only be one constraint per boundary side for each term. That is,
    /// each term may *either* have one equality constraint (:=) *or* one
    /// inequality constraint per side (one less / less equal, one
    /// greater / greater equal). Violating this will result in a
    /// `DuplicateConstraint` error.
    ///
    /// # Arguments
    /// - `lhs`: The left hand side of the constraint.
    /// - `op`: The operation of the constraint.
    /// - `rhs`: The right hand side of the constraint.
    /// - `id`: The identifier that the constraint system will use to refer to
    /// the constraint. Results will reference this id.
    ///
    ///
    /// # Errors
    /// - `TypeMismatch` if the type of `term` is different from the type of
    /// `rhs` and `op` is `ConstraintOp::Assign`
    /// - `MaxConstraintCountExceeded` if the maximum number of constraints has
    /// been exceeded.
    ErrorCode Context::add_constraint(FfiTerm lhs, ConstraintOp op, FfiTerm rhs, uint32_t id) {
        return abc_add_constraint(*this, lhs, op, rhs, id);
    }

    /// Mark a return statement.
    ///
    /// # Arguments
    /// - `retval`: The term that corresponds to the return value. If no value
    /// is returned, this *must* be the `Empty` term.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    ErrorCode Context::mark_return(FfiTerm retval) {
        return abc_mark_return(*this, retval);
    }

    /// Mark the length of an array's dimension. Only valid for dynamically
    /// sized arrays.
    ///
    ///
    /// It is *strongly* preferred to use the type system to mark the variable as
    /// a `SizedArray` type.
    /// This should *only* be used for dynamic arrays whose size is determined later.
    ///
    /// # Arguments
    /// - `term`: The array term to mark the dimension of
    /// - `dim`: The term that corresponds to the 0-based dimension of the array.
    /// - `size`: The size of the array.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    ErrorCode Context::mark_array_length(FfiTerm term, uint8_t dim, uint32_t size) {
        return abc_mark_array_length(*this, term, dim, size);
    }
    
    /// Mark the range of a runtime constant. Sugar for a pair of assumptions
    /// (var >= min) and (var <= max).
    ///
    /// The `lower` and `upper` terms must be literals of the same type.
    ///
    /// ## Notes
    /// - The current predicate block is ignored, though the constraints *are*
    /// added to the active summary (or global if no summary is active)
    /// - This is not meant to be used for loop variables.
    /// Use `mark_loop_variable` for that.
    ///
    /// # Arguments
    /// - `term`: The term to mark the range of.
    /// - `lower`: The lower bound of the range.
    /// - `upper`: The upper bound of the range.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidTerm` if the term does not exist in the context.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::WrongType` if the two literals are not the same variant.
    ErrorCode Context::mark_range(FfiTerm term, Literal lower, Literal upper) {
        return abc_mark_range(*this, term, lower, upper);
    }

    /// Mark the return type for the active summary.
    ///
    /// It is not necessary to call this method for functions that do not return
    /// a value.
    ///
    /// # Errors
    /// - `ErrorCode::PoisonedLock` if the lock on the global contexts is poisoned.
    /// - `ErrorCode::InvalidContext` if the context does not exist.
    /// - `ErrorCode::InvalidSummary` if there is no active summary
    /// - `ErrorCode::DuplicateType` if the return type has already been marked for the summary
    ErrorCode Context::mark_return_type(FfiAbcType ty) {
        return abc_mark_return_type(*this, ty);
    }

    FfiTerm Context::get_empty_term() {
        return abc_get_empty_term();
    }

    inline std::ostream& operator<<(std::ostream& os, SolverResult result) {
        switch (result) {
          case SolverResult::Yes: os << "Yes"; break;
          case SolverResult::Maybe: os << "Maybe"; break;
          case SolverResult::No: os << "No"; break;
        }
        return os;
      }
}