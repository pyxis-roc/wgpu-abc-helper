#![allow(unused)]

use std::sync::atomic::AtomicU32;

use abc_helper::{
    solvers::interval::SolverResult, AbcType, CmpOp, ConstraintHelper, ConstraintInterface, Handle,
    Term, Var,
};
use rstest::{fixture, rstest};
use serde::Serialize;
use serde_json::{
    ser::{Formatter, PrettyFormatter},
    Serializer,
};

#[fixture]
fn constraint_helper() -> ConstraintHelper {
    ConstraintHelper::default()
}

fn u32_ty() -> Handle<AbcType> {
    AbcType::mk_u32()
}

macro_rules! abc_new_array {
    ($ty:expr) => {
        AbcType::DynamicArray { ty: $ty }.into()
    };
    ($ty:expr, $size:expr) => {
        AbcType::SizedArray {
            ty: $ty,
            size: $size,
        }
        .into()
    };
}

fn u32_array_ty() -> Handle<AbcType> {
    abc_new_array!(u32_ty())
}

fn u32_array_sized(n: std::num::NonZeroU32) -> Handle<AbcType> {
    abc_new_array!(u32_ty(), n)
}

fn new_uniform_var(helper: &mut ConstraintHelper) -> Term {
    static GLOBAL_COUNTER: AtomicU32 = AtomicU32::new(0);

    let current_uniform_id = GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let id_as_str = current_uniform_id.to_string();
    let mut name = String::with_capacity(8 + id_as_str.len());
    name.push_str("uniform_");
    name.push_str(&id_as_str);
    let var = helper.declare_var(Var { name }).unwrap();
    helper.mark_type(&var, &u32_ty()).unwrap();
    helper.mark_uniform_var(&var).unwrap();
    var
}

fn new_runtime_array_var(helper: &mut ConstraintHelper, name: &str) -> Term {
    let var = helper.declare_var(Var { name: name.into() }).unwrap();
    helper.mark_type(&var, &u32_array_ty()).unwrap();
    helper.mark_uniform_var(&var).unwrap();
    var
}

fn new_static_array_var(helper: &mut ConstraintHelper, name: &str, size: u32) -> Term {
    let size = std::num::NonZeroU32::new(size).unwrap();
    let var = helper.declare_var(Var { name: name.into() }).unwrap();
    helper.mark_type(&var, &u32_array_sized(size)).unwrap();
    helper.mark_uniform_var(&var).unwrap();
    var
}

#[rstest]
fn test_uniform_indexing_static(mut constraint_helper: ConstraintHelper) {
    let uniform_var = new_uniform_var(&mut constraint_helper);

    // Start simple...
    constraint_helper.begin_summary("main".into(), 0u8).unwrap();
    let id: u32 = 0;
    constraint_helper
        .add_constraint(
            &uniform_var,
            abc_helper::ConstraintOp::Cmp(CmpOp::Lt),
            &Term::new_literal(64u32),
            id,
        )
        .unwrap();
    let idx = constraint_helper.end_summary().unwrap();

    let solver_result = constraint_helper.solve(idx).unwrap();

    // let mut s = serde_json::Serializer::with_formatter(std::io::stdout(), PrettyFormatter::new());
    // let result = constraint_helper.get_module().serialize(&mut s);

    assert_eq!(solver_result.len(), 1);
    let combined_results = ConstraintHelper::solution_to_result(&solver_result);
    assert!(combined_results
        .iter()
        .any(|(k, v)| { k == &id && *v == SolverResult::Maybe }));
}
