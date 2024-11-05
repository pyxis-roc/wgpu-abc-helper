/*!
The converter module contains several classes that process
a `ConstraintModule`
*/

#![allow(unused)] // Temporary, for development only.

use crate::{ConstraintModule, FastHashSet, Term};
/// Strips away assumptions and terms that are not used in any constraints.
/// This is a visitor.
struct Filter {}

// impl Filter {
//     fn filter(&self, module: &mut ConstraintModule) -> ConstraintModule {
//         // First, go through global constraints.
//         for term in module.global_constraints {

//         }
//     }
// }

// In: Summaries, each containing a set of constraints and assumptions.
// Out: A refined set of summaries, where each assumption is strictly related to a constraint
// or another assumption.

// First, we go through every single constraint and mark every single term that is referenced.
// To do this, we need a constraint visitor.

//

// Should this be per-function or global?
