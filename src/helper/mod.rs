/*!
The helper module contains several classes used to translate
 */

#![allow(unused)] // Temporary, for development only.

mod converter;
mod interface;

pub use interface::*;

mod visitor {
    use crate::{AbcExpression, FastHashMap, FastHashSet, Predicate, Term};

    struct DfsTermVisitor<'a> {
        visited: FastHashSet<&'a Term>,
        dependency_map: FastHashMap<&'a Term, FastHashSet<&'a Term>>,
        current_term: &'a Term,
    }

    impl<'a> DfsTermVisitor<'a> {
        fn visit(start: &'a Term) {
            let mut visitor = DfsTermVisitor {
                visited: FastHashSet::default(),
                dependency_map: FastHashMap::default(),
                current_term: start,
            };
        }

        fn visit_with_map(start: &'a Term, map: FastHashMap<&'a Term, FastHashSet<&'a Term>>) {
            let mut visited: FastHashSet<&Term> = map.keys().copied().collect();
            let mut visitor = DfsTermVisitor {
                visited: FastHashSet::default(),
                dependency_map: map,
                current_term: start,
            };
        }

        /// Visits `term`, then adds both it and its dependencies to those of `self.current_term`.
        fn visit_and_merge(&mut self, other: &'a Term) {
            if self.current_term == other {
                // Do nothing. A term cannot depend on itself.
                return;
            }
            let old_term = self.current_term;
            self.visit_term(other);
            self.current_term = old_term;
            // Save the old term

            // Until `get_many_mut` becomes stable (https://github.com/rust-lang/rust/issues/97601), we clone
            // unnecessarily.
            if let Some(other_deps) = self.dependency_map.get(other).cloned() {
                self.dependency_map
                    .entry(self.current_term)
                    .or_default()
                    .extend(other_deps);
            }
        }

        // Visit the sub-terms in the predicate.
        fn visit_predicate<T: AsRef<Predicate>>(&mut self, p: &'a T) {
            match *p.as_ref() {
                Predicate::And(ref p1, ref p2) | Predicate::Or(ref p1, ref p2) => {
                    self.visit_predicate(p1);
                    self.visit_predicate(p2);
                }
                Predicate::Not(ref p) => {
                    self.visit_predicate(p);
                }
                Predicate::Comparison(_, ref t1, ref t2) => {
                    self.visit_and_merge(t1);
                    self.visit_and_merge(t2);
                }
                Predicate::Unit(ref t) => self.visit_and_merge(t),
                Predicate::False | Predicate::True => {
                    // No terms to visit.
                }
            }
        }

        fn visit_expr<T: AsRef<AbcExpression>>(&mut self, e: &'a T) {
            use AbcExpression as E;
            match *e.as_ref() {
                // Expressions with one child term
                E::Splat(ref t, _)
                | E::ArrayLength(ref t)
                | E::Cast(ref t, _)
                | E::FieldAccess { base: ref t, .. }
                | E::UnaryOp(_, ref t)
                | E::Abs(ref t) => self.visit_and_merge(t),
                // Expressions with two child terms
                E::BinaryOp(_, ref t1, ref t2)
                | E::IndexAccess {
                    base: ref t1,
                    index: ref t2,
                }
                | E::StructStore {
                    base: ref t1,
                    value: ref t2,
                    ..
                }
                | E::Dot(ref t1, ref t2)
                | E::Max(ref t1, ref t2)
                | E::Min(ref t1, ref t2)
                | E::Pow {
                    base: ref t1,
                    exponent: ref t2,
                } => {
                    self.visit_and_merge(t1);
                    self.visit_and_merge(t2);
                }
                // Expressions with three child terms
                E::Select(ref t1, ref t2, ref t3)
                | E::Store {
                    base: ref t1,
                    index: ref t2,
                    value: ref t3,
                } => {
                    self.visit_and_merge(t1);
                    self.visit_and_merge(t2);
                    self.visit_and_merge(t3);
                }
                E::Matrix { ref components, .. } | E::Vector { ref components, .. } => {
                    for component in components {
                        self.visit_and_merge(component)
                    }
                }
            }
        }

        /// Visit a term. Add it to the visited set. Then, visit its children.
        /// The term visitor will mark all visited children to the
        fn visit_term(&mut self, t: &'a Term) {
            if self.visited.insert(t) {
                // Has already been visited. Do nothing.
                return;
            }
            match *t {
                Term::Var(_) | Term::Empty | Term::Literal(_) => {
                    // No children to visit.
                }
                Term::Predicate(ref p) => {
                    self.visit_predicate(p);
                }
                Term::Expr(ref e) => {
                    self.visit_expr(e);
                }
            }
        }
    }
}

//  Design:
//  For each term we visit, we append it to a vector
// For each term, its children is a slice of that vector.
