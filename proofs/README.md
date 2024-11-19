<!--
SPDX-FileCopyrightText: 2024 University of Rochester

SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Proofs
Included are smtlib proofs of the interval arithmetic operations on the "Basic" Intervals.
Basic intervals are those that are represented as `[low, high]` (as opposed to "Compound" intervals which are Unions of basic intervals).

## Methodology
Each operation is proved as follows:
1. Declare two intervals, `[a_low, a_high]`, and `[b_low, b_high]`these intervals.
2. Declare `a` and `b` and assert `a_low <= a <= a_high` and `b_low <= b <= b_high`
3. `low_bound` and `high_bound` are computed from `a_low`, `a_high`, `b_low`, and `b_high` in the same manner as the implementation.
4. Assert `(a op b) < low_bound` and check
5. Assert `(a op b) > high_bound` and check

If both checks return `unsat`, then our algorithm is correct.
