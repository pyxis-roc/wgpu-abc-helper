; SPDX-FileCopyrightText: 2024 University of Rochester
;
; SPDX-License-Identifier: MIT OR Apache-2.0

;; This file proves that the signed interval division logic is correct.
;; We implement our algorithm in SMTLIB, and then try
;; to use smt to find values where the result does not lie
;; within the computed bounds.
(set-logic BV)
(set-option :global-declarations true)
(define-sort bv4 () (_ BitVec 4))


(define-const MINVAL bv4 #x8)
(define-const MAXVAL bv4 #x7)

(define-const NEGTWO bv4 #xe)

(define-const NEGONE bv4 #xf)
(define-const ONE bv4 #x1)



; We have to implement signed division
; We will start with i8.

(define-fun wgsl_div ((a bv4) (b bv4)) bv4
    (ite (= b #x0) a (bvsdiv a b))
)

(define-const zero bv4 #x0)
; We want to make sure our algorithm is correct

(declare-const low_a bv4)
(declare-const high_a bv4)
(declare-const low_b bv4)
(declare-const high_b bv4)

(assert (bvsge high_a low_a))
(assert (bvsge high_b low_b))

(declare-const a bv4)
(declare-const b bv4)

(assert (and (bvsge a low_a) (bvsle a high_a)))
(assert (and (bvsge b low_b) (bvsle b high_b)))

(define-const res bv4 (wgsl_div a b))

;; We are going to prove this for int4.
;; The properties should hold for all values, though.
;; There is no reason that they wouldn't.

;; We need the result...



;; get the zero value

(define-fun min ((a bv4) (b bv4)) bv4
    (ite (bvslt a b) a b)
)

(define-fun max ((a bv4) (b bv4)) bv4
    (ite (bvslt a b) b a)
)


(define-const init_low bv4 #x8)
(define-const init_high bv4 #x7)


(push)
;; fastpath. If low_a == high_a and low_b == high_b


(echo "Fastpath correctness. Expecting unsat.")
(assert (not (=> 
    (and (= low_b high_b) (= low_a high_a))
    (ite
        (= low_b zero) (= res low_a)
        (ite
            (= low_a zero) (= res zero)
            (= res (wgsl_div low_a low_b))
    )
))))
(check-sat)
(pop)

;; Exclude the fastpath now.

;; Invert fastpath condition
(assert (not (and (= low_b high_b) (= low_a high_a))))

;; Exclude the special case.
(assert (not (and (= a MINVAL) (= b NEGONE))))

(define-const low_init bv4 #x7)
(define-const high_init bv4 #x8)


(define-fun has_value ((lo bv4) (hi bv4) (val bv4)) Bool
    (
        and
        (bvsle lo val)
        (bvsge hi val)
    )
)

;; zero check for rhs


(define-const b_zero_check_hi bv4
    (ite (has_value low_b high_b zero) high_a high_init)
)

(define-const step0_hi bv4
    (ite (has_value low_a high_a zero) (max zero b_zero_check_hi) b_zero_check_hi)
)


;; Get max mag positive a

(declare-const min_mag_pos_a bv4)
(declare-const max_mag_pos_a bv4)
(declare-const min_mag_pos_b bv4)
(declare-const max_mag_pos_b bv4)

(declare-const min_mag_neg_a bv4)
(declare-const max_mag_neg_a bv4)
(declare-const min_mag_neg_b bv4)
(declare-const max_mag_neg_b bv4)

(assert (= min_mag_pos_a (max low_a ONE)))
(assert (= min_mag_pos_b (max low_b ONE)))
(assert (= min_mag_neg_a (min high_a NEGONE)))
(assert (= min_mag_neg_b (min high_b NEGONE)))


(assert (= max_mag_pos_a high_a))
(assert (= max_mag_pos_b high_b))
(assert (= max_mag_neg_a low_a))
(assert (= max_mag_neg_b low_b))

(define-const maybe_a_neg Bool
    (bvsle low_a zero)
)

(define-const maybe_b_neg Bool 
    (bvsle low_b zero)
)

(define-const maybe_a_pos Bool
    (bvsge high_a zero)
)

(define-const maybe_b_pos Bool
    (bvsge high_b zero)
)

(define-const b_only_neg Bool
    (bvslt high_b zero)
)

(define-const b_only_pos Bool
    (bvsgt low_b zero)
)

(define-const a_only_neg Bool
    (bvslt high_a zero)
)

(define-const a_only_pos Bool
    (bvsgt low_a zero)
)


;; set high bound from positives..
(define-const step1_high bv4
    (ite
        (and maybe_a_pos maybe_b_pos)
        (max
            (bvsdiv max_mag_pos_a min_mag_pos_b)
            step0_hi
        )
        step0_hi
    )
)

;; Set the high bound from negatives..

(define-const step2_high bv4
    (ite (and maybe_a_neg maybe_b_neg)
        (ite (and (= max_mag_neg_a MINVAL) (= min_mag_neg_b NEGONE))
            (ite (distinct high_a MINVAL)
                MAXVAL
                (ite (distinct low_b NEGONE)
                    (bvsdiv MINVAL NEGTWO)
                    (ite
                        (and (bvsgt high_b zero) a_only_neg)
                        (bvsdiv MINVAL high_b)
                        step1_high)))
            (max
                (bvsdiv max_mag_neg_a min_mag_neg_b)
                step1_high))
        step1_high
    )
)



;;set the high bound when a is only neg and b is only pos.
(define-const step3_high bv4
    (ite
        (and a_only_neg b_only_pos)
        (max
            (bvsdiv min_mag_neg_a max_mag_pos_b)
            step2_high
        )
        step2_high
    )
)

(define-const high_bound bv4
    (ite
        (and a_only_pos b_only_neg)
        (max
            (bvsdiv min_mag_pos_a max_mag_neg_b)
            step3_high
        )
        step3_high
    )
)


;; low bound logic..

(define-const b_zero_check_lo bv4
    (ite (has_value low_b high_b zero) low_a low_init)
)
(define-const step0_lo bv4
    (ite (has_value low_a high_a zero) (min zero b_zero_check_lo) b_zero_check_lo)
)

(define-const low_bound bv4
    (ite (and a_only_neg b_only_neg)
        (min (bvsdiv min_mag_neg_a max_mag_neg_b) step0_lo)
        (ite (and a_only_pos b_only_pos)
            (min (bvsdiv min_mag_pos_a max_mag_pos_b) step0_lo)
            (ite (and maybe_a_pos maybe_b_neg)
                (min (bvsdiv max_mag_pos_a min_mag_neg_b) step0_lo)
                (ite (and maybe_a_neg maybe_b_pos)
                    (min (bvsdiv max_mag_neg_a min_mag_pos_b) step0_lo)
                    step0_lo))))
)

;; check that we have properly computed the high bound.

;; (assert (bvsge true_result low_bound))

(echo "[High Bound] Assert result > high_bound. (Expected unsat)")
(push)
(assert (bvsgt res high_bound))
(check-sat)
(pop)



(push)
(echo ("[Low Bound] Assert result < low_bound (Expected unsat)"))
(assert (bvslt res low_bound))
(check-sat)