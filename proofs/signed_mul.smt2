; SPDX-FileCopyrightText: 2024 University of Rochester
;
; SPDX-License-Identifier: MIT OR Apache-2.0
(set-logic ALL)

; (set-option :produce-models true)
; (set-option :incremental true)


;; pair
; (declare-datatypes (T U) ((Pair (mk-pair (first T) (second U)))))

(define-sort BV8 () (_ BitVec 8))
(define-sort BV16 () (_ BitVec 16))



(define-fun extended_mul ((a BV8) (b BV8) (signed Bool)) BV16
  (ite signed 
    (bvmul ((_ sign_extend 8) a) ((_ sign_extend 8) b))
    (bvmul ((_ zero_extend 8) a) ((_ zero_extend 8) b))
  )
)

;; the number of times the result of the multiplication overflows in 32 bits.
(define-fun wrap_count ((a BV16)) BV16
    (bvashr a (_ bv8 16))
)


;; Okay, so let's multiply x and y by some `z` in 32-bits

(declare-const lower_bound BV8)
(declare-const upper_bound BV8)
(assert (bvult lower_bound upper_bound))

; (check-sat)

(declare-const x BV8)

;; Check if we can multiply x by z, where they both overflow

(define-const wrap_low BV16 (wrap_count (extended_mul x lower_bound true)))
(define-const wrap_high BV16 (wrap_count (extended_mul x upper_bound true)))

;; So they wrap the same number of times. Now, check if we can get a normal result where the low is greater than high

(define-const wrapped_low BV8 (bvmul x lower_bound))
(define-const wrapped_high BV8 (bvmul x upper_bound))


;; Assert that x is lower than high

(push)
(echo "Test overflow can cause low*x to be greater than high * x")
(assert (bvslt wrapped_low wrapped_high))
(check-sat)
(pop)

(echo "Test that when wrap count of high is 1 more than wrap count of low, low * x can be < than high * x even though the gap between them is < 1 wrap.")
(push)
(assert (= (bvadd #x0001 wrap_low) wrap_high))
(assert (bvsle (bvsub (extended_mul x upper_bound true) (extended_mul x lower_bound true)) #x00ff))
; (assert (bvult wrapped_low wrapped_high))
(check-sat)
(pop)

(push)
(echo "Test that when wrap count of high is 1 more than wrap count of low, low * x can be > than high * x.
Expected: sat")
(assert (= (bvadd #x0001 wrap_low) wrap_high))
(assert (bvsle (bvsub (extended_mul x upper_bound true) (extended_mul x lower_bound true)) #x00ff))
(assert (bvsgt wrapped_low wrapped_high))
(check-sat)
(pop)

(push)
(assert (= wrap_low wrap_high))
(assert (bvsgt wrap_low (_ bv0 16)))
;; This should be unsat.
(assert (bvsgt wrapped_low wrapped_high))

(echo "Test low*x > high*x, when low*x and high*x wrap the same number of times. Expected: unsat")
(check-sat)
(pop)

(push)
(check-sat)
