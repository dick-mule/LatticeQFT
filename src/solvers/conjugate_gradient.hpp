#pragma once

/**
 * @file conjugate_gradient.hpp
 * @brief Templated conjugate-gradient solver for Hermitian positive-definite A.
 *
 * Given an A satisfying A = A† and (ψ, A ψ) > 0 for ψ ≠ 0, solves A x = b
 * by the textbook Hestenes–Stiefel iteration:
 *
 *     r₀ = b − A x₀         p₀ = r₀
 *     α_k = ‖r_k‖² / ⟨p_k, A p_k⟩
 *     x_{k+1} = x_k + α_k p_k
 *     r_{k+1} = r_k − α_k A p_k
 *     β_k     = ‖r_{k+1}‖² / ‖r_k‖²
 *     p_{k+1} = r_{k+1} + β_k p_k
 *
 * The matrix `A` is passed as a callable with signature
 *
 *     void apply_A(const Spinor& x, Spinor& y)
 *
 * which makes the solver agnostic to whether `A` is the Dirac D†D, a
 * gauge Laplacian, or any other Hermitian PSD operator. The Spinor type
 * is whatever supports the algebra primitives in spinor_field.hpp
 * (axpy, scale_add, copy, inner_product, norm_squared, zero).
 *
 * Stopping criterion: relative residual ‖r_k‖ / ‖b‖ < tol, with a hard
 * cap of `max_iters` to bound pathological cases.
 *
 * Future Phase 5 (lattice QCD) will need the deflated variant; the
 * interface here is intentionally narrow so deflation can wrap it without
 * rewriting the inner loop.
 */

#include "../fields/spinor_field.hpp"

#include <cmath>

namespace lqft::solvers
{

struct CGResult
{
    int    iterations;
    double final_residual;
    bool   converged;
};

/// Solve A x = b with the conjugate-gradient method.
/// On entry `x` is the initial guess (use `x.zero()` for a cold start).
template<int Dim, int Nc, typename ApplyA>
CGResult conjugateGradient(
    ApplyA&& apply_A,
    const SpinorField<Dim, Nc>& b,
    SpinorField<Dim, Nc>& x,
    double tolerance,
    int    max_iters,
    SpinorField<Dim, Nc>& r,
    SpinorField<Dim, Nc>& p,
    SpinorField<Dim, Nc>& Ap)
{
    using Complex = std::complex<double>;

    // r = b - A x
    apply_A(x, Ap);                  // Ap ← A x
    copy(b, r);                      // r  ← b
    axpy(Complex{-1.0, 0.0}, Ap, r); // r  ← b − A x
    copy(r, p);                      // p  ← r

    const double b_norm2 = norm_squared(b);
    if (b_norm2 == 0.0)
    {
        x.zero();
        return CGResult{ 0, 0.0, true };
    }

    double r_norm2 = norm_squared(r);
    const double tol2 = tolerance * tolerance * b_norm2;

    int k = 0;
    for (; k < max_iters; ++k)
    {
        if (r_norm2 <= tol2)
            return CGResult{ k, std::sqrt(r_norm2 / b_norm2), true };

        apply_A(p, Ap);
        // α = r_norm² / ⟨p, A p⟩.  ⟨p, A p⟩ is real (and positive) when A is HPD;
        // we take the real part to suppress accumulated round-off i ε.
        const Complex pAp_c = inner_product(p, Ap);
        const double  pAp   = pAp_c.real();
        if (!(pAp > 0.0))
            // A is not positive-definite (or pathological round-off). Bail.
            return CGResult{ k, std::sqrt(r_norm2 / b_norm2), false };

        const double alpha = r_norm2 / pAp;

        axpy(Complex{ alpha, 0.0}, p,  x); // x ← x + α p
        axpy(Complex{-alpha, 0.0}, Ap, r); // r ← r − α A p

        const double r_norm2_new = norm_squared(r);
        const double beta = r_norm2_new / r_norm2;
        scale_add(Complex{ beta, 0.0 }, r, p); // p ← r + β p

        r_norm2 = r_norm2_new;
    }

    return CGResult{ k, std::sqrt(r_norm2 / b_norm2), r_norm2 <= tol2 };
}

} // namespace lqft::solvers
