#pragma once

/**
 * @file deflation.hpp
 * @brief Spectral deflation for the conjugate gradient solver.
 *
 * Given a deflation basis `V` (n × k) of approximate low-eigenvalue
 * eigenvectors of the Hermitian PSD operator `A`:
 *
 *   - Precompute `H = V†AV ∈ ℝ^{k×k}` (small, symmetric PSD).
 *   - For each right-hand side `b`, start CG from the projected solution
 *
 *         x₀ = V · H⁻¹ · V†b
 *
 *     The initial residual `r₀ = b − A x₀ = b − A V (V†AV)⁻¹ V†b` then
 *     satisfies `V†r₀ = 0` exactly (a one-line algebraic check), which means
 *     the residual lives in `range(I − V(V†AV)⁻¹V†A)` — the A-orthogonal
 *     complement of `span(V)`. CG iterating from there sees the spectrum of
 *     `A` restricted to that complement, which lacks the `k` smallest
 *     eigenvalues that V approximates.
 *
 * Iteration-count improvement for a Hermitian PSD system: vanilla CG
 * converges in O(√(λ_max / λ_min)); deflation replaces `λ_min` with
 * `λ_{k+1}`, so the convergence factor becomes O(√(λ_max / λ_{k+1})). At
 * small fermion mass (near the chiral limit) where the lowest Dirac
 * eigenvalues collapse toward zero, this is a multiplicative speedup of
 * 5–20× in practice.
 *
 * In finite precision the projection of `V†r` is only approximately zero,
 * so we additionally re-deflate the search direction `p` against `V` at each
 * CG step — this prevents the low modes that were deflated out from being
 * reintroduced by accumulated round-off.
 */

#include "../fields/spinor_field.hpp"
#include "conjugate_gradient.hpp"
#include "lanczos.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace lqft::solvers
{

/// Encapsulates a deflation subspace and the cached `(V†AV)⁻¹` for it.
template<int Dim, int Nc>
class DeflationSubspace
{
public:
    using Spinor = SpinorField<Dim, Nc>;

    DeflationSubspace() = default;

    int size() const { return static_cast<int>(m_V.size()); }

    /// Build by running Lanczos on `A` for `lanczos_iters` iterations, then
    /// keeping the `k_smallest` Ritz vectors with the smallest Ritz values.
    template<typename ApplyA>
    void build(ApplyA&&      apply_A,
               const Spinor& seed_vec,
               int           k_smallest,
               int           lanczos_iters,
               Spinor&       w_scratch,
               Spinor&       tmp_scratch)
    {
        auto basis  = lanczos<Dim, Nc>(apply_A, seed_vec, lanczos_iters,
                                       w_scratch, tmp_scratch);
        auto ritz   = extractSmallestRitz<Dim, Nc>(basis, k_smallest);
        m_V         = std::move(ritz.vectors);
        m_lambda    = std::move(ritz.values);

        const int k = static_cast<int>(m_V.size());
        if (k == 0) return;

        // Compute AV[j] = A · V[j] and cache it for fast deflated-CG steps.
        m_AV.clear();
        m_AV.reserve(static_cast<std::size_t>(k));
        for (int j = 0; j < k; ++j)
        {
            Spinor av = m_V[static_cast<std::size_t>(j)]; // copy template
            apply_A(m_V[static_cast<std::size_t>(j)], av);
            m_AV.push_back(std::move(av));
        }

        // H = V†AV (k × k, symmetric). Invert (Cholesky-style by hand;
        // k is small so straight Gauss-Jordan is fine).
        m_H.assign(static_cast<std::size_t>(k),
                   std::vector<double>(static_cast<std::size_t>(k), 0.0));
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
                m_H[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    = inner_product(m_V[static_cast<std::size_t>(i)],
                                    m_AV[static_cast<std::size_t>(j)]).real();
        m_H_inv = invertSPD(m_H);
    }

    /// Return the small Ritz value estimates (debug / display).
    const std::vector<double>& ritzValues() const { return m_lambda; }

    /// Initial guess `x = V H⁻¹ V† b`; zeros if `b` is null in span(V).
    ///
    /// `V†b` is genuinely complex for complex spinor `b`; we keep both real
    /// and imaginary parts. `H = V†AV` is real symmetric (Hermitian A,
    /// Lanczos basis), so `H⁻¹` acts on Re and Im independently — solve once
    /// in real, twice.
    void projectInitial(const Spinor& b, Spinor& x) const
    {
        using Complex = std::complex<double>;
        const int k = size();
        x.zero();
        if (k == 0) return;
        std::vector<double> Re_Vtb(static_cast<std::size_t>(k), 0.0);
        std::vector<double> Im_Vtb(static_cast<std::size_t>(k), 0.0);
        for (int i = 0; i < k; ++i)
        {
            const Complex c = inner_product(m_V[static_cast<std::size_t>(i)], b);
            Re_Vtb[static_cast<std::size_t>(i)] = c.real();
            Im_Vtb[static_cast<std::size_t>(i)] = c.imag();
        }
        std::vector<double> Re_y(static_cast<std::size_t>(k), 0.0);
        std::vector<double> Im_y(static_cast<std::size_t>(k), 0.0);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
            {
                const double h
                    = m_H_inv[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                Re_y[static_cast<std::size_t>(i)] += h * Re_Vtb[static_cast<std::size_t>(j)];
                Im_y[static_cast<std::size_t>(i)] += h * Im_Vtb[static_cast<std::size_t>(j)];
            }
        for (int j = 0; j < k; ++j)
            axpy(Complex(Re_y[static_cast<std::size_t>(j)],
                         Im_y[static_cast<std::size_t>(j)]),
                 m_V[static_cast<std::size_t>(j)], x);
    }

    /// Project `p` onto the A-orthogonal complement of `span(V)`:
    ///     p ← p − V H⁻¹ V† (A p) = p − V H⁻¹ (AV)† p.
    /// Same complex-aware projection as `projectInitial`.
    void deflateSearch(const Spinor& Ap, Spinor& p) const
    {
        using Complex = std::complex<double>;
        const int k = size();
        if (k == 0) return;
        std::vector<double> Re_VtAp(static_cast<std::size_t>(k), 0.0);
        std::vector<double> Im_VtAp(static_cast<std::size_t>(k), 0.0);
        for (int i = 0; i < k; ++i)
        {
            const Complex c = inner_product(m_V[static_cast<std::size_t>(i)], Ap);
            Re_VtAp[static_cast<std::size_t>(i)] = c.real();
            Im_VtAp[static_cast<std::size_t>(i)] = c.imag();
        }
        std::vector<double> Re_y(static_cast<std::size_t>(k), 0.0);
        std::vector<double> Im_y(static_cast<std::size_t>(k), 0.0);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j)
            {
                const double h
                    = m_H_inv[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                Re_y[static_cast<std::size_t>(i)] += h * Re_VtAp[static_cast<std::size_t>(j)];
                Im_y[static_cast<std::size_t>(i)] += h * Im_VtAp[static_cast<std::size_t>(j)];
            }
        for (int j = 0; j < k; ++j)
            axpy(Complex(-Re_y[static_cast<std::size_t>(j)],
                         -Im_y[static_cast<std::size_t>(j)]),
                 m_V[static_cast<std::size_t>(j)], p);
    }

private:
    /// Invert a small symmetric positive-definite matrix by Gauss-Jordan
    /// elimination — the small-k assumption (k ≤ 60) means this is trivial.
    static std::vector<std::vector<double>>
    invertSPD(std::vector<std::vector<double>> H)
    {
        const int k = static_cast<int>(H.size());
        std::vector<std::vector<double>> I(static_cast<std::size_t>(k),
                                           std::vector<double>(static_cast<std::size_t>(k), 0.0));
        for (int i = 0; i < k; ++i)
            I[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
        for (int p = 0; p < k; ++p)
        {
            // Partial pivoting.
            int    pivot     = p;
            double pivot_val = std::abs(H[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)]);
            for (int r = p + 1; r < k; ++r)
            {
                const double v = std::abs(H[static_cast<std::size_t>(r)][static_cast<std::size_t>(p)]);
                if (v > pivot_val) { pivot = r; pivot_val = v; }
            }
            if (pivot_val < 1e-18)
                throw std::runtime_error("DeflationSubspace: H is singular");
            if (pivot != p)
            {
                std::swap(H[static_cast<std::size_t>(pivot)],
                          H[static_cast<std::size_t>(p)]);
                std::swap(I[static_cast<std::size_t>(pivot)],
                          I[static_cast<std::size_t>(p)]);
            }
            const double s = 1.0 / H[static_cast<std::size_t>(p)][static_cast<std::size_t>(p)];
            for (int j = 0; j < k; ++j)
            {
                H[static_cast<std::size_t>(p)][static_cast<std::size_t>(j)] *= s;
                I[static_cast<std::size_t>(p)][static_cast<std::size_t>(j)] *= s;
            }
            for (int r = 0; r < k; ++r)
            {
                if (r == p) continue;
                const double f = H[static_cast<std::size_t>(r)][static_cast<std::size_t>(p)];
                for (int j = 0; j < k; ++j)
                {
                    H[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] -=
                        f * H[static_cast<std::size_t>(p)][static_cast<std::size_t>(j)];
                    I[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] -=
                        f * I[static_cast<std::size_t>(p)][static_cast<std::size_t>(j)];
                }
            }
        }
        return I;
    }

    std::vector<Spinor>                          m_V;      // basis vectors (n × k)
    std::vector<Spinor>                          m_AV;     // cached A·V[j]
    std::vector<double>                          m_lambda; // Ritz value estimates
    std::vector<std::vector<double>>             m_H;      // V†AV (k × k)
    std::vector<std::vector<double>>             m_H_inv;  // (V†AV)⁻¹
};

/// Deflated CG: same interface as `conjugateGradient`, plus a deflation
/// subspace. The initial guess is the projected exact solution in span(V),
/// and the search direction is re-deflated each iteration to suppress
/// reintroduction of the low-mode subspace by round-off.
template<int Dim, int Nc, typename ApplyA>
CGResult deflatedConjugateGradient(
    ApplyA&&                          apply_A,
    const DeflationSubspace<Dim, Nc>& def,
    const SpinorField<Dim, Nc>&       b,
    SpinorField<Dim, Nc>&             x,
    double                            tolerance,
    int                               max_iters,
    SpinorField<Dim, Nc>&             r,
    SpinorField<Dim, Nc>&             p,
    SpinorField<Dim, Nc>&             Ap)
{
    using Complex = std::complex<double>;

    using Complex = std::complex<double>;

    // x₀ = V H⁻¹ V† b — captures the low-mode contribution to the solution
    // exactly; algebraically V† r₀ = 0 at the start.
    def.projectInitial(b, x);
    apply_A(x, Ap);
    copy(b, r);
    axpy(Complex(-1.0, 0.0), Ap, r);

    // p₀ = r₀ then deflated so that A p₀ ⊥ V (which keeps r in V⊥ after the
    // CG step). The re-deflation of p at every iteration is what actually
    // shrinks the effective condition number from λ_max/λ_min to
    // λ_max/λ_{k+1} — init-only deflation gives only the absolute-residual
    // headroom from x₀, not the rate change.
    copy(r, p);
    apply_A(p, Ap);
    def.deflateSearch(Ap, p);

    const double b_norm2 = norm_squared(b);
    if (b_norm2 == 0.0) { x.zero(); return CGResult{ 0, 0.0, true }; }
    double r_norm2 = norm_squared(r);
    const double tol2 = tolerance * tolerance * b_norm2;

    int k = 0;
    for (; k < max_iters; ++k)
    {
        if (r_norm2 <= tol2)
            return CGResult{ k, std::sqrt(r_norm2 / b_norm2), true };

        apply_A(p, Ap);
        const double pAp = inner_product(p, Ap).real();
        if (!(pAp > 0.0))
            return CGResult{ k, std::sqrt(r_norm2 / b_norm2), false };
        const double alpha = r_norm2 / pAp;

        axpy(Complex( alpha, 0.0), p,  x);
        axpy(Complex(-alpha, 0.0), Ap, r);

        const double r_norm2_new = norm_squared(r);
        const double beta = r_norm2_new / r_norm2;
        scale_add(Complex(beta, 0.0), r, p);   // p ← r + β p

        // Re-deflate the new search direction so the iterate stays in V⊥.
        apply_A(p, Ap);
        def.deflateSearch(Ap, p);

        r_norm2 = r_norm2_new;
    }
    return CGResult{ k, std::sqrt(r_norm2 / b_norm2), r_norm2 <= tol2 };
}

} // namespace lqft::solvers
