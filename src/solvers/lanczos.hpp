#pragma once

/**
 * @file lanczos.hpp
 * @brief Lanczos iteration and tridiagonal eigensolver.
 *
 * Standard symmetric Lanczos: given Hermitian PSD operator A and a starting
 * vector v₁, build an orthonormal basis V_m = [v₁ | v₂ | ... | v_m] and a
 * symmetric tridiagonal projection T_m such that A V_m = V_m T_m + β_m v_{m+1} e_m^T.
 *
 * Iteration:
 *
 *     β₀ = 0,  v₀ = 0,  v₁ = (initial) / ‖·‖
 *     for j = 1, 2, ..., m:
 *         w = A v_j − β_{j-1} v_{j-1}
 *         α_j = ⟨v_j, w⟩
 *         w = w − α_j v_j
 *         (optionally full re-orthogonalize w against {v_1, ..., v_j})
 *         β_j = ‖w‖
 *         v_{j+1} = w / β_j
 *
 * Diagonalizing T_m gives **Ritz pairs** (θ_i, V_m s_i) that approximate
 * eigenpairs of A — the smallest Ritz values are the best approximations to
 * A's smallest true eigenvalues after a few dozen iterations, which is
 * exactly what the deflation subspace needs.
 *
 * Re-orthogonalization is required in finite precision to keep V_m orthonormal
 * — without it, "ghost" duplicated eigenvalues appear in T_m's spectrum.
 * We do full re-orthogonalization (Gram-Schmidt against the whole basis)
 * each step; for the lattice sizes we actually run this on (8 ≤ L ≤ 16, m ≤ 60)
 * the cost is negligible compared to A · v_j.
 *
 * Tridiagonal eigensolver: implicit-shift QL with Wilkinson shifts (textbook,
 * Press et al. Numerical Recipes §11.3). Operates on a tridiagonal of size m;
 * we only ever apply it to m ≤ 60 so a single-precision implementation is
 * fine and gets every eigenpair in a few sweeps.
 */

#include "../fields/spinor_field.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

namespace lqft::solvers
{

template<int Dim, int Nc>
struct LanczosBasis
{
    std::vector<SpinorField<Dim, Nc>> v;     // basis vectors (m of them)
    std::vector<double>               alpha; // diagonal entries (m)
    std::vector<double>               beta;  // off-diagonal entries (m-1)
};

/// Build the Lanczos basis for the Hermitian PSD operator `A` driven by
/// `apply_A(in, out)`. `initial` is a starting guess (will be normalized;
/// must be non-zero). Returns up to `max_iters` Lanczos vectors and the
/// tridiagonal projection's diagonals.
template<int Dim, int Nc, typename ApplyA>
LanczosBasis<Dim, Nc> lanczos(
    ApplyA&&                     apply_A,
    const SpinorField<Dim, Nc>&  initial,
    int                          max_iters,
    SpinorField<Dim, Nc>&        w_scratch,
    SpinorField<Dim, Nc>&        tmp_scratch)
{
    using Complex = std::complex<double>;
    LanczosBasis<Dim, Nc> basis;
    basis.v.reserve(static_cast<std::size_t>(max_iters));
    basis.alpha.reserve(static_cast<std::size_t>(max_iters));
    basis.beta.reserve(static_cast<std::size_t>(max_iters - 1));

    // v_1 = initial / ‖initial‖
    {
        SpinorField<Dim, Nc> v1 = initial;
        const double n2 = norm_squared(v1);
        if (n2 < 1e-30) return basis;
        scale(Complex(1.0 / std::sqrt(n2), 0.0), v1, v1);
        basis.v.push_back(std::move(v1));
    }

    for (int j = 0; j < max_iters; ++j)
    {
        // w = A v_j
        apply_A(basis.v[static_cast<std::size_t>(j)], w_scratch);

        // w -= β_{j-1} v_{j-1}
        if (j > 0)
        {
            axpy(Complex(-basis.beta[static_cast<std::size_t>(j - 1)], 0.0),
                 basis.v[static_cast<std::size_t>(j - 1)], w_scratch);
        }

        // α_j = ⟨v_j, w⟩ (real for Hermitian A)
        const double alpha_j = inner_product(
            basis.v[static_cast<std::size_t>(j)], w_scratch).real();
        basis.alpha.push_back(alpha_j);

        // w -= α_j v_j
        axpy(Complex(-alpha_j, 0.0),
             basis.v[static_cast<std::size_t>(j)], w_scratch);

        // Full re-orthogonalization: project out every existing v_k from w.
        // For complex spinors `⟨v_k, w⟩` is complex in general; subtracting
        // only `Re ⟨v_k, w⟩ · v_k` leaves the imaginary projection behind.
        for (int k = 0; k <= j; ++k)
        {
            const Complex c = inner_product(
                basis.v[static_cast<std::size_t>(k)], w_scratch);
            if (std::abs(c) > 1e-12)
                axpy(-c, basis.v[static_cast<std::size_t>(k)], w_scratch);
        }

        if (j + 1 >= max_iters) break;

        const double w_norm = std::sqrt(norm_squared(w_scratch));
        if (w_norm < 1e-12) break; // invariant subspace exhausted
        basis.beta.push_back(w_norm);

        // v_{j+1} = w / β_j
        SpinorField<Dim, Nc> v_next = w_scratch; // copy
        scale(Complex(1.0 / w_norm, 0.0), v_next, v_next);
        basis.v.push_back(std::move(v_next));
    }
    return basis;
}

/// Diagonalize the symmetric tridiagonal matrix with diagonal `d[]` and
/// sub-diagonal `e[]` (size n-1). On entry `Z` must be the identity matrix
/// (size n × n); on exit `d` holds the eigenvalues and the columns of `Z`
/// are the corresponding eigenvectors.
///
/// Implicit-shift QL with Wilkinson shift — converges in ~30·n flops per
/// eigenvalue. Fine for n ≤ 200.
inline void tridiagQL(std::vector<double>&         d,
                      std::vector<double>&         e,
                      std::vector<std::vector<double>>& Z,
                      int                          max_sweeps = 60)
{
    const int n = static_cast<int>(d.size());
    if (n <= 1) return;
    e.push_back(0.0); // sentinel at the end

    for (int l = 0; l < n; ++l)
    {
        int iter = 0;
        while (true)
        {
            // Find a sub-diagonal small enough to split.
            int m;
            for (m = l; m < n - 1; ++m)
            {
                const double dd = std::abs(d[static_cast<std::size_t>(m)])
                                + std::abs(d[static_cast<std::size_t>(m + 1)]);
                if (std::abs(e[static_cast<std::size_t>(m)]) <= 1e-14 * dd) break;
            }
            if (m == l) break;
            if (++iter > max_sweeps) break;

            // Wilkinson shift.
            double g = (d[static_cast<std::size_t>(l + 1)]
                      - d[static_cast<std::size_t>(l)])
                     / (2.0 * e[static_cast<std::size_t>(l)]);
            double r = std::hypot(g, 1.0);
            g = d[static_cast<std::size_t>(m)]
              - d[static_cast<std::size_t>(l)]
              + e[static_cast<std::size_t>(l)] / (g + (g >= 0.0 ? r : -r));

            double s = 1.0, c = 1.0, p = 0.0;
            int    i;
            for (i = m - 1; i >= l; --i)
            {
                const double f = s * e[static_cast<std::size_t>(i)];
                const double b = c * e[static_cast<std::size_t>(i)];
                if (std::abs(f) >= std::abs(g))
                {
                    c = g / f;
                    r = std::hypot(c, 1.0);
                    e[static_cast<std::size_t>(i + 1)] = f * r;
                    s = 1.0 / r;
                    c *= s;
                }
                else
                {
                    s = f / g;
                    r = std::hypot(s, 1.0);
                    e[static_cast<std::size_t>(i + 1)] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }
                g = d[static_cast<std::size_t>(i + 1)] - p;
                r = (d[static_cast<std::size_t>(i)] - g) * s + 2.0 * c * b;
                p = s * r;
                d[static_cast<std::size_t>(i + 1)] = g + p;
                g = c * r - b;

                // Rotate eigenvector columns.
                for (int k = 0; k < n; ++k)
                {
                    const double t = Z[static_cast<std::size_t>(k)][static_cast<std::size_t>(i + 1)];
                    Z[static_cast<std::size_t>(k)][static_cast<std::size_t>(i + 1)]
                        = s * Z[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] + c * t;
                    Z[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)]
                        = c * Z[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)] - s * t;
                }
            }
            d[static_cast<std::size_t>(l)] -= p;
            e[static_cast<std::size_t>(l)]  = g;
            e[static_cast<std::size_t>(m)]  = 0.0;
        }
    }
    e.pop_back();
}

/// Output of `extractSmallestRitz`: the `k` smallest Ritz eigenpairs of A as
/// approximated from a Lanczos basis. `values[i]` = θ_i (eigenvalue),
/// `vectors[i]` = y_i = V_m s_i (eigenvector of A).
template<int Dim, int Nc>
struct RitzPairs
{
    std::vector<double>               values;
    std::vector<SpinorField<Dim, Nc>> vectors;
};

/// From a Lanczos basis (T_m, V_m), extract the k Ritz pairs corresponding to
/// the k smallest Ritz values.
template<int Dim, int Nc>
RitzPairs<Dim, Nc> extractSmallestRitz(
    const LanczosBasis<Dim, Nc>& basis,
    int                          k_smallest)
{
    using Complex = std::complex<double>;
    const int m = static_cast<int>(basis.v.size());
    RitzPairs<Dim, Nc> out;
    if (m == 0 || k_smallest <= 0) return out;
    const int k = std::min(k_smallest, m);

    // Build T_m and identity Z; diagonalize.
    std::vector<double> d  = basis.alpha;
    std::vector<double> e  = basis.beta;
    std::vector<std::vector<double>> Z(static_cast<std::size_t>(m),
                                       std::vector<double>(static_cast<std::size_t>(m), 0.0));
    for (int i = 0; i < m; ++i)
        Z[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;

    tridiagQL(d, e, Z);

    // Sort eigenvalues ascending and keep the smallest k.
    std::vector<int> order(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) order[static_cast<std::size_t>(i)] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b)
              { return d[static_cast<std::size_t>(a)]
                     < d[static_cast<std::size_t>(b)]; });

    out.values.reserve(static_cast<std::size_t>(k));
    out.vectors.reserve(static_cast<std::size_t>(k));
    for (int idx = 0; idx < k; ++idx)
    {
        const int i = order[static_cast<std::size_t>(idx)];
        out.values.push_back(d[static_cast<std::size_t>(i)]);

        // y_i = Σ_j Z[j, i] · v_j
        SpinorField<Dim, Nc> y = basis.v[0];
        y.zero();
        for (int j = 0; j < m; ++j)
            axpy(Complex(Z[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)], 0.0),
                 basis.v[static_cast<std::size_t>(j)], y);
        // Normalize for safety against drift.
        const double yn = std::sqrt(norm_squared(y));
        if (yn > 1e-20) scale(Complex(1.0 / yn, 0.0), y, y);
        out.vectors.push_back(std::move(y));
    }
    return out;
}

} // namespace lqft::solvers
