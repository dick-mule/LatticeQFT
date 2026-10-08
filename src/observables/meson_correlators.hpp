#pragma once

/**
 * @file meson_correlators.hpp
 * @brief Pseudoscalar meson correlators in the 2D Schwinger model from the exact
 *        (dense) fermion propagator.
 *
 * ## Why dense
 *
 * On the lattices this project runs (L ≤ 24 in 2D) the Wilson–Dirac operator is a
 * 2V × 2V complex matrix with 2V ≤ 1152. An LU factorisation costs O((2V)³) ≈ 10⁹
 * flops at the top end, well under a second, and gives the FULL propagator
 * S(x, y) = D⁻¹ for every source and sink at once. That is what the flavour-singlet
 * correlator needs: its disconnected ("hairpin") piece involves tr[γ⁵ S(x, x)] on
 * every site, which stochastic estimators only approximate. Translation averaging
 * over all V sources comes free.
 *
 * ## Correlators (time axis = lattice direction 1, spatial = direction 0)
 *
 * With P(x) = ψ̄ γ⁵ ψ and S the propagator, Wick's theorem gives two terms:
 *
 *   C_conn(t) = (1/V) Σ_y Σ_{x: t_x − t_y ≡ t} Σ_{a,b} |S_ab(x, y)|²
 *
 *       the connected piece −tr[γ⁵ S(x,y) γ⁵ S(y,x)] rewritten with γ⁵-Hermiticity,
 *       S(y, x) = γ⁵ S(x, y)† γ⁵ (sign dropped: the overall sign of ⟨PP⟩ is
 *       convention). This is the correlator of the flavour non-singlet "pion"
 *       (a two-flavour object); it is positive and decays with the pion mass.
 *
 *   C_disc(t) = (1/V) Σ_y Σ_{x: t_x − t_y ≡ t} tr[γ⁵ S(x,x)] · tr[γ⁵ S(y,y)]
 *
 *       the disconnected piece. tr[γ⁵ S(x,x)] is real by γ⁵-Hermiticity.
 *
 *   C_η(t) = C_conn(t) − C_disc(t)                       (N_f = 1)
 *
 *       the single-flavour pseudoscalar: in the Schwinger model this is THE
 *       Schwinger boson, whose mass stays finite at the chiral point,
 *       M = e / √π  (lattice units: M a = 1 / √(π β) for the Wilson U(1) action
 *       with β = 1 / (e a)²), while the connected "pion" goes massless.
 *
 * Effective masses are extracted downstream (scripts/analyze_schwinger_mass.py)
 * from the periodic cosh form of C(t).
 *
 * Conventions: the operator in math/dirac_2d.hpp uses γ⁰ = σ_x, γ¹ = σ_y, so
 * γ⁵ ∝ σ_z = diag(1, −1). Its sign cancels in both correlators.
 */

#include "../fields/link_field.hpp"
#include "../fields/spinor_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/dirac_2d.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace lqft::obs
{

using Complex = std::complex<double>;

/// Dense complex matrix, row-major, n × n.
struct DenseMatrix
{
    int n = 0;
    std::vector<Complex> a;
    DenseMatrix() = default;
    explicit DenseMatrix(int n_) : n(n_), a(static_cast<std::size_t>(n_) * n_, Complex{0, 0}) {}
    Complex&       operator()(int i, int j)       { return a[static_cast<std::size_t>(i) * n + j]; }
    const Complex& operator()(int i, int j) const { return a[static_cast<std::size_t>(i) * n + j]; }
};

/// Build the dense Wilson–Dirac matrix by applying D to the 2V unit vectors.
/// Index convention: row/column i = 2·site + spinor component.
inline DenseMatrix denseDirac(const dirac::WilsonDirac2D& D,
                              const LinkField<double, 2>& U,
                              const Lattice<2>& lattice)
{
    const int V = lattice.volume(), n = 2 * V;
    DenseMatrix M(n);
    SpinorField<2, 2> e(lattice), y(lattice);
    for (int j = 0; j < n; ++j)
    {
        e.zero();
        e(j / 2, j % 2) = Complex{1.0, 0.0};
        D.apply(U, e, y);
        for (int i = 0; i < n; ++i) M(i, j) = y(i / 2, i % 2);
    }
    return M;
}

/// In-place inverse by Gauss–Jordan elimination with partial pivoting.
/// Throws if the matrix is singular to working precision.
inline DenseMatrix denseInverse(DenseMatrix A)
{
    const int n = A.n;
    DenseMatrix Inv(n);
    for (int i = 0; i < n; ++i) Inv(i, i) = Complex{1.0, 0.0};
    for (int col = 0; col < n; ++col)
    {
        int    piv  = col;
        double best = std::abs(A(col, col));
        for (int r = col + 1; r < n; ++r)
            if (std::abs(A(r, col)) > best) { best = std::abs(A(r, col)); piv = r; }
        if (best < 1e-300) throw std::runtime_error("denseInverse: singular matrix");
        if (piv != col)
            for (int c = 0; c < n; ++c) { std::swap(A(col, c), A(piv, c)); std::swap(Inv(col, c), Inv(piv, c)); }
        const Complex invp = Complex{1.0, 0.0} / A(col, col);
        for (int c = 0; c < n; ++c) { A(col, c) *= invp; Inv(col, c) *= invp; }
        for (int r = 0; r < n; ++r)
        {
            if (r == col) continue;
            const Complex f = A(r, col);
            if (f == Complex{0.0, 0.0}) continue;
            for (int c = 0; c < n; ++c) { A(r, c) -= f * A(col, c); Inv(r, c) -= f * Inv(col, c); }
        }
    }
    return Inv;
}

/// The full propagator S = D⁻¹ on the gauge field U.
inline DenseMatrix densePropagator(const dirac::WilsonDirac2D& D,
                                   const LinkField<double, 2>& U,
                                   const Lattice<2>& lattice)
{
    return denseInverse(denseDirac(D, U, lattice));
}

struct MesonCorrelators
{
    std::vector<double> conn;   // C_conn(t), t = 0..L_t−1
    std::vector<double> disc;   // C_disc(t)
    std::vector<double> eta;    // C_conn − C_disc
};

/// Time-sliced pseudoscalar correlators from the full propagator, averaged over all
/// V source positions. Time axis = direction 1 (the lattice is L × L here, so L_t = L).
inline MesonCorrelators mesonCorrelators(const DenseMatrix& S, const Lattice<2>& lattice)
{
    const int V = lattice.volume();
    const int Lt = lattice.coords(V - 1)[1] + 1;
    MesonCorrelators C;
    C.conn.assign(static_cast<std::size_t>(Lt), 0.0);
    C.disc.assign(static_cast<std::size_t>(Lt), 0.0);
    C.eta.assign(static_cast<std::size_t>(Lt), 0.0);

    // tr[γ⁵ S(x,x)] = S_00(x,x) − S_11(x,x)  (real by γ⁵-Hermiticity)
    std::vector<double> loop(static_cast<std::size_t>(V));
    std::vector<int>    tOf(static_cast<std::size_t>(V));
    for (int x = 0; x < V; ++x)
    {
        loop[static_cast<std::size_t>(x)] = (S(2 * x, 2 * x) - S(2 * x + 1, 2 * x + 1)).real();
        tOf[static_cast<std::size_t>(x)]  = lattice.coords(x)[1];
    }
    for (int y = 0; y < V; ++y)
    {
        const int ty = tOf[static_cast<std::size_t>(y)];
        for (int x = 0; x < V; ++x)
        {
            const int t = ((tOf[static_cast<std::size_t>(x)] - ty) % Lt + Lt) % Lt;
            double c = 0.0;
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    c += std::norm(S(2 * x + a, 2 * y + b));
            C.conn[static_cast<std::size_t>(t)] += c;
            C.disc[static_cast<std::size_t>(t)] += loop[static_cast<std::size_t>(x)] * loop[static_cast<std::size_t>(y)];
        }
    }
    for (int t = 0; t < Lt; ++t)
    {
        C.conn[static_cast<std::size_t>(t)] /= static_cast<double>(V);
        C.disc[static_cast<std::size_t>(t)] /= static_cast<double>(V);
        C.eta[static_cast<std::size_t>(t)]   = C.conn[static_cast<std::size_t>(t)] - C.disc[static_cast<std::size_t>(t)];
    }
    return C;
}

/// Effective mass from the periodic cosh form, C(t) ∝ cosh(M (t − L_t/2)):
/// solves C(t)/C(t+1) = cosh(M(t − L_t/2)) / cosh(M(t + 1 − L_t/2)) by bisection.
/// Returns NaN where the ratio has no solution (noise or t at the symmetry point).
inline double effectiveMassCosh(double Ct, double Ct1, int t, int Lt)
{
    if (!(Ct > 0.0) || !(Ct1 > 0.0)) return std::nan("");
    const double target = Ct / Ct1;
    const double u = t - 0.5 * Lt, v = t + 1 - 0.5 * Lt;
    auto f = [&](double M) { return std::cosh(M * u) / std::cosh(M * v); };
    double lo = 1e-6, hi = 10.0;
    if ((f(lo) - target) * (f(hi) - target) > 0.0) return std::nan("");
    for (int it = 0; it < 200; ++it)
    {
        const double mid = 0.5 * (lo + hi);
        if ((f(mid) - target) * (f(lo) - target) <= 0.0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace lqft::obs
