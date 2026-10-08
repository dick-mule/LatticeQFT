/**
 * @file test_deflation.cpp
 * @brief Tests of the Lanczos eigensolver and deflated CG.
 *
 *   1. Lanczos on a diagonal operator recovers the diagonal entries as
 *      Ritz values to machine precision.
 *   2. Ritz pairs on the Wilson-Dirac D†D approximate eigenpairs to
 *      `‖A v − θ v‖ < tol`.
 *   3. Deflated CG converges to the same solution as vanilla CG.
 *   4. Deflated CG takes fewer iterations than vanilla CG on a near-massless
 *      Schwinger Dirac operator (where the low Dirac eigenvalues slow
 *      vanilla CG dramatically).
 */

#include "fields/link_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/u1.hpp"
#include "rng/rng.hpp"
#include "solvers/conjugate_gradient.hpp"
#include "solvers/deflation.hpp"
#include "solvers/lanczos.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <complex>

using namespace lqft;
using Complex = std::complex<double>;

namespace
{

void random_spinor(SpinorField<2, 2>& s, Rng& rng)
{
    const int N = s.totalSize();
    auto* p = s.data();
    for (int i = 0; i < N; ++i)
        p[i] = Complex{ rng.normal(), rng.normal() };
}

} // namespace

// -----------------------------------------------------------------------------
// 1. Lanczos on a diagonal operator.
// -----------------------------------------------------------------------------

TEST(Lanczos, DiagonalOperatorRecoversEigenvalues)
{
    // Diagonal "operator" with N entries (a, 2a, 3a, ..., Na). Lanczos should
    // identify the smallest entries as the smallest Ritz values to machine
    // precision after enough iterations.
    Lattice<2> lattice = Lattice<2>::cube(4);
    SpinorField<2, 2> seed(lattice), w(lattice), tmp(lattice);
    Rng rng(11u);
    random_spinor(seed, rng);

    const int N = seed.totalSize();
    std::vector<double> diag(N);
    for (int i = 0; i < N; ++i) diag[i] = 0.1 + static_cast<double>(i);

    auto apply_A = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        for (int i = 0; i < N; ++i) out.data()[i] = diag[i] * in.data()[i];
    };

    auto basis = solvers::lanczos<2, 2>(apply_A, seed, /*m*/N, w, tmp);
    auto ritz  = solvers::extractSmallestRitz<2, 2>(basis, /*k*/4);

    ASSERT_EQ(static_cast<int>(ritz.values.size()), 4);
    EXPECT_NEAR(ritz.values[0], 0.1, 1e-9);
    EXPECT_NEAR(ritz.values[1], 1.1, 1e-9);
    EXPECT_NEAR(ritz.values[2], 2.1, 1e-9);
    EXPECT_NEAR(ritz.values[3], 3.1, 1e-9);
}

// -----------------------------------------------------------------------------
// 2. Ritz pairs of Wilson-Dirac D†D satisfy A y ≈ θ y.
// -----------------------------------------------------------------------------

TEST(Lanczos, WilsonDiracRitzPairsAreEigenpairs)
{
    // L = 4 ⇒ V · Nc = 32 complex Lanczos space. Running m = 50 (> N)
    // exhausts the space and the Ritz values converge to true eigenvalues.
    Lattice<2> lattice = Lattice<2>::cube(4);
    LinkField<double, 2> U(lattice);
    Rng rng(42u);
    u1::U1Model<2>::hot(U, rng);
    dirac::WilsonDirac2D D(lattice, /*mass*/0.4);

    SpinorField<2, 2> seed(lattice), w(lattice), tmp(lattice), Ay(lattice);
    random_spinor(seed, rng);

    auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        D.applyDagD(U, in, out, tmp);
    };

    auto basis = solvers::lanczos<2, 2>(apply_DagD, seed, /*m*/50, w, tmp);
    auto ritz  = solvers::extractSmallestRitz<2, 2>(basis, /*k*/4);
    ASSERT_GE(static_cast<int>(ritz.vectors.size()), 4);

    for (int i = 0; i < 4; ++i)
    {
        const double theta = ritz.values[static_cast<std::size_t>(i)];
        const auto&  y     = ritz.vectors[static_cast<std::size_t>(i)];
        apply_DagD(y, Ay);
        // ‖A y − θ y‖
        SpinorField<2, 2> diff = Ay;
        axpy(Complex(-theta, 0.0), y, diff);
        const double res = std::sqrt(norm_squared(diff));
        EXPECT_LT(res, 1e-6) << "Ritz residual " << res
                              << " at θ = " << theta;
    }
}

// -----------------------------------------------------------------------------
// Controlled diagonal-operator test: spectrum where deflation should clearly help.
// -----------------------------------------------------------------------------

TEST(DeflatedCG, DiagonalOperatorWithClusteredLowModes)
{
    // Diagonal "operator" with 32 entries (matches the L=4 Nc=2 spinor
    // total_size). Spectrum has 4 very-low modes clustered near zero
    // (which kill vanilla CG's convergence rate) and 28 bulk modes
    // spread between 1 and 30. Deflating the 4 low modes should drop
    // κ from ~3·10^4 to ~30, i.e. √κ from ~175 to ~5.
    Lattice<2> lattice = Lattice<2>::cube(4);
    const int N = SpinorField<2, 2>(lattice).totalSize(); // 16 sites × 2 = 32

    std::vector<double> eigs(N);
    eigs[0] = 0.001;
    eigs[1] = 0.005;
    eigs[2] = 0.01;
    eigs[3] = 0.02;
    for (int i = 4; i < N; ++i) eigs[i] = 1.0 + (i - 4);
    // i.e. 0.001, 0.005, 0.01, 0.02, 1, 2, ..., 28

    auto apply_A = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        for (int i = 0; i < N; ++i) out.data()[i] = eigs[i] * in.data()[i];
    };

    SpinorField<2, 2> b(lattice), x_v(lattice), x_d(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice);
    SpinorField<2, 2> w(lattice), tmp(lattice), seed(lattice);
    Rng rng(7u);
    random_spinor(b, rng);

    // Vanilla CG
    x_v.zero();
    auto vanilla = solvers::conjugateGradient<2, 2>(
        apply_A, b, x_v, /*tol*/1e-10, /*max*/4000, r, p, Ap);
    ASSERT_TRUE(vanilla.converged);

    // Build deflation subspace (full Lanczos exhausts N=32-dim space).
    random_spinor(seed, rng);
    solvers::DeflationSubspace<2, 2> def;
    def.build(apply_A, seed, /*k_smallest*/4, /*lanczos_iters*/40, w, tmp);
    ASSERT_GE(def.size(), 4);

    // Deflated CG
    x_d.zero();
    auto deflated = solvers::deflatedConjugateGradient<2, 2>(
        apply_A, def, b, x_d, /*tol*/1e-10, /*max*/4000, r, p, Ap);
    ASSERT_TRUE(deflated.converged);

    // Solutions must agree.
    SpinorField<2, 2> diff = x_v;
    axpy(Complex(-1.0, 0.0), x_d, diff);
    const double rel = std::sqrt(norm_squared(diff) / norm_squared(x_v));
    EXPECT_LT(rel, 1e-7);

    // Deflation gives a measurable iteration-count reduction: vanilla wants
    // ~√(30/0.001) = 173 in continuous arithmetic but in finite precision
    // CG terminates near N = 32 due to Krylov-space exhaustion. Deflation
    // collapses the 4 clustered low modes; we expect at least a 25 % drop.
    EXPECT_LT(deflated.iterations, vanilla.iterations * 3 / 4)
        << "vanilla=" << vanilla.iterations
        << "  deflated=" << deflated.iterations;
}

// -----------------------------------------------------------------------------
// 3 + 4. Deflated CG correctness + iteration-count win on Wilson-Dirac.
// -----------------------------------------------------------------------------

TEST(DeflatedCG, MatchesVanillaCGAndUsesFewerIterations)
{
    // Near-massless Schwinger Dirac operator. L = 8 gives N = 128 (complex);
    // vanilla CG converges in ~√κ iterations well short of N, so there's
    // headroom for deflation to actually pay off. We run m = 100 Lanczos
    // (78 % of the space) which converges the smallest few Ritz values to
    // 1e-6.
    constexpr int    L     = 8;
    constexpr double mass  = 0.02;  // near-chiral; small λ_min, large κ
    Lattice<2> lattice = Lattice<2>::cube(L);
    LinkField<double, 2> U(lattice);
    Rng rng(99u);
    u1::U1Model<2>::hot(U, rng);
    dirac::WilsonDirac2D D(lattice, mass);

    SpinorField<2, 2> b(lattice), x_vanilla(lattice), x_def(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice), tmp(lattice);
    SpinorField<2, 2> w(lattice), tmp2(lattice);
    random_spinor(b, rng);

    auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        D.applyDagD(U, in, out, tmp);
    };

    // ---- Vanilla CG ----
    x_vanilla.zero();
    auto vanilla = solvers::conjugateGradient<2, 2>(
        apply_DagD, b, x_vanilla, /*tol*/1e-8, /*max*/4000, r, p, Ap);
    ASSERT_TRUE(vanilla.converged) << "vanilla CG didn't converge";

    // ---- Build deflation subspace ----
    solvers::DeflationSubspace<2, 2> def;
    SpinorField<2, 2> seed(lattice);
    random_spinor(seed, rng);
    def.build(apply_DagD, seed,
              /*k_smallest*/20, /*lanczos_iters*/120, w, tmp2);
    ASSERT_GT(def.size(), 0);

    // ---- Deflated CG ----
    x_def.zero();
    auto deflated = solvers::deflatedConjugateGradient<2, 2>(
        apply_DagD, def, b, x_def, /*tol*/1e-8, /*max*/4000, r, p, Ap);
    ASSERT_TRUE(deflated.converged) << "deflated CG didn't converge";

    // Solutions must agree to the tolerance.
    SpinorField<2, 2> diff = x_vanilla;
    axpy(Complex(-1.0, 0.0), x_def, diff);
    const double rel_err = std::sqrt(norm_squared(diff) / norm_squared(x_vanilla));
    EXPECT_LT(rel_err, 1e-6) << "vanilla and deflated solutions disagree: "
                              << rel_err;

    // Deflated should take strictly fewer iterations.
    EXPECT_LT(deflated.iterations, vanilla.iterations)
        << "vanilla=" << vanilla.iterations
        << "  deflated=" << deflated.iterations;
    // And by a meaningful factor at near-chiral mass.
    EXPECT_LT(deflated.iterations, vanilla.iterations * 3 / 4)
        << "vanilla=" << vanilla.iterations
        << "  deflated=" << deflated.iterations
        << " (expected ≥ 25% reduction at this mass)";
}
