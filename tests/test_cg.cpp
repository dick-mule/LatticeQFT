/**
 * @file test_cg.cpp
 * @brief Tests of the conjugate-gradient solver.
 *
 *   1. CG inverts a small diagonal HPD operator A = m · I exactly in one step.
 *   2. CG inverts the Wilson-Dirac D†[U] D[U] to relative tol 1e-9 on a
 *      hot U(1) gauge background — the round-trip identity ‖x_solve − x_true‖
 *      is small. This is the canonical "the fermion solver works" test.
 *   3. CG on a singular operator reports non-converged.
 */

#include "fields/link_field.hpp"
#include "fields/spinor_field.hpp"
#include "lattice/lattice.hpp"
#include "math/dirac_2d.hpp"
#include "models/u1.hpp"
#include "rng/rng.hpp"
#include "solvers/conjugate_gradient.hpp"

#include <gtest/gtest.h>
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

TEST(ConjugateGradient, ScalarMultipleConvergesInOneStep)
{
    Lattice<2> lattice = Lattice<2>::cube(4);
    SpinorField<2, 2> b(lattice), x(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice);
    Rng rng(7u);
    random_spinor(b, rng);

    const double m = 1.7;
    auto apply_A = [m](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        const int N = in.totalSize();
        for (int i = 0; i < N; ++i) out.data()[i] = m * in.data()[i];
    };

    auto res = solvers::conjugateGradient<2, 2>(
        apply_A, b, x, /*tol*/1e-12, /*max*/50, r, p, Ap);

    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 1); // exact in one step for any HPD scalar
    // x should be b / m.
    for (int i = 0; i < x.totalSize(); ++i)
    {
        const Complex expected = b.data()[i] / m;
        EXPECT_NEAR((x.data()[i] - expected).real(), 0.0, 1e-12);
        EXPECT_NEAR((x.data()[i] - expected).imag(), 0.0, 1e-12);
    }
}

TEST(ConjugateGradient, InvertsWilsonDiracDagDOnHotGaugeBackground)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    LinkField<double, 2> U(lattice);
    Rng rng(1234u);
    u1::U1Model<2>::hot(U, rng);

    dirac::WilsonDirac2D D(lattice, /*mass*/0.5);

    SpinorField<2, 2> x_true(lattice), b(lattice);
    SpinorField<2, 2> x_solve(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice);
    SpinorField<2, 2> tmp(lattice);

    random_spinor(x_true, rng);

    auto apply_DagD = [&](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        D.applyDagD(U, in, out, tmp);
    };

    // b = D†D x_true. Solve for x — should recover x_true.
    apply_DagD(x_true, b);

    auto res = solvers::conjugateGradient<2, 2>(
        apply_DagD, b, x_solve, /*tol*/1e-9, /*max*/500, r, p, Ap);

    EXPECT_TRUE(res.converged) << "iters=" << res.iterations
                                << " resid=" << res.final_residual;

    // Round-trip: ‖x_solve − x_true‖² / ‖x_true‖² small.
    double diff2 = 0.0;
    for (int i = 0; i < x_true.totalSize(); ++i)
        diff2 += std::norm(x_solve.data()[i] - x_true.data()[i]);
    const double rel = std::sqrt(diff2 / norm_squared(x_true));
    EXPECT_LT(rel, 1e-8) << "rel error=" << rel;
}

TEST(ConjugateGradient, ReportsNonConvergeOnSingularOperator)
{
    // A = projector onto component 0 ⇒ singular on component 1.
    // CG must not lie and call this converged when b has component-1 content.
    Lattice<2> lattice = Lattice<2>::cube(4);
    SpinorField<2, 2> b(lattice), x(lattice);
    SpinorField<2, 2> r(lattice), p(lattice), Ap(lattice);
    Rng rng(13u);
    random_spinor(b, rng);

    auto apply_A = [](const SpinorField<2, 2>& in, SpinorField<2, 2>& out)
    {
        const int V = in.volume();
        for (int s = 0; s < V; ++s)
        {
            out(s, 0) = in(s, 0);
            out(s, 1) = Complex{0.0, 0.0};
        }
    };

    auto res = solvers::conjugateGradient<2, 2>(
        apply_A, b, x, /*tol*/1e-10, /*max*/30, r, p, Ap);

    // pAp ≤ 0 once we exhaust the component-0 direction, so the solver bails.
    EXPECT_FALSE(res.converged);
}
