/**
 * @file test_u1.cpp
 * @brief Tests of the compact U(1) lattice-gauge Metropolis driver.
 *
 *   1. Exact-arithmetic:
 *      - Cold start (all θ = 0) has S = 0 exactly.
 *      - The model's `delta_action(...)` matches the literal action
 *        difference before and after `apply(...)` to machine precision.
 *      - Gauge invariance: θ_μ(x) → θ_μ(x) + α(x+ê_μ) − α(x) for any
 *        random α leaves the total action invariant.
 *
 *   2. Statistical (against the closed-form 2D Bessel ratio):
 *      - ⟨cos θ_□⟩ matches I_1(β) / I_0(β) on a 2D L = 12 lattice.
 *      - ⟨W(2, 1)⟩ matches (I_1/I_0)² in 2D (Wilson-loop area law).
 */

#include "fields/link_field.hpp"
#include "fields/site_field.hpp"
#include "lattice/lattice.hpp"
#include "models/u1.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/tuning.hpp"
#include "monte_carlo/u1_heatbath.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

// -----------------------------------------------------------------------------
// Exact-arithmetic
// -----------------------------------------------------------------------------

TEST(U12D, ColdStartHasZeroAction)
{
    Lattice<2> lattice = Lattice<2>::cube(6);
    LinkField<double, 2> field(lattice);
    u1::U1Model<2>::cold(field);

    u1::U1Model<2> model(lattice, /*β*/2.0);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
    EXPECT_DOUBLE_EQ(u1::averagePlaquette(lattice, field), 1.0);
}

TEST(U12D, DeltaSMatchesActionDifference)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    LinkField<double, 2> field(lattice);
    Rng rng(42);
    u1::U1Model<2>::hot(field, rng);

    u1::U1Model<2> model(lattice, /*β*/1.5);

    for (int unit : {0, 7, 23, 51, 99, 127})
    {
        const int    site      = u1::U1Model<2>::siteOfUnit(unit);
        const int    mu        = u1::U1Model<2>::muOfUnit(unit);
        const double theta_old = field(site, mu);
        const double proposal  = theta_old + 0.37;

        const double S_before  = model.totalAction(field);
        const double dS_pred   = model.delta_action(field, unit, proposal);
        model.apply(field, unit, proposal);
        const double S_after   = model.totalAction(field);

        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-9)
            << "unit " << unit << " (site " << site << ", μ=" << mu << ")";

        // Restore for the next iteration.
        model.apply(field, unit, theta_old);
    }
}

TEST(U12D, ActionIsGaugeInvariant)
{
    // Apply a random gauge transformation θ_μ(x) → θ_μ(x) + α(x+ê_μ) − α(x).
    // The Wilson plaquette action is gauge invariant; total action must not
    // change. Test is exact to round-off.
    Lattice<2> lattice = Lattice<2>::cube(10);
    LinkField<double, 2> field(lattice);
    Rng rng(2024u);
    u1::U1Model<2>::hot(field, rng);

    u1::U1Model<2> model(lattice, /*β*/2.5);
    const double S_before  = model.totalAction(field);
    const double Re_W_before = u1::averagePlaquette(lattice, field);

    // Random gauge angles α(x) ∈ [−π, π).
    SiteField<double, 2> alpha(lattice);
    constexpr double kPi = 3.14159265358979323846;
    for (int s = 0; s < lattice.volume(); ++s)
        alpha[s] = (2.0 * rng.uniform() - 1.0) * kPi;

    // Apply: θ_μ(x) += α(x+ê_μ) − α(x).
    for (int s = 0; s < lattice.volume(); ++s)
        for (int mu = 0; mu < 2; ++mu)
            field(s, mu) += alpha[lattice.forward(s, mu)] - alpha[s];

    const double S_after    = model.totalAction(field);
    const double Re_W_after = u1::averagePlaquette(lattice, field);

    EXPECT_NEAR(S_after, S_before,         1e-9);
    EXPECT_NEAR(Re_W_after, Re_W_before,   1e-12);
}

// -----------------------------------------------------------------------------
// Color partition: every color class has the right size, and every
// (μ, parity) pair has its own class.
// -----------------------------------------------------------------------------

TEST(U12D, ColorPartitionCovers)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    u1::U1Model<2> model(lattice, /*β*/1.0);
    ASSERT_EQ(model.numColors(), 4);

    long total = 0;
    for (int c = 0; c < model.numColors(); ++c)
        total += static_cast<long>(model.unitsOfColor(c).size());
    EXPECT_EQ(total, static_cast<long>(lattice.volume()) * 2);
}

// -----------------------------------------------------------------------------
// Statistical: 2D Bessel-ratio benchmark.
// -----------------------------------------------------------------------------

namespace
{

struct U1Stats
{
    double avg_plaquette;
    double avg_W21;
    double acceptance;
};

U1Stats run_u1_2d(int L, double beta, std::uint64_t seed,
                  int therm, int measure_sweeps, int sample_every)
{
    Lattice<2> lattice = Lattice<2>::cube(L);
    LinkField<double, 2> field(lattice);
    Rng rng(seed);
    u1::U1Model<2>::hot(field, rng);

    u1::U1Model<2> model(lattice, beta, /*step*/1.0);
    mc::autoTuneStepSize(model, field, rng,
                         /*batch*/80, /*max_iters*/15, /*target*/0.5, /*tol*/0.05);

    mc::MetropolisSweep<u1::U1Model<2>> sweeper(model);
    sweeper.sweepN(field, rng, therm);
    sweeper.resetCounters();

    obs::Mean plaq, w21;
    for (int s = 0; s < measure_sweeps; ++s)
    {
        sweeper.sweep(field, rng);
        if ((s % sample_every) == 0)
        {
            plaq.add(u1::averagePlaquette(lattice, field));
            w21 .add(u1::averageWilsonLoop(lattice, field, /*μ*/0, /*ν*/1, /*R*/2, /*T*/1));
        }
    }
    return { plaq.mean(), w21.mean(), sweeper.cumulativeAcceptance() };
}

} // namespace

TEST(U12D, PlaquetteMatchesBesselRatio)
{
    constexpr double beta = 2.0;
    const double analytic = u1::besselRatio(beta);  // ≈ 0.6977…

    auto s = run_u1_2d(/*L*/12, beta, /*seed*/1001,
                       /*therm*/300, /*measure*/1500, /*every*/3);
    const double rel = std::abs(s.avg_plaquette - analytic) / analytic;
    EXPECT_LT(rel, 0.05)
        << "measured=" << s.avg_plaquette
        << "  analytic=" << analytic
        << "  rel_err="  << rel;
}

TEST(U12D, WilsonLoop2x1MatchesBesselAreaLaw)
{
    // For pure 2D U(1), ⟨W(R, T)⟩ = (I_1/I_0)^{R*T} exactly. R=2, T=1 ⇒ (I_1/I_0)².
    constexpr double beta = 2.0;
    const double analytic = u1::besselRatio(beta) * u1::besselRatio(beta);

    auto s = run_u1_2d(/*L*/12, beta, /*seed*/2002,
                       /*therm*/300, /*measure*/2000, /*every*/2);
    const double rel = std::abs(s.avg_W21 - analytic) / analytic;
    EXPECT_LT(rel, 0.08)
        << "measured=" << s.avg_W21
        << "  analytic=" << analytic
        << "  rel_err="  << rel;
}

// =============================================================================
// 3D dimensional-lift tests — the algorithm and model are unchanged; we just
// instantiate U1Model<3>. If these pass, the "Dim is a free parameter, not a
// hard-coded 2" architectural commitment is real.
// =============================================================================

TEST(U13D, ColdStartHasZeroAction)
{
    Lattice<3> lattice = Lattice<3>::cube(4);
    LinkField<double, 3> field(lattice);
    u1::U1Model<3>::cold(field);

    u1::U1Model<3> model(lattice, /*β*/2.0);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
    EXPECT_DOUBLE_EQ(u1::averagePlaquette(lattice, field), 1.0);
}

TEST(U13D, DeltaSMatchesActionDifference)
{
    Lattice<3> lattice = Lattice<3>::cube(5);
    LinkField<double, 3> field(lattice);
    Rng rng(31337u);
    u1::U1Model<3>::hot(field, rng);

    u1::U1Model<3> model(lattice, /*β*/1.7);
    for (int unit : {0, 14, 51, 117, 244})
    {
        const int site = u1::U1Model<3>::siteOfUnit(unit);
        const int mu   = u1::U1Model<3>::muOfUnit(unit);
        const double theta_old = field(site, mu);
        const double proposal  = theta_old - 0.42;

        const double S_before  = model.totalAction(field);
        const double dS_pred   = model.delta_action(field, unit, proposal);
        model.apply(field, unit, proposal);
        const double S_after   = model.totalAction(field);

        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-9)
            << "unit " << unit;
        model.apply(field, unit, theta_old);
    }
}

TEST(U13D, ActionIsGaugeInvariant)
{
    Lattice<3> lattice = Lattice<3>::cube(5);
    LinkField<double, 3> field(lattice);
    Rng rng(2025u);
    u1::U1Model<3>::hot(field, rng);

    u1::U1Model<3> model(lattice, /*β*/2.0);
    const double S_before = model.totalAction(field);

    SiteField<double, 3> alpha(lattice);
    constexpr double kPi = 3.14159265358979323846;
    for (int s = 0; s < lattice.volume(); ++s)
        alpha[s] = (2.0 * rng.uniform() - 1.0) * kPi;

    for (int s = 0; s < lattice.volume(); ++s)
        for (int mu = 0; mu < 3; ++mu)
            field(s, mu) += alpha[lattice.forward(s, mu)] - alpha[s];

    EXPECT_NEAR(model.totalAction(field), S_before, 1e-8);
}

TEST(U13D, ColorPartitionIsSixColors)
{
    Lattice<3> lattice = Lattice<3>::cube(4);
    u1::U1Model<3> model(lattice, /*β*/1.0);
    EXPECT_EQ(model.numColors(), 6);

    long total = 0;
    for (int c = 0; c < model.numColors(); ++c)
        total += static_cast<long>(model.unitsOfColor(c).size());
    EXPECT_EQ(total, static_cast<long>(lattice.volume()) * 3);
}

// =============================================================================
// Heat-bath (Best-Fisher von Mises) tests.
// =============================================================================

TEST(U1HeatBath, VonMisesSamplerMatchesAnalyticMean)
{
    // For vM(μ, κ) the standard identity is ⟨cos(θ − μ)⟩ = I_1(κ)/I_0(κ).
    // Best-Fisher returns z ∈ [−π, π) drawn from vM(0, κ), so ⟨cos z⟩ should
    // equal the Bessel ratio. Sample many times, compare empirical mean.
    Rng rng(2025u);
    constexpr double kappa = 2.0;
    const double analytic = u1::besselRatio(kappa); // ≈ 0.6977…
    constexpr int N = 20000;
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += std::cos(u1::bestFisherSample(kappa, rng));
    const double empirical = sum / N;
    EXPECT_NEAR(empirical, analytic, 0.02)
        << "empirical=" << empirical << " analytic=" << analytic;
}

TEST(U1HeatBath, StrongCouplingPlaquetteMatchesBesselRatio)
{
    // At β = 0.5 Metropolis acceptance is poor and step size needs to
    // be huge. Heat-bath should still nail I_1(β)/I_0(β) = 0.2425
    // because it samples exactly from the conditional.
    constexpr double beta = 0.5;
    Lattice<2> lattice = Lattice<2>::cube(12);
    LinkField<double, 2> U(lattice);
    Rng rng(101u);
    u1::U1Model<2>::hot(U, rng);
    u1::U1Model<2> model(lattice, beta);

    // Thermalize.
    u1::heatBathSweepN(model, U, rng, 200);

    obs::Mean p;
    for (int s = 0; s < 600; ++s)
    {
        u1::heatBathSweep(model, U, rng);
        if ((s % 2) == 0) p.add(u1::averagePlaquette(lattice, U));
    }
    const double analytic = u1::besselRatio(beta);
    EXPECT_NEAR(p.mean(), analytic, 0.012)
        << "heatbath=" << p.mean() << " analytic=" << analytic;
}

TEST(U1HeatBath, AgreesWithMetropolisOnPlaquette)
{
    // Sanity: at moderate β heat-bath and Metropolis must converge to the
    // same plaquette expectation up to MC noise. They differ only in
    // autocorrelation, not equilibrium.
    constexpr int    L    = 10;
    constexpr double beta = 1.5;
    Lattice<2> lattice = Lattice<2>::cube(L);

    double p_metro, p_hb;
    {
        LinkField<double, 2> U(lattice);
        Rng rng(11u);
        u1::U1Model<2>::hot(U, rng);
        u1::U1Model<2> model(lattice, beta);
        mc::autoTuneStepSize(model, U, rng, 60, 12, 0.5, 0.05);
        mc::MetropolisSweep<u1::U1Model<2>> sw(model);
        sw.sweepN(U, rng, 400);
        obs::Mean p;
        for (int s = 0; s < 800; ++s) { sw.sweep(U, rng); if ((s % 2) == 0) p.add(u1::averagePlaquette(lattice, U)); }
        p_metro = p.mean();
    }
    {
        LinkField<double, 2> U(lattice);
        Rng rng(22u);
        u1::U1Model<2>::hot(U, rng);
        u1::U1Model<2> model(lattice, beta);
        u1::heatBathSweepN(model, U, rng, 200);
        obs::Mean p;
        for (int s = 0; s < 600; ++s)
        {
            u1::heatBathSweep(model, U, rng);
            if ((s % 2) == 0) p.add(u1::averagePlaquette(lattice, U));
        }
        p_hb = p.mean();
    }
    EXPECT_NEAR(p_metro, p_hb, 0.015) << "metro=" << p_metro << " hb=" << p_hb;
}

TEST(U13D, StrongAndWeakCouplingPlaquetteBoundsAreSane)
{
    // β = 0: action is identically 0; θ uniform on [-π, π); ⟨cos θ_□⟩ → 0
    //         (a sum of four independent uniform phases mod 2π is uniform).
    // β = 10 (weak coupling): leading-order Gaussian expansion gives
    //         ⟨cos θ_□⟩ ≈ 1 − 1/(2β) ≈ 0.95.
    // Test loose ordering — the point is that the algorithm produces sensible
    // numbers in both limits in 3D, not a quantitative dimensional fit.
    auto plaq_at = [](double beta, std::uint64_t seed) -> double
    {
        Lattice<3> lattice = Lattice<3>::cube(6);
        LinkField<double, 3> field(lattice);
        Rng rng(seed);
        u1::U1Model<3>::hot(field, rng);

        u1::U1Model<3> model(lattice, beta, /*step*/1.0);
        mc::autoTuneStepSize(model, field, rng, /*batch*/60, /*max*/12, /*target*/0.5, /*tol*/0.05);

        mc::MetropolisSweep<u1::U1Model<3>> sweeper(model);
        sweeper.sweepN(field, rng, /*therm*/250);

        obs::Mean acc;
        for (int s = 0; s < 700; ++s)
        {
            sweeper.sweep(field, rng);
            if ((s % 3) == 0) acc.add(u1::averagePlaquette(lattice, field));
        }
        return acc.mean();
    };

    const double p_strong = plaq_at(/*β*/0.1, 11u);
    const double p_weak   = plaq_at(/*β*/8.0, 13u);
    EXPECT_LT(p_strong, 0.20) << "p(β=0.1)=" << p_strong;
    EXPECT_GT(p_weak,   0.85) << "p(β=8.0)=" << p_weak;
    EXPECT_LT(p_strong, p_weak);
}
