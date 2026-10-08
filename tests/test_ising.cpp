/**
 * @file test_ising.cpp
 * @brief Quantitative tests of the 2D Ising Metropolis driver.
 *
 * Two kinds of test:
 *   1. Exact-arithmetic checks (no MC noise): action accounting, hot/cold
 *      starts, ΔS sign for trivial cases.
 *   2. Statistical tests against the Onsager critical point. These run
 *      short MC chains and use loose bounds so they are stable under CI
 *      noise — the point is the framework, not finite-size scaling.
 */

#include "fields/site_field.hpp"
#include "lattice/lattice.hpp"
#include "models/ising.hpp"
#include "monte_carlo/metropolis.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

// -----------------------------------------------------------------------------
// Exact-arithmetic
// -----------------------------------------------------------------------------

TEST(Ising2D, ColdConfigHasAllAlignedAction)
{
    // Cold (all +1) on L×L periodic lattice in 2D has 2 V nearest-neighbor
    // pairs (V links per direction × 2 directions, each counted once when
    // we walk only forward). All pairs are aligned (σσ = +1) so
    //     S = -β × (number of pairs) × 1 = -β × 2V
    constexpr int L = 6;
    Lattice<2> lattice = Lattice<2>::cube(L);
    SiteField<std::int8_t, 2> field(lattice);
    const double beta = 0.7;
    ising::IsingModel<2> model(lattice, beta);
    ising::IsingModel<2>::cold(field);

    const double expected = -beta * 2.0 * static_cast<double>(lattice.volume());
    EXPECT_NEAR(model.totalAction(field), expected, 1e-12);
}

TEST(Ising2D, DeltaSMatchesActionDifference)
{
    // Flip one spin in a hot configuration; ΔS computed by the model should
    // match the literal difference of total actions before and after.
    Lattice<2> lattice = Lattice<2>::cube(8);
    SiteField<std::int8_t, 2> field(lattice);
    Rng rng(42);
    ising::IsingModel<2>::hot(field, rng);

    ising::IsingModel<2> model(lattice, 0.4);
    for (int site : {0, 5, 17, 33, 60})
    {
        const double S_before = model.totalAction(field);
        const double dS_pred  = model.delta_action(field, site, ising::Flip{});
        model.apply(field, site, ising::Flip{});
        const double S_after  = model.totalAction(field);
        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-12) << "site " << site;
        // Restore.
        model.apply(field, site, ising::Flip{});
    }
}

// -----------------------------------------------------------------------------
// Statistical: walk across the Onsager transition and confirm the order
// parameter behaves correctly. Bounds are deliberately loose for a small
// L = 24 lattice with short chains so this is reproducible in CI.
// -----------------------------------------------------------------------------

namespace
{

struct Stats
{
    double abs_m_mean;
    double acceptance;
};

Stats run_2d_ising(int L, double beta, std::uint64_t seed,
                   int therm, int measure_sweeps, int sample_every)
{
    Lattice<2> lattice = Lattice<2>::cube(L);
    SiteField<std::int8_t, 2> field(lattice);
    Rng rng(seed);
    ising::IsingModel<2>::hot(field, rng);

    ising::IsingModel<2> model(lattice, beta);
    mc::MetropolisSweep<ising::IsingModel<2>> sweeper(model);
    sweeper.sweepN(field, rng, therm);
    sweeper.resetCounters();

    obs::Mean acc;
    for (int s = 0; s < measure_sweeps; ++s)
    {
        sweeper.sweep(field, rng);
        if ((s % sample_every) == 0)
            acc.add(obs::abs_magnetization(field));
    }
    return { acc.mean(), sweeper.cumulativeAcceptance() };
}

} // namespace

TEST(Ising2D, DisorderedAboveCritical)
{
    // β = 0.30 is well above T_c (i.e. well below β_c = 0.4407).
    // Order parameter should be small on L = 24.
    Stats s = run_2d_ising(/*L*/24, /*beta*/0.30,
                           /*seed*/123,
                           /*therm*/400,
                           /*measure*/1200,
                           /*every*/4);
    EXPECT_LT(s.abs_m_mean, 0.20) << "abs_m=" << s.abs_m_mean;
}

TEST(Ising2D, OrderedBelowCritical)
{
    // β = 0.55 is below T_c. Order parameter should be large.
    Stats s = run_2d_ising(/*L*/24, /*beta*/0.55,
                           /*seed*/456,
                           /*therm*/400,
                           /*measure*/1200,
                           /*every*/4);
    EXPECT_GT(s.abs_m_mean, 0.80) << "abs_m=" << s.abs_m_mean;
}

TEST(Ising2D, MonotoneAcrossTransition)
{
    // |m|(β) should be monotonically non-decreasing across the transition.
    // Sample at 5 β values straddling β_c and check ordering.
    constexpr double betas[] = { 0.30, 0.36, 0.42, 0.48, 0.55 };
    constexpr int N = sizeof(betas) / sizeof(betas[0]);
    double m[N];
    for (int i = 0; i < N; ++i)
        m[i] = run_2d_ising(20, betas[i], 1000u + i, 300, 800, 4).abs_m_mean;

    // Allow small inversions due to MC noise: require m[i+1] >= m[i] - 0.05.
    for (int i = 0; i + 1 < N; ++i)
        EXPECT_GE(m[i + 1], m[i] - 0.05) << "i=" << i
            << "  m[i]=" << m[i] << "  m[i+1]=" << m[i + 1];

    // And the endpoints should differ by a lot (transition is real).
    EXPECT_GT(m[N - 1] - m[0], 0.5);
}
