/**
 * @file test_phi4.cpp
 * @brief Quantitative tests of the 2D φ⁴ Metropolis driver.
 *
 *   1. Exact-arithmetic:
 *      - Cold start φ ≡ 0 has S = 0.
 *      - The model's `delta_action(...)` matches the literal action
 *        difference before and after `apply(...)` to machine precision.
 *
 *   2. Statistical (against the closed-form free Gaussian propagator):
 *      - λ = 0, m² > 0 ⇒ ⟨φ²⟩_MC matches the lattice Klein-Gordon sum
 *            ⟨φ²⟩ = (1/V) Σ_p 1 / (4 Σ_μ sin²(p_μ/2) + m²)
 *        within a few percent on a small L = 8 lattice.
 *
 *   3. Infrastructure (mc::autoTuneStepSize):
 *      - The tuner drives acceptance into [0.4, 0.6] within max_iters.
 */

#include "fields/site_field.hpp"
#include "lattice/lattice.hpp"
#include "models/phi4.hpp"
#include "monte_carlo/metropolis.hpp"
#include "monte_carlo/tuning.hpp"
#include "observables/observables.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace lqft;

// -----------------------------------------------------------------------------
// Exact-arithmetic
// -----------------------------------------------------------------------------

TEST(Phi42D, ColdStartHasZeroAction)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    SiteField<double, 2> field(lattice);
    phi4::Phi4Model<2>::cold(field);

    phi4::Phi4Model<2> model(lattice, /*m²*/0.7, /*λ*/1.2);
    EXPECT_DOUBLE_EQ(model.totalAction(field), 0.0);
}

TEST(Phi42D, DeltaSMatchesActionDifference)
{
    Lattice<2> lattice = Lattice<2>::cube(8);
    SiteField<double, 2> field(lattice);
    Rng rng(42);
    phi4::Phi4Model<2>::hot(field, rng, /*scale*/1.0);

    phi4::Phi4Model<2> model(lattice, /*m²*/0.5, /*λ*/1.0);

    // For each of several proposed updates at several sites, compare
    //   delta_action  vs  totalAction(after) - totalAction(before)
    // and require machine-precision agreement.
    for (int site : {0, 3, 11, 27, 50, 63})
    {
        const double phi_old   = field[site];
        const double proposal  = phi_old + 0.37; // arbitrary fixed displacement
        const double S_before  = model.totalAction(field);
        const double dS_pred   = model.delta_action(field, site, proposal);
        model.apply(field, site, proposal);
        const double S_after   = model.totalAction(field);
        EXPECT_NEAR(S_after - S_before, dS_pred, 1e-9) << "site " << site;
        // Restore so the next iteration starts from the same state.
        model.apply(field, site, phi_old);
    }
}

// -----------------------------------------------------------------------------
// Statistical: free-field Gaussian ⟨φ²⟩.
// -----------------------------------------------------------------------------

TEST(Phi42D, FreeFieldMatchesAnalyticPhiSquared)
{
    constexpr int    L     = 8;
    constexpr double m_sq  = 1.0;
    constexpr double lam   = 0.0;
    Lattice<2> lattice = Lattice<2>::cube(L);
    SiteField<double, 2> field(lattice);
    Rng rng(2024u);

    phi4::Phi4Model<2> model(lattice, m_sq, lam, /*step*/1.0);
    // Tune step before measuring — free-field acceptance peaks easily.
    mc::autoTuneStepSize(model, field, rng,
                         /*batch_sweeps*/50,
                         /*max_iters*/12,
                         /*target*/0.5,
                         /*tol*/0.05);

    mc::MetropolisSweep<phi4::Phi4Model<2>> sweeper(model);
    sweeper.sweepN(field, rng, /*thermalize*/600);

    obs::Mean phi2;
    constexpr int measure_sweeps = 4000;
    constexpr int sample_every   = 2;
    for (int s = 0; s < measure_sweeps; ++s)
    {
        sweeper.sweep(field, rng);
        if ((s % sample_every) == 0)
            phi2.add(obs::mean_field_squared(field));
    }

    const double analytic = phi4::freeFieldPhi2(lattice, m_sq);
    const double measured = phi2.mean();

    // Loose 8 % tolerance: V = 64, ~2000 samples, MC noise on ⟨φ²⟩ is a
    // few percent. The point is the agreement with the closed form.
    const double rel_err = std::abs(measured - analytic) / analytic;
    EXPECT_LT(rel_err, 0.08)
        << "measured=" << measured
        << "  analytic=" << analytic
        << "  rel_err=" << rel_err;
}

// -----------------------------------------------------------------------------
// Infrastructure: adaptive step-size tuner lands inside the target band.
// -----------------------------------------------------------------------------

TEST(Phi42D, AutoTuneLandsNearTargetAcceptance)
{
    Lattice<2> lattice = Lattice<2>::cube(12);
    SiteField<double, 2> field(lattice);
    Rng rng(777u);
    phi4::Phi4Model<2>::hot(field, rng, 1.0);

    // Pick an absurd starting step size so the tuner has to work.
    phi4::Phi4Model<2> model(lattice, /*m²*/0.5, /*λ*/1.0, /*step*/10.0);

    auto r = mc::autoTuneStepSize(model, field, rng,
                                  /*batch_sweeps*/80,
                                  /*max_iters*/25,
                                  /*target*/0.5,
                                  /*tol*/0.07);
    EXPECT_GT(r.final_acceptance, 0.40) << "step=" << r.final_step_size;
    EXPECT_LT(r.final_acceptance, 0.60) << "step=" << r.final_step_size;
    EXPECT_GT(r.final_step_size, 1e-3);
}
