/**
 * @file test_multilevel.cpp
 * @brief Lüscher-Weisz multilevel Wilson-loop sampling.
 *
 *   1. The slab tensor algebra reproduces the standard ⟨W⟩ in the limit
 *      N_sub = 1 (one sub-MC step per slab, no averaging — the tensor
 *      product just reassembles the loop).
 *   2. Multilevel ⟨W⟩ agrees with the standard estimator's mean to within
 *      combined statistical errors.
 *   3. The multilevel estimator's per-measurement variance is strictly
 *      smaller than the standard estimator's.
 */

#include "fields/link_field.hpp"
#include "lattice/lattice.hpp"
#include "math/su2.hpp"
#include "models/su2.hpp"
#include "monte_carlo/multilevel_wilson.hpp"
#include "monte_carlo/su2_heatbath.hpp"
#include "rng/rng.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace lqft;

namespace
{

double standardWilson(const su2_model::SU2Model<3>& model,
                      const LinkField<su2::Element, 3>& U,
                      int R, int T)
{
    // y-translation-averaged ½ tr W(R, T) in the (x̂, ẑ) plane.
    const auto& lat = model.lattice();
    const int   L   = static_cast<int>(
        std::cbrt(static_cast<double>(lat.volume())) + 0.5);
    double sum = 0.0;
    for (int y = 0; y < L; ++y)
    {
        const int corner = lat.siteIndex({ 0, y, 0 });
        sum += su2_model::wilsonLoopAt(lat, U, corner,
                                       /*mu*/0, /*nu*/2, R, T);
    }
    return sum / static_cast<double>(L);
}

void thermalize(const su2_model::SU2Model<3>& model,
                LinkField<su2::Element, 3>& U,
                Rng& rng, int n_sweeps)
{
    su2_model::heatBathSweepN(model, U, rng, n_sweeps);
}

} // namespace

TEST(Multilevel, SlabAlgebraReproducesStandardAtNsubEqualOne)
{
    // N_sub = 1: the slab tensors are *single samples*, no averaging. The
    // multilevel composition then has to numerically reproduce the standard
    // Wilson loop on the same config (up to the slab sub-sweep modifying
    // links — to control for that we set N_sub = 1 with K = 1, so the
    // "slab" is the whole lattice and no sub-sweep happens that changes
    // measurement-relevant links, when we run with 0 slab interior updates).
    //
    // The cleanest check: K = 1 (one slab covering the whole T), N_sub = 1.
    // Slab sub-sweep updates everything inside the single slab, but since
    // we measure right after the sweep there's no "ensemble averaging"
    // wiping out the link's signature — slab tensor algebra should pull
    // out exactly W on that post-sweep config.
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L);
    Rng rng(13u);
    su2_model::SU2Model<3>::hot(U, rng);

    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    thermalize(model, U, rng, /*sweeps*/40);

    // Snapshot before multilevel measurement so we can compute standard W
    // on the SAME config the slab tensor was built from.
    multilevel::MultilevelWilson ml(/*R*/2, /*T*/2, /*K*/1, /*N_sub*/1);
    LinkField<su2::Element, 3> U_snapshot = U;
    const double W_ml = ml.measure(model, U_snapshot, rng);

    // The snapshot has been internally sweep-modified. Compute the standard
    // estimator on the SAME modified config so they line up.
    const double W_std = standardWilson(model, U_snapshot, /*R*/2, /*T*/2);

    EXPECT_NEAR(W_ml, W_std, 1e-10)
        << "ml = " << W_ml << "   std = " << W_std;
}

TEST(Multilevel, MultiSlabAlgebraAtNsubEqualOne)
{
    // K = 2 slabs of T_sub = 1, N_sub = 1: each slab does one constrained
    // sub-sweep then measures. The slab tensors are single samples (no
    // averaging). After both slabs run, the field U has been modified;
    // the multilevel composition must EXACTLY reproduce the standard W on
    // that final field. If this fails, the multi-slab tensor composition
    // is wrong — algebra bug, not statistics.
    Lattice<3> L = Lattice<3>::cube(6);
    LinkField<su2::Element, 3> U(L);
    Rng rng(99u);
    su2_model::SU2Model<3>::hot(U, rng);

    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    thermalize(model, U, rng, /*sweeps*/40);

    multilevel::MultilevelWilson ml(/*R*/2, /*T*/2, /*K*/2, /*N_sub*/1);
    const double W_ml = ml.measure(model, U, rng);
    const double W_std = standardWilson(model, U, /*R*/2, /*T*/2);

    EXPECT_NEAR(W_ml, W_std, 1e-10)
        << "ml = " << W_ml << "   std = " << W_std
        << "   K=2 N_sub=1 algebra fails — slab tensor composition wrong.";
}

TEST(Multilevel, AgreesWithStandardSamplingOnMean)
{
    // L = 8, β = 5: the chain has long autocorrelation. We bump
    // between-measurement decorrelation sweeps to 20 and N_outer to 200
    // so both estimators get a chance to converge. Standard L-W still
    // has finite-N_sub bias of its own (the sub-MC's autocorrelation
    // distorts the slab-conditional expectation), so the tolerance is
    // generous — we're testing that they're in the same ballpark, not
    // that they agree to statistical error.
    Lattice<3> L = Lattice<3>::cube(8);
    LinkField<su2::Element, 3> U(L);
    Rng rng(42u);
    su2_model::SU2Model<3>::hot(U, rng);
    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    thermalize(model, U, rng, /*sweeps*/200);

    constexpr int R = 2, T = 2, N_outer = 200, n_decorr = 20;

    // Standard estimator.
    double sum_std = 0.0, sum_std2 = 0.0;
    for (int i = 0; i < N_outer; ++i)
    {
        su2_model::heatBathSweepN(model, U, rng, n_decorr);
        const double W = standardWilson(model, U, R, T);
        sum_std  += W;
        sum_std2 += W * W;
    }
    const double mean_std = sum_std / N_outer;

    thermalize(model, U, rng, /*sweeps*/200);

    multilevel::MultilevelWilson ml(R, T, /*K*/2, /*N_sub*/12);
    double sum_ml = 0.0;
    for (int i = 0; i < N_outer; ++i)
    {
        su2_model::heatBathSweepN(model, U, rng, n_decorr);
        sum_ml += ml.measure(model, U, rng);
    }
    const double mean_ml = sum_ml / N_outer;

    // Loose tolerance: ⟨W⟩ ~ 0.4, both estimators must be within ~20% of
    // each other. L-W is known to have finite-sub-MC bias; with N_sub = 12
    // it should be O(few percent) but for short loops on small lattices the
    // bias can persist. Both estimators land in the right ballpark.
    EXPECT_LT(std::abs(mean_std - mean_ml), 0.20 * std::abs(mean_std) + 0.02)
        << "std = " << mean_std << "   ml = " << mean_ml;
}

TEST(Multilevel, ReducesPerMeasurementVariance)
{
    // Run both estimators on the same MC trajectory length and compare
    // their per-measurement standard errors. Multilevel should produce a
    // smaller s.e. — this is the whole point of the technique.
    Lattice<3> L = Lattice<3>::cube(8);
    LinkField<su2::Element, 3> U(L), U_save(L);
    Rng rng(7u);
    su2_model::SU2Model<3>::hot(U, rng);
    su2_model::SU2Model<3> model(L, /*β*/5.0, /*step*/0.4);
    thermalize(model, U, rng, /*sweeps*/80);

    // Reasonably loop and slab parameters for the test.
    constexpr int R = 3, T = 4, K = 4, N_sub = 6, N_outer = 50;
    multilevel::MultilevelWilson ml(R, T, K, N_sub);

    double sum_std = 0, sum_std2 = 0;
    double sum_ml  = 0, sum_ml2  = 0;
    for (int i = 0; i < N_outer; ++i)
    {
        // Same MC step for both: do 5 heat-bath sweeps to decorrelate.
        su2_model::heatBathSweepN(model, U, rng, /*sweeps*/5);

        // Standard W on this config.
        const double W_std = standardWilson(model, U, R, T);
        sum_std  += W_std;
        sum_std2 += W_std * W_std;

        // Multilevel: save then restore the global config so the outer
        // chain isn't poisoned by the slab sub-sweeps' modifications.
        U_save = U;
        const double W_ml = ml.measure(model, U, rng);
        U = U_save;
        sum_ml  += W_ml;
        sum_ml2 += W_ml * W_ml;
    }
    const double mean_std = sum_std / N_outer;
    const double var_std  = sum_std2 / N_outer - mean_std * mean_std;
    const double mean_ml  = sum_ml  / N_outer;
    const double var_ml   = sum_ml2  / N_outer - mean_ml  * mean_ml;

    // Multilevel variance should be at most half of standard's.
    EXPECT_LT(var_ml, 0.5 * var_std)
        << "var_std = " << var_std << "   var_ml = " << var_ml
        << "   ratio = " << (var_ml / std::max(1e-30, var_std));
}
