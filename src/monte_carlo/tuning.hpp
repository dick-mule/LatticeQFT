#pragma once

/**
 * @file tuning.hpp
 * @brief Adaptive Metropolis step-size tuning.
 *
 * For continuous-DOF models (φ⁴, U(1), …) the Gaussian-perturbation step
 * size σ controls the per-update action jump and hence the acceptance
 * rate. Too small ⇒ acceptance ≈ 1 but trajectories crawl; too large ⇒
 * acceptance ≈ 0 and the chain stalls. The textbook optimum for single-site
 * Metropolis on a smooth target is around 44–55 % acceptance.
 *
 * This helper runs short MC bursts during thermalization, measures the
 * batch acceptance, and rescales the model's step size up or down until
 * acceptance lands within `tol` of `target`. Works with any model that
 * exposes
 *
 *     double stepSize() const;
 *     void   setStepSize(double);
 *
 * which is the convention `phi4::Phi4Model` (and any later continuous
 * model) follows. Discrete-DOF models (Ising) don't need this.
 */

#include "metropolis.hpp"
#include "../rng/rng.hpp"

#include <algorithm>
#include <cmath>

namespace lqft::mc
{

struct TuneResult
{
    int    iterations;
    double final_step_size;
    double final_acceptance;
};

template<typename Model>
TuneResult autoTuneStepSize(
    Model& model,
    typename Model::FieldT& field,
    Rng& rng,
    int    batch_sweeps = 100,
    int    max_iters   = 20,
    double target_acc  = 0.5,
    double tol         = 0.05,
    double min_step    = 1e-6,
    double max_step    = 1e3)
{
    MetropolisSweep<Model> sweeper(model);

    double step = model.stepSize();
    double acc  = 0.0;
    int    i    = 0;

    for (; i < max_iters; ++i)
    {
        sweeper.resetCounters();
        sweeper.sweepN(field, rng, batch_sweeps);
        acc = sweeper.cumulativeAcceptance();

        if (std::abs(acc - target_acc) <= tol) break;

        // Multiplicative rule: σ_new = σ_old · (acc / target)^α.
        // α ≈ 1 is the standard heuristic; bound the rescale per step to
        // avoid oscillation when batches are noisy.
        const double ratio        = acc / std::max(target_acc, 1e-6);
        const double bounded      = std::clamp(ratio, 0.5, 2.0);
        step = std::clamp(step * bounded, min_step, max_step);
        model.setStepSize(step);
    }

    return TuneResult{ i + 1, step, acc };
}

} // namespace lqft::mc
