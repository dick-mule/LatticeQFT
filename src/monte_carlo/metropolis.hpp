#pragma once

/**
 * @file metropolis.hpp
 * @brief Model-agnostic single-unit Metropolis sweep.
 *
 * The sweeper visits each "update unit" once per sweep in color order.
 * A color is a sublattice within which all update units can in principle
 * be processed in parallel without conflicting (no two units in the same
 * color share an action term). For site models (Ising, φ⁴) a color is
 * just the parity (2 colors total). For link gauge models a color is
 * (site_parity, direction μ) so that no two same-colored link updates
 * share a plaquette — 2·Dim colors total.
 *
 * Any Model that exposes the small interface
 *
 *     using FieldT   = ...;
 *     using Proposal = ...;
 *     int  numColors() const;
 *     const std::vector<int>& unitsOfColor(int c) const;
 *     Proposal propose(const FieldT&, int unit, Rng&) const;
 *     double   delta_action(const FieldT&, int unit, Proposal) const;
 *     void     apply(FieldT&, int unit, Proposal) const;
 *
 * can be driven by this engine. "unit" is an integer ID whose meaning is
 * private to the model: a site index for site models, an encoded
 * (site, μ) pair for link models.
 */

#include "../rng/rng.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lqft::mc
{

template<typename Model>
class MetropolisSweep
{
public:
    using FieldT   = typename Model::FieldT;
    using Proposal = typename Model::Proposal;

    explicit MetropolisSweep(const Model& model) : m_model(model) {}

    /// One full sweep over every update unit, color by color.
    /// Returns the number of accepted updates this sweep.
    int sweep(FieldT& field, Rng& rng)
    {
        int accepted_this_sweep = 0;
        const int n_colors = m_model.numColors();
        for (int c = 0; c < n_colors; ++c)
        {
            const auto& units = m_model.unitsOfColor(c);
            for (int unit : units)
            {
                const Proposal prop = m_model.propose(field, unit, rng);
                const double   dS   = m_model.delta_action(field, unit, prop);
                // exp(-ΔS) ≥ 1 fast-path (always accept) keeps us off exp() for
                // half of all updates near critical, and all updates well below.
                bool accept = (dS <= 0.0);
                if (!accept)
                    accept = (rng.uniform() < std::exp(-dS));
                if (accept)
                {
                    m_model.apply(field, unit, prop);
                    ++accepted_this_sweep;
                }
                ++m_attempted_total;
            }
        }
        m_accepted_total += static_cast<std::uint64_t>(accepted_this_sweep);
        return accepted_this_sweep;
    }

    /// Run `n_sweeps` sweeps; returns the average per-site acceptance rate.
    double sweepN(FieldT& field, Rng& rng, int n_sweeps)
    {
        const std::uint64_t acc_before = m_accepted_total;
        const std::uint64_t att_before = m_attempted_total;
        for (int i = 0; i < n_sweeps; ++i) sweep(field, rng);
        const auto dacc = m_accepted_total - acc_before;
        const auto datt = m_attempted_total - att_before;
        return datt == 0 ? 0.0 : static_cast<double>(dacc) / static_cast<double>(datt);
    }

    double cumulativeAcceptance() const
    {
        return m_attempted_total == 0
            ? 0.0
            : static_cast<double>(m_accepted_total) / static_cast<double>(m_attempted_total);
    }

    void resetCounters() { m_accepted_total = m_attempted_total = 0; }

private:
    const Model&       m_model;
    std::uint64_t      m_accepted_total  = 0;
    std::uint64_t      m_attempted_total = 0;
};

} // namespace lqft::mc
