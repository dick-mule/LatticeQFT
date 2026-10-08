#pragma once

/**
 * @file observables.hpp
 * @brief Free-function observables on lattice fields.
 *
 * Observables live outside the model so we can compose them freely (any
 * field of the right type works) and so swapping a model out doesn't touch
 * the analysis code. Phase 0 only needs magnetization and energy per site;
 * later phases add Wilson loops, plaquette field, hadron correlators, etc.
 */

#include "../fields/site_field.hpp"
#include "../lattice/lattice.hpp"

#include <cmath>
#include <cstdint>

namespace lqft::obs
{

// -----------------------------------------------------------------------------
// Generic mean-of-field observables.
// Work for any site-DOF type T: int8_t (Ising spins) and double (φ⁴, …).
// Accumulation in long double minimizes rounding for large V.
// -----------------------------------------------------------------------------

template<typename T, int Dim>
double mean_field(const SiteField<T, Dim>& field)
{
    long double sum = 0;
    const int V = field.volume();
    for (int s = 0; s < V; ++s)
        sum += static_cast<long double>(field[s]);
    return static_cast<double>(sum / static_cast<long double>(V));
}

template<typename T, int Dim>
double mean_field_squared(const SiteField<T, Dim>& field)
{
    long double sum = 0;
    const int V = field.volume();
    for (int s = 0; s < V; ++s)
    {
        const long double x = static_cast<long double>(field[s]);
        sum += x * x;
    }
    return static_cast<double>(sum / static_cast<long double>(V));
}

template<typename T, int Dim>
double abs_mean_field(const SiteField<T, Dim>& field)
{
    return std::abs(mean_field(field));
}

// -----------------------------------------------------------------------------
// Ising-specific aliases (named after the physics for clarity at the call site).
// -----------------------------------------------------------------------------

/// Mean spin ⟨σ⟩ = (1/V) Σ_x σ_x. Range [-1, +1].
template<int Dim>
double magnetization(const SiteField<std::int8_t, Dim>& field)
{
    return mean_field(field);
}

/// |⟨σ⟩|. Order parameter for the broken-symmetry phase; correct for the
/// fact that finite-volume MC can flip global sign during a long run.
template<int Dim>
double abs_magnetization(const SiteField<std::int8_t, Dim>& field)
{
    return abs_mean_field(field);
}

/// (1/V) Σ_x σ_x Σ_μ σ_{x+ê_μ}. Range [-Dim, +Dim] (ferromagnetic = +Dim).
/// This is the "energy per site over -β"; for Ising the action density is
/// just -β times this quantity.
template<int Dim>
double nn_pair_density(
    const Lattice<Dim>& lattice,
    const SiteField<std::int8_t, Dim>& field)
{
    long long sum = 0;
    const int V = lattice.volume();
    for (int s = 0; s < V; ++s)
    {
        const int si = field[s];
        for (int mu = 0; mu < Dim; ++mu)
            sum += static_cast<long long>(si) * field[lattice.forward(s, mu)];
    }
    return static_cast<double>(sum) / static_cast<double>(V);
}

/// Rolling-mean accumulator used by sweep loops.
class Mean
{
public:
    void add(double x)            { m_sum += x; m_sumsq += x * x; ++m_n; }
    long long count() const       { return m_n; }
    double mean() const           { return m_n == 0 ? 0.0 : m_sum / static_cast<double>(m_n); }
    double variance() const
    {
        if (m_n < 2) return 0.0;
        const double m = mean();
        return m_sumsq / static_cast<double>(m_n) - m * m;
    }
    /// Naive standard error assuming uncorrelated samples (per-sweep
    /// autocorrelation is real; this is a lower bound on the true error).
    double naiveStdErr() const
    {
        return m_n < 2 ? 0.0 : std::sqrt(variance() / static_cast<double>(m_n));
    }
    void reset() { m_sum = m_sumsq = 0.0; m_n = 0; }
private:
    double    m_sum   = 0.0;
    double    m_sumsq = 0.0;
    long long m_n     = 0;
};

} // namespace lqft::obs
