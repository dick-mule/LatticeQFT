#pragma once

/**
 * @file ising.hpp
 * @brief The Ising model on a Dim-dimensional periodic lattice.
 *
 * Spins σ_x ∈ {−1, +1} live on sites. Action
 *
 *     S[σ] = -β Σ_{<x,y>} σ_x σ_y                                       (*)
 *
 * where the sum runs over each unordered nearest-neighbor pair once. The
 * single-spin-flip Metropolis update at site x has
 *
 *     ΔS = 2 β σ_x Σ_{μ=0..Dim-1} (σ_{x+ê_μ} + σ_{x-ê_μ})
 *
 * This is the smallest non-trivial Monte Carlo problem; it serves as the
 * test bench for the lattice + parity + RNG + Metropolis infrastructure.
 *
 * Exact 2D Onsager critical inverse temperature:
 *     β_c = ½ ln(1 + √2) ≈ 0.4406867935...
 *
 * The Model interface is the same one every later phase will implement:
 *   - using FieldT     = SiteField<int8_t, Dim>;
 *   - using Proposal   = lqft::ising::Flip   (trivial; just "flip this spin")
 *   - propose(field, site, rng)        -> Proposal
 *   - delta_action(field, site, prop)  -> double
 *   - apply(field, site, proposal)     -> void
 */

#include "../fields/site_field.hpp"
#include "../lattice/lattice.hpp"
#include "../rng/rng.hpp"

#include <cstdint>

namespace lqft::ising
{

/// Trivial proposal type: the only single-site update for an Ising spin is
/// a flip. Carrying a struct here (rather than overloading on `void`) keeps
/// the interface uniform with continuous models (φ⁴, U(1), …).
struct Flip {};

template<int Dim>
class IsingModel
{
public:
    using FieldT     = SiteField<std::int8_t, Dim>;
    using Proposal   = Flip;
    using LatticeT   = Lattice<Dim>;

    static constexpr int dimension() { return Dim; }

    IsingModel(const LatticeT& lattice, double beta)
        : m_lattice(lattice), m_beta(beta) {}

    double beta() const { return m_beta; }
    void   setBeta(double beta) { m_beta = beta; }

    const LatticeT& lattice() const { return m_lattice; }

    // -----------------------------------------------------------------------
    // Color partition: 2 colors (even / odd parity).
    // -----------------------------------------------------------------------

    int numColors() const { return 2; }
    const std::vector<int>& unitsOfColor(int c) const
    {
        return m_lattice.sitesOfParity(c);
    }

    // -----------------------------------------------------------------------
    // Model interface
    // -----------------------------------------------------------------------

    Proposal propose(const FieldT& /*field*/, int /*site*/, Rng& /*rng*/) const
    {
        // Single-spin flip is the only move.
        return Flip{};
    }

    /// Σ_μ (σ_{x+ê_μ} + σ_{x-ê_μ}). Integer sum, kept as int to avoid float
    /// noise; multiplied into a double in delta_action().
    int neighborSum(const FieldT& field, int site) const
    {
        int s = 0;
        for (int mu = 0; mu < Dim; ++mu)
        {
            s += field[m_lattice.forward(site, mu)];
            s += field[m_lattice.backward(site, mu)];
        }
        return s;
    }

    double delta_action(const FieldT& field, int site, Proposal /*p*/) const
    {
        // ΔS = 2 β σ_x Σ_neighbors σ_y
        return 2.0 * m_beta * static_cast<double>(field[site])
                            * static_cast<double>(neighborSum(field, site));
    }

    void apply(FieldT& field, int site, Proposal /*p*/) const
    {
        field[site] = static_cast<std::int8_t>(-field[site]);
    }

    // -----------------------------------------------------------------------
    // Convenience: total action S[σ] for the current field.
    // Used by tests and by observables that need <E>. O(Dim * V).
    // -----------------------------------------------------------------------

    double totalAction(const FieldT& field) const
    {
        // Sum each unordered pair once by only walking forward links.
        long long sum = 0;
        const int V = m_lattice.volume();
        for (int s = 0; s < V; ++s)
        {
            const int si = field[s];
            for (int mu = 0; mu < Dim; ++mu)
                sum += static_cast<long long>(si) * field[m_lattice.forward(s, mu)];
        }
        return -m_beta * static_cast<double>(sum);
    }

    // -----------------------------------------------------------------------
    // Field initializers.
    // -----------------------------------------------------------------------

    /// Cold start: all spins +1 (lowest-action configuration).
    static void cold(FieldT& field)
    {
        field.fill(static_cast<std::int8_t>(+1));
    }

    /// Hot start: each spin independently uniform on {-1, +1}.
    static void hot(FieldT& field, Rng& rng)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            field[s] = (rng.uniform() < 0.5) ? std::int8_t{-1} : std::int8_t{+1};
    }

private:
    const LatticeT& m_lattice;
    double          m_beta;
};

/// Exact 2D Ising critical inverse temperature (Onsager).
inline constexpr double kOnsagerBetaC2D = 0.44068679350977147; // ½ ln(1 + √2)

} // namespace lqft::ising
