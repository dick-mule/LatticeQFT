#pragma once

/**
 * @file su3.hpp (models)
 * @brief Pure-gauge SU(3) Yang-Mills on a Dim-dimensional periodic lattice.
 *
 * Wilson plaquette action:
 *
 *     S[U] = β Σ_□ (1 − (1/N_c) Re tr U_□),     N_c = 3
 *          = β Σ_□ (1 − (1/3) Re tr U_□)
 *
 * with U_□(x; μ, ν) = U_μ(x) · U_ν(x+ê_μ) · U_μ†(x+ê_ν) · U_ν†(x).
 *
 * ## Single-link Metropolis (Cabibbo-Marinari sub-group proposal)
 *
 * Direct random-walk Metropolis with a near-identity SU(3) proposal requires
 * an 8-parameter generator (Gell-Mann matrices) plus a matrix-exponential.
 * Cabibbo-Marinari sidesteps both: pick one of the three SU(2) sub-groups
 * embedded in row/column pairs (0,1), (0,2), (1,2) uniformly at random,
 * generate a near-identity SU(2) rotation in that sub-block, embed it in
 * SU(3), and use it as the proposal ΔU.
 *
 * The proposal is symmetric under ΔU → ΔU† (each SU(2) sub-group rotation
 * sample is symmetric under that flip, and the random sub-group choice
 * doesn't change), so detailed balance holds with the simple Metropolis
 * accept probability `min(1, exp(−ΔS))`. Three sub-groups are enough to
 * span SU(3) ergodically.
 *
 * ## ΔS evaluation
 *
 * For SU(2) we had `ΔS = −(β/2) Re tr[(U_new − U_old) · A]`. The N_c = 3
 * version replaces the prefactor:
 *
 *     ΔS = −(β/N_c) Re tr[(U_new − U_old) · A]
 *
 * with `A` the same 2(Dim−1)-term staple sum as in SU(2). `Re tr` is now
 * computed directly on 3×3 complex matrices (`su3::real_trace_product`).
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su3.hpp"
#include "../rng/rng.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace lqft::su3_model
{

template<int Dim>
class SU3Model
{
public:
    using FieldT   = LinkField<su3::Element, Dim>;
    using Proposal = su3::Element;
    using LatticeT = Lattice<Dim>;

    static constexpr int dimension() { return Dim; }
    static_assert(Dim >= 2, "SU(3) gauge theory needs ≥ 2 directions for plaquettes");

    SU3Model(const LatticeT& lattice, double beta, double step_size = 0.3)
        : m_lattice(lattice), m_beta(beta), m_step_size(step_size)
    {
        buildColorPartition();
    }

    const LatticeT& lattice() const { return m_lattice; }
    double beta()     const { return m_beta; }
    double stepSize() const { return m_step_size; }
    void   setBeta(double b)     { m_beta = b; }
    void   setStepSize(double s) { m_step_size = s; }

    // ---------------------------------------------------------------------
    // Unit / color encoding (identical to U1Model / SU2Model).
    // ---------------------------------------------------------------------

    int numColors() const { return 2 * Dim; }
    const std::vector<int>& unitsOfColor(int c) const { return m_units_by_color[c]; }

    static constexpr int siteOfUnit(int unit) { return unit / Dim; }
    static constexpr int muOfUnit  (int unit) { return unit % Dim; }
    static constexpr int unitOf(int site, int mu) { return site * Dim + mu; }

    // ---------------------------------------------------------------------
    // Staple sum at link (site, μ) — same topology as SU(2), just on 3×3
    // complex matrices.
    // ---------------------------------------------------------------------

    su3::Element staple(const FieldT& field, int site, int mu) const
    {
        su3::Element total = su3::Element::zero();
        for (int nu = 0; nu < Dim; ++nu)
        {
            if (nu == mu) continue;

            const int y_pmu     = m_lattice.forward(site, mu);
            const int y_pnu     = m_lattice.forward(site, nu);
            const int y_mnu     = m_lattice.backward(site, nu);
            const int y_pmu_mnu = m_lattice.backward(y_pmu, nu);

            // Forward plaquette staple:
            //   S_+ = U_ν(y+μ) · U_μ†(y+ν) · U_ν†(y)
            su3::Element fwd = su3::multiply(
                field(y_pmu, nu), su3::dagger(field(y_pnu, mu)));
            fwd = su3::multiply(fwd, su3::dagger(field(site, nu)));
            total = su3::add(total, fwd);

            // Backward plaquette staple:
            //   S_- = U_ν†(y+μ−ν) · U_μ†(y−ν) · U_ν(y−ν)
            su3::Element bwd = su3::multiply(
                su3::dagger(field(y_pmu_mnu, nu)),
                su3::dagger(field(y_mnu, mu)));
            bwd = su3::multiply(bwd, field(y_mnu, nu));
            total = su3::add(total, bwd);
        }
        return total;
    }

    // ---------------------------------------------------------------------
    // Model interface.
    // ---------------------------------------------------------------------

    Proposal propose(const FieldT& field, int unit, Rng& rng) const
    {
        const int site = siteOfUnit(unit);
        const int mu   = muOfUnit(unit);
        const su3::Element delta = randomSubgroupNearIdentity(m_step_size, rng);
        return su3::multiply(delta, field(site, mu));
    }

    /// ΔS = −(β/3) [Re tr(U_new · A) − Re tr(U_old · A)].
    double delta_action(const FieldT& field, int unit, Proposal U_new) const
    {
        const int site = siteOfUnit(unit);
        const int mu   = muOfUnit(unit);
        const su3::Element& U_old = field(site, mu);
        const su3::Element A      = staple(field, site, mu);

        const double dRetr = su3::real_trace_product(U_new, A)
                           - su3::real_trace_product(U_old, A);
        return -(m_beta / 3.0) * dRetr;
    }

    void apply(FieldT& field, int unit, Proposal U_new) const
    {
        field(siteOfUnit(unit), muOfUnit(unit)) = U_new;
    }

    // ---------------------------------------------------------------------
    // Total action — Σ_□ over μ < ν, each plaquette once.
    // ---------------------------------------------------------------------

    double totalAction(const FieldT& field) const
    {
        double S = 0.0;
        const int V = m_lattice.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                for (int nu = mu + 1; nu < Dim; ++nu)
                {
                    const int s_mu = m_lattice.forward(s, mu);
                    const int s_nu = m_lattice.forward(s, nu);
                    su3::Element P = su3::multiply(field(s, mu), field(s_mu, nu));
                    P = su3::multiply(P, su3::dagger(field(s_nu, mu)));
                    P = su3::multiply(P, su3::dagger(field(s, nu)));
                    S += m_beta * (1.0 - su3::real_trace(P) / 3.0);
                }
        return S;
    }

    // ---------------------------------------------------------------------
    // Initializers.
    // ---------------------------------------------------------------------

    static void cold(FieldT& field)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = su3::Element::identity();
    }

    /// Hot start: compose three near-identity-but-large SU(2) sub-group
    /// rotations and re-project to make every link a genuine SU(3) element.
    /// Not bias-free uniform on SU(3) (true uniform sampling needs the Haar
    /// measure via QR of a complex Gaussian matrix), but uniform enough that
    /// the action is unbiased and equilibrium is reached after thermalization.
    static void hot(FieldT& field, Rng& rng)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = randomSU3(rng);
    }

    // ---------------------------------------------------------------------
    // Sampling helpers.
    // ---------------------------------------------------------------------

    /// Random SU(3) by composing three "wide" SU(2) sub-group rotations
    /// (σ = π/2 each) and Gram-Schmidt projecting. Cheap, covers SU(3).
    static su3::Element randomSU3(Rng& rng)
    {
        su3::Element U = su3::Element::identity();
        for (int sub = 0; sub < 3; ++sub)
            U = su3::multiply(randomSubgroupNearIdentity(/*σ*/1.5, rng, sub), U);
        return su3::project_to_su3(U);
    }

    /// Sub-group Metropolis kernel: pick a uniformly-random SU(2) sub-block
    /// of SU(3) and a near-identity SU(2) rotation in it with scale `sigma`.
    /// Symmetric under ΔU → ΔU† so detailed balance is automatic.
    static su3::Element randomSubgroupNearIdentity(double sigma, Rng& rng,
                                                  int sub_index = -1)
    {
        if (sub_index < 0)
            sub_index = std::min(2, static_cast<int>(rng.uniform() * 3.0));
        const std::array<std::array<int, 2>, 3> pairs = { {
            {0, 1}, {0, 2}, {1, 2}
        } };
        const int i = pairs[static_cast<std::size_t>(sub_index)][0];
        const int j = pairs[static_cast<std::size_t>(sub_index)][1];

        const double t0 = sigma * rng.normal();
        const double t1 = sigma * rng.normal();
        const double t2 = sigma * rng.normal();
        const double mag = std::sqrt(t0 * t0 + t1 * t1 + t2 * t2);
        if (mag < 1e-12) return su3::Element::identity();
        const double half = 0.5 * mag;
        const double s    = std::cos(half);
        const double v_scale = std::sin(half) / mag;

        return su3::embedSU2(i, j, s,
                             v_scale * t0, v_scale * t1, v_scale * t2);
    }

private:
    void buildColorPartition()
    {
        m_units_by_color.assign(static_cast<std::size_t>(2 * Dim), {});
        const int V = m_lattice.volume();
        for (auto& v : m_units_by_color)
            v.reserve(static_cast<std::size_t>(V / 2 + 1));
        for (int s = 0; s < V; ++s)
        {
            const int p = m_lattice.parityOf(s);
            for (int mu = 0; mu < Dim; ++mu)
                m_units_by_color[p * Dim + mu].push_back(unitOf(s, mu));
        }
    }

    const LatticeT& m_lattice;
    double          m_beta;
    double          m_step_size;
    std::vector<std::vector<int>> m_units_by_color;
};

// ----------------------------------------------------------------------------
// Average plaquette: ⟨(1/N_c) Re tr U_□⟩ over every oriented plaquette.
// Cold (all-identity) configuration → 1; deep-disorder limit → 0.
// ----------------------------------------------------------------------------

template<int Dim>
double averagePlaquette(const Lattice<Dim>& lattice,
                        const LinkField<su3::Element, Dim>& field)
{
    double sum = 0.0;
    long long N = 0;
    const int V = lattice.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < Dim; ++mu)
            for (int nu = mu + 1; nu < Dim; ++nu)
            {
                const int s_mu = lattice.forward(s, mu);
                const int s_nu = lattice.forward(s, nu);
                su3::Element P = su3::multiply(field(s, mu), field(s_mu, nu));
                P = su3::multiply(P, su3::dagger(field(s_nu, mu)));
                P = su3::multiply(P, su3::dagger(field(s, nu)));
                sum += su3::real_trace(P) / 3.0;
                ++N;
            }
    return sum / static_cast<double>(N);
}

// ----------------------------------------------------------------------------
// Wilson loop W(R, T) at one corner: (1/N_c) Re tr of the path-ordered
// product around a rectangular loop in plane (μ, ν).
// ----------------------------------------------------------------------------

template<int Dim>
double wilsonLoopAt(
    const Lattice<Dim>& lat,
    const LinkField<su3::Element, Dim>& field,
    int corner, int mu, int nu, int R, int T)
{
    su3::Element P = su3::Element::identity();
    int s = corner;
    for (int i = 0; i < R; ++i) { P = su3::multiply(P, field(s, mu)); s = lat.forward(s, mu); }
    for (int j = 0; j < T; ++j) { P = su3::multiply(P, field(s, nu)); s = lat.forward(s, nu); }
    for (int i = 0; i < R; ++i) { s = lat.backward(s, mu); P = su3::multiply(P, su3::dagger(field(s, mu))); }
    for (int j = 0; j < T; ++j) { s = lat.backward(s, nu); P = su3::multiply(P, su3::dagger(field(s, nu))); }
    return su3::real_trace(P) / 3.0;
}

} // namespace lqft::su3_model
