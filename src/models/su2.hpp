#pragma once

/**
 * @file su2.hpp (models)
 * @brief Pure-gauge SU(2) Yang-Mills on a Dim-dimensional periodic lattice.
 *
 * Each directed link carries an SU(2) matrix `U_μ(x) ∈ SU(2)` stored as a
 * unit quaternion (`su2::Element`). The Wilson plaquette action is
 *
 *     S[U] = β Σ_□ (1 − ½ tr U_□)
 *          = β Σ_□ (1 − [U_□].s)
 *
 * with `U_□(x; μ, ν) = U_μ(x) · U_ν(x+ê_μ) · U_μ†(x+ê_ν) · U_ν†(x)`.
 *
 * ## Update strategy
 *
 * Single-link Metropolis proposes `U → ΔU · U` with `ΔU = exp(i θ·σ/2)`
 * and `θ ∼ N(0, σ²·I)`. The symmetry `θ → −θ ⇔ ΔU → ΔU†` makes the
 * proposal symmetric. The action change collapses to
 *
 *     ΔS = − (β / 2) · Re tr[(U_new − U_old) · A_μ(x)]
 *
 * where the staple `A_μ(x)` is the sum of the 2(Dim − 1) products of three
 * neighboring links that close into a plaquette through this link. The
 * staple is not in SU(2) but lives in the same 4-vector representation
 * so all the algebra reuses `su2::multiply`.
 *
 * ## Color partition
 *
 * Same as U(1): 2·Dim colors indexed by `(site_parity, μ)`. The encoding
 * `unit_id = site · Dim + μ` is identical, so the generic `MetropolisSweep`
 * runs unchanged.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"
#include "../rng/rng.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace lqft::su2_model
{

template<int Dim>
class SU2Model
{
public:
    using FieldT   = LinkField<su2::Element, Dim>;
    using Proposal = su2::Element;
    using LatticeT = Lattice<Dim>;

    static constexpr int dimension() { return Dim; }
    static_assert(Dim >= 2, "SU(2) gauge theory needs ≥ 2 directions for plaquettes");

    SU2Model(const LatticeT& lattice, double beta, double step_size = 0.4)
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
    // Unit / color encoding (same as U1Model).
    // ---------------------------------------------------------------------

    int numColors() const { return 2 * Dim; }
    const std::vector<int>& unitsOfColor(int c) const { return m_units_by_color[c]; }

    static constexpr int siteOfUnit(int unit) { return unit / Dim; }
    static constexpr int muOfUnit  (int unit) { return unit % Dim; }
    static constexpr int unitOf(int site, int mu) { return site * Dim + mu; }

    // ---------------------------------------------------------------------
    // Staple sum at link (site, μ).
    //   A_μ(y) = Σ_{ν ≠ μ}
    //              [ U_ν(y+ê_μ) U_μ†(y+ê_ν) U_ν†(y)
    //              + U_ν†(y+ê_μ−ê_ν) U_μ†(y−ê_ν) U_ν(y−ê_ν) ]
    // The two terms per ν correspond to the "forward" and "backward"
    // plaquettes that contain link (y, μ).
    // ---------------------------------------------------------------------

    su2::Element staple(const FieldT& field, int site, int mu) const
    {
        su2::Element total = su2::Element::zero();
        for (int nu = 0; nu < Dim; ++nu)
        {
            if (nu == mu) continue;

            const int y_pmu = m_lattice.forward(site, mu);
            const int y_pnu = m_lattice.forward(site, nu);
            const int y_mnu = m_lattice.backward(site, nu);
            // (y + ê_μ − ê_ν): step backward in ν from y + ê_μ.
            const int y_pmu_mnu = m_lattice.backward(y_pmu, nu);

            // Forward plaquette staple.
            su2::Element fwd = su2::multiply(
                field(y_pmu, nu),
                su2::dagger(field(y_pnu, mu))
            );
            fwd = su2::multiply(fwd, su2::dagger(field(site, nu)));
            total = su2::add(total, fwd);

            // Backward plaquette staple.
            su2::Element bwd = su2::multiply(
                su2::dagger(field(y_pmu_mnu, nu)),
                su2::dagger(field(y_mnu, mu))
            );
            bwd = su2::multiply(bwd, field(y_mnu, nu));
            total = su2::add(total, bwd);
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
        const su2::Element delta = randomNearIdentity(m_step_size, rng);
        return su2::multiply(delta, field(site, mu));
    }

    double delta_action(const FieldT& field, int unit, Proposal U_new) const
    {
        const int site = siteOfUnit(unit);
        const int mu   = muOfUnit(unit);
        const su2::Element& U_old = field(site, mu);
        const su2::Element A      = staple(field, site, mu);

        // ΔS = − (β / 2) · [ Re tr(U_new A) − Re tr(U_old A) ].
        const double dRetr = su2::real_trace_product(U_new, A)
                           - su2::real_trace_product(U_old, A);
        return -0.5 * m_beta * dRetr;
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
                    su2::Element P = su2::multiply(field(s, mu), field(s_mu, nu));
                    P = su2::multiply(P, su2::dagger(field(s_nu, mu)));
                    P = su2::multiply(P, su2::dagger(field(s, nu)));
                    // 1 − ½ tr U_□ = 1 − P.s
                    S += m_beta * (1.0 - P.s);
                }
        return S;
    }

    // ---------------------------------------------------------------------
    // Initializers.
    // ---------------------------------------------------------------------

    /// All links = I → S = 0 exactly.
    static void cold(FieldT& field)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = su2::Element::identity();
    }

    /// Uniform on S³ via Marsaglia 4D sphere-rejection.
    static void hot(FieldT& field, Rng& rng)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = randomUniformSU2(rng);
    }

    // ---------------------------------------------------------------------
    // Sampling helpers.
    // ---------------------------------------------------------------------

    /// Marsaglia sphere-rejection: sample uniform on S³.
    static su2::Element randomUniformSU2(Rng& rng)
    {
        double x[4];
        double r2;
        do
        {
            x[0] = 2.0 * rng.uniform() - 1.0;
            x[1] = 2.0 * rng.uniform() - 1.0;
            x[2] = 2.0 * rng.uniform() - 1.0;
            x[3] = 2.0 * rng.uniform() - 1.0;
            r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3];
        } while (r2 > 1.0 || r2 < 1e-12);
        const double r = std::sqrt(r2);
        return { x[0] / r, { x[1] / r, x[2] / r, x[3] / r } };
    }

    /// SU(2) Metropolis proposal kernel: ΔU = exp(i θ·σ / 2) with
    /// θ ∼ N(0, sigma²·I). For small `sigma` this is close to the identity;
    /// the proposal is symmetric under θ → −θ ⇔ ΔU → ΔU†.
    static su2::Element randomNearIdentity(double sigma, Rng& rng)
    {
        const double t0 = sigma * rng.normal();
        const double t1 = sigma * rng.normal();
        const double t2 = sigma * rng.normal();
        const double mag = std::sqrt(t0*t0 + t1*t1 + t2*t2);
        if (mag < 1e-12) return su2::Element::identity();
        const double half = 0.5 * mag;
        const double c    = std::cos(half);
        const double s    = std::sin(half) / mag;
        return { c, { s * t0, s * t1, s * t2 } };
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
// Plaquette observable: ⟨½ tr U_□⟩ averaged over every oriented plaquette.
// Range [−1, +1]; equals 1 in the cold (all-identity) configuration.
// ----------------------------------------------------------------------------

template<int Dim>
double averagePlaquette(const Lattice<Dim>& lattice,
                        const LinkField<su2::Element, Dim>& field)
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
                su2::Element P = su2::multiply(field(s, mu), field(s_mu, nu));
                P = su2::multiply(P, su2::dagger(field(s_nu, mu)));
                P = su2::multiply(P, su2::dagger(field(s, nu)));
                sum += P.s;
                ++N;
            }
    return sum / static_cast<double>(N);
}

// ----------------------------------------------------------------------------
// Wilson loop W(R, T) at one corner: ½ tr of the path-ordered product around
// the rectangular loop with sides R · ê_μ and T · ê_ν.
//
// Gauge invariance: the closed-loop product cyclically eats the gauge-rotation
// matrices Ω(x) at every vertex, so ½ tr W is invariant under
// U_μ(x) → Ω(x) U_μ(x) Ω†(x + ê_μ).
//
// Area law in confining phase:
//     ⟨W(R, T)⟩ ~ exp(−σ · R · T − μ · (R + T) + const)
// the perimeter μ piece is the self-energy of the static charges; σ is the
// string tension that survives in the (R, T) → ∞ limit.
// ----------------------------------------------------------------------------

template<int Dim>
double wilsonLoopAt(
    const Lattice<Dim>& lat,
    const LinkField<su2::Element, Dim>& field,
    int corner, int mu, int nu, int R, int T)
{
    su2::Element P = su2::Element::identity();
    int s = corner;
    // forward μ
    for (int i = 0; i < R; ++i)
    {
        P = su2::multiply(P, field(s, mu));
        s = lat.forward(s, mu);
    }
    // forward ν
    for (int j = 0; j < T; ++j)
    {
        P = su2::multiply(P, field(s, nu));
        s = lat.forward(s, nu);
    }
    // backward μ — traverse with dagger
    for (int i = 0; i < R; ++i)
    {
        s = lat.backward(s, mu);
        P = su2::multiply(P, su2::dagger(field(s, mu)));
    }
    // backward ν
    for (int j = 0; j < T; ++j)
    {
        s = lat.backward(s, nu);
        P = su2::multiply(P, su2::dagger(field(s, nu)));
    }
    return P.s; // ½ tr W
}

template<int Dim>
double averageWilsonLoop(
    const Lattice<Dim>& lat,
    const LinkField<su2::Element, Dim>& field,
    int mu, int nu, int R, int T)
{
    double sum = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s)
        sum += wilsonLoopAt(lat, field, s, mu, nu, R, T);
    return sum / static_cast<double>(V);
}

/// Creutz ratio
///     χ(R, T) = − log[ W(R, T) · W(R−1, T−1) / (W(R−1, T) · W(R, T−1)) ]
/// On a confining Wilson-area-law Ansatz this isolates the string tension:
/// `χ(R, T) → σ` at large `R, T`. Numerically the cleanest version is
/// `χ(R, R)` along the diagonal.
inline double creutzRatio(double W_RT, double W_R1T1, double W_R1T, double W_RT1)
{
    return -std::log( (W_RT * W_R1T1) / (W_R1T * W_RT1) );
}

} // namespace lqft::su2_model
