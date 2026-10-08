#pragma once

/**
 * @file u1.hpp
 * @brief Compact U(1) lattice gauge theory.
 *
 * Link variables θ_μ(x) ∈ ℝ (treated modulo 2π — cos/sin are periodic so
 * we don't bother wrapping the stored value) live on directed links.
 * The group element on link (x → x + ê_μ) is U_μ(x) = exp(i θ_μ(x)).
 *
 * Wilson plaquette action:
 *
 *     S[θ] = β Σ_□ (1 − cos θ_□)
 *
 * where each oriented plaquette (x; μ, ν) with μ < ν contributes once and
 *
 *     θ_□(x; μ, ν) = θ_μ(x) + θ_ν(x + ê_μ) − θ_μ(x + ê_ν) − θ_ν(x).
 *
 * ## Local Metropolis update
 *
 * Updating one link (y, μ) by θ_μ(y) → θ' affects the 2 · (Dim − 1)
 * plaquettes containing it (one "positive" plaquette (y; μ, ν) and one
 * "negative" plaquette (y − ê_ν; μ, ν) for every ν ≠ μ). The action
 * change collapses to
 *
 *     ΔS = β · Re[(e^{iθ_old} − e^{iθ_new}) · A]
 *
 * where the (complex) staple A = Σ_{ν ≠ μ} (e^{iφ_+(ν)} + e^{−iφ_−(ν)})
 * encodes the three other-link contributions in each of the 2(D − 1)
 * plaquettes. This is the same algebraic structure that gives the heat-bath
 * the von-Mises conditional we will eventually exploit.
 *
 * ## Color partition (for parallelism)
 *
 * Two link updates are independent iff they share no plaquette. For
 * fixed direction μ, links at sites of the same parity don't share a
 * plaquette — but links of *different* directions sharing a corner do.
 * So the safe coloring is (site_parity, μ) → 2 · Dim colors. We compute
 * it once at construction and the generic MetropolisSweep iterates colors
 * in order. The CPU sweep doesn't depend on this coloring being correct,
 * but it is the structure the GPU port will need.
 *
 * ## Closed-form 2D benchmark
 *
 * The 2D partition function factorizes over plaquettes; the one-plaquette
 * expectation is exactly
 *
 *     ⟨cos θ_□⟩ = I_1(β) / I_0(β)
 *
 * and Wilson loops obey ⟨W(R, T)⟩ = (I_1/I_0)^{RT}. This is the analog of
 * the Onsager / free-Gaussian closed forms in earlier phases.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../rng/rng.hpp"

#include <cmath>
#include <vector>

namespace lqft::u1
{

template<int Dim>
class U1Model
{
public:
    using FieldT   = LinkField<double, Dim>;
    using Proposal = double; // proposed new value of θ_μ(y)
    using LatticeT = Lattice<Dim>;

    static constexpr int dimension() { return Dim; }
    static_assert(Dim >= 2, "U(1) gauge theory needs at least 2 directions for plaquettes");

    U1Model(const LatticeT& lattice, double beta, double step_size = 1.0)
        : m_lattice(lattice), m_beta(beta), m_step_size(step_size)
    {
        buildColorPartition();
    }

    const LatticeT& lattice() const { return m_lattice; }
    double beta()     const { return m_beta; }
    double stepSize() const { return m_step_size; }
    void   setBeta(double b)     { m_beta = b; }
    void   setStepSize(double s) { m_step_size = s; }

    // -----------------------------------------------------------------------
    // Unit encoding: unit_id = site * Dim + μ.
    // Color: c = site_parity * Dim + μ  ∈ {0, …, 2·Dim − 1}.
    // -----------------------------------------------------------------------

    int numColors() const { return 2 * Dim; }
    const std::vector<int>& unitsOfColor(int c) const { return m_units_by_color[c]; }

    static constexpr int siteOfUnit(int unit) { return unit / Dim; }
    static constexpr int muOfUnit  (int unit) { return unit % Dim; }
    static constexpr int unitOf(int site, int mu) { return site * Dim + mu; }

    // -----------------------------------------------------------------------
    // Staple at (site, μ) — Σ_{ν ≠ μ} (e^{iφ_+(ν)} + e^{−iφ_−(ν)}).
    //   φ_+(ν) = θ_ν(y+μ) − θ_μ(y+ν) − θ_ν(y)
    //   φ_−(ν) = θ_μ(y−ν) + θ_ν(y−ν+μ) − θ_ν(y−ν)
    // -----------------------------------------------------------------------

    struct Staple { double re; double im; };

    Staple staple(const FieldT& links, int site, int mu) const
    {
        double Are = 0.0, Aim = 0.0;
        for (int nu = 0; nu < Dim; ++nu)
        {
            if (nu == mu) continue;

            // Positive plaquette (y; μ, ν): θ_□ = θ_μ(y) + φ_+(ν)
            const int y_plus_mu = m_lattice.forward(site, mu);
            const int y_plus_nu = m_lattice.forward(site, nu);
            const double phi_plus =
                links(y_plus_mu, nu) - links(y_plus_nu, mu) - links(site, nu);
            Are += std::cos(phi_plus);
            Aim += std::sin(phi_plus);

            // Negative plaquette (y − ν; μ, ν): θ_□ = −θ_μ(y) + φ_−(ν).
            // Contributes cos(θ_μ(y) − φ_−) ≡ Re[e^{iθ} · e^{−iφ_−}].
            const int y_minus_nu       = m_lattice.backward(site, nu);
            const int y_minus_nu_plus_mu = m_lattice.forward(y_minus_nu, mu);
            const double phi_minus =
                links(y_minus_nu, mu) + links(y_minus_nu_plus_mu, nu) - links(y_minus_nu, nu);
            Are += std::cos(phi_minus); // cos(-x) = cos(x)
            Aim -= std::sin(phi_minus); // sin(-x) = -sin(x)
        }
        return { Are, Aim };
    }

    // -----------------------------------------------------------------------
    // Model interface
    // -----------------------------------------------------------------------

    Proposal propose(const FieldT& field, int unit, Rng& rng) const
    {
        const int site = siteOfUnit(unit);
        const int mu   = muOfUnit(unit);
        return field(site, mu) + m_step_size * rng.normal();
    }

    double delta_action(const FieldT& field, int unit, Proposal theta_new) const
    {
        const int site = siteOfUnit(unit);
        const int mu   = muOfUnit(unit);
        const double theta_old = field(site, mu);
        const Staple A = staple(field, site, mu);

        // F(θ) = Re[e^{iθ} · A] = cos θ · A_re − sin θ · A_im
        // S_local = const − β · F(θ)  ⇒  ΔS = β · (F_old − F_new)
        const double F_old = std::cos(theta_old) * A.re - std::sin(theta_old) * A.im;
        const double F_new = std::cos(theta_new) * A.re - std::sin(theta_new) * A.im;
        return m_beta * (F_old - F_new);
    }

    void apply(FieldT& field, int unit, Proposal theta_new) const
    {
        field(siteOfUnit(unit), muOfUnit(unit)) = theta_new;
    }

    // -----------------------------------------------------------------------
    // Gauge force F_μ(x) = -∂S/∂θ_μ(x) for HMC.
    //   S_local = const − β · Re[e^{iθ} · A_μ(y)]
    //   ∂S/∂θ   =     β · (sin θ · A_re + cos θ · A_im)
    //   F       =    −β · (sin θ · A_re + cos θ · A_im).
    // -----------------------------------------------------------------------

    double gaugeForce(const FieldT& field, int site, int mu) const
    {
        const double theta = field(site, mu);
        const Staple A     = staple(field, site, mu);
        return -m_beta * (std::sin(theta) * A.re + std::cos(theta) * A.im);
    }

    /// Fill `force` with F_μ(x) for every link in the lattice.
    void computeAllGaugeForces(const FieldT& field, FieldT& force) const
    {
        const int V = m_lattice.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                force(s, mu) = gaugeForce(field, s, mu);
    }

    // -----------------------------------------------------------------------
    // Total action — sum β · (1 − cos θ_□) over every oriented plaquette
    // with μ < ν, each counted once.
    // -----------------------------------------------------------------------

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
                    const double theta_plaq =
                        field(s, mu) + field(s_mu, nu)
                        - field(s_nu, mu) - field(s, nu);
                    S += m_beta * (1.0 - std::cos(theta_plaq));
                }
        return S;
    }

    // -----------------------------------------------------------------------
    // Field initializers.
    // -----------------------------------------------------------------------

    /// Cold: all θ = 0 ⇒ all plaquettes 1 ⇒ S = 0.
    static void cold(FieldT& field) {
        // LinkField has no fill() helper — write the loop directly.
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = 0.0;
    }

    /// Hot: every link θ uniform on [−π, π).
    static void hot(FieldT& field, Rng& rng)
    {
        constexpr double kTwoPi = 6.28318530717958647692;
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            for (int mu = 0; mu < Dim; ++mu)
                field(s, mu) = kTwoPi * (rng.uniform() - 0.5);
    }

private:
    void buildColorPartition()
    {
        m_units_by_color.assign(static_cast<std::size_t>(2 * Dim), {});
        const int V = m_lattice.volume();
        // Reserve roughly to avoid rehashing.
        for (auto& v : m_units_by_color) v.reserve(static_cast<std::size_t>(V / 2 + 1));
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
// Wilson-loop and plaquette observables.
// ----------------------------------------------------------------------------

/// Re U_□ at one corner: cos of the oriented plaquette phase.
template<int Dim>
double plaquetteAt(
    const Lattice<Dim>& lat,
    const LinkField<double, Dim>& links,
    int corner, int mu, int nu)
{
    const int c_mu = lat.forward(corner, mu);
    const int c_nu = lat.forward(corner, nu);
    const double theta_plaq =
        links(corner, mu) + links(c_mu, nu)
        - links(c_nu, mu) - links(corner, nu);
    return std::cos(theta_plaq);
}

/// Average of cos θ_□ over every corner and every (μ < ν) plane. This is
/// the dimension-independent generalization of ⟨W(1, 1)⟩.
template<int Dim>
double averagePlaquette(
    const Lattice<Dim>& lat,
    const LinkField<double, Dim>& links)
{
    double sum   = 0.0;
    long long n  = 0;
    const int V  = lat.volume();
    for (int s = 0; s < V; ++s)
        for (int mu = 0; mu < Dim; ++mu)
            for (int nu = mu + 1; nu < Dim; ++nu)
            {
                sum += plaquetteAt(lat, links, s, mu, nu);
                ++n;
            }
    return sum / static_cast<double>(n);
}

/// Re W(R, T) at one corner, sides R · ê_μ and T · ê_ν.
template<int Dim>
double wilsonLoopAt(
    const Lattice<Dim>& lat,
    const LinkField<double, Dim>& links,
    int corner, int mu, int nu, int R, int T)
{
    double phase = 0.0;
    int s = corner;
    for (int i = 0; i < R; ++i) { phase += links(s, mu); s = lat.forward(s, mu); }
    for (int j = 0; j < T; ++j) { phase += links(s, nu); s = lat.forward(s, nu); }
    for (int i = 0; i < R; ++i) { s = lat.backward(s, mu); phase -= links(s, mu); }
    for (int j = 0; j < T; ++j) { s = lat.backward(s, nu); phase -= links(s, nu); }
    return std::cos(phase);
}

/// Average Re W(R, T) over all corners at fixed (μ, ν, R, T).
template<int Dim>
double averageWilsonLoop(
    const Lattice<Dim>& lat,
    const LinkField<double, Dim>& links,
    int mu, int nu, int R, int T)
{
    double sum = 0.0;
    const int V = lat.volume();
    for (int s = 0; s < V; ++s)
        sum += wilsonLoopAt(lat, links, s, mu, nu, R, T);
    return sum / static_cast<double>(V);
}

// ----------------------------------------------------------------------------
// Modified Bessel functions I_0(β) and I_1(β) by power series.
// Used to evaluate the closed-form 2D U(1) plaquette / Wilson-loop benchmark
//     ⟨cos θ_□⟩ = I_1(β) / I_0(β)
// Stable for β up to ~30 in IEEE-double; we only ever test small β.
// ----------------------------------------------------------------------------

inline double besselI0(double x)
{
    const double y = x * x / 4.0;
    double term = 1.0, sum = 1.0;
    for (int k = 1; k < 200; ++k)
    {
        term *= y / static_cast<double>(k * k);
        sum  += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

inline double besselI1(double x)
{
    const double y = x * x / 4.0;
    double term = x / 2.0, sum = term;
    for (int k = 1; k < 200; ++k)
    {
        term *= y / static_cast<double>(k * (k + 1));
        sum  += term;
        if (term < 1e-16 * sum) break;
    }
    return sum;
}

inline double besselRatio(double x) { return besselI1(x) / besselI0(x); }

} // namespace lqft::u1
