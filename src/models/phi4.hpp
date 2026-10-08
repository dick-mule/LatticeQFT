#pragma once

/**
 * @file phi4.hpp
 * @brief Lattice φ⁴ scalar field theory on a Dim-dimensional periodic lattice.
 *
 * A real scalar φ(x) ∈ ℝ lives on each site. Continuum action:
 *
 *     S[φ] = ∫ dᴰx [ ½ (∂φ)² + ½ m² φ² + (λ/4!) φ⁴ ]
 *
 * Lattice discretization with the forward difference ∂_μ φ(x) = φ(x + ê_μ) − φ(x)
 * and lattice spacing a = 1:
 *
 *     S[φ] = Σ_x [ ½ Σ_μ (φ_{x+ê_μ} − φ_x)²
 *                + ½ m² φ_x²
 *                + (λ/24) φ_x⁴ ]
 *
 * Single-site Metropolis update φ_x → φ_x + δ:
 *
 *     ΔS = -δ · S_nn(x) + D · (φ'² − φ²)
 *                       + ½ m² (φ'² − φ²)
 *                       + (λ/24)(φ'⁴ − φ⁴)
 *
 * where S_nn(x) = Σ_μ (φ_{x+ê_μ} + φ_{x-ê_μ}) and D = Dim.
 *
 * Proposal distribution: φ_x' = φ_x + σ · N(0, 1) with σ tunable to target
 * a chosen acceptance rate (see mc::autoTuneStepSize).
 *
 * For λ = 0 (free Gaussian field) the model is exactly solvable:
 *     ⟨|φ̃_p|²⟩ = 1 / (4 Σ_μ sin²(p_μ/2) + m²)
 * which gives ⟨φ_x²⟩ = (1/V) Σ_p G(p). Used as the primary validation test.
 *
 * For λ > 0 and m² sufficiently negative the Z₂ φ → −φ symmetry breaks
 * spontaneously; ⟨|φ|⟩ becomes a useful order parameter (analogous to the
 * Ising ⟨|m|⟩).
 */

#include "../fields/site_field.hpp"
#include "../lattice/lattice.hpp"
#include "../rng/rng.hpp"

namespace lqft::phi4
{

template<int Dim>
class Phi4Model
{
public:
    using FieldT   = SiteField<double, Dim>;
    using Proposal = double; // proposed new value of φ_x
    using LatticeT = Lattice<Dim>;

    static constexpr int dimension() { return Dim; }

    Phi4Model(const LatticeT& lattice,
              double m_sq,
              double lambda,
              double step_size = 1.0)
        : m_lattice(lattice)
        , m_m_sq(m_sq)
        , m_lambda(lambda)
        , m_step_size(step_size)
    {}

    const LatticeT& lattice() const { return m_lattice; }
    double mSquared()   const { return m_m_sq; }
    double lambda()     const { return m_lambda; }
    double stepSize()   const { return m_step_size; }
    void   setStepSize(double s) { m_step_size = s; }
    void   setMSquared(double m_sq) { m_m_sq = m_sq; }
    void   setLambda(double lam)    { m_lambda = lam; }

    // -----------------------------------------------------------------------
    // Color partition: 2 colors (even / odd parity).
    // -----------------------------------------------------------------------

    int numColors() const { return 2; }
    const std::vector<int>& unitsOfColor(int c) const
    {
        return m_lattice.sitesOfParity(c);
    }

    // -----------------------------------------------------------------------
    // Model interface — symmetric with IsingModel.
    // -----------------------------------------------------------------------

    double neighborSum(const FieldT& field, int site) const
    {
        double s = 0.0;
        for (int mu = 0; mu < Dim; ++mu)
        {
            s += field[m_lattice.forward(site, mu)];
            s += field[m_lattice.backward(site, mu)];
        }
        return s;
    }

    Proposal propose(const FieldT& field, int site, Rng& rng) const
    {
        return field[site] + m_step_size * rng.normal();
    }

    double delta_action(const FieldT& field, int site, Proposal phi_new) const
    {
        const double phi_old = field[site];
        const double delta   = phi_new - phi_old;
        const double Snn     = neighborSum(field, site);

        const double phi2_new = phi_new * phi_new;
        const double phi2_old = phi_old * phi_old;
        const double phi4_new = phi2_new * phi2_new;
        const double phi4_old = phi2_old * phi2_old;

        const double d_phi2 = phi2_new - phi2_old;
        const double d_phi4 = phi4_new - phi4_old;

        return -delta * Snn
             + static_cast<double>(Dim) * d_phi2
             + 0.5 * m_m_sq * d_phi2
             + (m_lambda / 24.0) * d_phi4;
    }

    void apply(FieldT& field, int site, Proposal phi_new) const
    {
        field[site] = phi_new;
    }

    // -----------------------------------------------------------------------
    // Total action (for tests and observables that need ⟨S⟩).
    // -----------------------------------------------------------------------

    double totalAction(const FieldT& field) const
    {
        double S = 0.0;
        const int V = m_lattice.volume();
        for (int s = 0; s < V; ++s)
        {
            const double phi = field[s];
            // Forward links from s only — each link counted once.
            for (int mu = 0; mu < Dim; ++mu)
            {
                const double d = field[m_lattice.forward(s, mu)] - phi;
                S += 0.5 * d * d;
            }
            const double phi2 = phi * phi;
            S += 0.5 * m_m_sq * phi2;
            S += (m_lambda / 24.0) * phi2 * phi2;
        }
        return S;
    }

    // -----------------------------------------------------------------------
    // Field initializers.
    // -----------------------------------------------------------------------

    /// Cold start: φ ≡ 0 (action minimum at m² > 0, λ ≥ 0).
    static void cold(FieldT& field) { field.fill(0.0); }

    /// Hot start: every φ_x independent N(0, scale²).
    static void hot(FieldT& field, Rng& rng, double scale = 1.0)
    {
        const int V = field.volume();
        for (int s = 0; s < V; ++s)
            field[s] = scale * rng.normal();
    }

private:
    const LatticeT& m_lattice;
    double          m_m_sq;
    double          m_lambda;
    double          m_step_size;
};

// ----------------------------------------------------------------------------
// Free-field analytic ⟨φ²⟩ on an L^Dim periodic lattice at λ = 0.
// Used by tests and as a closed-form benchmark for any new MC infrastructure.
//
//     ⟨φ²⟩_free = (1/V) Σ_p  1 / (4 Σ_μ sin²(p_μ/2) + m²)
//
// with p_μ ∈ {2π k_μ / L : k_μ = 0, 1, …, L-1}. The zero mode (all k_μ = 0)
// contributes 1/m², so this only converges for m² > 0.
// ----------------------------------------------------------------------------

template<int Dim>
double freeFieldPhi2(const Lattice<Dim>& lattice, double m_sq)
{
    using std::sin;
    constexpr double kPi = 3.14159265358979323846;
    const auto& shape = lattice.shape();
    const int   V     = lattice.volume();

    double sum = 0.0;
    std::array<int, Dim> k{};
    for (int s = 0; s < V; ++s)
    {
        // Decompose s into multi-index over k_0 .. k_{Dim-1}.
        int r = s;
        for (int d = 0; d < Dim; ++d) { k[d] = r % shape[d]; r /= shape[d]; }

        double kinetic = 0.0;
        for (int d = 0; d < Dim; ++d)
        {
            const double half_p = kPi * static_cast<double>(k[d])
                                / static_cast<double>(shape[d]);
            const double sn = sin(half_p);
            kinetic += 4.0 * sn * sn;
        }
        sum += 1.0 / (kinetic + m_sq);
    }
    return sum / static_cast<double>(V);
}

} // namespace lqft::phi4
