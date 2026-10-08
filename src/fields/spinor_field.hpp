#pragma once

/**
 * @file spinor_field.hpp
 * @brief Container for an N-component complex Dirac spinor at every site.
 *
 * SpinorField<Dim, Nc> stores `Nc` complex values per lattice site,
 * laid out contiguously per-site so `data() + Nc * site` is the spinor
 * at `site`. For 2D Wilson-Dirac fermions Nc = 2; for 4D Wilson-Dirac
 * Nc = 4. Color indices (when we add SU(N)) will multiply Nc → Nc·Ncolor.
 *
 * The header also exposes the small set of linear-algebra primitives
 * the conjugate-gradient solver needs:
 *   - axpy, scale, copy
 *   - real Hermitian inner product   ⟨a, b⟩ = Σ_i ā_i b_i
 *   - squared norm                   ‖a‖² = Σ_i |a_i|²
 *   - zero
 * These operate uniformly over (site × component) pairs and don't depend
 * on the lattice topology.
 */

#include "../lattice/lattice.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace lqft
{

template<int Dim, int Nc>
class SpinorField
{
public:
    using Complex = std::complex<double>;

    static constexpr int components() { return Nc; }
    static constexpr int dimension()  { return Dim; }

    explicit SpinorField(const Lattice<Dim>& lattice)
        : m_data(static_cast<std::size_t>(lattice.volume()) * Nc, Complex{0.0, 0.0})
        , m_volume(lattice.volume())
    {}

    int volume()    const { return m_volume; }
    int totalSize() const { return m_volume * Nc; }

    Complex& operator()(int site, int comp)
    {
        assert(site >= 0 && site < m_volume);
        assert(comp >= 0 && comp < Nc);
        return m_data[static_cast<std::size_t>(site) * Nc + comp];
    }
    const Complex& operator()(int site, int comp) const
    {
        assert(site >= 0 && site < m_volume);
        assert(comp >= 0 && comp < Nc);
        return m_data[static_cast<std::size_t>(site) * Nc + comp];
    }

    Complex*       data()       { return m_data.data(); }
    const Complex* data() const { return m_data.data(); }

    void zero() { std::fill(m_data.begin(), m_data.end(), Complex{0.0, 0.0}); }

private:
    std::vector<Complex> m_data;
    int                  m_volume;
};

// ----------------------------------------------------------------------------
// Spinor algebra primitives (free functions for composability).
// ----------------------------------------------------------------------------

/// y ← α x + y
template<int Dim, int Nc>
void axpy(std::complex<double> alpha,
          const SpinorField<Dim, Nc>& x,
          SpinorField<Dim, Nc>& y)
{
    const int N = x.totalSize();
    auto*       py = y.data();
    const auto* px = x.data();
    for (int i = 0; i < N; ++i) py[i] += alpha * px[i];
}

/// y ← x + β y           (used by CG's p ← r + β p)
template<int Dim, int Nc>
void scale_add(std::complex<double> beta,
               const SpinorField<Dim, Nc>& x,
               SpinorField<Dim, Nc>& y)
{
    const int N = x.totalSize();
    auto*       py = y.data();
    const auto* px = x.data();
    for (int i = 0; i < N; ++i) py[i] = px[i] + beta * py[i];
}

/// y ← x
template<int Dim, int Nc>
void copy(const SpinorField<Dim, Nc>& x, SpinorField<Dim, Nc>& y)
{
    const int N = x.totalSize();
    std::copy(x.data(), x.data() + N, y.data());
}

/// y ← α x
template<int Dim, int Nc>
void scale(std::complex<double> alpha,
           const SpinorField<Dim, Nc>& x,
           SpinorField<Dim, Nc>& y)
{
    const int N = x.totalSize();
    auto*       py = y.data();
    const auto* px = x.data();
    for (int i = 0; i < N; ++i) py[i] = alpha * px[i];
}

/// Hermitian inner product ⟨a, b⟩ = Σ_i ā_i b_i.  Complex-valued in general.
template<int Dim, int Nc>
std::complex<double> inner_product(const SpinorField<Dim, Nc>& a,
                                   const SpinorField<Dim, Nc>& b)
{
    const int N = a.totalSize();
    const auto* pa = a.data();
    const auto* pb = b.data();
    std::complex<double> s{0.0, 0.0};
    for (int i = 0; i < N; ++i) s += std::conj(pa[i]) * pb[i];
    return s;
}

/// Squared L² norm ‖a‖² = Σ_i |a_i|².  Real, non-negative.
template<int Dim, int Nc>
double norm_squared(const SpinorField<Dim, Nc>& a)
{
    const int N = a.totalSize();
    const auto* pa = a.data();
    double s = 0.0;
    for (int i = 0; i < N; ++i) s += std::norm(pa[i]);
    return s;
}

} // namespace lqft
