#pragma once

/**
 * @file lattice.hpp
 * @brief Dimension-templated periodic lattice topology.
 *
 * The Lattice<Dim> class owns the integer arithmetic of stepping around an
 * L_0 × L_1 × … × L_{Dim-1} periodic box: site <-> coordinate conversion,
 * nearest-neighbor traversal, and parity (sum-of-coords mod 2) bookkeeping.
 *
 * Algorithms never touch the linear layout directly — they go through
 * Lattice<Dim>::neighbor() and Lattice<Dim>::sitesOfParity(). That makes
 * the layout an implementation detail we can change later (e.g. swap to a
 * blocked layout for SIMD or split for GPU) without rewriting any model.
 *
 * The class is intentionally header-only and templated — all hot loops want
 * Dim as a compile-time constant.
 */

#include <array>
#include <cassert>
#include <cstddef>
#include <vector>

namespace lqft
{

template<int Dim>
class Lattice
{
    static_assert(Dim >= 1 && Dim <= 4, "Phase plan supports Dim in [1, 4]");

public:
    using Coord = std::array<int, Dim>;

    /// Construct an L_0 × L_1 × … box with periodic boundaries.
    explicit Lattice(const Coord& shape)
    {
        m_shape = shape;
        m_strides[0] = 1;
        for (int d = 1; d < Dim; ++d)
            m_strides[d] = m_strides[d - 1] * shape[d - 1];
        m_volume = m_strides[Dim - 1] * shape[Dim - 1];

        // Precompute parity partition. Used by every MC sweep; cheap to keep.
        m_even_sites.reserve(static_cast<std::size_t>(m_volume) / 2 + 1);
        m_odd_sites.reserve(static_cast<std::size_t>(m_volume) / 2 + 1);
        for (int s = 0; s < m_volume; ++s)
        {
            if (parityOf(s) == 0) m_even_sites.push_back(s);
            else                  m_odd_sites.push_back(s);
        }
    }

    /// Convenience: cubic lattice of side L in every direction.
    static Lattice cube(int L)
    {
        Coord s{};
        for (int d = 0; d < Dim; ++d) s[d] = L;
        return Lattice(s);
    }

    static constexpr int dimension() { return Dim; }
    int volume() const { return m_volume; }
    int extent(int d) const { assert(d >= 0 && d < Dim); return m_shape[d]; }
    const Coord& shape() const { return m_shape; }

    // -----------------------------------------------------------------------
    // Site <-> coordinate
    // -----------------------------------------------------------------------

    int siteIndex(const Coord& x) const
    {
        int idx = 0;
        for (int d = 0; d < Dim; ++d) idx += x[d] * m_strides[d];
        return idx;
    }

    Coord coords(int site) const
    {
        assert(site >= 0 && site < m_volume);
        Coord x{};
        int r = site;
        for (int d = 0; d < Dim; ++d)
        {
            x[d] = r % m_shape[d];
            r   /= m_shape[d];
        }
        return x;
    }

    // -----------------------------------------------------------------------
    // Neighbor in direction mu by signed step (+1 or -1), periodic.
    // Implemented via stride arithmetic — O(1), no full decompose/recompose.
    // -----------------------------------------------------------------------

    int neighbor(int site, int mu, int sign) const
    {
        assert(mu >= 0 && mu < Dim);
        assert(sign == +1 || sign == -1);
        const int Lmu     = m_shape[mu];
        const int stride  = m_strides[mu];
        const int coord_mu = (site / stride) % Lmu;
        const int new_mu  = (coord_mu + sign + Lmu) % Lmu;
        return site + (new_mu - coord_mu) * stride;
    }

    /// Forward neighbor in direction mu (site + e_mu). Convenience wrapper.
    int forward(int site, int mu) const { return neighbor(site, mu, +1); }
    /// Backward neighbor in direction mu (site - e_mu).
    int backward(int site, int mu) const { return neighbor(site, mu, -1); }

    // -----------------------------------------------------------------------
    // Parity (checkerboard)
    // -----------------------------------------------------------------------

    int parityOf(int site) const
    {
        const Coord x = coords(site);
        int p = 0;
        for (int d = 0; d < Dim; ++d) p += x[d];
        return p & 1;
    }

    const std::vector<int>& sitesOfParity(int p) const
    {
        assert(p == 0 || p == 1);
        return p == 0 ? m_even_sites : m_odd_sites;
    }

private:
    Coord m_shape{};
    Coord m_strides{};
    int   m_volume = 0;
    std::vector<int> m_even_sites;
    std::vector<int> m_odd_sites;
};

} // namespace lqft
