#pragma once

/**
 * @file link_field.hpp
 * @brief Container for Dim degrees of freedom per lattice site (one per
 *        directed link emerging in the forward direction).
 *
 * On a periodic lattice, every directed link can be uniquely labeled by the
 * site at its tail and the forward direction μ ∈ {0, …, Dim − 1}. We store
 * Dim values per site contiguously — `link(site, mu)`.
 *
 * Phase-0 doesn't use this yet; it lives here so Phase-2 (compact U(1))
 * can drop link variables in without restructuring containers.
 */

#include "../lattice/lattice.hpp"
#include <cassert>
#include <cstddef>
#include <vector>

namespace lqft
{

template<typename T, int Dim>
class LinkField
{
public:
    using value_type = T;
    static constexpr int dimension() { return Dim; }

    explicit LinkField(const Lattice<Dim>& lattice, T fill = T{})
        : m_data(static_cast<std::size_t>(lattice.volume()) * Dim, fill)
        , m_volume(lattice.volume())
    {}

    int volume() const { return m_volume; }

    /// Link at (site, μ).
    T& operator()(int site, int mu)
    {
        assert(site >= 0 && site < m_volume);
        assert(mu >= 0 && mu < Dim);
        return m_data[static_cast<std::size_t>(site) * Dim + mu];
    }
    const T& operator()(int site, int mu) const
    {
        assert(site >= 0 && site < m_volume);
        assert(mu >= 0 && mu < Dim);
        return m_data[static_cast<std::size_t>(site) * Dim + mu];
    }

    T*       data()       { return m_data.data(); }
    const T* data() const { return m_data.data(); }

private:
    std::vector<T> m_data;
    int            m_volume;
};

} // namespace lqft
