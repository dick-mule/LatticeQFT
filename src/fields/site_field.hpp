#pragma once

/**
 * @file site_field.hpp
 * @brief Container for one degree of freedom per lattice site.
 *
 * Used for the Ising spin field (T = int8_t, values ±1) and the φ⁴ scalar
 * field (T = double). Link fields (one DOF per directed link) live in
 * link_field.hpp.
 */

#include "../lattice/lattice.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

namespace lqft
{

template<typename T, int Dim>
class SiteField
{
public:
    using value_type = T;
    static constexpr int dimension() { return Dim; }

    explicit SiteField(const Lattice<Dim>& lattice, T fill = T{})
        : m_data(static_cast<std::size_t>(lattice.volume()), fill)
        , m_volume(lattice.volume())
    {}

    int volume() const { return m_volume; }

    T& operator[](int site)
    {
        assert(site >= 0 && site < m_volume);
        return m_data[static_cast<std::size_t>(site)];
    }
    const T& operator[](int site) const
    {
        assert(site >= 0 && site < m_volume);
        return m_data[static_cast<std::size_t>(site)];
    }

    T*       data()       { return m_data.data(); }
    const T* data() const { return m_data.data(); }

    void fill(T v) { std::fill(m_data.begin(), m_data.end(), v); }

private:
    std::vector<T> m_data;
    int            m_volume;
};

} // namespace lqft
