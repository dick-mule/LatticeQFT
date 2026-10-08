#pragma once

/**
 * @file rng.hpp
 * @brief CPU-side RNG abstraction.
 *
 * Phase-0 uses `std::mt19937_64` for simplicity. The MC engine takes an
 * `Rng&` and never names the underlying engine, so when we lift to GPU we
 * swap to a counter-based generator (Philox / Threefry) seeded by
 * (site_index, mc_step, key) without touching the algorithms.
 *
 * The interface intentionally exposes only what Monte Carlo updates need
 * (`uniform()`, `randint()`, `normal()`) — keeps the contract small enough
 * that a GPU implementation can match it.
 */

#include <cstdint>
#include <random>

namespace lqft
{

class Rng
{
public:
    explicit Rng(std::uint64_t seed) : m_engine(seed) {}

    /// Uniform on [0, 1).
    double uniform()
    {
        return std::generate_canonical<double, 53>(m_engine);
    }

    /// Uniform integer in [0, n).
    int randint(int n)
    {
        return std::uniform_int_distribution<int>(0, n - 1)(m_engine);
    }

    /// Standard normal N(0, 1).
    double normal()
    {
        return std::normal_distribution<double>(0.0, 1.0)(m_engine);
    }

    std::uint64_t raw() { return m_engine(); }

private:
    std::mt19937_64 m_engine;
};

} // namespace lqft
