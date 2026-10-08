#pragma once

/**
 * @file su2.hpp
 * @brief SU(2) group elements stored as unit quaternions.
 *
 * A general SU(2) matrix has the form
 *
 *     U = s · I + i · (v_0 σ_1 + v_1 σ_2 + v_2 σ_3),   s² + |v|² = 1,
 *
 * with σ the Pauli matrices. We carry the 4-tuple (s, v_0, v_1, v_2) ∈ S³
 * in a small POD struct; the matrix form is reconstructed only when needed.
 *
 * Useful identities (all derivable from σᵢσⱼ = δᵢⱼ I + i εᵢⱼₖ σₖ):
 *
 *     (s₁I + i v₁·σ)(s₂I + i v₂·σ)
 *         = (s₁s₂ − v₁·v₂) I + i (s₁v₂ + s₂v₁ − v₁×v₂) · σ.
 *
 *     U† = s · I − i v·σ   ↔   (s, −v).
 *     tr U = 2s.
 *     Re tr(U V) = 2 (s_U s_V − v_U · v_V).
 *
 * Sums of SU(2)'s (e.g. the gauge "staple") are not in SU(2) — `s² + |v|²`
 * is no longer 1 — but they live in the same 4-vector representation;
 * `Element` therefore doubles as the staple type. Multiplication is still
 * defined by the formula above and behaves correctly for general 2×2
 * complex matrices written in this basis.
 */

#include <array>
#include <cmath>

namespace lqft::su2
{

/// SU(2) element (when |s² + |v|² − 1| ≈ 0) or general 4-vector in the
/// {I, iσ_x, iσ_y, iσ_z} basis (when not).
struct Element
{
    double s = 1.0;
    std::array<double, 3> v = { 0.0, 0.0, 0.0 };

    static constexpr Element identity() { return { 1.0, { 0.0, 0.0, 0.0 } }; }
    static constexpr Element zero()     { return { 0.0, { 0.0, 0.0, 0.0 } }; }
};

/// (s₁I + i v₁·σ)(s₂I + i v₂·σ) = (s₁s₂ − v₁·v₂) I + i(s₁v₂ + s₂v₁ − v₁×v₂)·σ.
inline Element multiply(const Element& A, const Element& B)
{
    const double a0 = A.s, a1 = A.v[0], a2 = A.v[1], a3 = A.v[2];
    const double b0 = B.s, b1 = B.v[0], b2 = B.v[1], b3 = B.v[2];
    return Element{
        a0 * b0 - (a1 * b1 + a2 * b2 + a3 * b3),
        {
            a0 * b1 + b0 * a1 - (a2 * b3 - a3 * b2),
            a0 * b2 + b0 * a2 - (a3 * b1 - a1 * b3),
            a0 * b3 + b0 * a3 - (a1 * b2 - a2 * b1),
        }
    };
}

/// Hermitian conjugate. For SU(2) this coincides with the matrix inverse.
inline Element dagger(const Element& A)
{
    return Element{ A.s, { -A.v[0], -A.v[1], -A.v[2] } };
}

inline Element add(const Element& A, const Element& B)
{
    return Element{ A.s + B.s,
        { A.v[0] + B.v[0], A.v[1] + B.v[1], A.v[2] + B.v[2] } };
}

inline Element scale(double k, const Element& A)
{
    return Element{ k * A.s, { k * A.v[0], k * A.v[1], k * A.v[2] } };
}

/// tr U = 2s.
inline double real_trace(const Element& A) { return 2.0 * A.s; }

/// Re tr(A B) = 2 (a₀ b₀ − a·b).
inline double real_trace_product(const Element& A, const Element& B)
{
    return 2.0 * ( A.s * B.s
                 - A.v[0] * B.v[0]
                 - A.v[1] * B.v[1]
                 - A.v[2] * B.v[2] );
}

/// |A|² = s² + |v|². Equal to 1 for SU(2), generally not for staples.
inline double norm_squared(const Element& A)
{
    return A.s * A.s
         + A.v[0] * A.v[0]
         + A.v[1] * A.v[1]
         + A.v[2] * A.v[2];
}

inline double norm(const Element& A) { return std::sqrt(norm_squared(A)); }

/// Renormalize a near-SU(2) element back to S³. Used to eliminate round-off
/// drift after many multiplications; not needed every update but cheap.
inline Element project_to_su2(const Element& A)
{
    const double n2 = norm_squared(A);
    if (n2 < 1e-30) return Element::identity();
    const double inv = 1.0 / std::sqrt(n2);
    return scale(inv, A);
}

} // namespace lqft::su2
