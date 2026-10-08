#pragma once

/**
 * @file su3.hpp
 * @brief SU(3) group elements stored as 3×3 complex matrices.
 *
 * Unlike SU(2) (which fits as a unit quaternion on S³), SU(3) is 8-parameter
 * and admits no compact real representation. We just carry the 3×3 complex
 * matrix entries directly. The cost per multiply is 27 complex mults vs 4 for
 * SU(2) — the difference dwarfs everything else in real lattice runs.
 *
 * Group structure (for reference):
 *
 *     U ∈ SU(3) ⇔ U U† = I, det U = 1.
 *     dim(SU(3)) = 8.
 *     Generators: 8 Gell-Mann matrices λ_a, with tr(λ_a λ_b) = 2 δ_ab.
 *     Cabibbo-Marinari: SU(3) is covered by 3 SU(2) subgroups embedded in
 *     the (0,1), (0,2), (1,2) row/column pairs. Picking one at random and
 *     doing a near-identity SU(2) rotation in it gives an ergodic update
 *     with no need for full Gell-Mann arithmetic.
 *
 * Sums of SU(3)'s (gauge staples) are general 3×3 complex matrices; the
 * `Element` struct doubles as the staple type — the same way su2::Element
 * doubles for SU(2).
 */

#include <array>
#include <cmath>
#include <complex>

namespace lqft::su3
{

using Complex = std::complex<double>;

/// SU(3) element when U U† = I and det U = 1 (or a general 3×3 complex
/// matrix when used as a staple sum).
struct Element
{
    std::array<std::array<Complex, 3>, 3> m;

    static Element identity()
    {
        Element E{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                E.m[i][j] = (i == j) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
        return E;
    }
    static Element zero()
    {
        Element E{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                E.m[i][j] = Complex(0.0, 0.0);
        return E;
    }
};

/// (A B)_{ij} = Σ_k A_{ik} B_{kj}.
inline Element multiply(const Element& A, const Element& B)
{
    Element C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < 3; ++k)
                sum += A.m[i][k] * B.m[k][j];
            C.m[i][j] = sum;
        }
    return C;
}

/// Hermitian conjugate: (A†)_{ij} = conj(A_{ji}).
inline Element dagger(const Element& A)
{
    Element D{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D.m[i][j] = std::conj(A.m[j][i]);
    return D;
}

inline Element add(const Element& A, const Element& B)
{
    Element C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C.m[i][j] = A.m[i][j] + B.m[i][j];
    return C;
}

inline Element scale(double k, const Element& A)
{
    Element C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C.m[i][j] = k * A.m[i][j];
    return C;
}

inline Complex trace(const Element& A)
{
    return A.m[0][0] + A.m[1][1] + A.m[2][2];
}

/// Re tr A.
inline double real_trace(const Element& A) { return trace(A).real(); }

/// Re tr(A B) = Σ_{ik} Re[A_{ik} B_{ki}].
inline double real_trace_product(const Element& A, const Element& B)
{
    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            sum += (A.m[i][k] * B.m[k][i]).real();
    return sum;
}

/// Frobenius norm squared: ‖A‖² = tr(A† A) = Σ |A_{ij}|².
/// For U ∈ SU(3) this equals 3 (= N_c).
inline double norm_squared(const Element& A)
{
    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            sum += std::norm(A.m[i][j]);
    return sum;
}

inline double norm(const Element& A) { return std::sqrt(norm_squared(A)); }

/// det A — 3×3 expanded directly (we only call this for re-projection or in
/// tests, never on the inner loop).
inline Complex determinant(const Element& A)
{
    return A.m[0][0] * (A.m[1][1] * A.m[2][2] - A.m[1][2] * A.m[2][1])
         - A.m[0][1] * (A.m[1][0] * A.m[2][2] - A.m[1][2] * A.m[2][0])
         + A.m[0][2] * (A.m[1][0] * A.m[2][1] - A.m[1][1] * A.m[2][0]);
}

/// Re-project a matrix back onto SU(3): Gram-Schmidt orthonormalize the first
/// two columns, then form the third as (col1 × col2)* so that det = 1
/// exactly. Cleans up round-off drift after many multiplications. Not the
/// "nearest SU(3) to A" in the Frobenius sense — that would require polar
/// decomposition — but it's exact for U ∈ SU(3) and cheap. For APE-style
/// projection of staple sums in SU(3), the iterative Cabibbo-Marinari
/// procedure would be preferred; we add that when we need it.
inline Element project_to_su3(const Element& A)
{
    Element U = A;

    // Column 0: normalize.
    {
        double n2 = 0.0;
        for (int i = 0; i < 3; ++i) n2 += std::norm(U.m[i][0]);
        if (n2 < 1e-30) return Element::identity();
        const double inv = 1.0 / std::sqrt(n2);
        for (int i = 0; i < 3; ++i) U.m[i][0] *= inv;
    }

    // Column 1: subtract projection onto column 0, then normalize.
    {
        Complex dot(0.0, 0.0);
        for (int i = 0; i < 3; ++i) dot += std::conj(U.m[i][0]) * U.m[i][1];
        for (int i = 0; i < 3; ++i) U.m[i][1] -= dot * U.m[i][0];
        double n2 = 0.0;
        for (int i = 0; i < 3; ++i) n2 += std::norm(U.m[i][1]);
        if (n2 < 1e-30) return Element::identity();
        const double inv = 1.0 / std::sqrt(n2);
        for (int i = 0; i < 3; ++i) U.m[i][1] *= inv;
    }

    // Column 2 = conj(col0 × col1): ensures unitarity and det(U) = +1.
    {
        const Complex c0 = std::conj(U.m[1][0] * U.m[2][1] - U.m[2][0] * U.m[1][1]);
        const Complex c1 = std::conj(U.m[2][0] * U.m[0][1] - U.m[0][0] * U.m[2][1]);
        const Complex c2 = std::conj(U.m[0][0] * U.m[1][1] - U.m[1][0] * U.m[0][1]);
        U.m[0][2] = c0;
        U.m[1][2] = c1;
        U.m[2][2] = c2;
    }
    return U;
}

/// Embed an SU(2) element (parameterized by a 4-tuple (s, v₀, v₁, v₂) with
/// s² + |v|² = 1) into SU(3) at the (i, j) row/column pair. The remaining
/// row/column stays the identity. Used by the Cabibbo-Marinari Metropolis
/// proposal and (later) heat-bath update.
///
///                       ┌                    ┐
///                       │ s + i v₂   v₁ + iv₀│
///     (i, j) sub-block: │−v₁ + iv₀   s − iv₂ │
///                       └                    ┘
inline Element embedSU2(int i, int j, double s,
                       double v0, double v1, double v2)
{
    Element E = Element::identity();
    E.m[i][i] = Complex(s,  v2);
    E.m[i][j] = Complex(v1,  v0);
    E.m[j][i] = Complex(-v1, v0);
    E.m[j][j] = Complex(s, -v2);
    return E;
}

} // namespace lqft::su3
