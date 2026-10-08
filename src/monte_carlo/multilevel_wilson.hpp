#pragma once

/**
 * @file multilevel_wilson.hpp
 * @brief Lüscher-Weisz multilevel sampling for the 3D SU(2) Wilson loop.
 *
 * Reference: M. Lüscher & P. Weisz, Phys. Lett. B 519 (2001) 168.
 *
 * ## What and why
 *
 * The straightforward estimator
 *     Ŵ_std = (1/N) Σ_{config} W(R, T)[config]
 * has variance O(1)/N while the signal ⟨W⟩ ~ exp(−σRT − μ(R+T)) is
 * exponentially small. Signal-to-noise on a fixed budget is
 * exponentially bad in the loop area.
 *
 * Multilevel sampling reorganizes the lattice sum so the variance shrinks
 * **per time slab** instead of per global configuration. Partition the
 * temporal direction into K slabs of thickness T_sub = T/K. Freeze the
 * spatial links on the K−1 interior boundary t-slices, then sub-MC each
 * slab independently. The slab contribution
 *     M_s[(a_in, c_in), (a_out, c_out)]
 *         = ⟨ T_+(s)[a_in, a_out] · T_−(s)[c_out, c_in] ⟩_sub
 * is a 4×4 matrix in the double color-index space (a = right-side line,
 * c = left-side line). Composing the K slabs as 4×4 matrix multiplication
 * and contracting against the spatial closing pieces gives ⟨W⟩.
 *
 * Variance per outer measurement: roughly (1/N_sub)^K per slab "energy"
 * scale — geometric in K instead of arithmetic. With K=4 and N_sub=10 we
 * expect ~10× to ~100× variance reduction.
 *
 * ## What's included here
 *
 *   - `MultilevelWilson<3>`: orchestrator that, given a thermalized gauge
 *     field, performs the K-slab sub-MC and returns one multilevel ⟨W⟩
 *     estimate. Y-translation averaging is built in.
 *
 *   - `slabSweep`: a constrained heat-bath sweep that updates only links
 *     **inside** a given time slab — spatial links at the slab boundary
 *     t-slices are NOT touched. Temporal links inside the slab and
 *     spatial links at interior z-slices ARE updated.
 *
 *   - `composeWilson`: assembles ⟨W⟩ from the K slab tensors and the
 *     spatial closing-path matrices at z = 0 and z = T.
 *
 * ## What's deferred
 *
 * Multilevel for the BSS connected correlator ⟨W·ρ(x)⟩ needs per-voxel
 * slab tensors (one extra ρ-modified slab object per spatial site that
 * lives in the slab). That's a follow-up — this round demonstrates
 * variance reduction for ⟨W⟩ which is the load-bearing observable.
 */

#include "../fields/link_field.hpp"
#include "../lattice/lattice.hpp"
#include "../math/su2.hpp"
#include "../models/su2.hpp"
#include "../monte_carlo/su2_heatbath.hpp"
#include "../rng/rng.hpp"

#include <array>
#include <complex>
#include <vector>

namespace lqft::multilevel
{

using Complex = std::complex<double>;

/// 2×2 complex matrix. Layout: m[row][col].
using Mat2 = std::array<std::array<Complex, 2>, 2>;

/// 4×4 complex matrix in the (a, c) double-index space. Encoding:
///     row = 2·a_in + c_in,  col = 2·a_out + c_out.
/// So row/col ∈ {0, 1, 2, 3} where (a, c) = (0,0), (0,1), (1,0), (1,1).
using Mat4 = std::array<std::array<Complex, 4>, 4>;

// ----------------------------------------------------------------------------
// SU(2) ↔ Mat2 conversion. SU(2) element (s, v) ≡  s·I + i v·σ.
// ----------------------------------------------------------------------------

inline Mat2 toMat2(const su2::Element& U)
{
    Mat2 M{};
    M[0][0] = Complex(U.s,  U.v[2]);
    M[0][1] = Complex(U.v[1],  U.v[0]);
    M[1][0] = Complex(-U.v[1], U.v[0]);
    M[1][1] = Complex(U.s, -U.v[2]);
    return M;
}

inline Mat2 daggerMat2(const Mat2& A)
{
    Mat2 D{};
    D[0][0] = std::conj(A[0][0]);
    D[0][1] = std::conj(A[1][0]);
    D[1][0] = std::conj(A[0][1]);
    D[1][1] = std::conj(A[1][1]);
    return D;
}

inline Mat2 mulMat2(const Mat2& A, const Mat2& B)
{
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        {
            Complex sum(0.0, 0.0);
            for (int k = 0; k < 2; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    return C;
}

inline Mat2 identityMat2()
{
    return { {{ Complex(1,0), Complex(0,0) }, { Complex(0,0), Complex(1,0) }} };
}

inline Mat4 zeroMat4()
{
    Mat4 M{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) M[i][j] = Complex(0, 0);
    return M;
}

/// Slab tensor accumulator: M[(a_in, c_in), (a_out, c_out)] += T_+[a_in,a_out] · T_−[c_out,c_in].
/// T_- in our convention is `T_L_forward† = (∏ U_z(0, y, t))†`, indexed (top, bottom).
inline void accumulateSlabTensor(Mat4& M,
                                  const Mat2& T_plus,
                                  const Mat2& T_minus_dag)
{
    for (int a_in = 0; a_in < 2; ++a_in)
        for (int c_in = 0; c_in < 2; ++c_in)
        {
            const int row = 2 * a_in + c_in;
            for (int a_out = 0; a_out < 2; ++a_out)
                for (int c_out = 0; c_out < 2; ++c_out)
                {
                    const int col = 2 * a_out + c_out;
                    M[row][col]
                        += T_plus[a_in][a_out] * T_minus_dag[c_out][c_in];
                }
        }
}

inline void scaleMat4(Mat4& M, double s)
{
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) M[i][j] *= s;
}

inline Mat4 mulMat4(const Mat4& A, const Mat4& B)
{
    Mat4 C{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
        {
            Complex sum(0, 0);
            for (int k = 0; k < 4; ++k) sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    return C;
}

// ----------------------------------------------------------------------------
// Constrained slab sub-sweep.
// ----------------------------------------------------------------------------

/// One Kennedy-Pendleton heat-bath sweep, restricted to links **inside** the
/// time slab `[t_lo, t_hi)`. The spatial links at t = t_lo and t = t_hi are
/// **not** touched (those are the frozen slab-boundary spatial links that
/// keep slab interiors conditionally independent). Temporal links at
/// t = t_lo (which connect z = t_lo and z = t_lo+1) ARE updated — they
/// live inside the slab. Spatial links at interior t ∈ (t_lo, t_hi) are
/// also updated.
template<int Dim>
void slabSweep(const su2_model::SU2Model<Dim>& model,
               LinkField<su2::Element, Dim>& field,
               Rng& rng,
               int t_lo, int t_hi,
               int t_axis = 2)
{
    const auto& lattice = model.lattice();
    const int   V       = lattice.volume();
    for (int s = 0; s < V; ++s)
    {
        const auto coords = lattice.coords(s);
        const int t       = coords[static_cast<std::size_t>(t_axis)];

        for (int mu = 0; mu < Dim; ++mu)
        {
            // Spatial links are updated only strictly INSIDE the slab: the boundary
            // slices t_lo and t_hi are frozen, and slices outside the slab belong to
            // other slabs (or to the closing lines). The old test `t == t_lo || t == t_hi`
            // skipped just the two boundary slices and so updated spatial links
            // everywhere else - slab 1's sub-MC then moved the z = 0 closing line that
            // slab 0's tensor had been conditioned on, which breaks the conditional
            // independence the method rests on (a 27% low bias on <W(2,2)> at beta = 5).
            if (mu != t_axis)
            {
                if (t <= t_lo || t >= t_hi) continue;
            }
            // Temporal link U_z(s) connects t and t+1. It lives inside the
            // slab [t_lo, t_hi) iff t ∈ [t_lo, t_hi). The link out of the
            // top boundary (t = t_hi − 1 → t_hi) IS in this slab.
            else
            {
                if (t < t_lo || t >= t_hi) continue;
            }
            field(s, mu) = su2_model::heatBathUpdateOneLink(
                model, field, s, mu, rng);
        }
    }
}

// ----------------------------------------------------------------------------
// Building blocks for the Wilson loop assembly.
// ----------------------------------------------------------------------------

/// Forward product of temporal links along the line at (x, y), from
/// z = t_start (inclusive) through z = t_end (exclusive). Returns an SU(2)
/// element if the slab is empty (identity).
inline su2::Element temporalLine(const Lattice<3>& lat,
                                  const LinkField<su2::Element, 3>& U,
                                  int x, int y, int t_start, int t_end)
{
    su2::Element P = su2::Element::identity();
    for (int t = t_start; t < t_end; ++t)
    {
        const int site = lat.siteIndex({ x, y, t });
        P = su2::multiply(P, U(site, /*ẑ*/2));
    }
    return P;
}

/// Forward spatial product at fixed y, z, from x = 0 to x = R − 1 (R links).
inline su2::Element spatialLine(const Lattice<3>& lat,
                                 const LinkField<su2::Element, 3>& U,
                                 int y, int z, int R)
{
    su2::Element P = su2::Element::identity();
    for (int x = 0; x < R; ++x)
    {
        const int site = lat.siteIndex({ x, y, z });
        P = su2::multiply(P, U(site, /*x̂*/0));
    }
    return P;
}

// ----------------------------------------------------------------------------
// Multilevel orchestrator.
// ----------------------------------------------------------------------------

class MultilevelWilson
{
public:
    /// Parameters fully describe the slab partition. T must be divisible by K.
    MultilevelWilson(int R, int T, int K, int N_sub)
        : m_R(R), m_T(T), m_K(K), m_N_sub(N_sub), m_T_sub(T / K)
    {
        if (K <= 0 || N_sub <= 0 || R <= 0 || T <= 0)
            throw std::runtime_error("MultilevelWilson: bad parameters");
        if (T % K != 0)
            throw std::runtime_error(
                "MultilevelWilson: T must be divisible by K");
    }

    int R()     const { return m_R; }
    int T()     const { return m_T; }
    int K()     const { return m_K; }
    int Nsub()  const { return m_N_sub; }
    int Tsub()  const { return m_T_sub; }

    /// One multilevel measurement on the current `field`. Performs K slabs
    /// of N_sub sub-MC sweeps each, accumulates the slab tensors, then
    /// composes ⟨W(R, T)⟩ (already y-averaged inside).
    ///
    /// The sub-MC modifies `field` — caller is responsible for restoring it
    /// if a non-multilevel outer measurement on the same config is required.
    double measure(const su2_model::SU2Model<3>& model,
                   LinkField<su2::Element, 3>&    field,
                   Rng&                            rng)
    {
        const auto& lat = model.lattice();
        const int   L   = lat.volume() == 0 ? 0 : static_cast<int>(
            std::cbrt(static_cast<double>(lat.volume())) + 0.5);

        // Accumulators: one Mat4 per (slab, y). We average over the L
        // values of y as the translation average.
        std::vector<std::vector<Mat4>> M(
            m_K, std::vector<Mat4>(static_cast<std::size_t>(L), zeroMat4()));

        for (int s = 0; s < m_K; ++s)
        {
            const int t_lo = s * m_T_sub;
            const int t_hi = (s + 1) * m_T_sub;

            for (int n = 0; n < m_N_sub; ++n)
            {
                slabSweep<3>(model, field, rng, t_lo, t_hi);

                for (int y = 0; y < L; ++y)
                {
                    // T_+(s) at x = R: product of U_z(R, y, t) for t in slab.
                    const su2::Element T_R
                        = temporalLine(lat, field, m_R, y, t_lo, t_hi);
                    // T_L_forward at x = 0: product of U_z(0, y, t) for t in slab.
                    const su2::Element T_L
                        = temporalLine(lat, field,    0, y, t_lo, t_hi);
                    // T_-(s) used in the loop = T_L†.
                    const Mat2 Mp = toMat2(T_R);
                    const Mat2 Mm_dag = daggerMat2(toMat2(T_L));

                    accumulateSlabTensor(M[static_cast<std::size_t>(s)]
                                          [static_cast<std::size_t>(y)],
                                          Mp, Mm_dag);
                }
            }
            // Average.
            for (int y = 0; y < L; ++y)
                scaleMat4(M[static_cast<std::size_t>(s)]
                            [static_cast<std::size_t>(y)],
                          1.0 / static_cast<double>(m_N_sub));
        }

        // Compose ⟨W⟩, averaged over y.
        double sum_W = 0.0;
        for (int y = 0; y < L; ++y)
        {
            // Multiply slab tensors in order.
            Mat4 prod = M[0][static_cast<std::size_t>(y)];
            for (int s = 1; s < m_K; ++s)
                prod = mulMat4(prod,
                               M[static_cast<std::size_t>(s)]
                                [static_cast<std::size_t>(y)]);

            // Spatial closures at z = 0 (bottom) and z = m_T (top).
            const su2::Element B_su2   = spatialLine(lat, field, y, 0,    m_R);
            const su2::Element Top_su2_fwd
                                       = spatialLine(lat, field, y, m_T, m_R);
            // "Top" in the loop is the dagger of the forward top product.
            const Mat2 B   = toMat2(B_su2);
            const Mat2 Top = daggerMat2(toMat2(Top_su2_fwd));

            // ⟨W⟩_y = Σ B[c0, a0] · prod[(a0, c0), (a_K, c_K)] · Top[a_K, c_K]
            // with the closing condition i = c0 already absorbed.
            double Wy = 0.0;
            for (int a0 = 0; a0 < 2; ++a0)
                for (int c0 = 0; c0 < 2; ++c0)
                    for (int aK = 0; aK < 2; ++aK)
                        for (int cK = 0; cK < 2; ++cK)
                        {
                            const int row = 2 * a0 + c0;
                            const int col = 2 * aK + cK;
                            const Complex term
                                = B[c0][a0] * prod[row][col] * Top[aK][cK];
                            Wy += term.real();
                        }
            sum_W += 0.5 * Wy;   // ½ tr for SU(2)
        }
        return sum_W / static_cast<double>(L);
    }

private:
    int m_R, m_T, m_K, m_N_sub, m_T_sub;
};

} // namespace lqft::multilevel
