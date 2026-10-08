#pragma once

/**
 * @file gpu_cg_2d.hpp
 * @brief Fully-GPU conjugate-gradient solver for the 2D Wilson-Dirac D†D.
 *
 * Solves
 *     D†[U] D[U] x = b
 * for x given a U(1) gauge field U and a right-hand-side spinor b, with all
 * spinors and vector ops resident on the GPU between iterations. Only the
 * scalars (α, β, ⟨r,r⟩, ⟨p,Ap⟩) round-trip to the host — one reduction
 * readback per inner-product per CG iteration.
 *
 * Three compute pipelines:
 *
 *   - `wilson_dirac_2d.comp` — D / D† apply  (shared SPIR-V with GpuWilsonDirac2D)
 *   - `spinor_axpby_real.comp` — y = a·x + b·y on complex spinor pairs
 *   - `spinor_inner_real.comp` — workgroup-partial reductions of Re ⟨a, b⟩
 *
 * Per CG iteration:
 *
 *   1. D p → tmp ;  D† tmp → Ap ;  inner ⟨p, Ap⟩ → partials_pap
 *      ── wait, sum partials, α = r² / ⟨p,Ap⟩
 *
 *   2. x += α p ;  r −= α Ap ;  inner ⟨r, r⟩ → partials_rr
 *      ── wait, sum partials, β = r²_new / r²_old ;  convergence check
 *
 *   3. p = r + β p
 *
 * On each phase we re-record the command buffer with the current push
 * constants (α, β) and reuse the seven pre-allocated descriptor sets — one
 * per (pipeline, buffer-binding) pair — so the buffers themselves stay
 * resident the entire solve.
 *
 * `selfValidateCG()` runs the same solve on CPU (via `solvers::conjugateGradient`)
 * and on GPU, then returns the relative L²-norm error.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "../../fields/link_field.hpp"
#include "../../fields/spinor_field.hpp"
#include "../../lattice/lattice.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuCG2D
{
public:
    GpuCG2D(vk::PhysicalDevice  phys,
            vk::Device          device,
            vk::Queue           queue,
            std::uint32_t       queue_family,
            vk::CommandPool     cmd_pool,
            int                 L,
            const std::string&  shader_dir);
    ~GpuCG2D();

    GpuCG2D(const GpuCG2D&)            = delete;
    GpuCG2D& operator=(const GpuCG2D&) = delete;

    int latticeSize() const { return m_L; }

    struct Result
    {
        int    iterations = 0;
        double final_res2 = 0.0;   // ⟨r, r⟩ at exit
        bool   converged  = false;
    };

    /// Solve D†D x = b for x. The gauge field `U` and right-hand side `b`
    /// are uploaded internally; the solution `x` is downloaded into the
    /// provided output spinor.
    Result solve(const LinkField<double, 2>& U,
                 const SpinorField<2, 2>&    b,
                 double                       mass,
                 double                       tol,
                 int                          max_iters,
                 SpinorField<2, 2>&           x_out);

    /// Compare GPU CG against CPU CG on a deterministic random configuration.
    /// Returns relative L² error between the two solution spinors. Values
    /// below ~1e-4 confirm the GPU pipeline is correct (the floor is set by
    /// single-precision GPU arithmetic vs double-precision CPU).
    double selfValidateCG(double mass        = 0.4,
                          double tol         = 1e-7,
                          int    max_iters   = 4000,
                          std::uint64_t seed = 11u);

    // ---- Buffer accessors for downstream pipelines (e.g. GpuSchwingerForce).
    // Read-only access to the gauge and solution buffers so an outer class
    // can bind them into its own descriptor sets without copying.
    vk::Buffer      gaugeBuffer()    const { return m_gauge.buf;   }
    vk::DeviceSize  gaugeBytes()     const { return m_gauge.bytes; }
    vk::Buffer      solutionBuffer() const { return m_x.buf;       }
    vk::DeviceSize  spinorBytes()    const { return m_b.bytes;     }
    int             nComponents()    const { return m_n_components; }

private:
    // ----- Vulkan resource construction -----
    void createBuffers();
    void createPipelinesAndDescriptors(const std::string& shader_dir);
    void allocateCommandBuffer();

    // ----- per-iteration helpers -----
    /// Record + submit phase 1 (apply A = D†D to p, compute ⟨p, Ap⟩).
    /// Returns ⟨p, Ap⟩ summed on the host.
    double dispatchPhaseAp(float mass);

    /// Record + submit phase 2 (update x and r, compute ⟨r, r⟩).
    /// Returns ⟨r, r⟩ summed on the host.
    double dispatchPhaseUpdate(float alpha);

    /// Record + submit phase 3 (update p = r + β p).
    void dispatchPhaseRecurse(float beta);

    /// Sum the partials buffer (one float per workgroup).
    double sumPartials() const;

    // ----- IO helpers -----
    void uploadGauge   (const LinkField<double, 2>& U);
    void uploadSpinor  (vk::DeviceMemory  mem, void* mapped,
                        const SpinorField<2, 2>& s);
    void downloadSpinor(void* mapped, SpinorField<2, 2>& s) const;

    static std::vector<char> readFile(const std::string& path);
    std::uint32_t findMemoryType(std::uint32_t type_bits,
                                 vk::MemoryPropertyFlags props) const;

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    int                m_L;
    int                m_V;
    int                m_n_components;   // V · Nc = V · 2 complex entries
    std::uint32_t      m_n_workgroups;   // ceil(n_components / 64)

    // ----- storage buffers -----
    struct Buffer
    {
        vk::Buffer       buf;
        vk::DeviceMemory mem;
        void*            mapped = nullptr;
        vk::DeviceSize   bytes  = 0;
    };
    Buffer m_gauge, m_b, m_x, m_r, m_p, m_Ap, m_tmp, m_partials;

    // ----- descriptor pool + sets -----
    vk::DescriptorSetLayout m_wd_layout;     // wilson_dirac: 3 storage buffers
    vk::DescriptorSetLayout m_axpby_layout;  // axpby: 2 storage buffers
    vk::DescriptorSetLayout m_inner_layout;  // inner: 3 storage buffers

    vk::DescriptorPool m_desc_pool;

    // Seven persistent descriptor sets bound to fixed buffer combinations:
    vk::DescriptorSet m_set_wd_Dp;     // (gauge, p,  tmp): D p → tmp
    vk::DescriptorSet m_set_wd_DtAp;   // (gauge, tmp, Ap): D† tmp → Ap
    vk::DescriptorSet m_set_axpby_x;   // (p, x):       x  += α·p
    vk::DescriptorSet m_set_axpby_r;   // (Ap, r):      r  −= α·Ap
    vk::DescriptorSet m_set_axpby_p;   // (r, p):       p  = r + β·p
    vk::DescriptorSet m_set_inner_pap; // (p,  Ap, partials)
    vk::DescriptorSet m_set_inner_rr;  // (r,  r,  partials)

    // ----- pipelines -----
    vk::PipelineLayout m_wd_pipe_layout;
    vk::PipelineLayout m_axpby_pipe_layout;
    vk::PipelineLayout m_inner_pipe_layout;

    vk::Pipeline m_wd_pipeline;
    vk::Pipeline m_axpby_pipeline;
    vk::Pipeline m_inner_pipeline;

    vk::ShaderModule m_wd_shader;
    vk::ShaderModule m_axpby_shader;
    vk::ShaderModule m_inner_shader;

    vk::CommandBuffer m_cmd;
    vk::Fence         m_fence;
};

} // namespace lqft::vis
