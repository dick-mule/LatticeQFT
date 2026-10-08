#pragma once

/**
 * @file gpu_schwinger_force.hpp
 * @brief GPU computation of the Schwinger HMC force on every link.
 *
 * Force = gauge force + pseudofermion force:
 *
 *     F(x, μ) = −∂S_gauge/∂θ_μ(x) − ∂S_pf/∂θ_μ(x)
 *
 * with:
 *
 *     S_gauge = β Σ_□ (1 − cos φ_□)
 *     S_pf    = φ† (D†D)⁻¹ φ                       (Schwinger pseudofermion)
 *
 * Per `computeForce(φ, U) → F` call:
 *
 *   1. Upload φ to the inner GpuCG2D as the right-hand-side.
 *   2. CG-solve for ψ = (D†D)⁻¹ φ on GPU.
 *   3. Apply D to ψ to get η = D ψ on GPU.
 *   4. Dispatch `gauge_force_u1.comp` writing F_gauge into the force buffer.
 *   5. Dispatch `fermion_force_u1.comp` *adding* F_pf to the force buffer.
 *   6. Download the force to the user-provided LinkField.
 *
 * `selfValidate()` runs the same operation on CPU (via `SchwingerProvider`)
 * and on GPU from the same configuration, returns relative L² error in F.
 */

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include "../../fields/link_field.hpp"
#include "../../fields/spinor_field.hpp"
#include "../../lattice/lattice.hpp"
#include "../../rng/rng.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lqft::vis
{

class GpuCG2D; // forward declaration

class GpuSchwingerForce
{
public:
    GpuSchwingerForce(vk::PhysicalDevice  phys,
                      vk::Device          device,
                      vk::Queue           queue,
                      std::uint32_t       queue_family,
                      vk::CommandPool     cmd_pool,
                      int                 L,
                      const std::string&  shader_dir);
    ~GpuSchwingerForce();

    GpuSchwingerForce(const GpuSchwingerForce&)            = delete;
    GpuSchwingerForce& operator=(const GpuSchwingerForce&) = delete;

    int latticeSize() const { return m_L; }

    /// One total-force computation on GPU. Uploads U and φ; downloads the
    /// resulting force vector to `force_out`. CG tol/max_iters control the
    /// inner solve. `mass` and `beta` are the Wilson and Wilson-Dirac
    /// parameters.
    void computeForce(const LinkField<double, 2>&    U,
                      const SpinorField<2, 2>&       phi,
                      double                          mass,
                      double                          beta,
                      double                          cg_tol,
                      int                             cg_max_iters,
                      LinkField<double, 2>&          force_out);

    /// Compare GPU force to CPU force from the same configuration. Returns
    /// relative L² error ‖F_gpu − F_cpu‖ / ‖F_cpu‖.
    double selfValidate(double beta              = 2.0,
                        double mass              = 0.4,
                        double cg_tol            = 1e-7,
                        int    cg_max_iters      = 4000,
                        std::uint64_t seed       = 42u);

    // ---- Buffer accessors (kept narrow on purpose; downstream classes
    // bind to CG's gauge + solution via GpuCG2D's own accessors).
    GpuCG2D&       cg()                  { return *m_cg;       }
    const GpuCG2D& cg()            const { return *m_cg;       }

    /// Run one HMC trajectory on CPU (via the existing `SchwingerProvider`
    /// + `GaugeHMC`) and one on GPU (host-orchestrated leapfrog, but every
    /// force evaluation runs `computeForce(U, φ, …)` end-to-end on GPU) with
    /// identical initial conditions and identical (π, χ) Gaussian draws.
    /// Returns the absolute difference in the trajectory's final ΔH —
    /// values below ~1e-3 confirm the GPU force pipeline integrates to the
    /// same trajectory as the CPU within single-precision GPU rounding.
    double selfValidateTrajectory(double beta              = 2.0,
                                  double mass              = 0.4,
                                  double dt                = 0.04,
                                  int    n_steps           = 20,
                                  double cg_tol            = 1e-7,
                                  int    cg_max_iters      = 4000,
                                  std::uint64_t seed       = 271828u);

    /// One full GPU-backed HMC trajectory. Samples π and χ on host, computes
    /// φ = D†χ once on host, runs the leapfrog with every force evaluation
    /// dispatched to the GPU (CG + Wilson-Dirac + force kernels), evaluates
    /// the Hamiltonian before and after, and Metropolis accepts/rejects.
    /// `theta_io` is updated to the accepted proposal (or left unchanged on
    /// reject). Returns ΔH and the accept decision.
    struct Result { double dH; bool accepted; };
    Result trajectory(LinkField<double, 2>& theta_io,
                      double beta,
                      double mass,
                      double dt,
                      int    n_steps,
                      double cg_tol,
                      int    cg_max_iters,
                      Rng&   rng);

private:
    void createBuffers();
    void createPipelinesAndDescriptors(const std::string& shader_dir);
    void allocateCommandBuffer();

    void uploadPhi(const SpinorField<2, 2>& phi);
    void downloadForce(LinkField<double, 2>& force_out) const;

    void dispatchEta(float mass);    // η = D ψ
    void dispatchForces(float beta); // gauge force then fermion force

    static std::vector<char> readFile(const std::string& path);
    std::uint32_t findMemoryType(std::uint32_t type_bits,
                                 vk::MemoryPropertyFlags props) const;

    struct WdPushConstants
    {
        int   Lx; int Ly; float mass; int dagger;
    };
    struct GaugeForcePushConstants
    {
        int Lx; int Ly; float beta;
    };
    struct FermionForcePushConstants
    {
        int Lx; int Ly;
    };

    vk::PhysicalDevice m_phys;
    vk::Device         m_device;
    vk::Queue          m_queue;
    std::uint32_t      m_queue_family;
    vk::CommandPool    m_cmd_pool;
    int                m_L;
    int                m_V;

    // Inner CG (owns the gauge + spinor buffers we operate on).
    std::unique_ptr<GpuCG2D> m_cg;

    // Owned buffers
    struct Buffer
    {
        vk::Buffer       buf;
        vk::DeviceMemory mem;
        void*            mapped = nullptr;
        vk::DeviceSize   bytes  = 0;
    };
    Buffer m_phi;    // pseudofermion (CG b)
    Buffer m_eta;    // η = D ψ
    Buffer m_force;  // V·2 floats

    // Pipelines
    vk::DescriptorSetLayout m_wd_layout;
    vk::DescriptorSetLayout m_gforce_layout;
    vk::DescriptorSetLayout m_fforce_layout;
    vk::DescriptorPool      m_desc_pool;
    vk::DescriptorSet       m_set_wd;
    vk::DescriptorSet       m_set_gforce;
    vk::DescriptorSet       m_set_fforce;

    vk::PipelineLayout m_wd_pl, m_gforce_pl, m_fforce_pl;
    vk::Pipeline       m_wd_pipe, m_gforce_pipe, m_fforce_pipe;
    vk::ShaderModule   m_wd_mod, m_gforce_mod, m_fforce_mod;

    vk::CommandBuffer m_cmd;
    vk::Fence         m_fence;
};

} // namespace lqft::vis
