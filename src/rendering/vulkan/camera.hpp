#pragma once

/**
 * @file camera.hpp
 * @brief Orbit camera around the origin for the 3D volume viewer.
 *
 * Parameters:
 *   - distance              radial distance from the lattice center
 *   - polar (θ ∈ (0, π))     angle from +Y, clamped away from poles to avoid
 *                            singularities in the look-at matrix
 *   - azimuthal (φ ∈ ℝ)      angle in the XZ-plane, free
 *   - fov                    vertical field-of-view (radians)
 *
 * The view matrix is `lookAt(position(), origin, +Y)`. The projection is
 * `perspective(fov, aspect, near, far)` with the standard Vulkan Y-flip
 * (`[1][1] *= -1`) applied to match clip-space conventions.
 *
 * Mouse handling lives in the parent app, which calls `onMouseDrag()` /
 * `onScroll()` between frames. The class itself is purely state; no GLFW.
 */

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace lqft::vis
{

class Camera
{
public:
    Camera() { reset(); }

    void reset()
    {
        m_distance  = 4.5f;
        m_polar     = 1.2f;          // ~70° from +Y
        m_azimuthal = 0.785f;        // π/4
        m_fov       = glm::radians(45.0f);
    }

    void onMouseDrag(float dx_pixels, float dy_pixels)
    {
        constexpr float kSensitivity = 0.0065f;
        m_azimuthal -= dx_pixels * kSensitivity;
        m_polar      = glm::clamp(m_polar - dy_pixels * kSensitivity,
                                  0.10f, 3.04f);
    }

    void onScroll(float dy)
    {
        m_distance = glm::clamp(m_distance * std::pow(1.10f, -dy),
                                1.2f, 80.0f);
    }

    glm::vec3 position() const
    {
        const float sp = std::sin(m_polar);
        const float cp = std::cos(m_polar);
        const float sa = std::sin(m_azimuthal);
        const float ca = std::cos(m_azimuthal);
        return m_distance * glm::vec3(sp * ca, cp, sp * sa);
    }

    glm::mat4 viewMatrix() const
    {
        return glm::lookAt(position(),
                           glm::vec3(0.0f),
                           glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 projectionMatrix(float aspect) const
    {
        glm::mat4 P = glm::perspective(m_fov, aspect, 0.05f, 200.0f);
        P[1][1] *= -1.0f;  // Vulkan clip-space Y flip.
        return P;
    }

    /// `inverse(P · V)` — useful for un-projecting a clip-space point in the
    /// ray-march fragment shader.
    glm::mat4 inverseViewProj(float aspect) const
    {
        return glm::inverse(projectionMatrix(aspect) * viewMatrix());
    }

    float distance() const   { return m_distance; }
    float polar() const      { return m_polar; }
    float azimuthal() const  { return m_azimuthal; }
    float fov() const        { return m_fov; }

private:
    float m_distance;
    float m_polar;
    float m_azimuthal;
    float m_fov;
};

} // namespace lqft::vis
