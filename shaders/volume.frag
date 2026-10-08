#version 450

// Volumetric ray-marcher for the 3D lattice action-density field.
//
// Inputs:
//   - u_volume: R32_SFLOAT 3D texture, sum_{μ<ν} (1 − cos θ_□) per site
//   - push:    inv(proj · view), camera world-pos, transfer-function knobs
//
// Algorithm (single-pass):
//   1. Reconstruct the world-space ray for this pixel by un-projecting a
//      clip-space point at NDC z = 1 through inv_view_proj.
//   2. Intersect that ray with the lattice's [-1, +1]^3 bounding cube.
//   3. March from t_near to t_far in N_STEPS uniform steps.
//      At each step:
//        - sample the 3D texture (lattice in [0, 1]^3 texcoords),
//        - apply the transfer function (linear-after-threshold opacity,
//          viridis color),
//        - composite front-to-back.
//      Early-out once accumulated alpha saturates near 1.

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler3D u_volume;

layout(push_constant) uniform PC
{
    mat4 inv_view_proj;
    vec3 cam_pos;
    float density_scale;        // multiplies opacity per step
    float opacity_threshold;    // subtract from sample before scaling
    float gamma;                // colormap exponent (1.0 = linear)
} pc;

vec3 viridis(float t)
{
    const vec3 c0 = vec3( 0.2777273,  0.0057731,  0.3340999);
    const vec3 c1 = vec3( 0.1059625,  1.4045418,  1.3847944);
    const vec3 c2 = vec3(-0.3308618,  0.2148293,  0.0935573);
    const vec3 c3 = vec3(-4.6342050, -5.7991189,-19.3324190);
    const vec3 c4 = vec3( 6.2287682, 14.1798714, 56.6905352);
    const vec3 c5 = vec3( 4.7763263,-13.7456095,-65.3539479);
    const vec3 c6 = vec3(-5.4350286,  4.6450957, 26.3124243);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

// Ray vs axis-aligned bounding box; returns whether the ray hits the box
// and writes the near/far parametric intersection ts.
bool intersect_box(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max,
                   out float t_near, out float t_far)
{
    vec3 inv_rd = 1.0 / rd;
    vec3 t1 = (box_min - ro) * inv_rd;
    vec3 t2 = (box_max - ro) * inv_rd;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    t_near = max(max(tmin.x, tmin.y), tmin.z);
    t_far  = min(min(tmax.x, tmax.y), tmax.z);
    return t_far >= max(t_near, 0.0);
}

void main()
{
    // Construct ray from screen-space uv → world-space ray direction.
    vec4 ndc       = vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world_pt  = pc.inv_view_proj * ndc;
    world_pt      /= world_pt.w;
    vec3 ro        = pc.cam_pos;
    vec3 rd        = normalize(world_pt.xyz - ro);

    // Lattice lives in [-1, +1]^3 in world space.
    const vec3 box_min = vec3(-1.0);
    const vec3 box_max = vec3( 1.0);
    float t_near, t_far;
    if (!intersect_box(ro, rd, box_min, box_max, t_near, t_far))
    {
        // Background — neutral dark to make ImGui pop, matches clear color.
        frag_color = vec4(0.08, 0.10, 0.14, 1.0);
        return;
    }
    t_near = max(t_near, 0.0);

    const int N_STEPS = 192;
    float dt   = (t_far - t_near) / float(N_STEPS);
    vec3  step = rd * dt;
    vec3  p    = ro + t_near * rd;

    vec4 accum = vec4(0.0);
    for (int i = 0; i < N_STEPS; ++i)
    {
        vec3 tex_coord = clamp(p * 0.5 + 0.5, vec3(0.0), vec3(1.0));
        float v = texture(u_volume, tex_coord).r;

        float thresholded = max(v - pc.opacity_threshold, 0.0);
        float alpha       = clamp(thresholded * pc.density_scale * dt, 0.0, 1.0);

        // Colormap on a 0..1-ish range. v in 3D U(1) action density runs
        // roughly 0..3 (3 plaquettes × max ~2); divide by 3 to keep colormap
        // sweep across the typical dynamic range.
        float t_color = clamp(pow(v / 3.0, pc.gamma), 0.0, 1.0);
        vec3  col     = viridis(t_color);

        accum.rgb += (1.0 - accum.a) * col * alpha;
        accum.a   += (1.0 - accum.a) * alpha;
        if (accum.a > 0.99) break;
        p += step;
    }

    // Composite over the background so any uncovered ray still produces
    // something visible.
    vec3 bg = vec3(0.08, 0.10, 0.14);
    frag_color = vec4(accum.rgb + (1.0 - accum.a) * bg, 1.0);
}
