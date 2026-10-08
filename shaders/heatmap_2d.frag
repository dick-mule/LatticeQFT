#version 450

// 2D heatmap fragment shader.
// Reuses `fullscreen.vert` for the screen triangle. Samples a R32_SFLOAT 2D
// texture, applies a thresholded-linear gain, viridis colormap (signed
// version: blue = negative, yellow = positive, dark = near zero so the
// Ising ±1 spins read as cool/warm against a dark background).

layout(location = 0) in  vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D u_field;

layout(push_constant) uniform PC
{
    float gain;        // multiplies the texel before colormap lookup
    float bias;        // shifts the texel before gain (e.g. for Ising: 0)
    float gamma;       // colormap exponent
    int   signed_map;  // 0 = positive-only (viridis), 1 = diverging (blue/white/yellow)
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

// Diverging map: -1 → cool blue, 0 → dark, +1 → warm yellow.
// Built by mapping |v| through viridis then tinting by sign.
vec3 diverging(float v)
{
    float a = clamp(pow(abs(v), pc.gamma), 0.0, 1.0);
    vec3  warm = viridis(0.55 + 0.45 * a);              // yellow-green tail
    vec3  cool = vec3(0.05, 0.20, 0.50) + a * vec3(0.10, 0.55, 0.50); // blue
    return v >= 0.0 ? warm : cool;
}

void main()
{
    float v = texture(u_field, v_uv).r;
    v = (v + pc.bias) * pc.gain;
    if (pc.signed_map != 0)
    {
        frag_color = vec4(diverging(v), 1.0);
    }
    else
    {
        float t = clamp(pow(max(v, 0.0), pc.gamma), 0.0, 1.0);
        frag_color = vec4(viridis(t), 1.0);
    }
}
