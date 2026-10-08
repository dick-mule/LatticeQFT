#version 450

// Fullscreen-triangle vertex shader.
// Emits 3 vertices covering the clip-space [-1, 1]² rectangle so that a
// triangle list of 3 vertices with no vertex buffer rasterizes the entire
// viewport. Standard trick:
//   gl_VertexIndex = 0 → (-1, -1)   uv = (0, 0)
//   gl_VertexIndex = 1 → ( 3, -1)   uv = (2, 0)
//   gl_VertexIndex = 2 → (-1,  3)   uv = (0, 2)
// The triangle covers the screen exactly once; the part outside [0, 1]²
// in UV is clipped before reaching the fragment shader.

layout(location = 0) out vec2 v_uv;

void main()
{
    v_uv        = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
