#version 420

out vec4 out_Color;
in vec3 grad;

void main()
{
    float grad_norm = clamp(length(grad) * 1000.f, 0.f, 1.f);
    out_Color = vec4(grad_norm * (normalize(grad) * 0.5f + vec3(0.5f)), 1.0f);
}