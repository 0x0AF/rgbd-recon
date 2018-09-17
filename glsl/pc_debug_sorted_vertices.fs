#version 420

out vec4 out_Color;

in vec3 pass_Position;
in vec3 pass_Error;

void main()
{
    float error = clamp(pass_Error.x / 0.01f, 0.f, 1.f);
    out_Color = vec4(error, 0.f, 1.f - error, 1.0f);
}