#version 420

out vec4 out_Color;

in float pass_Length;
in vec3 pass_Color;

void main()
{
    if(pass_Length > 0.f)
    {
        out_Color = vec4(pass_Color + vec3(pass_Length, 0.f, 0.f), 1.0f);
    }
    else
    {
        out_Color = vec4(pass_Color + vec3(pass_Length, 0.f, 0.f), 1.0f);
    }
}