#version 420

in vec3 pass_Position;
in vec3 pass_Normal;

out vec4 out_Color;

void main()
{
    out_Color = vec4(pass_Normal, 1.);
}