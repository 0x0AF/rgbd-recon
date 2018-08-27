#version 420

in vec3 pass_Position;
in vec3 pass_Normal;

out vec4 out_Color;

void main()
{
    out_Color =  vec4(normalize(pass_Normal) * 0.5f + vec3(0.5f), 1.0f);
}