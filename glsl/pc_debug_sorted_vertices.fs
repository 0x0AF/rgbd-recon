#version 420

out vec4 out_Color;

in vec3 pass_Position;
in float pass_BrickIdColor;
in vec3 pass_Normal;
in float pass_EDCellIdColor;

void main()
{
    out_Color = vec4(0.1f, pass_EDCellIdColor, pass_BrickIdColor, 1.0f);
}