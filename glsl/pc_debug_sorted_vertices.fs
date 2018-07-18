#version 420

out vec4 out_Color;

in vec3 pass_Position;
in float pass_BrickIdColor;
in float pass_EDCellIdColor;
in float pass_IsED;

void main()
{
    if(pass_IsED > 0.f)
    {
        out_Color = vec4(1.f, pass_EDCellIdColor, pass_BrickIdColor, 1.f);
    }
    else
    {
        out_Color = vec4(0.1f, pass_EDCellIdColor, pass_BrickIdColor, 1.f);
    }
}