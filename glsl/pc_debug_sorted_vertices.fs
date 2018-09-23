#version 420

out vec4 out_Color;

in vec3 pass_Position;
in float pass_Error;
in float pass_Range;

float colormap_red(float x)
{
    if(x < 0.7)
    {
        return 4.0 * x - 1.5;
    }
    else
    {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x)
{
    if(x < 0.5)
    {
        return 4.0 * x - 0.5;
    }
    else
    {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x)
{
    if(x < 0.3)
    {
        return 4.0 * x + 0.5;
    }
    else
    {
        return -4.0 * x + 2.5;
    }
}

vec4 colormap(float x)
{
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main()
{
    float error = clamp(pass_Error / pass_Range, 0.f, 1.f);
    out_Color = colormap(error);
}