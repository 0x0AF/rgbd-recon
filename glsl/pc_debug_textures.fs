#version 420

#extension GL_ARB_explicit_attrib_location : require
uniform sampler2DArray texture_2d_array;
layout(location = 0) out vec4 fragColor;
in vec2 v_uv;

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
    int layer = 0;
    vec2 uv = v_uv;

    if(v_uv.x < 0.5f && v_uv.y < 0.5f)
    {
        uv = v_uv;
        uv = uv * 2.;
        layer = 0;
    }

    if(v_uv.x < 0.5f && v_uv.y > 0.5f)
    {
        uv = v_uv - vec2(0., 0.5);
        uv = uv * 2.;
        layer = 1;
    }

    if(v_uv.x > 0.5f && v_uv.y < 0.5f)
    {
        uv = v_uv - vec2(0.5, 0.);
        uv = uv * 2.;
        layer = 2;
    }

    if(v_uv.x > 0.5f && v_uv.y > 0.5f)
    {
        uv = v_uv - vec2(0.5);
        uv = uv * 2.;
        layer = 3;
    }

    float rot = radians(270.0);
    uv -= .5;
    mat2 m = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    uv = m * uv;
    uv += .5;

    ivec3 dims = textureSize(texture_2d_array, 0);
    ivec3 coordinate = ivec3(uv.x * dims.x, uv.y * dims.y, layer);
    fragColor = texelFetch(texture_2d_array, coordinate, 0);

    float error = fragColor.r;

    fragColor = colormap(error);

    // fragColor = vec4(uv.xy, float(layer) / 4., 1.);
}