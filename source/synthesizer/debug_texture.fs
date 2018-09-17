#version 420

#extension GL_ARB_explicit_attrib_location : require
uniform sampler2DArray texture_2d_array;
uniform int mode;
layout(location = 0) out vec4 fragColor;
in vec2 v_uv;

#define TWO_PI 6.28318530718

vec3 hsv2rgb(in vec3 c)
{
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);

    return c.z * mix(vec3(1.0), rgb, c.y);
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

    /*float rot = radians(270.);
    uv -= .5;
    mat2 m = mat2(cos(rot), -sin(rot), sin(rot), cos(rot));
    uv = m * uv;
    uv += .5;*/

    ivec3 dims = textureSize(texture_2d_array, 0);
    ivec3 coordinate = ivec3(uv.x * dims.x, uv.y * dims.y, layer);

    fragColor = texelFetch(texture_2d_array, coordinate, 0);

    switch(mode)
    {
    case 1:
        fragColor = fragColor / (3. - 0.5);
        break;
    case 3:
        float angle = atan(fragColor.y, fragColor.x);
        float radius = length(vec2(fragColor.x, fragColor.y)) * 2.f;

        vec3 color = hsv2rgb(vec3((angle / TWO_PI) + 0.5f, radius, 1.f));
        fragColor = vec4(color, 1.f);
        break;
    }
}