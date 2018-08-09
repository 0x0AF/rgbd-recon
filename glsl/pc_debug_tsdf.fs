#version 420

out vec4 out_Color;

in vec3 pass_Position;

uniform sampler3D volume_tsdf;
uniform float limit;

void main()
{
    float tsdf = texture(volume_tsdf, pass_Position).r;

    float inverted_dist = abs(tsdf) / limit;
    if(tsdf > 0.f)
    {
        out_Color = vec4(1.0f - inverted_dist, 0.0f, 0.0f, 1.0f);
    }
    else
    {
        out_Color = vec4(0.0f, 1.0f - inverted_dist, 0.0f, 1.0f);
    }

    if(tsdf > limit)
    {
        out_Color = vec4(0.0f, 0.0f, 1.0f, 1.0f);
    }

    if(tsdf <= -limit || tsdf == 0)
    {
        discard;
    }
}