#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_image_load_store : require

in vec3 pass_Position;
in vec3 pass_Normal;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform mat4 NormalMatrix;

uniform sampler2DArray kinect_colors;
uniform sampler2DArray kinect_depths;
uniform sampler2DArray kinect_qualities;
uniform sampler2DArray kinect_normals;

uniform sampler3D[5] cv_xyz_inv;
uniform sampler3D[5] cv_uv;
uniform uint num_kinects;
uniform float limit;

uniform mat4 vol_to_world;

#include </shading.glsl>

out vec4 out_Color;

float[5] getWeights(const in vec3 sample_pos)
{
    float weights[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    for(uint i = 0u; i < num_kinects; ++i)
    {
        vec3 pos_calib = texture(cv_xyz_inv[i], sample_pos).xyz;
        float depth = texture(kinect_depths, vec3(pos_calib.xy, float(i))).r;
        float quality = 0.0;
        // blend if in valid depth range
        float dist = abs(depth - pos_calib.z);
        if(dist < limit)
        {
            quality = texture(kinect_qualities, vec3(pos_calib.xy, float(i))).r;
        }

        weights[i] = quality;
    }
    return weights;
}

vec3[5] getNormals(const in vec3 sample_pos)
{
    vec3 normals[5] = vec3[5](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    for(uint i = 0u; i < num_kinects; ++i)
    {
        vec3 pos_calib = texture(cv_xyz_inv[i], sample_pos).xyz;
        normals[i] = texture(kinect_normals, vec3(pos_calib.xy, float(i))).rgb;
    }
    return normals;
}

vec4 blendColors(const in vec3 sample_pos)
{
    vec3 total_color = vec3(0.0);
    vec3 total_color2 = vec3(0.0);
    float total_weight = 0.0;
    float total_weight2 = 0.0;
    vec3 colors[5] = vec3[5](vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    float distances[5] = float[5](0.0, 0.0, 0.0, 0.0, 0.0);
    for(uint i = 0u; i < num_kinects; ++i)
    {
        vec3 pos_calib = texture(cv_xyz_inv[i], sample_pos).xyz;
        vec2 pos_color = texture(cv_uv[i], pos_calib).xy;
        colors[i] = texture(kinect_colors, vec3(pos_color.xy, float(i))).rgb;

        float depth = texture(kinect_depths, vec3(pos_calib.xy, float(i))).r;
        float quality = 0.0;
        distances[i] = abs(depth - pos_calib.z);
        // blend if in valid depth range
        if(distances[i] < limit)
        {
            quality = texture(kinect_qualities, vec3(pos_calib.xy, float(i))).r;
        }
        // quality plus inverse distance
        total_color += colors[i] * quality / (distances[i] + 0.01);
        total_weight += quality / (distances[i] + 0.01);
        // fallback color blending
        total_color2 += colors[i] / distances[i];
        total_weight2 += 1.0 / distances[i];
    }
    if(total_weight > 0.0)
    {
        total_color /= total_weight;
        return vec4(total_color, 1.0);
    }
    else
    {
        total_color2 /= total_weight2;
        // return vec4(vec3(0.0), -1.0);
        return vec4(total_color2, -1.0);
    }
}

vec3 blendNormals(const in vec3 sample_pos)
{
    vec3 total_normal = vec3(0.0);
    vec3[5] normals = getNormals(sample_pos);
    float total_weight = 0.0;
    float[5] weights = getWeights(sample_pos);
    for(uint i = 0u; i < num_kinects; ++i)
    {
        total_normal += normals[i] * weights[i];
        total_weight += weights[i];
    }

    total_normal /= total_weight;
    return total_normal;
}

vec3 blendCameras(const in vec3 sample_pos)
{
    vec3 total_color = vec3(0.0);
    float total_weight = 0.0;
    float[5] weights = getWeights(sample_pos);
    for(uint i = 0u; i < num_kinects; ++i)
    {
        vec3 pos_calib = texture(cv_xyz_inv[i], sample_pos).xyz;
        vec3 color = camera_colors[i];

        total_color += color * weights[i];
        total_weight += weights[i];
    }

    total_color /= total_weight;
    if(total_weight <= 0.0)
        total_color = vec3(1.0);
    return total_color;
}

vec4 get_color(vec3 sample_pos)
{
    vec3 view_normal = normalize((NormalMatrix * vec4(blendNormals(sample_pos), 0.0)).xyz);
    vec3 view_pos = (gl_ModelViewMatrix * vol_to_world * vec4(sample_pos, 1.0)).xyz;

    if(g_shade_mode == 3)
    {
        return vec4(blendCameras(sample_pos), 1.0);
    }
    else
    {
        vec4 diffuseColor = blendColors(sample_pos);
        return vec4(shade(view_pos, view_normal, diffuseColor.rgb), diffuseColor.a);
    }
}

void main() { out_Color = get_color(pass_Position); }