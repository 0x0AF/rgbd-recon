#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_image_load_store : require

in vec3 pass_Position;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform mat4 vol_to_world;

uniform sampler3D volume_tsdf;

uniform mat4 img_to_eye_curr;

out vec4 out_Color;

float sample_volume(const vec3 pos) { return texture(volume_tsdf, pos).r; }

void main()
{
    out_Color = vec4(normalize(pass_Position), 1.0f);

    //float value = sample_volume(pass_Position);

    //out_Color = vec4(value * 256.0f, 0.1f, 0.1f, 1.0f);
}