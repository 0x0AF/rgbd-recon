#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_image_load_store : require

in vec3 pass_Position;
in vec3 pass_Normal;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

out vec4 out_Color;

void main()
{
    out_Color = vec4(pass_Normal, 1.0f);
}