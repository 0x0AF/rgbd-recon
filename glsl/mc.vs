#version 420

layout(location = 0) in vec3 in_Position;

uniform mat4 TextureMatrix;
uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;

out vec3 in_pass_Position;

void main()
{
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix *  vec4(in_Position, 1.0f);
    in_pass_Position = in_Position;
}
