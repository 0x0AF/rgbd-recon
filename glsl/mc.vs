#version 420

layout(location = 0) in vec3 in_Position;

uniform mat4 TextureMatrix;
uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

out vec3 pass_Position;

void main()
{
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(in_Position, 1.0f);
    pass_Position = in_Position;
}
