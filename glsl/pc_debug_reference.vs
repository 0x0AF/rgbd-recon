#version 420

in vec3 in_Position;
out vec3 geo_Position;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;

void main() {
  geo_Position = in_Position;
  gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(in_Position, 1.0);
}