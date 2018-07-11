#version 420

out vec4 out_Color;

in vec3 pass_Position;
in vec3 pass_Normal;

void main() {
  out_Color =  vec4(pass_Normal, 1.0f);
}