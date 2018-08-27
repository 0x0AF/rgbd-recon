#version 420

out vec4 out_Color;

in vec3 pass_Position;
in vec3 pass_Normal;

void main() {
  out_Color =  vec4(normalize(pass_Normal) * 0.5f + vec3(0.5f), 1.0f);
}