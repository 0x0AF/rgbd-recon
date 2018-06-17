#version 420

in vec3 pass_Position;
in vec3 pass_Normal;

out vec4 out_Color;

const vec3 solid_diffuse = vec3(0.5f);

const vec3 LightPosition = vec3(1.f, -1.f, -1.f);
const vec3 LightDiffuse = vec3(1.0f, 0.9f, 0.7f);
const vec3 LightAmbient = LightDiffuse * 0.2f;
const vec3 LightSpecular = vec3(1.0f);

const float ks = 0.5f;
const float n = 20.0f;

vec2 phongDiffSpec(const vec3 position, const vec3 normal, const float n, const vec3 lightPos) {
  vec3 toLight = normalize(lightPos - position);
  float lightAngle = dot(normal, toLight);
  // if fragment is not directly lit, use only ambient light
  if (lightAngle <= 0.0f) {
    return vec2(0.0f);
  }

  float diffContribution = max(lightAngle, 0.0f);

  vec3 toViewer = normalize(-position);
  vec3 halfwayVector = normalize(toLight + toViewer);
  float reflectedAngle = dot(halfwayVector, normal);
  float specLight = pow(reflectedAngle, n);

  // fade out specular hightlights towards edge of lit region
  float a = (1.0f - lightAngle) * ( 1.0f - lightAngle);
  specLight *= 1.0f - a * a * a;

  return vec2(diffContribution, specLight);
}

void main()
{
    vec2 diffSpec = phongDiffSpec(pass_Position, pass_Normal, n, LightPosition);
    out_Color =  vec4(LightAmbient * solid_diffuse + LightDiffuse * solid_diffuse * diffSpec.x + LightSpecular * ks * diffSpec.y, 1.0f);
}