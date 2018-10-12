#version 420

in vec3 pass_Position;
in vec3 pass_Normal;

out vec4 out_Color;

const vec3 solid_diffuse = vec3(0.5f);

const vec3 LightPosition = vec3(1.5f, 1.0f, 1.0f);
const vec3 LightAmbient = vec3(0.135, 0.2225, 0.1575);
const vec3 LightDiffuse = vec3(0.54, 0.89, 0.63);
const vec3 LightSpecular = vec3(0.316228, 0.316228, 0.316228);

const float ks = 0.1f;
const float n = 12.0f;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 world_to_vol;
uniform mat4 vol_to_world;

vec2 phongDiffSpec(const vec3 position, const vec3 normal, const float n, const vec3 lightPos)
{
    vec3 toLight = normalize(lightPos - position);
    float lightAngle = dot(normal, toLight);
    // if fragment is not directly lit, use only ambient light
    if(lightAngle <= 0.0f)
    {
        return vec2(0.0f);
    }

    float diffContribution = max(lightAngle, 0.0f);

    vec3 toViewer = normalize(-position);
    vec3 halfwayVector = normalize(toLight + toViewer);
    float reflectedAngle = dot(halfwayVector, normal);
    float specLight = pow(reflectedAngle, n);

    // fade out specular hightlights towards edge of lit region
    float a = (1.0f - lightAngle) * (1.0f - lightAngle);
    specLight *= 1.0f - a * a * a;

    return vec2(diffContribution, specLight);
}

void main()
{
    vec3 normal = clamp(pass_Normal * 0.5 + 0.5, vec3(0.f), vec3(1.f, 1.f, 1.f));
    out_Color = vec4(normal, 1.0f);
    //vec3 position = (gl_ModelViewMatrix * inverse(world_to_vol) * vol_to_world * vec4(pass_Position,1.f)).xyz;
    //vec2 diffSpec = phongDiffSpec(position, normal, n, LightPosition);
    //out_Color = vec4(LightAmbient * solid_diffuse + 0.5 * (LightDiffuse * solid_diffuse * diffSpec.x + LightSpecular * ks * diffSpec.y), 1.0f);
}