#version 420

out vec4 out_Color;

in vec3 pass_Position;
in vec3 pass_Translation;

vec3 get_color_for_translation(float translation)
{
    if(translation > 1.f)
    {
        return vec3(1.f, 0.f, 0.f);
    }

    vec3 color = vec3(0.3f, 1.f, 0.3f);
    color = color * (0.5 + clamp(translation / 2.f, 0.f, 0.5f));
    return color;
}

void main() { out_Color = vec4(get_color_for_translation(length(pass_Translation)), 1.0f); }