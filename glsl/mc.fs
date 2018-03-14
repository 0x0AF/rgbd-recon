#version 420
#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_image_load_store : require

in vec3 pass_Position;
// calibration
uniform float limit;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform mat4 NormalMatrix;
uniform mat4 vol_to_world;

uniform sampler3D volume_tsdf;
uniform vec3 CameraPos;

uniform sampler2D depth_peels;
uniform vec2 viewport_offset;
uniform mat4 img_to_eye_curr;

layout(r32f) uniform image2D tex_num_samples;

float sampleDistance = limit * 0.5f;
const float IsoValue = 0.0;
const int refinement_num = 4;

out vec4 out_Color;
out float gl_FragDepth;

float sample_volume(const vec3 pos) { return texture(volume_tsdf, pos).r; }

bool intersectBox(const vec3 origin, const vec3 dir, out float t0, out float t1)
{
    vec3 invR = 1.0 / dir;
    vec3 tbot = invR * (vec3(0.0) - origin);
    vec3 ttop = invR * (vec3(1.0) - origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    t1 = min(t.x, t.y);
    return t0 <= t1;
}

vec3 screenToVol(vec3 frag_coord)
{
    vec4 position_curr = img_to_eye_curr * vec4(frag_coord, 1.0);
    vec4 position_curr_es = vec4(position_curr.xyz / position_curr.w, 1.0);
    vec4 position_curr_ws = inverse(gl_ModelViewMatrix) * position_curr_es;
    vec3 position_vol = (inverse(vol_to_world) * position_curr_ws).xyz;
    return position_vol;
}

vec4 getStartPos(ivec2 coords)
{
    vec3 depthMinMax = texelFetch(depth_peels, coords, 0).rgb;
    // if closest back face is closest face -> front face culled
    depthMinMax.r = (depthMinMax.r >= depthMinMax.b) ? gl_DepthRange.near : depthMinMax.r;
    vec3 pos_front = screenToVol(vec3(gl_FragCoord.xy - viewport_offset, depthMinMax.r));
    vec3 pos_back = screenToVol(vec3(gl_FragCoord.xy - viewport_offset, -depthMinMax.g));
    // no valid closest face found
    pos_back = (depthMinMax.r >= 1.0) ? pos_front : pos_back;
    return vec4(pos_front, distance(pos_front, pos_back));
}

void writeNumSamples(uint num_samples)
{
    float samples = float(num_samples) * 0.0027;
    imageStore(tex_num_samples, ivec2(gl_FragCoord.xy), vec4(samples, 0.0, 0.0, 0.0));
}

void main()
{
    // multiply with dimensions to scale direction by dimension relation
    vec3 sampleStep = normalize(pass_Position - CameraPos) * sampleDistance;

    uint max_num_samples = 0u;
    vec3 sample_pos = vec3(0.0);

    // get ray beginning in volume cube
    float t0, t1 = 0.0;
    bool is_t0 = intersectBox(CameraPos, sampleStep, t0, t1);
    float t_near = (is_t0 ? t0 : t1);
    // if camera is within cube, start from camera, else move inside a little
    t_near = (t_near < 0.0 ? 0.0 : t_near);
    float t_far = (is_t0 ? t1 : t0);

    sample_pos = CameraPos + sampleStep * t_near;
    max_num_samples = uint(ceil(abs(t_far - t_near)));

    // initial sample is assumed to be outside the object
    float prev_density = -limit;

    uint num_samples = 0u;
    while(num_samples < max_num_samples)
    {
        num_samples += 1u;
        // get sample
        float density = sample_volume(sample_pos);

        // check if cell is inside contour
        if(density > IsoValue)
        {
            // approximate ray-cell intersection
            sample_pos = (sample_pos - sampleStep) - sampleStep * (prev_density / (density - prev_density));

            out_Color = vec4(density, density, density, 1.0f);
            vec3 view_pos = (gl_ModelViewMatrix * vol_to_world * vec4(sample_pos, 1.0)).xyz;
            gl_FragDepth = (gl_ProjectionMatrix[2].z *view_pos.z + gl_ProjectionMatrix[3].z) / -view_pos.z * 0.5f + 0.5f;

            writeNumSamples(num_samples);
            return;
        }

        prev_density = density;
        sample_pos += sampleStep;
    }

    // no surface found
    writeNumSamples(num_samples);
    discard;
}