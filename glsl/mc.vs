#version 420

layout(location = 0) in vec3 in_Position;

uniform mat4 TextureMatrix;
uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;

uniform sampler2DArray kinect_depths;
uniform sampler2DArray kinect_qualities;
uniform sampler2DArray kinect_silhouettes;

uniform sampler3D[5] cv_xyz_inv;

uniform float limit;
uniform uint num_kinects;

uniform float iso;
uniform float size_voxel;

out vec3 in_pass_Position;
out float in_valid_vertex;

float sample_volume(const vec3 position)
{
    float weighted_tsd = limit;
    float total_weight = 0;
    for(uint i = 0u; i < num_kinects; ++i)
    {
        vec3 pos_calib = texture(cv_xyz_inv[i], position).xyz;
        float silhouette = texture(kinect_silhouettes, vec3(pos_calib.xy, float(i))).r;
        if(silhouette < 1.0f)
        {
            // no write yet -> voxel outside of surface
            if(weighted_tsd >= limit)
            {
                weighted_tsd = -limit;
                continue;
            }
        }
        float depth = texture(kinect_depths, vec3(pos_calib.xy, float(i))).r;
        float sdist = pos_calib.z - depth;
        if(sdist <= -limit)
        {
            weighted_tsd = -limit;
            // break;
        }
        else if(sdist >= limit)
        {
            // do nothing
        }
        else
        {
            float weight = texture(kinect_qualities, vec3(pos_calib.xy, float(i))).r;

            weighted_tsd = (weighted_tsd * total_weight + weight * sdist) / (total_weight + weight);
            total_weight += weight;
        }
    }

    return weighted_tsd;
}

bool check_bounds_3()
{
    float size_voxel_x2 = size_voxel * 2;
    float size_voxel_min = -size_voxel;

    bool bounds[8] = bool[8](0., 0., 0., 0., 0., 0., 0., 0.);

    bounds[0] = sample_volume(in_pass_Position + vec3(size_voxel_x2, size_voxel_x2, size_voxel_x2)) > iso;
    bounds[1] = sample_volume(in_pass_Position + vec3(size_voxel_x2, size_voxel_x2, size_voxel_min)) > iso;
    bounds[2] = sample_volume(in_pass_Position + vec3(size_voxel_x2, size_voxel_min, size_voxel_x2)) > iso;
    bounds[3] = sample_volume(in_pass_Position + vec3(size_voxel_min, size_voxel_x2, size_voxel_x2)) > iso;
    bounds[4] = sample_volume(in_pass_Position + vec3(size_voxel_x2, size_voxel_min, size_voxel_min)) > iso;
    bounds[5] = sample_volume(in_pass_Position + vec3(size_voxel_min, size_voxel_min, size_voxel_x2)) > iso;
    bounds[6] = sample_volume(in_pass_Position + vec3(size_voxel_min, size_voxel_x2, size_voxel_min)) > iso;
    bounds[7] = sample_volume(in_pass_Position + vec3(size_voxel_min, size_voxel_min, size_voxel_min)) > iso;

    bool some_inside = false;
    bool all_inside = true;

    for(uint i = 0u; i < 8; i++)
    {
        if(bounds[i])
        {
            some_inside = true;
        }
        else
        {
            all_inside = false;
            if(some_inside)
            {
                return true;
            }
        }
    }

    return some_inside && !all_inside;
}

void main()
{
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(in_Position, 1.0f);
    in_pass_Position = in_Position;
    in_valid_vertex = check_bounds_3() ? 1.0f : -1.0f;
}
