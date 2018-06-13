#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

#include </bricks.glsl>

#include </mc.glsl>

// #define PASS_NORMALS

layout(points) in;
layout(triangle_strip, max_vertices = 36) out;

in vec3 geo_Position[];
in uint geo_Id[];

out vec3 pass_Position;
#ifdef PASS_NORMALS
out vec3 pass_Normal;
#endif

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform float size_voxel;
uniform uvec3 res_tsdf;

uniform mat4 vol_to_world;
uniform sampler3D volume_tsdf;

float sample_volume(const vec3 position) { return texture(volume_tsdf, position).r; }

void sample_cube(const vec3 pos, inout float cube[8])
{
    for(uint i = 0u; i < 8; i++)
    {
        cube[i] = sample_volume(pos + vertex_offsets[i] * size_voxel / res_tsdf * max(res_tsdf.x, max(res_tsdf.y, res_tsdf.z)));
    }
}

void make_face(vec3 a, vec3 b, vec3 c)
{
#ifdef PASS_NORMALS
    vec3 normal;

    vec3 u = b - a;
    vec3 v = c - a;

    normal.x = u.y * v.z - u.z * v.y;
    normal.y = u.z * v.x - u.x * v.z;
    normal.z = u.x * v.y - u.y * v.x;

    normal = normalize(normal);
    pass_Normal = normal;
#endif

    pass_Position = c;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(c, 1.0);

    EmitVertex();

    pass_Position = b;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(b, 1.0);

    EmitVertex();

    pass_Position = a;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(a, 1.0);

    EmitVertex();

    EndPrimitive();
}

void main()
{
    uvec3 index = index_3d(geo_Id[0]);
    if(!brick_occupied(get_id(index)))
    {
        return;
    }

    // MC START

    float cube[8] = float[8](0., 0., 0., 0., 0., 0., 0., 0.);
    vec3 center = geo_Position[0] + size_voxel * vec3(-1., -1., -1.) / res_tsdf * max(res_tsdf.x, max(res_tsdf.y, res_tsdf.z));

    sample_cube(center, cube);

    int i = 0;
    int flag_storage = 0;
    vec3 edge_vertices[12];

    for(i = 0; i < 8; i++)
    {
        if(cube[i] > 0.f)
        {
            flag_storage = flag_storage | (1 << i);
        }
    }

    if(flag_storage != 0)
    {
        int edge_flags = cube_edge_flags[flag_storage];

        for(i = 0; i < 12; i++)
        {
            if((edge_flags & (1 << i)) != 0)
            {
                float offset = get_offset(cube[edge_connections[i].x], cube[edge_connections[i].y]);

                edge_vertices[i] = center + (vertex_offsets[edge_connections[i].x] + offset * edge_directions[i]) * size_voxel / res_tsdf * max(res_tsdf.x, max(res_tsdf.y, res_tsdf.z));
            }
        }

        for(i = 0; i < 5; i++)
        {
            int triplet_position = flag_storage * 16 + 3 * i;
            if(triangle_connections[triplet_position] != -1)
            {
                make_face(edge_vertices[triangle_connections[triplet_position]], edge_vertices[triangle_connections[triplet_position + 1]], edge_vertices[triangle_connections[triplet_position + 2]]);
            }
        }
    }

    // MC END
}