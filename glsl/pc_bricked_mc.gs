#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require

#include </bricks.glsl>

#include </mc.glsl>

layout(points) in;
layout(triangle_strip, max_vertices = 36) out;

in vec3 geo_Position[];
in uint geo_Id[];

out uint pass_Index;
out vec3 pass_Normal;
out vec3 pass_Position;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform float size_voxel;

uniform mat4 vol_to_world;
uniform sampler3D volume_tsdf;

float sample_volume(const vec3 position) { return texture(volume_tsdf, position).r; }

void sample_cube(const vec3 pos, inout float cube[8])
{
    for(uint i = 0u; i < 8; i++)
    {
        cube[i] = sample_volume(pos + vertex_offsets[i] * size_voxel);
    }
}

void make_face(vec3 a, vec3 b, vec3 c)
{
    vec3 u = b - a;
    vec3 v = c - a;

    pass_Normal.x = u.y * v.z - u.z * v.y;
    pass_Normal.y = u.z * v.x - u.x * v.z;
    pass_Normal.z = u.x * v.y - u.y * v.x;

    pass_Normal = normalize(pass_Normal);
    pass_Index = atomicCounterIncrement(face_counter);

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
    vec3 center = geo_Position[0] + size_voxel * vec3(-1., -1., -1.);

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

                edge_vertices[i] = center + (vertex_offsets[edge_connections[i].x] + offset * edge_directions[i]) * size_voxel;
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