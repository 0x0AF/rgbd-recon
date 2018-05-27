#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

#include </bricks.glsl>

#include </mc.glsl>

layout(points) in;
layout(triangle_strip, max_vertices = 36) out;

layout(binding = 6) uniform atomic_uint face_counter;
layout(binding = 7) uniform atomic_uint vertex_counter;

struct Vertex
{
    vec3 position;
    vec3 normal;
};

layout(std430, binding = 8) restrict buffer ReferenceMeshVertexBuffer
{
    Vertex vertices[];
};

layout(std430, binding = 9) restrict buffer ReferenceMeshFaceBuffer { uvec3 faces[]; };

in vec3 geo_Position[];
in uint geo_Id[];

out vec3 pass_Position;

uniform float size_voxel;
uniform sampler3D volume_tsdf;

float sample_volume(const vec3 position) { return texture(volume_tsdf, position).r; }

void sample_cube(const vec3 pos, inout float cube[8])
{
    for(uint i = 0u; i < 8; i++)
    {
        cube[i] = sample_volume(pos + vertex_offsets[i] * size_voxel);
    }
}

void store_face(uvec3 face, inout vec3 edge_vertices[12], inout uint edge_vertices_indices[12], inout vec3 edge_vertices_normals[12])
{
    uint face_buffer_index = atomicCounterIncrement(face_counter);

    vec3 normal;

    vec3 u = edge_vertices[face.y] - edge_vertices[face.x];
    vec3 v = edge_vertices[face.z] - edge_vertices[face.x];

    normal.x = u.y * v.z - u.z * v.y;
    normal.y = u.z * v.x - u.x * v.z;
    normal.z = u.x * v.y - u.y * v.x;

    normal = normalize(normal);

    faces[face_buffer_index] = uvec3(edge_vertices_indices[face.z], edge_vertices_indices[face.y], edge_vertices_indices[face.x]);

    edge_vertices_normals[face.x] = (edge_vertices_normals[face.x] + normal) / 2.f;
    edge_vertices_normals[face.y] = (edge_vertices_normals[face.y] + normal) / 2.f;
    edge_vertices_normals[face.z] = (edge_vertices_normals[face.z] + normal) / 2.f;
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

    uint edge_vertices_indices[12];
    vec3 edge_vertices_normals[12];

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

                edge_vertices_indices[i] = atomicCounterIncrement(vertex_counter);
                vertices[edge_vertices_indices[i]].position = edge_vertices[i];
            }
        }

        for(i = 0; i < 5; i++)
        {
            int triplet_position = flag_storage * 16 + 3 * i;
            if(triangle_connections[triplet_position] != -1)
            {
                uvec3 face = uvec3(triangle_connections[triplet_position], triangle_connections[triplet_position + 1], triangle_connections[triplet_position + 2]);
                store_face(face, edge_vertices, edge_vertices_indices, edge_vertices_normals);
            }
        }

        for(i = 0; i < 12; i++)
        {
            if((edge_flags & (1 << i)) != 0)
            {
                vertices[edge_vertices_indices[i]].normal = edge_vertices_normals[i];
            }
        }
    }

    // MC END
}