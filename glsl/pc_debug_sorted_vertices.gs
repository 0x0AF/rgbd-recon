#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

layout(points) in;
layout(points, max_vertices = 128) out;

struct EDNode
{
    vec3 position;
    uint brick_id;
    vec3 translation;
    uint ed_cell_id;
    vec4 affine;

    uint vx_offset;
    uint vx_length;
    uint pad[2];
};

layout(std430, binding = 8) restrict buffer EDNodeBuffer { EDNode ed_nodes[]; };

struct Vertex
{
    vec3 position;
    uint brick_id;
    vec3 normal;
    uint ed_cell_id;
};

layout(std430, binding = 9) restrict buffer SortedVertexBuffer { Vertex vertices[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];

out vec3 pass_Position;
out float pass_BrickIdColor;
out float pass_EDCellIdColor;
out float pass_IsED;

#include </inc_bbox_test.glsl>

vec3 qtransform(vec4 q, vec3 v) { return v + 2.f * cross(cross(v, q.xyz) + q.w * v, q.xyz); }

void main()
{
    EDNode ed_node = ed_nodes[uint(geo_Position[0].x * 100000.f)];

    pass_Position = ed_node.position;
    pass_BrickIdColor = (ed_node.brick_id % 256) / 256.0f;
    pass_EDCellIdColor = (ed_node.ed_cell_id % 27) / 27.0f;
    pass_IsED = 1.f;

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(ed_node.position + bbox_min, 1.0);
    gl_PointSize = 3.0f;

    EmitVertex();
    EndPrimitive();

    for(uint i = 0; i < ed_node.vx_length; i++){
        Vertex vertex = vertices[ed_node.vx_offset + i];

        pass_Position = vertex.position;
        pass_BrickIdColor = (vertex.brick_id % 256) / 256.0f;
        pass_EDCellIdColor = (vertex.ed_cell_id % 27) / 27.0f;
        pass_IsED = -1.f;

        vec3 dist = ed_node.position - vertex.position;
        float skinning_weight = 1.f; // exp(length(dist) * length(dist) / 0.00005f); // TODO: generalize based on voxel size
        vec3 warped_position = skinning_weight * (qtransform(ed_node.affine, dist) + ed_node.position + ed_node.translation);

        gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vec4(warped_position + bbox_min, 1.0);
        gl_PointSize = 2.0f;

        EmitVertex();
        EndPrimitive();
    }
}