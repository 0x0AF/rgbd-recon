#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

layout(points) in;
layout(points, max_vertices = 1) out;

struct Vertex
{
    vec3 position;
    uint brick_id;
    vec3 normal;
    uint pad_2;
};

layout(std430, binding = 7) restrict buffer ReferenceMeshVertexBuffer { Vertex vertices[]; };

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];

out vec3 pass_Position;
out vec3 pass_Normal;

#include </inc_bbox_test.glsl>

void main()
{
    Vertex vertex = vertices[uint(geo_Position[0].x * 1000000.f)];

    pass_Position = vertex.position;
    pass_Normal = vertex.normal;

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(pass_Position, 1.0);
    gl_PointSize = 2.5f;

    EmitVertex();
    EndPrimitive();
}