#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

#include </bricks.glsl>

layout(points) in;
layout(points, max_vertices = 1) out;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

in vec3 geo_Position[];
in uint geo_Id[];

out vec3 pass_Position;

#include </inc_bbox_test.glsl>

void main()
{
    uvec3 index = index_3d(geo_Id[0]);
    if(!brick_occupied(get_id(index)))
    {
        return;
    }

    pass_Position = geo_Position[0];
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(geo_Position[0], 1.0);
    EmitVertex();
    EndPrimitive();
}