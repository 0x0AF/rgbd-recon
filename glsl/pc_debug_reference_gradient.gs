#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_shader_atomic_counters : enable
#extension GL_ARB_uniform_buffer_object : enable

#include </bricks.glsl>

layout(points) in;
layout(line_strip, max_vertices = 2) out;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 vol_to_world;

uniform sampler3D volume_grad;

in vec3 geo_Position[];
in uint geo_Id[];

out vec3 grad;

#include </inc_bbox_test.glsl>

void main()
{
    uvec3 index = index_3d(geo_Id[0]);
    if(!brick_occupied(get_id(index)))
    {
        return;
    }

    grad = texture(volume_grad, geo_Position[0]).rgb * 0.001f;

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(geo_Position[0], 1.0);
    EmitVertex();

    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(geo_Position[0] + grad, 1.0);
    EmitVertex();

    EndPrimitive();
}