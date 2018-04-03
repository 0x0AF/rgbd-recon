#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require
#extension GL_ARB_uniform_buffer_object : enable

layout(points) in;
layout(triangle_strip, max_vertices = 120) out;

in vec3 in_pass_Position[];
in float in_valid_vertex[];

out vec3 pass_Position;

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform sampler2DArray kinect_depths;
uniform sampler2DArray kinect_qualities;
uniform sampler2DArray kinect_silhouettes;

uniform sampler3D[5] cv_xyz_inv;

uniform float limit;
uniform uint num_kinects;

uniform float iso;
uniform float size_voxel;

uniform mat4 vol_to_world;

int[] cube_edge_flags =
    int[](0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
          0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af,
          0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 0x5f0,
          0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56,
          0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5,
          0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 0xaf0, 0xbf9,
          0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a,
          0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
          0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 0xf00, 0xe09, 0xd03,
          0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000);

ivec2 edge_connections[] = ivec2[](ivec2(0, 1), ivec2(1, 2), ivec2(2, 3), ivec2(3, 0), ivec2(4, 5), ivec2(5, 6), ivec2(6, 7), ivec2(7, 4), ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7));

vec3 edge_directions[] = vec3[](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f),
                                vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, 0.0f, 1.0f));

layout(std430, binding = 5) buffer tri_table { int[] triangle_connections; };

vec3 vertex_offsets[] = vec3[](vec3(0, 0, 0), vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 1), vec3(1, 1, 1), vec3(0, 1, 1));

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

void sample_cube(const vec3 pos, inout float cube[8])
{
    for(uint i = 0u; i < 8; i++)
    {
        cube[i] = sample_volume(pos + vertex_offsets[i] * size_voxel);
    }
}

float get_offset(float v1, float v2)
{
    float delta = v2 - v1;
    return (delta == 0.0f) ? 0.5f : (iso - v1) / delta;
}

void make_face(vec3 a, vec3 b, vec3 c)
{
    pass_Position = a;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(a, 1.0);
    EmitVertex();

    pass_Position = b;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(b, 1.0);
    EmitVertex();

    pass_Position = c;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * vol_to_world * vec4(c, 1.0);
    EmitVertex();

    EndPrimitive();
}

void main()
{
    // MC START

    if(in_valid_vertex[0] < 0.f)
    {
        return;
    }

    for(int x = 0; x < 3; x++)
    {
        for(int y = 0; y < 3; y++)
        {
            for(int z = 0; z < 3; z++)
            {
                float cube[8] = float[8](0., 0., 0., 0., 0., 0., 0., 0.);

                vec3 center = in_pass_Position[0] + size_voxel * vec3(x - 1, y - 1, z - 1);

                sample_cube(center, cube);

                int i = 0;
                int flag_storage = 0;
                vec3 cube_vertices[12];

                for(i = 0; i < 8; i++)
                    if(cube[i] > iso)
                        flag_storage |= 1 << i;

                if(flag_storage != 0)
                {
                    int edgeFlags = cube_edge_flags[flag_storage];

                    for(i = 0; i < 12; i++)
                    {
                        if((edgeFlags & (1 << i)) != 0)
                        {
                            float offset = get_offset(cube[edge_connections[i].x], cube[edge_connections[i].y]);

                            cube_vertices[i] = center + (vertex_offsets[edge_connections[i].x] + offset * edge_directions[i]) * size_voxel;
                        }
                    }

                    for(i = 0; i < 5; i++)
                    {
                        if(triangle_connections[flag_storage * 16 + 3 * i] >= 0)
                        {
                            make_face(cube_vertices[triangle_connections[flag_storage * 16 + 3 * i + 0]], cube_vertices[triangle_connections[flag_storage * 16 + 3 * i + 1]],
                                      cube_vertices[triangle_connections[flag_storage * 16 + 3 * i + 2]]);
                        }
                    }
                }
            }
        }
    }

    // MC END

    //    if(sample_volume(in_pass_Position[0]) > iso)
    //    {
    //        make_face(in_pass_Position[0] + vertex_offset[0] / 256.0f, in_pass_Position[0] + vertex_offset[1] / 256.0f, in_pass_Position[0] + vertex_offset[2] / 256.0f);
    //    }
}