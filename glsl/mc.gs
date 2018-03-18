#version 420

#extension GL_ARB_shading_language_include : require
#extension GL_ARB_shader_storage_buffer_object : require

layout(points) in;
layout(points, max_vertices = 1) out;

// in vec3 pass_Position[];

uniform mat4 gl_ModelViewMatrix;
uniform mat4 gl_ProjectionMatrix;
uniform mat4 gl_NormalMatrix;

uniform mat4 NormalMatrix;
uniform mat4 vol_to_world;

uniform sampler3D volume_tsdf;

float sample_volume(const vec3 pos) { return texture(volume_tsdf, pos).r; }

void main() {
  for(uint i = 0u; i < gl_in.length; ++i) {
    gl_Position = gl_in[i].gl_Position;

    //if(sample_volume(gl_Position) > 0.1){
        EmitVertex();
    //}
  }
}