#ifndef RECON_PC_CUDA
#define RECON_PC_CUDA

struct struct_native_handles
{
  GLuint buffer_bricks;
  GLuint buffer_occupied;

  GLuint buffer_vertex_counter;
  GLuint buffer_reference_vertices;

  GLuint volume_tsdf_reference;
  GLuint volume_tsdf_data;

  GLuint array2d_kinect_depths;
};

struct struct_vertex
{
  glm::vec3 position;
  int pad_1;
  glm::vec3 normal;
  int pad_2;
};

struct struct_ed_node
{
  bool set;
  glm::vec3 position;
  glm::mat3 affine;
  glm::vec3 translation;
};

#endif // RECON_PC_CUDA