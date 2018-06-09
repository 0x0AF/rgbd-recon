#ifndef RECON_PC_CUDA
#define RECON_PC_CUDA

struct struct_native_handles
{
  GLuint buffer_bricks;
  GLuint buffer_occupied;

  GLuint buffer_vertex_counter;
  GLuint buffer_reference_vertices;

  GLuint volume_tsdf_data;

  GLuint array2d_kinect_depths;
};

struct struct_vertex
{
  glm::vec3 position;
  unsigned int brick_id; // TODO: fill at extraction time
  glm::vec3 normal;
  unsigned int ed_cell_id;
};

struct struct_ed_node
{
  glm::vec3 position{-1.f};
  glm::quat affine;
  glm::vec3 translation;
};

#endif // RECON_PC_CUDA