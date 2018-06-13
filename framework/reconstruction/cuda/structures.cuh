#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_CELL_RES = 1u;
const unsigned ED_CELL_VOXEL_DIM = 4u;
const unsigned ED_CELL_VOXELS = 64u;
const unsigned BRICK_VOXEL_DIM = 4u;
const unsigned BRICK_VOXELS = 64u;

const unsigned BRICK_RES_X = 6u;
const unsigned BRICK_RES_Y = 5u;
const unsigned BRICK_RES_Z = 6u;

const unsigned VOLUME_VOXEL_DIM_X = 71u;
const unsigned VOLUME_VOXEL_DIM_Y = 56u;
const unsigned VOLUME_VOXEL_DIM_Z = 65u;

struct struct_native_handles
{
  unsigned int buffer_bricks;
  unsigned int buffer_occupied;

  unsigned int buffer_vertex_counter;
  unsigned int buffer_reference_vertices;

  unsigned int volume_tsdf_data;

  unsigned int array2d_kinect_depths;

  unsigned int volume_cv_xyz_inv[4];
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
  glm::vec3 position;
  glm::quat affine;
  glm::vec3 translation;
};

#endif // RECON_PC_CUDA_STRUCTURES