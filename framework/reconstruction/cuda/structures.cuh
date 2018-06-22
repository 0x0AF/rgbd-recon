#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 10u;

const unsigned ED_CELL_RES = 9u;
const unsigned ED_CELL_VOXEL_DIM = 2u;
const unsigned ED_CELL_VOXELS = 8u;

const unsigned BRICK_VOXEL_DIM = 18u;
const unsigned BRICK_VOXELS = 5832u;

const unsigned BRICK_RES_X = 8u;
const unsigned BRICK_RES_Y = 7u;
const unsigned BRICK_RES_Z = 8u;

const unsigned VOLUME_VOXEL_DIM_X = 141u;
const unsigned VOLUME_VOXEL_DIM_Y = 111u;
const unsigned VOLUME_VOXEL_DIM_Z = 130u;

struct struct_native_handles
{
    unsigned int buffer_bricks;
    unsigned int buffer_occupied;

    unsigned int buffer_vertex_counter;
    unsigned int buffer_reference_vertices;

    unsigned int volume_tsdf_data;

    unsigned int array2d_kinect_depths;

    unsigned int volume_cv_xyz_inv[4];
    unsigned int volume_cv_xyz[4];
};

struct struct_measures
{
    glm::uvec2 depth_resolution{0u};
    glm::fvec2 depth_limits[4];
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
    glm::vec3 position{0.f};
    glm::quat affine{glm::mat4(1.f)};
    glm::vec3 translation{0.f};
};

struct struct_ed_dense_index_entry
{
    unsigned int brick_id;
    unsigned int ed_cell_id;
    unsigned long long int vx_offset;
    unsigned int vx_length;
};

#endif // RECON_PC_CUDA_STRUCTURES