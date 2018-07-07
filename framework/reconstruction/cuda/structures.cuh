#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 10u;

const unsigned ED_CELL_RES = 3u; // implied in 27-neighborhood!
const unsigned ED_CELL_VOXEL_DIM = 3u;
const unsigned ED_CELL_VOXELS = 9u;

const unsigned BRICK_VOXEL_DIM = 9u;
const unsigned BRICK_VOXELS = 729u;

const unsigned BRICK_RES_X = 16u;
const unsigned BRICK_RES_Y = 16u;
const unsigned BRICK_RES_Z = 16u;

const unsigned VOLUME_VOXEL_DIM_X = 141u;
const unsigned VOLUME_VOXEL_DIM_Y = 140u;
const unsigned VOLUME_VOXEL_DIM_Z = 140u;

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __host__ __device__
#else
#define CUDA_HOST_DEVICE
#endif

// #define DEBUG_NANS

struct struct_native_handles
{
    unsigned int buffer_bricks;
    unsigned int buffer_occupied;

    unsigned int buffer_vertex_counter;
    unsigned int buffer_reference_vertices;

    unsigned int volume_tsdf_data;

    unsigned int texture_kinect_rgbs;
    unsigned int texture_kinect_depths;
    unsigned int texture_kinect_silhouettes;

    unsigned int volume_cv_xyz_inv[4];
    unsigned int volume_cv_xyz[4];
};

struct struct_measures
{
    glm::uvec2 color_resolution{0u};
    glm::uvec2 depth_resolution{0u};
    glm::fvec2 depth_limits[4];
    glm::fvec3 bbox_translation{0.f, 0.f, 0.f};
    glm::fvec3 bbox_dimensions{0.f, 0.f, 0.f};
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

struct struct_vertex_weights
{
    float skinning_weights[27];
};

struct struct_ed_meta_entry
{
    unsigned int brick_id;
    unsigned int ed_cell_id;
    unsigned long long int vx_offset;
    unsigned int vx_length;
    bool rejected;
    int neighbors[27];
};

#endif // RECON_PC_CUDA_STRUCTURES