#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 10u;

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __device__ __host__
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
    unsigned int buffer_ed_nodes_debug;
    unsigned int buffer_sorted_vertices_debug;

    unsigned int volume_tsdf_data;
    unsigned int volume_tsdf_ref;

    unsigned int texture_kinect_rgbs;
    unsigned int texture_kinect_depths;
    unsigned int texture_kinect_silhouettes;

    unsigned int volume_cv_xyz_inv[4];
    unsigned int volume_cv_xyz[4];
};

struct struct_measures
{
    float size_voxel{0.f};
    float size_ed_cell{0.f};
    float size_brick{0.f};

    /** Constant values strictly define the available relationships between resolutions, dimensions and sizes **/

    const unsigned int ed_cell_dim_voxels = 3u;
    const unsigned int brick_dim_ed_cells = 3u;
    const unsigned int brick_dim_voxels = 9u;

    const unsigned int ed_cell_num_voxels = 27u;
    const unsigned int brick_num_ed_cells = 27u; // implied in 27-neighborhood!
    const unsigned int brick_num_voxels = 729u;

    glm::uvec2 color_res{0u};
    glm::uvec2 depth_res{0u};

    glm::uvec3 data_volume_res{0u, 0u, 0u};
    glm::uvec3 data_volume_bricked_res{0u, 0u, 0u};
    unsigned int data_volume_num_bricks{0u};
    glm::uvec3 cv_xyz_res{0u, 0u, 0u};
    glm::uvec3 cv_xyz_inv_res{0u, 0u, 0u};

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

struct struct_ed_node_debug
{
    glm::vec3 position;
    unsigned int brick_id;
    glm::vec3 translation;
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