#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 10u;

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE __device__ __host__
#else
#define CUDA_HOST_DEVICE
#endif

#ifdef __CUDACC__
#define CUDA_ALIGN_8 __align__(8)
#else
#define CUDA_ALIGN_8
#endif

// #define VERBOSE
#define DEBUG_NANS

#define SIFT_MAX_CORRESPONDENCES 2048
#define SIFT_MINIMAL_SCORE 0.95f
#define SIFT_FILTER_MAX_MOTION 0.2f
#define SIFT_USE_COLOR
// #define SIFT_USE_SILHOUETTES

#define SIFT_OCTAVES 5
#define SIFT_BLUR 0.f
#define SIFT_THRESHOLD 0.01f
#define SIFT_LOWEST_SCALE 0.01f
#define SIFT_UPSCALE false

#define EVALUATE_DATA
// #define EVALUATE_VISUAL_HULL
// #define EVALUATE_ED_REGULARIZATION

// #define ED_NODES_ROBUSTIFY
// #define FAST_QUAT_OPS

// #define DEBUG_JTJ
// #define DEBUG_JTJ_COO
// #define DEBUG_JTJ_DENSE
// #define DEBUG_JTJ_PUSH_ORDERED_INTEGERS
// #define DEBUG_JTF
// #define DEBUG_H

#define SOLVER_DIRECT_CHOL
// #define SOLVER_DIRECT_QR
// #define SOLVER_PCG

struct struct_native_handles
{
    unsigned int buffer_bricks;
    unsigned int buffer_occupied;

    unsigned int buffer_vertex_counter;
    unsigned int buffer_reference_vertices;
    unsigned int buffer_ed_nodes_debug;
    unsigned int buffer_sorted_vertices_debug;
    unsigned int buffer_correspondences_debug;

    unsigned int volume_tsdf_data;
    unsigned int volume_tsdf_ref;

    unsigned int pbo_kinect_rgbs;
    unsigned int pbo_kinect_depths;
    unsigned int pbo_kinect_silhouettes;

    unsigned int pbo_kinect_silhouettes_debug;
    unsigned int pbo_kinect_intens_debug;

    unsigned int pbo_cv_xyz_inv[4];
    unsigned int pbo_cv_xyz[4];
};

struct struct_measures
{
    float size_voxel{0.f};
    float size_ed_cell{0.f};
    float size_brick{0.f};

    float sigma{0.f};

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

struct CUDA_ALIGN_8 struct_vertex
{
    glm::vec3 position;
    unsigned int brick_id; // TODO: fill at reference extraction time
    glm::vec3 normal;
    unsigned int ed_cell_id;
};

struct CUDA_ALIGN_8 struct_ed_node_debug
{
    glm::vec3 position;
    unsigned int brick_id;
    glm::vec3 translation;
    unsigned int ed_cell_id;
    glm::quat affine;

    unsigned int vx_offset;
    unsigned int vx_length;
    unsigned int pad[2];
};

struct CUDA_ALIGN_8 struct_ed_node
{
    glm::vec3 position;
    glm::quat affine;
    glm::vec3 translation;
};

struct CUDA_ALIGN_8 struct_ed_meta_entry
{
    unsigned int brick_id;
    unsigned int ed_cell_id;
    unsigned long long int vx_offset;
    unsigned int vx_length;
    bool rejected;
    int neighbors[27];
};

struct CUDA_ALIGN_8 struct_correspondence
{
    glm::vec3 previous;
    unsigned int pad1;
    glm::vec3 current;
    unsigned int pad2;
};

#endif // RECON_PC_CUDA_STRUCTURES