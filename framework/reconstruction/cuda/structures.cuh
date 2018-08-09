#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 10u;

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

#define __CUDACC__
#include <device_functions.h>

#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_cmath.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#endif

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
// #define DEBUG_NANS

// #define PIPELINE_DEBUG_TEXTURE_COLORS
// #define PIPELINE_DEBUG_TEXTURE_DEPTHS
#define PIPELINE_DEBUG_TEXTURE_SILHOUETTES
#define PIPELINE_DEBUG_CORRESPONDENCE_FIELD
#define PIPELINE_DEBUG_REFERENCE_VOLUME
#define PIPELINE_DEBUG_REFERENCE_MESH
#define PIPELINE_DEBUG_ED_SAMPLING
#define PIPELINE_DEBUG_SORTED_VERTICES
#define PIPELINE_DEBUG_SORTED_VERTICES_CONNECTIONS
#define PIPELINE_DEBUG_GRADIENT_FIELD
#define PIPELINE_DEBUG_WARPED_REFERENCE_VOLUME

#define PIPELINE_TEXTURES_PREPROCESS
#define PIPELINE_SAMPLE
#define PIPELINE_CORRESPONDENCE
#define PIPELINE_ALIGN
#define PIPELINE_FUSE

#define MAX_REFERENCE_VERTICES 262144

#define SIFT_MAX_CORRESPONDENCES 2048
#define SIFT_USE_COLOR

/// #define SIFT_MINIMAL_SCORE 0.95f
/// #define SIFT_FILTER_MAX_MOTION 0.1f

/// #define SIFT_OCTAVES 5
/// #define SIFT_BLUR 0.f
/// #define SIFT_THRESHOLD 0.01f
/// #define SIFT_LOWEST_SCALE 0.01f
/// #define SIFT_UPSCALE false

/// #define WEIGHT_DATA 1.0f
/// #define WEIGHT_VISUAL_HULL 0.08f
/// #define WEIGHT_ED_REGULARIZATION 0.02f
/// #define WEIGHT_CORRESPONDENCE_FIELD 0.01f

#define EVALUATE_DATA
#define EVALUATE_VISUAL_HULL
#define EVALUATE_ED_REGULARIZATION
#define EVALUATE_CORRESPONDENCE_FIELD

#define ED_NODES_ROBUSTIFY
#define FAST_QUAT_OPS

#define REJECT_MISALIGNED

// #define DEBUG_JTJ
// #define DEBUG_JTJ_COO
// #define DEBUG_JTJ_DENSE
// #define DEBUG_JTJ_PUSH_ORDERED_INTEGERS
// #define DEBUG_JTF
// #define DEBUG_H

// #define SOLVER_DIRECT_CHOL
// #define SOLVER_DIRECT_QR
#define SOLVER_CG
// #define SOLVER_PCG

struct Configuration
{
    // bool verbose = false;
    // bool debug_nan = false;

    int reset_frame_count = 64;

    bool debug_wipe_data = false;
    bool use_bricks = true;
    bool draw_bricks = false;

    bool debug_texture_silhouettes = false;
    bool debug_correspondence_field = false;
    bool debug_reference_volume = false;
    bool debug_reference_mesh = false;
    bool debug_ed_sampling = false;
    bool debug_sorted_vertices = false;
    bool debug_sorted_vertices_connections = false;
    bool debug_gradient_field = false;
    bool debug_warped_reference_volume_value = false;
    bool debug_warped_reference_volume_surface = false;

    bool pipeline_preprocess_textures = false;
    bool pipeline_sample = false;
    bool pipeline_correspondence = false;
    bool pipeline_align = false;
    bool pipeline_fuse = false;

    int textures_silhouettes_iterations = 4;
    int textures_SIFT_octaves = 5;
    float textures_SIFT_blur = 0.2f;
    float textures_SIFT_threshold = 0.025f;
    float textures_SIFT_lowest_scale = 0.01f;
    bool textures_SIFT_upscale = false;
    float textures_SIFT_min_score = 0.95f;
    float textures_SIFT_max_motion = 0.1f;

    float weight_data = 1.0f;
    float weight_hull = 0.08f;
    float weight_correspondence = 0.02f;
    float weight_regularization = 0.01f;

    float solver_mu = 0.2f;
    float solver_mu_step = 0.05f;
    int solver_lma_steps = 4;
    int solver_cg_steps = 4;
};

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
    unsigned int volume_tsdf_ref_grad;

    unsigned int pbo_kinect_rgbs;
    unsigned int pbo_kinect_depths;
    unsigned int pbo_kinect_silhouettes;

    unsigned int pbo_kinect_silhouettes_debug;
    unsigned int pbo_tsdf_ref_warped_debug;

    unsigned int pbo_cv_xyz_inv[4];
    unsigned int pbo_cv_xyz[4];
};

struct struct_measures
{
    float size_voxel{0.f};
    float size_ed_cell{0.f};
    float size_brick{0.f};
    unsigned int size_depth_cell{0u};

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
    glm::uvec2 depth_cell_res{0u};
    unsigned int num_depth_cells{0u};

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
    unsigned int brick_id;
    glm::vec3 normal;
    unsigned int ed_cell_id;
};

struct CUDA_ALIGN_8 struct_projection
{
    glm::vec2 projection[4];
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
    unsigned int layer;
    glm::vec3 current;
    unsigned int cell_id;
    glm::vec2 previous_proj;
    glm::vec2 current_proj;
};

struct CUDA_ALIGN_8 struct_depth_cell_meta
{
    unsigned int cp_offset;
    unsigned int cp_length;
};

#endif // RECON_PC_CUDA_STRUCTURES