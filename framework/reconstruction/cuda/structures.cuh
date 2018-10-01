#ifndef RECON_PC_CUDA_STRUCTURES
#define RECON_PC_CUDA_STRUCTURES

const unsigned ED_COMPONENT_COUNT = 6u;

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

#ifdef __CUDACC__
#define CUDA_ALIGN_4 __align__(4)
#else
#define CUDA_ALIGN_4
#endif

// #define VERBOSE
// #define DEBUG_NANS

// #define UNIT_TEST_NRA
// #define UNIT_TEST_REF_WARP
// #define UNIT_TEST_ALIGNMENT_ERROR_MAP
// #define UNIT_TEST_ERROR_INFORMED_BLENDING_IDENTICAL
// #define UNIT_TEST_ERROR_INFORMED_BLENDING_COMPLEMENTARY

#define PIPELINE_DEBUG_TEXTURE_SILHOUETTES
#define PIPELINE_DEBUG_TEXTURE_ALIGNMENT_ERROR
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

#define MAX_REFERENCE_VERTICES 65536

#define EVALUATE_DATA
#define EVALUATE_VISUAL_HULL
#define EVALUATE_ED_REGULARIZATION
#define EVALUATE_CORRESPONDENCE_FIELD

#define ED_NODES_ROBUSTIFY
#define FAST_QUAT_OPS
// #define JTJ_HESSIAN_DIAG

#define REJECT_MISALIGNED

// #define DEBUG_JTJ
// #define DEBUG_JTJ_COO
// #define DEBUG_JTJ_DENSE
// #define DEBUG_JTJ_PUSH_ORDERED_INTEGERS
// #define DEBUG_JTF
// #define DEBUG_H
// #define DEBUG_CONVERGENCE

#define SOLVER_DIRECT_QR
// #define SOLVER_DIRECT_CHOL
// #define SOLVER_CG
// #define SOLVER_PCG

enum class IsoSurfaceVolume
{
    Data,
    Reference,
    WarpedReference,
    Fused
};

struct Configuration
{
    // bool verbose = false;
    // bool debug_nan = false;

    int frame = 0;

    int reset_frame_count = 2;

    bool use_bricks = true;
    bool draw_bricks = false;

    bool debug_texture_silhouettes = false;
    bool debug_texture_alignment_error = false;
    bool debug_correspondence_field = false;
    bool debug_reference_volume = false;
    bool debug_reference_mesh = false;
    bool debug_ed_sampling = false;
    bool debug_sorted_vertices = false;
    bool debug_sorted_vertices_connections = false;
    bool debug_gradient_field = false;
    bool debug_warped_reference_volume_surface = false;

    int debug_sorted_vertices_mode = 0;
    int debug_sorted_vertices_traces = 0;

    bool pipeline_preprocess_textures = true;
    bool pipeline_sample = true;
    bool pipeline_correspondence = true;
    bool pipeline_align = true;
    bool pipeline_fuse = true;

    int textures_silhouettes_iterations = 100;

    float weight_data = 1.f;
    float weight_hull = 0.25f;
    float weight_correspondence = 0.9f;
    float weight_regularization = 0.01f;

    float solver_mu = 1000.f;
    float solver_mu_step = 50.f;
    int solver_lma_max_iter = 10;
    int solver_cg_steps = 12;

    float rejection_threshold = 0.01f;

    double time_copy_reference = 0.;
    double time_sample_ed = 0.;
    double time_preprocess = 0.;
    double time_correspondence = 0.;
    double time_nra = 0.;
    double time_fuse = 0.;
};

struct struct_native_handles
{
    unsigned int buffer_bricks;
    unsigned int buffer_occupied;

    unsigned int buffer_ed_nodes_debug;
    unsigned int buffer_sorted_vertices_debug;

    unsigned int volume_tsdf_data;
    unsigned int volume_tsdf_ref_grad;

    unsigned int pbo_kinect_depths;
    unsigned int pbo_kinect_normals;
    unsigned int pbo_kinect_silhouettes;
    unsigned int pbo_opticflow;

    unsigned int pbo_kinect_silhouettes_debug;
    unsigned int pbo_kinect_alignment_error_debug;

    unsigned int pbo_cv_xyz_inv[4];
    unsigned int pbo_cv_xyz[4];

    unsigned int posvbo;
    unsigned int normalvbo;
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

    glm::fmat4 mc_2_norm{1.f};
};

struct CUDA_ALIGN_8 struct_vertex
{
    glm::fvec3 position;
    unsigned int brick_id;
    glm::fvec3 normal;
    unsigned int ed_cell_id;
};

struct CUDA_ALIGN_8 struct_projection
{
    glm::vec2 projection[4];
};

struct CUDA_ALIGN_8 struct_ed_node_debug
{
    glm::fvec3 position;
    unsigned int brick_id;
    glm::fvec3 translation;
    unsigned int ed_cell_id;
    glm::fquat affine;

    unsigned int vx_offset;
    unsigned int vx_length;
    float misalignment_error;
    float data_term;
    float hull_term;
    float correspondence_term;
    float regularization_term;
    int pad;
};

struct CUDA_ALIGN_8 struct_ed_node
{
    float translation[3];
    float rotation[3];
};

struct CUDA_ALIGN_8 struct_ed_meta_entry
{
    glm::fvec3 position;
    unsigned int brick_id;
    unsigned int ed_cell_id;
    unsigned long long int vx_offset;
    unsigned int vx_length;
    float misalignment_error;
    float data_term;
    float hull_term;
    float correspondence_term;
    float regularization_term;
    int pad;
};

struct CUDA_ALIGN_8 struct_correspondence
{
    glm::fvec3 previous;
    unsigned int layer;
    glm::fvec3 current;
    unsigned int cell_id;
    glm::vec2 previous_proj;
    glm::vec2 current_proj;
};

#endif // RECON_PC_CUDA_STRUCTURES