#ifndef RECON_PC_HPP
#define RECON_PC_HPP

#include "recon_integration.hpp"
#include "reconstruction.hpp"
#include "view.hpp"
#include "view_lod.hpp"
#include "volume_sampler.hpp"
#include <LocalKinectArray.h>
#include <TextureArray.h>

#include <globjects/base/ref_ptr.h>

#include <algorithm>
#include <chrono>
#include <iostream>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtx/transform.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>

#include <glbinding-aux/ContextInfo.h>
#include <glbinding-aux/types_to_string.h>
#include <glbinding/Version.h>
#include <glbinding/gl/gl.h>

#include <globjects/base/File.h>
#include <globjects/globjects.h>
#include <globjects/logging.h>

#include <globjects/Buffer.h>
#include <globjects/Program.h>
#include <globjects/Query.h>
#include <globjects/Shader.h>
#include <globjects/TransformFeedback.h>
#include <globjects/Uniform.h>
#include <globjects/VertexArray.h>
#include <globjects/VertexAttributeBinding.h>

#include <tinyply.h>

#include <reconstruction/cuda/structures.cuh>

namespace globjects
{
class Texture;
class Program;
class Framebuffer;
} // namespace globjects

#include <atomic>
#include <memory>

namespace kinect
{
class ReconPerformanceCapture : public Reconstruction
{
  public:
    ReconPerformanceCapture(LocalKinectArray &nka, CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbo, float limit, float size, float ed_size);
    ~ReconPerformanceCapture();

    void pause(bool pause);

    void draw() override;
    void drawF() override;
    void drawComparison() override;
    void integrate_data_frame();

    float occupiedRatio() const;
    float getBrickSize() const;

    void setMinVoxelsPerBrick(unsigned i);
    void setTsdfLimit(float limit);

    void clearOccupiedBricks() const;
    void updateOccupiedBricks();
    void drawOccupiedBricks() const;

    static int TRI_TABLE[4096];
    static std::string TIMER_DATA_VOLUME_INTEGRATION, TIMER_REFERENCE_MESH_EXTRACTION, TIMER_DATA_MESH_DRAW;

    Configuration _conf;

  private:
    LocalKinectArray *_nka = nullptr;

    struct_native_handles _native_handles;
    struct_measures _measures;

    globjects::Buffer *_buffer_bricks, *_buffer_occupied;

    globjects::ref_ptr<globjects::Buffer> _buffer_pbo_cv_xyz[4], _buffer_pbo_cv_xyz_inv[4];

    glm::uvec3 _res_volume, _res_bricks;

    VolumeSampler *_sampler;

    glm::fmat4 _mat_vol_to_world, _mat_world_to_vol;

    globjects::Program *_program_pc_fast_mc, *_program_integration, *_program_solid, *_program_bricks, *_program_pc_debug_tsdf;

    GLuint _volume_tsdf_data, _volume_tsdf_ref_grad;

    globjects::VertexArray *_vao_fast_mc, *_vao_debug, *_vao_fsquad_debug;
    globjects::Buffer *_buffer_fast_mc_pos, *_buffer_fast_mc_normal, *_buffer_debug, *_buffer_fsquad_debug;
    std::vector<glm::fvec3> _vec_debug;

    globjects::Texture *_texture2darray_debug;

    globjects::Buffer *_buffer_ed_nodes_debug, *_buffer_sorted_vertices_debug, *_buffer_pbo_silhouettes_debug, *_buffer_pbo_alignment_debug;
    globjects::Program *_program_pc_debug_correspondences, *_program_pc_debug_textures, *_program_pc_debug_opticflow, *_program_pc_debug_draw_ref, *_program_pc_debug_draw_ref_grad, *_program_pc_debug_sorted_vertices,
        *_program_pc_debug_sorted_vertices_connections, *_program_pc_debug_ed_sampling, *_program_pc_debug_reference;

    std::vector<brick> _bricks;
    std::vector<unsigned> _active_bricks;
    std::vector<unsigned> _bricks_occupied;

    bool _is_paused = false;

    float _limit;
    float _voxel_size;
    float _ed_cell_size;
    float _brick_size;
    float _ratio_occupied;
    unsigned _min_voxels_per_brick;

    std::atomic<uint64_t> _frame_number;

    void init(float limit, float size, float ed_cell_size);
    void init_shaders();
    void divideBox();
    void extract_reference_mesh();
    void draw_data();
    void draw_fused();
    void draw_debug_reference_volume();
    void draw_debug_reference_volume_warped();
    void draw_debug_reference_gradient();
    void draw_debug_reference_mesh();
    void draw_debug_ed_sampling();
    void draw_debug_sorted_vertices();
    void draw_debug_texture();

    // privatized temporarily
    void setVoxelSize(float size);
    void setBrickSize(float limit);
};

}; // namespace kinect

#endif // #ifndef RECON_PC_HPP