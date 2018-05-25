#ifndef RECON_MC_HPP
#define RECON_MC_HPP

#include "recon_integration.hpp"
#include "reconstruction.hpp"
#include "view.hpp"
#include "view_lod.hpp"
#include "volume_sampler.hpp"

#include <globjects/base/ref_ptr.h>

#include <algorithm>
#include <chrono>
#include <iostream>

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

namespace globjects
{
class Texture;
class Program;
class Framebuffer;
}

#include <memory>
#include <atomic>

namespace kinect
{
class ReconPerformanceCapture : public Reconstruction
{
  public:
    ReconPerformanceCapture(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbo, float limit, float size);
    ~ReconPerformanceCapture();

    void draw() override;
    void drawF() override;
    void integrate();
    void setVoxelSize(float size);
    void setTsdfLimit(float limit);
    void setUseBricks(bool active);
    void setDrawBricks(bool active);
    void setBrickSize(float limit);

    float occupiedRatio() const;
    float getBrickSize() const;

    void clearOccupiedBricks() const;
    void updateOccupiedBricks();
    void setMinVoxelsPerBrick(unsigned i);
    void drawOccupiedBricks() const;

    static int TRI_TABLE[4096];

  private:
    void divideBox();
    void init(float d, float d1);

    globjects::Buffer *_tri_table_buffer, *_uv_counter_buffer, *_buffer_bricks, *_buffer_occupied, *_mesh_feedback_buffer, *_vertex_counter_buffer;
    globjects::TransformFeedback *_transformFeedback;
    glm::uvec3 _res_volume, _res_bricks;

    VolumeSampler *_sampler;

    glm::fmat4 _mat_vol_to_world;

    globjects::Program *_program, *_progra_integration, *_progra_solid, *_progra_bricks;
    globjects::Texture *_volume_tsdf;

    std::vector<brick> _bricks;
    std::vector<unsigned> _active_bricks;
    std::vector<unsigned> _bricks_occupied;

    float _limit;
    float _voxel_size;
    float _brick_size;
    bool _use_bricks;
    bool _draw_bricks;
    float _ratio_occupied;
    unsigned _min_voxels_per_brick;

    bool _capture_mesh = true;
    std::atomic<uint> _frame_number;

    struct struct_vertex
    {
        GLuint _index;
        glm::vec3 _position;
        glm::vec3 _normal;
        float _dummy;
    };
};
}

#endif // #ifndef RECON_MC_HPP