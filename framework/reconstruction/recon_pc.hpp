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
    static std::string TIMER_DATA_VOLUME_INTEGRATION, TIMER_REFERENCE_MESH_EXTRACTION, TIMER_DATA_MESH_DRAW;

  private:
    void divideBox();
    void init(float d, float d1);

    globjects::Buffer *_tri_table_buffer, *_buffer_bricks, *_buffer_occupied;
    globjects::Buffer *_buffer_vertex_counter, *_buffer_face_counter, *_buffer_reference_mesh_vertices, *_buffer_reference_mesh_faces;

    glm::uvec3 _res_volume, _res_bricks;

    VolumeSampler *_sampler;

    glm::fmat4 _mat_vol_to_world;

    globjects::Program *_program_pc_draw_data, *_program_pc_extract_reference, *_program_integration, *_program_solid, *_program_bricks;
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

    std::atomic<uint64_t> _frame_number;

    /**
     * Structures kept for reference, none of the following should reside on CPU

        struct struct_vertex
        {
            glm::vec3 _position;
            glm::vec3 _normal;
        };

        struct struct_ED_node
        {
            glm::vec3 _position;
            glm::mat3 _transformation;
            glm::vec3 _translation;
        };

        struct struct_global_deformation
        {
            glm::mat3 _rotation;
            glm::vec3 _translation;
        };

        std::vector<struct_vertex> _reference_vx;
        std::vector<struct_ED_node> _ed_nodes;
        std::vector<float> _skinning_weights;
        struct_global_deformation _global_deformation;
    */

    void extract_ref_mesh();
    void draw_data();
    void init_shaders();
};
} // namespace kinect

#endif // #ifndef RECON_MC_HPP