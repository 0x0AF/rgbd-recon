#ifndef RECON_MC_HPP
#define RECON_MC_HPP

#include "recon_integration.hpp"
#include "reconstruction.hpp"
#include "view.hpp"
#include "view_lod.hpp"
#include "volume_sampler.hpp"

#include <globjects/base/ref_ptr.h>
namespace globjects
{
class Texture;
class Program;
class Framebuffer;
}

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

  private:
    void divideBox();
    globjects::Buffer *m_tri_table_buffer, *m_uv_counter_buffer, *m_buffer_bricks, *m_buffer_occupied;

    globjects::ref_ptr<globjects::Program> m_program, m_program_integration, m_program_solid, m_program_bricks;

    glm::uvec3 m_res_volume, m_res_bricks;
    VolumeSampler m_sampler;

    glm::fmat4 m_mat_vol_to_world;

    globjects::ref_ptr<globjects::Texture> m_volume_tsdf;

    std::vector<brick> m_bricks;
    std::vector<unsigned> m_active_bricks;
    std::vector<unsigned> m_bricks_occupied;

    float m_limit;
    float m_voxel_size;
    float m_brick_size;
    bool m_use_bricks;
    bool m_draw_bricks;
    float m_ratio_occupied;
    unsigned m_min_voxels_per_brick;
};
}

#endif // #ifndef RECON_MC_HPP