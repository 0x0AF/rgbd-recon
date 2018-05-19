#ifndef RECON_MC_HPP
#define RECON_MC_HPP

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
    void setVoxelSize(float size);
    void setTsdfLimit(float limit);
    void setIso(float iso);

    static int TRI_TABLE[4096];

  private:
    globjects::VertexArray *m_point_grid;
    globjects::Buffer *m_point_buffer, *m_tri_table_buffer, *m_uv_counter_buffer;

    globjects::ref_ptr<globjects::Program> m_program;

    glm::uvec3 m_res_volume;
    glm::fmat4 m_mat_vol_to_world;

    float m_limit;
    float m_voxel_size;

    float m_iso;
};
}

#endif // #ifndef RECON_MC_HPP