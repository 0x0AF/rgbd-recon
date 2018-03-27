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
class ReconMC : public Reconstruction
{
  public:
    ReconMC(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbo, float limit, float size);
    ~ReconMC();

    void integrate();

    void draw() override;
    void drawF() override;
    void setVoxelSize(float size);
    void setTsdfLimit(float limit);
    void setIso(float iso);
    void setSizeMCVoxel(float size_mc_voxel);

    void resize(std::size_t width, std::size_t height) override;

    static int TRI_TABLE[4096];

  private:
    globjects::VertexArray *m_point_grid;
    globjects::Buffer *m_point_buffer, *m_tri_table_buffer;

    globjects::ref_ptr<globjects::Program> m_program;
    globjects::ref_ptr<globjects::Program> m_program_integration;

    glm::uvec3 m_res_volume;
    VolumeSampler m_sampler;
    globjects::ref_ptr<globjects::Texture> m_volume_tsdf;
    globjects::ref_ptr<globjects::Texture> m_tex_num_samples;

    glm::fmat4 m_mat_vol_to_world;
    float m_limit;
    float m_voxel_size;

    float m_iso;
    float m_size_mc_voxel;
};
}

#endif // #ifndef RECON_MC_HPP