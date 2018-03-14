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
struct brick_mc {
    brick_mc(glm::fvec3 const& p, glm::fvec3 const& s)
        :pos{p}
        ,size{s}
        ,indices{}
        ,baseVoxel{0}
    {}

    glm::fvec3 pos;
    glm::fvec3 size;
    std::vector<unsigned> indices;
    unsigned baseVoxel;
};

class ReconMC : public Reconstruction
{
  public:
    ReconMC(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbo, float limit, float size);

    void draw() override;
    void drawF() override;
    void setVoxelSize(float size);
    void setTsdfLimit(float limit);
    void setBrickSize(float limit);

    float occupiedRatio() const;
    float getBrickSize() const;

    void clearOccupiedBricks() const;
    void updateOccupiedBricks();
    void resize(std::size_t width, std::size_t height) override;

    // override from Reconstruction
    void setViewportOffset(float x, float y);

  private:
    void divideBox();

    globjects::ref_ptr<globjects::Buffer> m_buffer_bricks;
    globjects::ref_ptr<globjects::Buffer> m_buffer_occupied;

    globjects::ref_ptr<globjects::Program> m_program;

    glm::uvec3 m_res_volume;
    glm::uvec3 m_res_bricks;
    VolumeSampler m_sampler;
    VolumeSampler m_sampler_brick;
    globjects::ref_ptr<globjects::Texture> m_volume_tsdf;
    globjects::ref_ptr<globjects::Texture> m_tex_num_samples;

    glm::fmat4 m_mat_vol_to_world;
    std::vector<brick_mc> m_bricks;
    std::vector<unsigned> m_active_bricks;
    std::vector<unsigned> m_bricks_occupied;
    float m_limit;
    float m_voxel_size;
    float m_brick_size;
    float m_ratio_occupied;
};
}

#endif // #ifndef RECON_MC_HPP