#include "recon_mc.hpp"

#include "CalibVolumes.hpp"
#include "calibration_files.hpp"
#include "screen_quad.hpp"
#include "texture_blitter.hpp"
#include "timer_database.hpp"
#include "unit_cube.hpp"
#include "view_lod.hpp"
#include <KinectCalibrationFile.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/io.hpp>

#include <glbinding/gl/gl.h>
using namespace gl;
#include <globjects/Buffer.h>
#include <globjects/Framebuffer.h>
#include <globjects/Program.h>
#include <globjects/Texture.h>
#include <globjects/VertexArray.h>
#include <globjects/VertexAttributeBinding.h>

#include <globjects/Shader.h>
#include <globjects/globjects.h>

namespace kinect
{
static int start_image_unit = 3;
ReconMC::ReconMC(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size)
    : Reconstruction(cfs, cv, bbox), m_program{new globjects::Program()}, m_program_integration{new globjects::Program()}, m_res_volume{0}, m_sampler{glm::uvec3{0}},
      m_volume_tsdf{globjects::Texture::createDefault(GL_TEXTURE_3D)}, m_tex_num_samples{globjects::Texture::createDefault(GL_TEXTURE_2D)}, m_mat_vol_to_world{1.0f}, m_limit{limit},
      m_point_grid{new globjects::VertexArray()}, m_point_buffer{new globjects::Buffer()}, m_voxel_size{size}

{
    m_program->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/mc.vs"),globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/mc.gs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER,
                                                                                                                                                                             "glsl/mc.fs"));

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    m_mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    m_mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * m_mat_vol_to_world;

    m_program->setUniform("vol_to_world", m_mat_vol_to_world);
    m_program->setUniform("volume_tsdf", 29);

    m_program_integration->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/tsdf_integration.vs"));
    m_program_integration->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    m_program->setUniform("volume_tsdf", 29);

    m_program_integration->setUniform("volume_tsdf", start_image_unit);
    m_program_integration->setUniform("kinect_colors", 1);
    m_program_integration->setUniform("kinect_depths", 2);
    m_program_integration->setUniform("kinect_qualities", 3);
    m_program_integration->setUniform("kinect_normals", 4);
    m_program_integration->setUniform("kinect_silhouettes", 5);

    m_program_integration->setUniform("num_kinects", m_num_kinects);
    m_program_integration->setUniform("res_depth", glm::uvec2{m_cf->getWidth(), m_cf->getHeight()});
    m_program_integration->setUniform("limit", m_limit);

    std::vector<glm::fvec3> data{};
    float stepX = bbox_dimensions.x / 256.0f;
    float stepY = bbox_dimensions.y / 256.0f;
    float stepZ = bbox_dimensions.z / 256.0f;
    for(unsigned x = 0; x < 256; ++x)
    {
        for(unsigned y = 0; y < 256; ++y)
        {
            for(unsigned z = 0; z < 256; ++z)
            {
                data.emplace_back(x * stepX + bbox_translation.x, y * stepY + bbox_translation.y, z * stepZ + bbox_translation.z);
            }
        }
    }

    m_point_buffer->setData(data, GL_STATIC_DRAW);

    m_point_grid->enable(0);
    m_point_grid->binding(0)->setAttribute(0);
    m_point_grid->binding(0)->setBuffer(m_point_buffer, 0, sizeof(glm::fvec3));
    m_point_grid->binding(0)->setFormat(3, GL_FLOAT);

    setVoxelSize(m_voxel_size);
}

ReconMC::~ReconMC()
{
    m_point_grid->destroy();
    m_point_buffer->destroy();
}

void ReconMC::drawF()
{
    Reconstruction::drawF();

    // bind to units for displaying in gui
    m_tex_num_samples->bindActive(17);
}

void ReconMC::draw()
{
    m_program->use();

    gloost::Matrix projection_matrix;
    glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix.data());
    gloost::Matrix viewport_translate;
    viewport_translate.setIdentity();
    viewport_translate.setTranslate(1.0, 1.0, 1.0);
    gloost::Matrix viewport_scale;
    viewport_scale.setIdentity();

    glm::uvec4 viewport_vals{getViewport()};
    viewport_scale.setScale(viewport_vals[2] * 0.5, viewport_vals[3] * 0.5, 0.5f);
    gloost::Matrix image_to_eye = viewport_scale * viewport_translate * projection_matrix;
    image_to_eye.invert();
    m_program->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());

    m_point_grid->drawArrays(GL_POINTS, 0, 256 * 256 * 256);

    m_program->release();
}

void ReconMC::setVoxelSize(float size)
{
    m_voxel_size = size;
    m_res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / m_voxel_size);
    m_sampler.resize(m_res_volume);
    m_program_integration->setUniform("res_tsdf", m_res_volume);
    m_volume_tsdf->image3D(0, GL_R32F, glm::ivec3{m_res_volume}, 0, GL_RED, GL_FLOAT, nullptr);
    m_volume_tsdf->bindActive(GL_TEXTURE0 + 29);
    std::cout << "resolution " << m_res_volume.x << ", " << m_res_volume.y << ", " << m_res_volume.z << " - " << (m_res_volume.x * m_res_volume.y * m_res_volume.z) / 1000 << "k voxels" << std::endl;
}

void ReconMC::setTsdfLimit(float limit) { m_limit = limit; }

void ReconMC::resize(std::size_t width, std::size_t height) { m_tex_num_samples->image2D(0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr); }

void ReconMC::integrate()
{
    glEnable(GL_RASTERIZER_DISCARD);
    m_program_integration->use();

    // clearing costs 0,4 ms on titan, filling from pbo 9
    float negative = -m_limit;
    m_volume_tsdf->clearImage(0, GL_RED, GL_FLOAT, &negative);

    m_volume_tsdf->bindImageTexture(start_image_unit, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    m_sampler.sample();

    m_program_integration->release();
    glDisable(GL_RASTERIZER_DISCARD);
}
}
