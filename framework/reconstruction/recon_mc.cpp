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

#include <globjects/Shader.h>
#include <globjects/globjects.h>

namespace kinect
{
static int start_image_unit = 3;

ReconMC::ReconMC(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size)
    : Reconstruction(cfs, cv, bbox), m_buffer_bricks{new globjects::Buffer()}, m_buffer_occupied{new globjects::Buffer()}, m_program{new globjects::Program()}, m_res_volume{0},
      m_res_bricks{0}, m_sampler{glm::uvec3{0}}, m_sampler_brick{glm::uvec3{0}}, m_volume_tsdf{globjects::Texture::createDefault(GL_TEXTURE_3D)},
      m_tex_num_samples{globjects::Texture::createDefault(GL_TEXTURE_2D)}, m_mat_vol_to_world{1.0f}, m_limit{limit}, m_voxel_size{size}, m_brick_size{0.1f}
{
    m_program->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/mc.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/mc.fs"));

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    m_mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    m_mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * m_mat_vol_to_world;

    m_program->setUniform("vol_to_world", m_mat_vol_to_world);
    m_program->setUniform("camera_positions", m_cv->getCameraPositions());
    m_program->setUniform("limit", m_limit);
    m_program->setUniform("depth_peels", 17);
    m_program->setUniform("tex_num_samples", start_image_unit + 1);
    m_program->setUniform("viewport_offset", glm::fvec2(0.0, 0.0));

    m_program->setUniform("volume_tsdf", 29);

    setVoxelSize(m_voxel_size);
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
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * m_mat_vol_to_world);
    m_program->setUniform("NormalMatrix", normal_matrix);
    // upload camera pos in volume space for correct raycasting dir
    glm::fvec4 camera_world{glm::inverse(model_view) * glm::fvec4{0.0f, 0.0f, 0.0f, 1.0f}};
    glm::vec3 camera_texturespace{glm::inverse(m_mat_vol_to_world) * camera_world};

    m_program->setUniform("CameraPos", camera_texturespace);
    // bind texture for sample counts
    static const float zero = 0.0f;
    m_tex_num_samples->clearImage(0, GL_RED, GL_FLOAT, &zero);
    m_tex_num_samples->bindImageTexture(start_image_unit + 1, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

    // begin special thing for anaglyph rendering
    if(m_color_mask_mode == 1)
    {
        glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
    }
    if(m_color_mask_mode == 2)
    {
        glColorMask(GL_FALSE, GL_TRUE, GL_TRUE, GL_FALSE);
    }
    // end special thing for anaglyph rendering
    glDisable(GL_CULL_FACE);

    UnitCube::draw();

    glEnable(GL_CULL_FACE);
    m_program->release();

    // begin special thing for anaglyph rendering
    if(m_color_mask_mode > 0)
    {
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    }
    // end special thing for anaglyph rendering
}

void ReconMC::clearOccupiedBricks() const
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // clear active bricks
    static unsigned zerou = 0;
    m_buffer_bricks->clearSubData(GL_R32UI, sizeof(unsigned) * 8, m_bricks.size() * sizeof(unsigned), GL_RED_INTEGER, GL_UNSIGNED_INT, &zerou);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}

void ReconMC::setVoxelSize(float size)
{
    m_voxel_size = size;
    m_res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / m_voxel_size);
    m_sampler.resize(m_res_volume);
    m_volume_tsdf->image3D(0, GL_R32F, glm::ivec3{m_res_volume}, 0, GL_RED, GL_FLOAT, nullptr);
    m_volume_tsdf->bindActive(GL_TEXTURE0 + 29);
    std::cout << "resolution " << m_res_volume.x << ", " << m_res_volume.y << ", " << m_res_volume.z << " - " << (m_res_volume.x * m_res_volume.y * m_res_volume.z) / 1000 << "k voxels" << std::endl;
    // update brick size to match
    setBrickSize(m_brick_size);
}

void ReconMC::divideBox()
{
    m_bricks.clear();
    glm::fvec3 min{m_bbox.getPMin()};
    glm::fvec3 size{glm::fvec3{m_bbox.getPMax()} - min};
    glm::fvec3 start{min};
    m_res_bricks = glm::uvec3{0};
    while(size.z - start.z + min.z > 0.0f)
    {
        while(size.y - start.y + min.y > 0.0f)
        {
            while(size.x - start.x + min.x > 0.0f)
            {
                m_bricks.emplace_back(start, glm::min(glm::fvec3{m_brick_size}, size - start + min));
                auto &curr_brick = m_bricks.back();
                curr_brick.indices = m_sampler.containedVoxels((curr_brick.pos - min) / size, curr_brick.size / size);
                curr_brick.baseVoxel = m_sampler.baseVoxel((curr_brick.pos - min) / size, curr_brick.size / size);
                start.x += m_brick_size;
                if(m_res_bricks.z == 0 && m_res_bricks.y == 0)
                {
                    ++m_res_bricks.x;
                }
            }
            start.x = min.x;
            start.y += m_brick_size;
            if(m_res_bricks.z == 0)
            {
                ++m_res_bricks.y;
            }
        }
        start.y = min.y;
        start.z += m_brick_size;
        ++m_res_bricks.z;
    }
    std::vector<unsigned> bricks(m_bricks.size() + 8, 0);
    std::memcpy(&bricks[0], &m_brick_size, sizeof(float));
    std::memcpy(&bricks[4], &m_res_bricks, sizeof(unsigned) * 3);
    // bricks[1] = m_brick_size;
    // for(unsigned j = 0; j < m_bricks.size() && j < 2; ++j) {
    //   std::cout << "original" << std::endl;
    //   for(auto const& i : m_bricks[j].indices) {
    //     std::cout << (m_sampler.voxelPositions()[i] - m_bricks[j].pos) / m_brick_size << ", ";
    //   }
    //   std::cout << std::endl;
    // }
    m_buffer_bricks->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_COPY);
    m_buffer_bricks->bindRange(GL_SHADER_STORAGE_BUFFER, 3, 0, sizeof(unsigned) * bricks.size());
    m_active_bricks.resize(m_bricks.size());

    m_buffer_occupied->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_DRAW);
    std::cout << "brick res " << m_res_bricks.x << ", " << m_res_bricks.y << ", " << m_res_bricks.z << " - " << m_bricks.front().indices.size() << " voxels per brick" << std::endl;
}

void ReconMC::updateOccupiedBricks()
{
    // load occupied brick info
    m_buffer_bricks->getSubData(sizeof(unsigned) * 8, m_active_bricks.size() * sizeof(unsigned), m_active_bricks.data());
    m_bricks_occupied.clear();

    for(unsigned i = 0; i < m_active_bricks.size(); ++i)
    {
        m_bricks_occupied.emplace_back(i);
    }
    m_ratio_occupied = float(m_bricks_occupied.size()) / float(m_active_bricks.size());
    m_buffer_occupied->setSubData(0, sizeof(unsigned) * m_bricks_occupied.size(), m_bricks_occupied.data());
    if(m_bricks_occupied.size() > 0)
    {
        m_buffer_occupied->bindRange(GL_SHADER_STORAGE_BUFFER, 4, 0, sizeof(unsigned) * m_bricks_occupied.size());
    }
}

void ReconMC::setTsdfLimit(float limit)
{
    m_limit = limit;
    m_program->setUniform("limit", m_limit);
}

void ReconMC::setBrickSize(float size)
{
    m_brick_size = m_voxel_size * glm::round(size / m_voxel_size);
    m_sampler_brick.resize(glm::uvec3{m_brick_size / m_voxel_size});
    std::cout << "adjusted bricksize from " << size << " to " << m_brick_size << std::endl;
    ;
    // std::cout << "brick" << std::endl;
    // for(auto const& i : m_sampler_brick.voxelPositions()) {
    //   std::cout << i << ", ";
    // }
    // std::cout << std::endl;
    divideBox();
}

float ReconMC::getBrickSize() const { return m_brick_size; }

float ReconMC::occupiedRatio() const { return m_ratio_occupied; }

void ReconMC::resize(std::size_t width, std::size_t height) { m_tex_num_samples->image2D(0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr); }

void ReconMC::setViewportOffset(float x, float y)
{
    m_program->setUniform("viewport_offset", glm::fvec2(x, y));
    // currently not working
    // m_program_colorfill->setUniform("viewport_offset", glm::fvec2(x, y));
    // m_program_inpaint->setUniform("viewport_offset", glm::fvec2(x, y));
}
}
