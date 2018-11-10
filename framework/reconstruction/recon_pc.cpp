#include "recon_pc.hpp"

#include "CalibVolumes.hpp"
#include "calibration_files.hpp"
#include "screen_quad.hpp"
#include "texture_blitter.hpp"
#include "timer_database.hpp"
#include "unit_cube.hpp"
#include "view_lod.hpp"
#include <KinectCalibrationFile.h>

#define GLM_ENABLE_EXPERIMENTAL

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/io.hpp>

#include <glbinding/gl/gl.h>
using namespace gl;
#include <globjects/Buffer.h>
#include <globjects/Framebuffer.h>
#include <globjects/Program.h>
#include <globjects/Texture.h>
#include <globjects/TextureHandle.h>
#include <globjects/VertexArray.h>
#include <globjects/VertexAttributeBinding.h>

#include <globjects/NamedString.h>
#include <globjects/Shader.h>
#include <globjects/globjects.h>

#include "/usr/local/cuda/include/cuda_runtime.h"
#include <globjects/Sync.h>
#include "/usr/local/cuda/include/vector_types.h"

extern "C" double copy_reference();
extern "C" double perform_brick_indexing();
extern "C" double sample_ed_nodes();
extern "C" double preprocess_textures();
extern "C" double pcg_solve();
extern "C" double fuse_data();

extern "C" double write_ply(int frame_number, IsoSurfaceVolume target);

extern "C" double extract_data();

extern "C" unsigned long int compute_isosurface(IsoSurfaceVolume target);
extern "C" double compute_isosurface_timed(IsoSurfaceVolume target);

extern "C" void init_cuda(glm::uvec3 &volume_res, struct_measures &measures, struct_native_handles &native_handles);
extern "C" void update_configuration(Configuration &configuration);
extern "C" void deinit_cuda();
extern "C" unsigned int push_debug_ed_nodes();
extern "C" unsigned long push_debug_sorted_vertices();

#define PASS_NORMALS

#define DATA_IMAGE_UNIT 4
#define REF_GRAD_IMAGE_UNIT 5

namespace kinect
{
using namespace globjects;
std::string ReconPerformanceCapture::TIMER_DATA_VOLUME_INTEGRATION = "TIMER_DATA_VOLUME_INTEGRATION";
std::string ReconPerformanceCapture::TIMER_DATA_MESH_DRAW = "TIMER_DATA_MESH_DRAW";

ReconPerformanceCapture::ReconPerformanceCapture(LocalKinectArray &nka, CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size,
                                                 float ed_cell_size)
    : Reconstruction(cfs, cv, bbox)
{
    _nka = &nka;

    init(limit, size, ed_cell_size);

    init_shaders();

    setVoxelSize(_voxel_size);
    setBrickSize(_brick_size);

    _native_handles.buffer_bricks = _buffer_bricks->id();
    _native_handles.buffer_occupied = _buffer_occupied->id();

    _native_handles.buffer_ed_nodes_debug = _buffer_ed_nodes_debug->id();
    _native_handles.buffer_sorted_vertices_debug = _buffer_sorted_vertices_debug->id();

    _native_handles.volume_tsdf_data = _volume_tsdf_data;
    _native_handles.volume_tsdf_ref_grad = _volume_tsdf_ref_grad;

    _native_handles.pbo_kinect_normals = _nka->getNormalsHandle();
    _native_handles.pbo_kinect_depths = _nka->getDepthHandle();
    _native_handles.pbo_kinect_silhouettes = _nka->getSilhouetteHandle();

    _measures.size_voxel = _voxel_size;
    _measures.sigma = 1.732f * 3.f / _res_volume.x;
    _measures.size_ed_cell = _ed_cell_size;
    _measures.size_brick = _brick_size;
    _measures.size_depth_cell = 8u;

    _measures.color_res = nka.getColorResolution();
    _measures.depth_res = nka.getDepthResolution();
    _measures.depth_cell_res = _measures.depth_res / _measures.size_depth_cell;
    _measures.num_depth_cells = _measures.depth_res.x * _measures.depth_res.y;

    _measures.data_volume_res = _res_volume;
    _measures.data_volume_bricked_res = _res_bricks;

    _measures.data_volume_num_bricks = _res_bricks.x * _res_bricks.y * _res_bricks.z;

    _measures.cv_xyz_res = cv->getVolumeResXYZ();
    _measures.cv_xyz_inv_res = cv->getVolumeRes();

    _texture2darray_debug = globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY);
    _texture2darray_debug->image3D(0, GL_R32F, _measures.depth_res.x, _measures.depth_res.y, 4, 0, GL_RED, GL_FLOAT, (void *)nullptr);

    _buffer_pbo_silhouettes_debug->setData(_measures.depth_res.x * _measures.depth_res.y * 4 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    _buffer_pbo_silhouettes_debug->bind(GL_PIXEL_UNPACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_UNPACK_BUFFER_ARB);

    _buffer_pbo_alignment_debug->setData(_measures.depth_res.x * _measures.depth_res.y * 4 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
    _buffer_pbo_alignment_debug->bind(GL_PIXEL_UNPACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_UNPACK_BUFFER_ARB);

    glBindTextureUnit(9, _texture2darray_debug->id());
    glBindTextureUnit(32, _volume_tsdf_data);
    glBindTextureUnit(34, _volume_tsdf_ref_grad);

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    _native_handles.pbo_kinect_silhouettes_debug = _buffer_pbo_silhouettes_debug->id();
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_ALIGNMENT_ERROR
    _native_handles.pbo_kinect_alignment_error_debug = _buffer_pbo_alignment_debug->id();
#endif

    _native_handles.pbo_opticflow = _nka->getFlowTextureHandle();

    for(uint8_t i = 0; i < m_num_kinects; i++)
    {
        _buffer_pbo_cv_xyz[i] = new Buffer();
        _buffer_pbo_cv_xyz[i]->setData(_measures.cv_xyz_res.x * _measures.cv_xyz_res.y * _measures.cv_xyz_res.z * sizeof(float4), nullptr, GL_DYNAMIC_COPY);
        _buffer_pbo_cv_xyz[i]->bind(GL_PIXEL_PACK_BUFFER_ARB);
        globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

        globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

        glBindTexture(GL_TEXTURE_3D, cv->getVolumesXYZ().at(i)->id());
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _buffer_pbo_cv_xyz[i]->id());
        glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, 0);
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
        glBindTexture(GL_TEXTURE_3D, 0);

        globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

        _buffer_pbo_cv_xyz_inv[i] = new Buffer();
        _buffer_pbo_cv_xyz_inv[i]->setData(_measures.cv_xyz_inv_res.x * _measures.cv_xyz_inv_res.y * _measures.cv_xyz_inv_res.z * sizeof(float4), nullptr, GL_DYNAMIC_COPY);
        _buffer_pbo_cv_xyz_inv[i]->bind(GL_PIXEL_PACK_BUFFER_ARB);
        globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

        globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

        glBindTexture(GL_TEXTURE_3D, cv->getVolumesXYZInv().at(i)->id());
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _buffer_pbo_cv_xyz_inv[i]->id());
        glGetTexImage(GL_TEXTURE_3D, 0, GL_RGBA, GL_FLOAT, 0);
        glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
        glBindTexture(GL_TEXTURE_3D, 0);

        globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

        _native_handles.pbo_cv_xyz_inv[i] = _buffer_pbo_cv_xyz_inv[i]->id();
        _native_handles.pbo_cv_xyz[i] = _buffer_pbo_cv_xyz[i]->id();
        _measures.depth_limits[i] = cv->getDepthLimits(i);
    }

    glFlush();

    _native_handles.posvbo = _buffer_fast_mc_pos->id();
    _native_handles.normalvbo = _buffer_fast_mc_normal->id();

    printf("\nres volume (%u,%u,%u)\n", _res_volume.x, _res_volume.y, _res_volume.z);
    printf("\nres bricks (%u,%u,%u)\n", _res_bricks.x, _res_bricks.y, _res_bricks.z);

    _measures.mc_2_norm = glm::inverse(_mat_vol_to_world) * _mat_world_to_vol;

    init_cuda(_res_volume, _measures, _native_handles);

    TimerDatabase::instance().addTimer(TIMER_DATA_VOLUME_INTEGRATION);
    TimerDatabase::instance().addTimer(TIMER_DATA_MESH_DRAW);
}
void ReconPerformanceCapture::init(float limit, float size, float ed_cell_size)
{
    _buffer_bricks = new Buffer();
    _buffer_occupied = new Buffer();
    _buffer_debug = new Buffer();
    _vao_debug = new VertexArray();
    _buffer_fsquad_debug = new Buffer();
    _vao_fsquad_debug = new VertexArray();
    _buffer_ed_nodes_debug = new Buffer();
    _buffer_sorted_vertices_debug = new Buffer();

    _buffer_pbo_silhouettes_debug = new Buffer();
    _buffer_pbo_alignment_debug = new Buffer();

    _vao_fast_mc = new VertexArray();
    _buffer_fast_mc_pos = new Buffer();
    _buffer_fast_mc_normal = new Buffer();

    _vec_debug = std::vector<glm::vec3>(MAX_REFERENCE_VERTICES);

    _program_pc_debug_textures = new Program();
    _program_pc_debug_opticflow = new Program();
    _program_pc_debug_draw_ref = new Program();
    _program_pc_debug_draw_ref_grad = new Program();
    _program_pc_debug_reference = new Program();
    _program_pc_debug_ed_sampling = new Program();
    _program_pc_debug_sorted_vertices = new Program();
    _program_pc_debug_sorted_vertices_connections = new Program();
    _program_pc_debug_correspondences = new Program();
    _program_pc_debug_tsdf = new Program();
    _program_integration = new Program();
    _program_solid = new Program();
    _program_bricks = new Program();

    _program_pc_fast_mc = new Program();

    _res_volume = glm::uvec3(64, 64, 64);
    _res_bricks = glm::uvec3(8, 8, 8);
    _sampler = new VolumeSampler(_res_volume);

    glGenTextures(1, &_volume_tsdf_data);
    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, _res_volume.x, _res_volume.y, _res_volume.z, 0, GL_RG, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_3D, 0);

    glGenTextures(1, &_volume_tsdf_ref_grad);
    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_ref_grad);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _res_volume.x, _res_volume.y, _res_volume.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_3D, 0);

    _mat_vol_to_world = glm::fmat4(1.0f);
    _limit = limit;
    _brick_size = size * 9;
    _voxel_size = size;
    _ed_cell_size = ed_cell_size;
    _ratio_occupied = 0.0f;
    _min_voxels_per_brick = 32;

    _frame_number.store(0);

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    _measures.bbox_translation = bbox_translation;
    _measures.bbox_dimensions = bbox_dimensions;

    _mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    _mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * _mat_vol_to_world;

    _mat_world_to_vol = glm::fmat4{1.0f};
    _mat_world_to_vol = glm::translate(_mat_world_to_vol, bbox_translation);
    _mat_world_to_vol = glm::scale(_mat_world_to_vol, 1.f / bbox_dimensions);
    _mat_world_to_vol = glm::translate(_mat_world_to_vol, glm::fvec3{1.f});

    _buffer_ed_nodes_debug->setData(sizeof(struct_ed_node_debug) * 16384, nullptr, GL_STREAM_COPY);
    _buffer_ed_nodes_debug->bindBase(GL_SHADER_STORAGE_BUFFER, 8);

    _buffer_sorted_vertices_debug->setData(sizeof(struct_vertex) * MAX_REFERENCE_VERTICES, nullptr, GL_STREAM_COPY);
    _buffer_sorted_vertices_debug->bindBase(GL_SHADER_STORAGE_BUFFER, 9);

    for(size_t i = 0; i < MAX_REFERENCE_VERTICES; i++)
    {
        auto vec = glm::vec3(1, 1, 1) * ((float)i) / 1000000.f;
        memcpy(&_vec_debug[i], &vec, sizeof(float) * 3);
    }

    _buffer_debug->bind(GL_ARRAY_BUFFER);
    _buffer_debug->setData(_vec_debug, GL_STATIC_DRAW);
    Buffer::unbind(GL_ARRAY_BUFFER);

    _vao_debug->enable(0);
    _vao_debug->binding(0)->setAttribute(0);
    _vao_debug->binding(0)->setBuffer(_buffer_debug, 0, sizeof(float) * 3);
    _vao_debug->binding(0)->setFormat(3, GL_FLOAT);

    const std::array<glm::vec2, 4> raw{{glm::vec2(1.f, -1.f), glm::vec2(1.f, 1.f), glm::vec2(-1.f, -1.f), glm::vec2(-1.f, 1.f)}};

    _buffer_fsquad_debug->setData(raw, GL_STATIC_DRAW); // needed for some drivers

    auto binding = _vao_fsquad_debug->binding(0);
    binding->setAttribute(0);
    binding->setBuffer(_buffer_fsquad_debug, 0, sizeof(float) * 2);
    binding->setFormat(2, GL_FLOAT, GL_FALSE, 0);
    _vao_fsquad_debug->enable(0);

    _buffer_fast_mc_pos->bind(GL_ARRAY_BUFFER);
    _buffer_fast_mc_pos->setData(MAX_REFERENCE_VERTICES * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    Buffer::unbind(GL_ARRAY_BUFFER);

    _buffer_fast_mc_normal->bind(GL_ARRAY_BUFFER);
    _buffer_fast_mc_normal->setData(MAX_REFERENCE_VERTICES * sizeof(float) * 4, nullptr, GL_DYNAMIC_DRAW);
    Buffer::unbind(GL_ARRAY_BUFFER);

    _vao_fast_mc->enable(0);
    _vao_fast_mc->binding(0)->setAttribute(0);
    _vao_fast_mc->binding(0)->setBuffer(_buffer_fast_mc_pos, 0, sizeof(float) * 4);
    _vao_fast_mc->binding(0)->setFormat(4, GL_FLOAT);
    _vao_fast_mc->enable(1);
    _vao_fast_mc->binding(1)->setAttribute(1);
    _vao_fast_mc->binding(1)->setBuffer(_buffer_fast_mc_normal, 0, sizeof(float) * 4);
    _vao_fast_mc->binding(1)->setFormat(4, GL_FLOAT);
}
void ReconPerformanceCapture::init_shaders()
{
    NamedString::create("/mc.glsl", new File("glsl/inc_mc.glsl"));

    _program_pc_debug_draw_ref_grad->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"));
    _program_pc_debug_draw_ref_grad->attach(Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_reference_gradient.gs"));
    _program_pc_debug_draw_ref_grad->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_reference_gradient.fs"));
    _program_pc_debug_draw_ref_grad->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_debug_draw_ref_grad->setUniform("size_voxel", _voxel_size);
    _program_pc_debug_draw_ref_grad->setUniform("volume_grad", 34);

    _program_pc_debug_reference->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/fast_mc.vs"));
    _program_pc_debug_reference->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_reference.fs"));
    _program_pc_debug_reference->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_debug_reference->setUniform("world_to_vol", _mat_world_to_vol);

    _program_pc_debug_ed_sampling->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_ed_sampling.vs"));
    _program_pc_debug_ed_sampling->attach(Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_ed_sampling.gs"));
    _program_pc_debug_ed_sampling->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_ed_sampling.fs"));
    _program_pc_debug_ed_sampling->setUniform("vol_to_world", _mat_vol_to_world);

    _program_pc_debug_sorted_vertices->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_reference.vs"));
    _program_pc_debug_sorted_vertices->attach(Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_sorted_vertices.gs"));
    _program_pc_debug_sorted_vertices->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_sorted_vertices.fs"));
    _program_pc_debug_sorted_vertices->setUniform("vol_to_world", _mat_vol_to_world);

    _program_pc_debug_sorted_vertices_connections->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_reference.vs"));
    _program_pc_debug_sorted_vertices_connections->attach(Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_sorted_vertices_connections.gs"));
    _program_pc_debug_sorted_vertices_connections->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_sorted_vertices.fs"));
    _program_pc_debug_sorted_vertices_connections->setUniform("vol_to_world", _mat_vol_to_world);

    _program_integration->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/tsdf_integration.vs"));
    _program_integration->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    _program_integration->setUniform("volume_tsdf", DATA_IMAGE_UNIT);
    _program_integration->setUniform("kinect_colors", 1);
    _program_integration->setUniform("kinect_depths", 2);
    _program_integration->setUniform("kinect_qualities", 3);
    _program_integration->setUniform("kinect_normals", 4);
    _program_integration->setUniform("kinect_silhouettes", 5);
    _program_integration->setUniform("num_kinects", m_num_kinects);
    _program_integration->setUniform("res_depth", glm::uvec2{m_cf->getWidth(), m_cf->getHeight()});
    _program_integration->setUniform("limit", _limit);

    _program_pc_debug_textures->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_textures.vs"), Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_textures.fs"));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    _program_pc_debug_textures->setUniform("texture_2d_array", 9);
#endif

    _program_pc_debug_opticflow->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_textures.vs"), Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_opticflow.fs"));

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    _program_pc_debug_opticflow->setUniform("opticflow", 8);
#endif

#ifdef PIPELINE_DEBUG_CORRESPONDENCE_FIELD
    _program_pc_debug_correspondences->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/pc_debug_correspondences.vs"));
    _program_pc_debug_correspondences->attach(Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_correspondences.gs"));
    _program_pc_debug_correspondences->attach(Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_correspondences.fs"));
    _program_pc_debug_correspondences->setUniform("kinect_depths", 2);
    _program_pc_debug_correspondences->setUniform("cv_xyz", m_cv->getXYZVolumeUnits());
    _program_pc_debug_correspondences->setUniform("vol_to_world", _mat_vol_to_world);
#endif

    _program_solid->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/solid.fs"));
    _program_bricks->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/bricks.gs"));

    _program_pc_debug_tsdf->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"));
    _program_pc_debug_tsdf->attach(globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_debug_tsdf.gs"));
    _program_pc_debug_tsdf->attach(globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_debug_tsdf.fs"));

    _program_pc_debug_tsdf->setUniform("volume_tsdf", 35);
    _program_pc_debug_tsdf->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_debug_tsdf->setUniform("limit", _limit);

    _program_pc_fast_mc->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/fast_mc.vs"));
#ifdef PASS_NORMALS
    _program_pc_fast_mc->attach(globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/fast_mc.fs"));
#else
    _program_pc_fast_mc->attach(globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_texture_blending.fs"));
#endif

    _program_pc_fast_mc->setUniform("world_to_vol", _mat_world_to_vol);
    _program_pc_fast_mc->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_fast_mc->setUniform("kinect_colors", 1);
    _program_pc_fast_mc->setUniform("kinect_depths", 2);
    _program_pc_fast_mc->setUniform("kinect_qualities", 3);
    _program_pc_fast_mc->setUniform("kinect_normals", 4);
    _program_pc_fast_mc->setUniform("kinect_silhouettes", 5);
    _program_pc_fast_mc->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    _program_pc_fast_mc->setUniform("cv_uv", m_cv->getUVVolumeUnits());
    _program_pc_fast_mc->setUniform("num_kinects", m_num_kinects);
    _program_pc_fast_mc->setUniform("limit", _limit);
}
ReconPerformanceCapture::~ReconPerformanceCapture()
{
    deinit_cuda();

    for(int i = 0; i < 4; i++)
    {
        _buffer_pbo_cv_xyz[i]->destroy();
        _buffer_pbo_cv_xyz_inv[i]->destroy();
    }

    _buffer_bricks->destroy();
    _buffer_occupied->destroy();

    _buffer_fast_mc_pos->destroy();
    _buffer_fast_mc_normal->destroy();
    _vao_fast_mc->destroy();

    _vao_debug->destroy();
    _vao_fsquad_debug->destroy();
    _buffer_debug->destroy();
    _buffer_fsquad_debug->destroy();
    _buffer_ed_nodes_debug->destroy();
    _buffer_sorted_vertices_debug->destroy();

    _buffer_pbo_silhouettes_debug->destroy();
    _buffer_pbo_alignment_debug->destroy();

    _texture2darray_debug->destroy();
    glDeleteTextures(1, &_volume_tsdf_data);
    glDeleteTextures(1, &_volume_tsdf_ref_grad);
}

void ReconPerformanceCapture::drawF()
{
    Reconstruction::drawF();
    if(_conf.draw_bricks)
    {
        drawOccupiedBricks();
    }
}

void ReconPerformanceCapture::drawComparison()
{
    update_configuration(_conf);

    extract_data(); // TODO: timer

    draw_data();

#ifdef OUTPUT_PLY_SEQUENCE_DATA
    if(!_is_paused)
    {
        write_ply((int)_frame_number.load(), IsoSurfaceVolume::Data);
    }
#endif
}

void ReconPerformanceCapture::draw()
{
    update_configuration(_conf);

    if(!_is_paused)
    {
        integrate_data_frame();

#ifdef PIPELINE_TEXTURES_PREPROCESS

        if(_conf.pipeline_preprocess_textures)
        {
            _conf.time_preprocess = preprocess_textures();

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES

            if(_conf.debug_texture_silhouettes)
            {
                globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, _buffer_pbo_silhouettes_debug->id());
                glBindTexture(GL_TEXTURE_2D_ARRAY, _texture2darray_debug->id());
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, _measures.depth_res.x, _measures.depth_res.y, 4, GL_RED, GL_FLOAT, 0);
                glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

                globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);
            }
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_ALIGNMENT_ERROR

            if(_conf.debug_texture_alignment_error)
            {
                globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, _buffer_pbo_alignment_debug->id());
                glBindTexture(GL_TEXTURE_2D_ARRAY, _texture2darray_debug->id());
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, 0, _measures.depth_res.x, _measures.depth_res.y, 4, GL_RED, GL_FLOAT, 0);
                glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

                globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);
            }
#endif
        }
#endif

#ifdef PIPELINE_SAMPLE

        if(_conf.pipeline_sample)
        {
            perform_brick_indexing();

            if(_frame_number.load() % _conf.reset_frame_count == 0)
            {
                _conf.time_copy_reference = copy_reference();

                _frame_number.store(0);
            }

            extract_reference_mesh();

            _conf.time_sample_ed = sample_ed_nodes();
        }

#endif

#ifdef PIPELINE_ALIGN

        if(_conf.pipeline_align && _frame_number.load() != 0)
        {
            _conf.time_nra = pcg_solve();
        }

#endif

#ifdef PIPELINE_FUSE

        if(_conf.pipeline_fuse)
        {
            glMemoryBarrier(GL_ALL_BARRIER_BITS);
            float4 empty{0.f, 0.f, 0.f, 0.f};
            glClearTexImage(_volume_tsdf_ref_grad, 0, GL_RGBA, GL_FLOAT, &empty);
            glBindImageTexture(REF_GRAD_IMAGE_UNIT, _volume_tsdf_ref_grad, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RGBA32F);
            glMemoryBarrier(GL_ALL_BARRIER_BITS);

            _conf.time_fuse = fuse_data();
        }

#endif

#ifdef OUTPUT_PLY_SEQUENCE_FUSED
        write_ply((int)_frame_number.load(), IsoSurfaceVolume::Fused);
#endif

        _frame_number.store(_frame_number.load() + 1);
    }

#ifdef PIPELINE_DEBUG_REFERENCE_VOLUME
    if(_conf.debug_reference_volume)
    {
        draw_debug_reference_volume();
    }
#endif

#ifdef PIPELINE_DEBUG_REFERENCE_MESH
    if(_conf.debug_reference_mesh)
    {
        draw_debug_reference_mesh();
    }
#endif

#ifdef PIPELINE_DEBUG_ED_SAMPLING
    if(_conf.debug_ed_sampling)
    {
        draw_debug_ed_sampling();
    }
#endif

#ifdef PIPELINE_DEBUG_SORTED_VERTICES
    if(_conf.debug_sorted_vertices)
    {
        draw_debug_sorted_vertices();
    }
#endif

    if(!_conf.debug_hide_fused && !_conf.debug_reference_volume && !_conf.debug_warped_reference_volume_surface)
    {
        draw_fused();
    }

#ifdef PIPELINE_DEBUG_TEXTURE_SILHOUETTES
    if(_conf.debug_texture_silhouettes)
    {
        draw_debug_texture();
    }
#endif

#ifdef PIPELINE_DEBUG_TEXTURE_ALIGNMENT_ERROR
    if(_conf.debug_texture_alignment_error)
    {
        draw_debug_texture();
    }
#endif

#ifdef PIPELINE_DEBUG_CORRESPONDENCE_FIELD
    if(_conf.debug_correspondence_field)
    {
        // TODO: draw_debug_correspondences();
    }
#endif

#ifdef PIPELINE_DEBUG_GRADIENT_FIELD
    if(_conf.debug_gradient_field)
    {
        draw_debug_reference_gradient();
    }
#endif

#ifdef PIPELINE_DEBUG_WARPED_REFERENCE_VOLUME
    if(_conf.debug_warped_reference_volume_surface)
    {
        draw_debug_reference_volume_warped();
    }
#endif
}
void ReconPerformanceCapture::extract_reference_mesh() {
    compute_isosurface_timed(IsoSurfaceVolume::Reference);
}
void ReconPerformanceCapture::draw_data()
{
    TimerDatabase::instance().begin(TIMER_DATA_MESH_DRAW);

    _program_pc_fast_mc->use();

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
    _program_pc_fast_mc->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_fast_mc->setUniform("NormalMatrix", normal_matrix);

    auto total_verts = (unsigned int)compute_isosurface(IsoSurfaceVolume::Data);

    _vao_fast_mc->bind();
    _vao_fast_mc->drawArrays(GL_TRIANGLES, 0, total_verts);
    globjects::VertexArray::unbind();

    Program::release();

    TimerDatabase::instance().end(TIMER_DATA_MESH_DRAW);
}
void ReconPerformanceCapture::draw_fused()
{
    TimerDatabase::instance().begin(TIMER_DATA_MESH_DRAW);

    _program_pc_fast_mc->use();

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
    _program_pc_fast_mc->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_fast_mc->setUniform("NormalMatrix", normal_matrix);

    auto total_verts = (unsigned int)compute_isosurface(IsoSurfaceVolume::Fused);

    _vao_fast_mc->bind();
    _vao_fast_mc->drawArrays(GL_TRIANGLES, 0, total_verts);
    globjects::VertexArray::unbind();

    Program::release();

    TimerDatabase::instance().end(TIMER_DATA_MESH_DRAW);
}
void ReconPerformanceCapture::draw_debug_reference_volume_warped()
{
    _program_pc_fast_mc->use();

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
    _program_pc_fast_mc->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_fast_mc->setUniform("NormalMatrix", normal_matrix);

    auto total_verts = (unsigned int)compute_isosurface(IsoSurfaceVolume::WarpedReference);

    _vao_fast_mc->bind();
    _vao_fast_mc->drawArrays(GL_TRIANGLES, 0, total_verts);
    globjects::VertexArray::unbind();

    Program::release();
}
void ReconPerformanceCapture::draw_debug_reference_volume()
{
    _program_pc_fast_mc->use();

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
    _program_pc_fast_mc->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_fast_mc->setUniform("NormalMatrix", normal_matrix);

    auto total_verts = (unsigned int)compute_isosurface(IsoSurfaceVolume::Reference);

    _vao_fast_mc->bind();
    _vao_fast_mc->drawArrays(GL_TRIANGLES, 0, total_verts);
    globjects::VertexArray::unbind();

    Program::release();
}
void ReconPerformanceCapture::draw_debug_reference_gradient()
{
    _program_pc_debug_draw_ref_grad->use();

    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_ref_grad);

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
    _program_pc_debug_draw_ref_grad->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_debug_draw_ref_grad->setUniform("NormalMatrix", normal_matrix);

    _sampler->sample();

    glBindTexture(GL_TEXTURE_3D, 0);
    Program::release();
}
void ReconPerformanceCapture::draw_debug_reference_mesh()
{
    _program_pc_debug_reference->use();

    gloost::Matrix projection_matrix;
    glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix.data());
    gloost::Matrix viewport_translate;
    viewport_translate.setIdentity();
    viewport_translate.setTranslate(1.0, 1.0, 1.0);
    gloost::Matrix viewport_scale;
    viewport_scale.setIdentity();

    // std::cout << vx_count << std::endl;

    _vao_fast_mc->bind();
    _vao_fast_mc->drawArrays(GL_POINTS, 0, MAX_REFERENCE_VERTICES);
    globjects::VertexArray::unbind();

    Program::release();
}
void ReconPerformanceCapture::draw_debug_ed_sampling()
{
    _program_pc_debug_ed_sampling->use();
    _program_pc_debug_ed_sampling->setUniform("mode", _conf.debug_sorted_vertices_mode);

    gloost::Matrix projection_matrix;
    glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix.data());
    gloost::Matrix viewport_translate;
    viewport_translate.setIdentity();
    viewport_translate.setTranslate(1.0, 1.0, 1.0);
    gloost::Matrix viewport_scale;
    viewport_scale.setIdentity();

    float zero = 0.f;
    _buffer_ed_nodes_debug->clearData(GL_R8, GL_RED, GL_FLOAT, &zero);
    auto ed_count = push_debug_ed_nodes();

    // std::cout << ed_count << std::endl;

    _vao_debug->bind();
    _vao_debug->drawArrays(GL_POINTS, 0, ed_count);
    globjects::VertexArray::unbind();

    Program::release();
}
void ReconPerformanceCapture::draw_debug_sorted_vertices()
{
    gloost::Matrix projection_matrix;
    glGetFloatv(GL_PROJECTION_MATRIX, projection_matrix.data());
    gloost::Matrix viewport_translate;
    viewport_translate.setIdentity();
    viewport_translate.setTranslate(1.0, 1.0, 1.0);
    gloost::Matrix viewport_scale;
    viewport_scale.setIdentity();

    float zero = 0.f;
    _buffer_sorted_vertices_debug->clearData(GL_R8, GL_RED, GL_FLOAT, &zero);
    auto vx_count = push_debug_sorted_vertices();
    _buffer_ed_nodes_debug->clearData(GL_R8, GL_RED, GL_FLOAT, &zero);
    auto ed_count = push_debug_ed_nodes();

    // std::cout << vx_count << std::endl;

    _program_pc_debug_sorted_vertices->use();
    _program_pc_debug_sorted_vertices->setUniform("mode", _conf.debug_sorted_vertices_mode);

    _vao_debug->bind();
    _vao_debug->drawArrays(GL_POINTS, 0, ed_count);
    globjects::VertexArray::unbind();

    Program::release();

#ifdef PIPELINE_DEBUG_SORTED_VERTICES_CONNECTIONS
    if(_conf.debug_sorted_vertices_connections)
    {
        _program_pc_debug_sorted_vertices_connections->use();
        _program_pc_debug_sorted_vertices_connections->setUniform("mode", _conf.debug_sorted_vertices_mode);
        _program_pc_debug_sorted_vertices_connections->setUniform("traces", _conf.debug_sorted_vertices_traces);

        _vao_debug->bind();
        _vao_debug->drawArrays(GL_POINTS, 0, ed_count);
        globjects::VertexArray::unbind();

        Program::release();
    }
#endif
}
void ReconPerformanceCapture::draw_debug_texture()
{
    _program_pc_debug_textures->use();

    glBindTexture(GL_TEXTURE_2D_ARRAY, _texture2darray_debug->id());

    _vao_fsquad_debug->bind();
    _vao_fsquad_debug->drawArrays(GL_TRIANGLE_STRIP, 0, 4);
    globjects::VertexArray::unbind();

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    Program::release();
}
void ReconPerformanceCapture::setVoxelSize(float size)
{
    _voxel_size = size;
    _res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / _voxel_size);

    _sampler->resize(_res_volume);

    _program_pc_debug_draw_ref->setUniform("res_tsdf", _res_volume);
    _program_pc_debug_draw_ref->setUniform("size_voxel", _voxel_size * 0.75f);

    _program_integration->setUniform("res_tsdf", _res_volume);

    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, _res_volume.x, _res_volume.y, _res_volume.z, 0, GL_RG, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_3D, 0);

    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_ref_grad);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA32F, _res_volume.x, _res_volume.y, _res_volume.z, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_3D, 0);

    setBrickSize(_brick_size);
}
void ReconPerformanceCapture::setTsdfLimit(float limit) { _limit = limit; }
void ReconPerformanceCapture::integrate_data_frame()
{
    TimerDatabase::instance().begin(TIMER_DATA_VOLUME_INTEGRATION);

    glEnable(GL_RASTERIZER_DISCARD);
    _program_integration->use();

    // clearing costs 0,4 ms on titan, filling from pbo 9
    float2 negative{-_limit, 0.f};
    glClearTexImage(_volume_tsdf_data, 0, GL_RG, GL_FLOAT, &negative);
    glBindImageTexture(DATA_IMAGE_UNIT, _volume_tsdf_data, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RG32F);

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    if(_conf.use_bricks)
    {
        for(auto const &index : _bricks_occupied)
        {
            _sampler->sample(_bricks[index].indices);
        }
    }
    else
    {
        _sampler->sample();
    }

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    Program::release();
    glDisable(GL_RASTERIZER_DISCARD);

    TimerDatabase::instance().end(TIMER_DATA_VOLUME_INTEGRATION);
}
void ReconPerformanceCapture::setBrickSize(float size)
{
    _brick_size = _voxel_size * glm::round(size / _voxel_size);
    std::cout << "adjusted bricksize from " << size << " to " << _brick_size << std::endl;
    divideBox();
}
float ReconPerformanceCapture::occupiedRatio() const { return _ratio_occupied; }
float ReconPerformanceCapture::getBrickSize() const { return _brick_size; }
void ReconPerformanceCapture::clearOccupiedBricks() const
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // clear active bricks
    static unsigned zerou = 0;
    _buffer_bricks->clearSubData(GL_R32UI, sizeof(unsigned) * 8, _bricks.size() * sizeof(unsigned), GL_RED_INTEGER, GL_UNSIGNED_INT, &zerou);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
void ReconPerformanceCapture::updateOccupiedBricks()
{
    // load occupied brick info
    _buffer_bricks->getSubData(sizeof(unsigned) * 8, _active_bricks.size() * sizeof(unsigned), _active_bricks.data());
    _bricks_occupied.clear();

    for(unsigned i = 0; i < _active_bricks.size(); ++i)
    {
        if(_active_bricks[i] >= _min_voxels_per_brick)
        {
            _bricks_occupied.emplace_back(i);
        }
    }
    _ratio_occupied = float(_bricks_occupied.size()) / float(_active_bricks.size());
    _buffer_occupied->setSubData(0, sizeof(unsigned) * _bricks_occupied.size(), _bricks_occupied.data());
    if(!_bricks_occupied.empty())
    {
        _buffer_occupied->bindRange(GL_SHADER_STORAGE_BUFFER, 4, 0, sizeof(unsigned) * _bricks_occupied.size());
    }
}
void ReconPerformanceCapture::setMinVoxelsPerBrick(unsigned num) { _min_voxels_per_brick = num; }
void ReconPerformanceCapture::drawOccupiedBricks() const
{
    _program_solid->use();
    _program_solid->setUniform("Color", glm::fvec3{1.0f, 0.0f, 0.0f});

    UnitCube::drawWireInstanced(_bricks_occupied.size());

    Program::release();
}
void ReconPerformanceCapture::divideBox()
{
    _bricks.clear();
    glm::fvec3 min{m_bbox.getPMin()};
    glm::fvec3 size{glm::fvec3{m_bbox.getPMax()} - min};
    glm::fvec3 start{min};
    _res_bricks = glm::uvec3{0};
    while(size.z - start.z + min.z > 0.0f)
    {
        while(size.y - start.y + min.y > 0.0f)
        {
            while(size.x - start.x + min.x > 0.0f)
            {
                _bricks.emplace_back(start, glm::min(glm::fvec3{_brick_size}, size - start + min));
                auto &curr_brick = _bricks.back();
                curr_brick.indices = _sampler->containedVoxels((curr_brick.pos - min) / size, curr_brick.size / size);
                curr_brick.baseVoxel = _sampler->baseVoxel((curr_brick.pos - min) / size, curr_brick.size / size);
                start.x += _brick_size;
                if(_res_bricks.z == 0 && _res_bricks.y == 0)
                {
                    ++_res_bricks.x;
                }
            }
            start.x = min.x;
            start.y += _brick_size;
            if(_res_bricks.z == 0)
            {
                ++_res_bricks.y;
            }
        }
        start.y = min.y;
        start.z += _brick_size;
        ++_res_bricks.z;
    }
    std::vector<unsigned> bricks(_bricks.size() + 8, 0);
    std::memcpy(&bricks[0], &_brick_size, sizeof(float));
    std::memcpy(&bricks[4], &_res_bricks, sizeof(unsigned) * 3);
    _buffer_bricks->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_COPY);
    _buffer_bricks->bindRange(GL_SHADER_STORAGE_BUFFER, 3, 0, sizeof(unsigned) * bricks.size());
    _active_bricks.resize(_bricks.size());

    _buffer_occupied->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_DRAW);
    std::cout << "brick res " << _res_bricks.x << ", " << _res_bricks.y << ", " << _res_bricks.z << " - " << _bricks.front().indices.size() << " voxels per brick" << std::endl;
}
void ReconPerformanceCapture::pause(bool pause) { _is_paused = pause; }

} // namespace kinect
