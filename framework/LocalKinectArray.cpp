#include "LocalKinectArray.h"

#include "CalibVolumes.hpp"
#include "calibration_files.hpp"
#include "screen_quad.hpp"
#include "timer_database.hpp"
#include <DXTCompressor.h>
#include <FileBuffer.h>
#include <KinectCalibrationFile.h>
#include <TextureArray.h>

#include <glbinding/gl/gl.h>
using namespace gl;

#include <globjects/Framebuffer.h>
#include <globjects/Sync.h>

#include <globjects/NamedString.h>
#include <globjects/Query.h>
#include <globjects/Shader.h>
#include <globjects/base/File.h>
#include <globjects/logging.h>

#include "/home/fusion_4d/Desktop/rgbd-recon/external/squish/squish.h"
#include <zmq.h>
#include <zmq.hpp>

#include <thread>

namespace kinect
{
static const std::size_t s_nu_bg_frames = 20;
LocalKinectArray::LocalKinectArray(std::string &file_name, std::string &file_name_flow, CalibrationFiles const *calibs, CalibVolumes const *vols, bool readfromfile)
    : _numLayers(4), _colorArray(),
      _depthArray_raw(), _textures_depth{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)}, _textures_depth_b{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)},
      _textures_depth2{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY), globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)},
      _textures_quality{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)}, _textures_normal{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)},
      _textures_color{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)}, _textures_bg{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY),
                                                                                            globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)},
      _textures_silhouette{globjects::Texture::createDefault(GL_TEXTURE_2D_ARRAY)}, _fbo{new globjects::Framebuffer()}, _colorArray_back(), _colorsize(0), _depthsize(0), _pbo_colors(), _pbo_depths(),
      _running(true), _filter_textures(true), _refine_bound(true), _curr_frametime{0.0}, _use_processed_depth{true}, _start_texture_unit(1), _calib_files{calibs}, _calib_vols{vols}
{
    _file_buffer = new sys::FileBuffer(file_name.c_str());
    if(!_file_buffer->open("r"))
    {
        std::cerr << "error opening " << file_name << " exiting..." << std::endl;
        exit(1);
    }
    _file_buffer->setLooping(true);

    _file_buffer_flow = new sys::FileBuffer(file_name_flow.c_str());
    if(!_file_buffer_flow->open("r"))
    {
        std::cerr << "error opening " << file_name_flow << " exiting..." << std::endl;
        exit(1);
    }
    _file_buffer_flow->setLooping(true);

    _out_pbo_colors = new globjects::Buffer();
    _out_pbo_normals = new globjects::Buffer();
    _out_pbo_depths = new globjects::Buffer();
    _out_pbo_silhouettes = new globjects::Buffer();
    _out_pbo_flows = new globjects::Buffer();

    _programs.emplace("filter", new globjects::Program());
    _programs.emplace("normal", new globjects::Program());
    _programs.emplace("quality", new globjects::Program());
    _programs.emplace("boundary", new globjects::Program());
    _programs.emplace("morph", new globjects::Program());
    // must happen before thread launching
    init();

    if(readfromfile)
    {
        readFromFiles(0);
    }

    globjects::NamedString::create("/bricks.glsl", new globjects::File("glsl/inc_bricks.glsl"));

    _programs.at("filter")->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/texture_passthrough.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pre_depth.fs"));
    _programs.at("normal")->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/texture_passthrough.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pre_normal.fs"));
    _programs.at("quality")->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/texture_passthrough.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pre_quality.fs"));
    _programs.at("boundary")->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/texture_passthrough.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pre_boundary.fs"));
    _programs.at("morph")->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/texture_passthrough.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pre_morph.fs"));
}

bool LocalKinectArray::init()
{
    _numLayers = _calib_files->num();
    _resolution_depth = glm::uvec2{_calib_files->getWidth(), _calib_files->getHeight()};
    _resolution_color = glm::uvec2{_calib_files->getWidthC(), _calib_files->getHeightC()};

    if(_calib_files->isCompressedRGB() == 1)
    {
        mvt::DXTCompressor dxt;
        dxt.init(_calib_files->getWidthC(), _calib_files->getHeightC(), FORMAT_DXT1);
        _colorsize = dxt.getStorageSize();
    }
    else if(_calib_files->isCompressedRGB() == 5)
    {
        std::cerr << "LocalKinectArray: using DXT5" << std::endl;
        _colorsize = 307200;
    }
    else
    {
        _colorsize = _resolution_color.x * _resolution_color.y * 3 * sizeof(byte);
    }

    _pbo_colors = double_pbo{_colorsize * _numLayers};

    if(_calib_files->isCompressedDepth())
    {
        _pbo_depths.size = _resolution_depth.x * _resolution_depth.y * _numLayers * sizeof(byte);
        _depthsize = _resolution_depth.x * _resolution_depth.y * sizeof(byte);
    }
    else
    {
        _pbo_depths.size = _resolution_depth.x * _resolution_depth.y * _numLayers * sizeof(float);
        _depthsize = _resolution_depth.x * _resolution_depth.y * sizeof(float);
    }

    std::cout << "Colorsize: " << _colorsize << std::endl;
    std::cout << "Color res: " << _resolution_color.x << "," << _resolution_color.y << std::endl;

    std::cout << "Depthsize: " << _depthsize << std::endl;
    std::cout << "Depth res: " << _resolution_depth.x << "," << _resolution_depth.y << std::endl;

    _pbo_depths = double_pbo{_depthsize * _numLayers};

    _pbo_flow = double_pbo{2048 * 1696 * sizeof(float) * _numLayers * 2};
    _flowArray = std::unique_ptr<TextureArray>{new TextureArray(2048, 1696, 4, GL_RG32F, GL_RG, GL_FLOAT)};
    _flowArray->setMAGMINFilter(GL_LINEAR);

    /* kinect color: GL_RGB32F, GL_RGB, GL_FLOAT*/
    /* kinect depth: GL_LUMINANCE32F_ARB, GL_RED, GL_FLOAT*/
    // _colorArray = new TextureArray(_resolution_depth.x, _resolution_depth.y, _numLayers, GL_RGB32F, GL_RGB, GL_FLOAT);
    if(_calib_files->isCompressedRGB() == 1)
    {
        std::cout << "Color DXT 1 compressed" << std::endl;
        _colorArray = std::unique_ptr<TextureArray>{
            new TextureArray(_resolution_color.x, _resolution_color.y, _numLayers, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, GL_UNSIGNED_BYTE, _colorsize)};
    }
    else if(_calib_files->isCompressedRGB() == 5)
    {
        std::cout << "Color DXT 5 compressed" << std::endl;
        _colorArray = std::unique_ptr<TextureArray>{
            new TextureArray(_resolution_color.x, _resolution_color.y, _numLayers, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_UNSIGNED_BYTE, _colorsize)};
    }
    else
    {
        _colorArray = std::unique_ptr<TextureArray>{new TextureArray(_resolution_color.x, _resolution_color.y, _numLayers, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)};
    }
    _colorArray_back = std::unique_ptr<TextureArray>{new TextureArray(_resolution_color.x, _resolution_color.y, _numLayers, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE)};
    _textures_color->image3D(0, GL_RGBA32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RGBA, GL_FLOAT, (void *)nullptr);

    _textures_quality->image3D(0, GL_LUMINANCE32F_ARB, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RED, GL_FLOAT, (void *)nullptr);
    _textures_normal->image3D(0, GL_RGBA32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RGBA, GL_FLOAT, (void *)nullptr);

    std::vector<glm::fvec2> empty_bg_tex(_resolution_depth.x * _resolution_depth.y * _numLayers, glm::fvec2{0.0f});
    _textures_bg.front->image3D(0, GL_RG32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RG, GL_FLOAT, empty_bg_tex.data());
    _textures_bg.back->image3D(0, GL_RG32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RG, GL_FLOAT, empty_bg_tex.data());
    _textures_silhouette->image3D(0, GL_R32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RED, GL_FLOAT, (void *)nullptr);

    _out_pbo_colors->setData(_depthsize * _numLayers * 4, nullptr, GL_DYNAMIC_COPY);
    _out_pbo_colors->bind(GL_PIXEL_PACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

    _out_pbo_normals->setData(_depthsize * _numLayers * 4, nullptr, GL_DYNAMIC_COPY);
    _out_pbo_normals->bind(GL_PIXEL_PACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

    _out_pbo_depths->setData(_depthsize * _numLayers * 2, nullptr, GL_DYNAMIC_COPY);
    _out_pbo_depths->bind(GL_PIXEL_PACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

    _out_pbo_flows->setData(2048 * 1696 * sizeof(float) * _numLayers * 2, nullptr, GL_DYNAMIC_COPY);
    _out_pbo_flows->bind(GL_PIXEL_PACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

    _out_pbo_silhouettes->setData(_depthsize * _numLayers, nullptr, GL_DYNAMIC_COPY);
    _out_pbo_silhouettes->bind(GL_PIXEL_PACK_BUFFER_ARB);
    globjects::Buffer::unbind(GL_PIXEL_PACK_BUFFER_ARB);

    if(_calib_files->isCompressedDepth())
    {
        _depthArray_raw = std::unique_ptr<TextureArray>{new TextureArray(_resolution_depth.x, _resolution_depth.y, _numLayers, GL_LUMINANCE, GL_RED, GL_UNSIGNED_BYTE)};
    }
    else
    {
        _depthArray_raw = std::unique_ptr<TextureArray>{new TextureArray(_resolution_depth.x, _resolution_depth.y, _numLayers, GL_LUMINANCE32F_ARB, GL_RED, GL_FLOAT)};
    }
    _textures_depth->image3D(0, GL_RG32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RG, GL_FLOAT, (void *)nullptr);
    _textures_depth_b->image3D(0, GL_RG32F, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RG, GL_FLOAT, (void *)nullptr);
    _textures_depth2.front->image3D(0, GL_LUMINANCE32F_ARB, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RED, GL_FLOAT, (void *)nullptr);
    _textures_depth2.back->image3D(0, GL_LUMINANCE32F_ARB, _resolution_depth.x, _resolution_depth.y, _numLayers, 0, GL_RED, GL_FLOAT, (void *)nullptr);

    _depthArray_raw->setMAGMINFilter(GL_NEAREST);
    _textures_depth_b->setParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _textures_depth_b->setParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    _textures_depth->setParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _textures_depth->setParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    _textures_depth2.front->setParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _textures_depth2.front->setParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    _textures_depth2.back->setParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _textures_depth2.back->setParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    _texture_unit_offsets.emplace("morph_input", 42);
    _texture_unit_offsets.emplace("raw_depth", 40);
    // _texture_unit_offsets.emplace("bg_depth", 41);

    _programs.at("filter")->setUniform("kinect_depths", getTextureUnit("raw_depth"));
    glm::fvec2 tex_size_inv{1.0f / _resolution_depth.x, 1.0f / _resolution_depth.y};
    _programs.at("filter")->setUniform("texSizeInv", tex_size_inv);
    _programs.at("normal")->setUniform("texSizeInv", tex_size_inv);
    _programs.at("quality")->setUniform("texSizeInv", tex_size_inv);
    _programs.at("quality")->setUniform("camera_positions", _calib_vols->getCameraPositions());

    _programs.at("morph")->setUniform("texSizeInv", tex_size_inv);
    _programs.at("morph")->setUniform("kinect_depths", getTextureUnit("morph_input"));
    _programs.at("boundary")->setUniform("texSizeInv", tex_size_inv);
    // _programs.at("bg")->setUniform("bg_depths", getTextureUnit("bg_depth"));

    globjects::NamedString::create("/inc_bbox_test.glsl", new globjects::File("glsl/inc_bbox_test.glsl"));
    globjects::NamedString::create("/inc_color.glsl", new globjects::File("glsl/inc_color.glsl"));

    TimerDatabase::instance().addTimer("morph");
    TimerDatabase::instance().addTimer("bilateral");
    TimerDatabase::instance().addTimer("boundary");
    TimerDatabase::instance().addTimer("normal");
    TimerDatabase::instance().addTimer("quality");
    TimerDatabase::instance().addTimer("1preprocess");

    return true;
}

LocalKinectArray::~LocalKinectArray()
{
    _running = false;

    _colorArray->unbind();
    _colorArray_back->unbind();
    _depthArray_raw->unbind();
    _flowArray->unbind();

    _fbo->destroy();

    _textures_bg.front->destroy();
    _textures_bg.back->destroy();

    _textures_depth2.front->destroy();
    _textures_depth2.back->destroy();

    _textures_depth->destroy();
    _textures_color->destroy();
    _textures_normal->destroy();
    _textures_quality->destroy();
    _textures_silhouette->destroy();

    _out_pbo_colors->destroy();
    _out_pbo_normals->destroy();
    _out_pbo_depths->destroy();
    _out_pbo_silhouettes->destroy();
    _out_pbo_flows->destroy();
}

bool LocalKinectArray::update(int frame_number)
{
    readFromFiles(frame_number);

    _colorArray->fillLayersFromPBO(_pbo_colors.get()->id());
    _depthArray_raw->fillLayersFromPBO(_pbo_depths.get()->id());
    _flowArray->fillLayersFromPBO(_pbo_flow.get()->id());

    return true;
}

glm::uvec2 LocalKinectArray::getDepthResolution() const { return _resolution_depth; }
glm::uvec2 LocalKinectArray::getColorResolution() const { return _resolution_color; }

int LocalKinectArray::getTextureUnit(std::string const &name) const { return _texture_unit_offsets.at(name); }

void LocalKinectArray::processDepth()
{
    _fbo->setDrawBuffers({GL_COLOR_ATTACHMENT0});

    glActiveTexture(GL_TEXTURE0 + getTextureUnit("morph_input"));
    _depthArray_raw->bind();
    _programs.at("morph")->use();
    _programs.at("morph")->setUniform("cv_xyz", _calib_vols->getXYZVolumeUnits());
    // erode
    _programs.at("morph")->setUniform("mode", 0u);
    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_depth2.back, 0, i);

        _programs.at("morph")->setUniform("layer", i);

        ScreenQuad::draw();
    }

    // dilate
    _programs.at("morph")->setUniform("mode", 1u);
    _textures_depth2.swapBuffers();
    _textures_depth2.front->bindActive(getTextureUnit("morph_input"));
    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_depth2.back, 0, i);

        _programs.at("morph")->setUniform("layer", i);

        ScreenQuad::draw();
    }

    _textures_depth2.front->unbindActive(getTextureUnit("morph_input"));

    _programs.at("morph")->release();

    _textures_depth2.swapBuffers();
    _textures_depth2.front->bindActive(getTextureUnit("morph_depth"));

    if(_use_processed_depth)
    {
        _textures_depth2.front->bindActive(getTextureUnit("raw_depth"));
    }
}

void LocalKinectArray::processTextures()
{
    TimerDatabase::instance().begin("1preprocess");
    GLint current_fbo;
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &current_fbo);
    GLsizei old_vp_params[4];
    glGetIntegerv(GL_VIEWPORT, old_vp_params);
    glViewport(0, 0, _resolution_depth.x, _resolution_depth.y);

    glActiveTexture(GL_TEXTURE0 + getTextureUnit("raw_depth"));
    _depthArray_raw->bind();

    _fbo->bind();
    TimerDatabase::instance().begin("morph");
    processDepth();
    TimerDatabase::instance().end("morph");

    TimerDatabase::instance().begin("bilateral");

    _fbo->setDrawBuffers({GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1});

    _programs.at("filter")->use();
    _programs.at("filter")->setUniform("filter_textures", _filter_textures);
    _programs.at("filter")->setUniform("processed_depth", _use_processed_depth);
    _programs.at("filter")->setUniform("cv_xyz", _calib_vols->getXYZVolumeUnits());
    _programs.at("filter")->setUniform("cv_uv", _calib_vols->getUVVolumeUnits());

    // depth and old quality
    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _programs.at("filter")->setUniform("cv_min_ds", _calib_vols->getDepthLimits(i).x);
        _programs.at("filter")->setUniform("cv_max_ds", _calib_vols->getDepthLimits(i).y);

        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_depth, 0, i);
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT1, _textures_color, 0, i);
        _programs.at("filter")->setUniform("layer", i);
        _programs.at("filter")->setUniform("compress", _calib_files->getCalibs()[i].isCompressedDepth());
        const float near = _calib_files->getCalibs()[i].getNear();
        const float far = _calib_files->getCalibs()[i].getFar();
        const float scale = (far - near);
        _programs.at("filter")->setUniform("scale", scale);
        _programs.at("filter")->setUniform("near", near);
        _programs.at("filter")->setUniform("scaled_near", scale / 255.0f);

        ScreenQuad::draw();
    }

    _programs.at("filter")->release();
    TimerDatabase::instance().end("bilateral");

    // boundary
    TimerDatabase::instance().begin("boundary");

    _programs.at("boundary")->use();

    _programs.at("boundary")->setUniform("cv_uv", _calib_vols->getUVVolumeUnits());
    _programs.at("boundary")->setUniform("refine", _refine_bound);
    _textures_depth->bindActive(getTextureUnit("depth"));

    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_depth_b, 0, i);
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT1, _textures_silhouette, 0, i);

        _programs.at("boundary")->setUniform("layer", i);

        ScreenQuad::draw();
    }
    _programs.at("boundary")->release();
    TimerDatabase::instance().end("boundary");

    _textures_depth_b->bindActive(getTextureUnit("depth"));
    // normals
    TimerDatabase::instance().begin("normal");
    _programs.at("normal")->use();
    _programs.at("normal")->setUniform("cv_xyz", _calib_vols->getXYZVolumeUnits());
    _programs.at("normal")->setUniform("cv_uv", _calib_vols->getUVVolumeUnits());

    _fbo->setDrawBuffers({GL_COLOR_ATTACHMENT0});

    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_normal, 0, i);

        _programs.at("normal")->setUniform("layer", i);

        ScreenQuad::draw();
    }

    _programs.at("normal")->release();
    TimerDatabase::instance().end("normal");

    // quality
    TimerDatabase::instance().begin("quality");
    _fbo->setDrawBuffers({GL_COLOR_ATTACHMENT0});
    _programs.at("quality")->use();
    _programs.at("quality")->setUniform("cv_xyz", _calib_vols->getXYZVolumeUnits());
    _programs.at("quality")->setUniform("processed_depth", _use_processed_depth);

    for(unsigned i = 0; i < _calib_files->num(); ++i)
    {
        _fbo->attachTextureLayer(GL_COLOR_ATTACHMENT0, _textures_quality, 0, i);

        _programs.at("quality")->setUniform("layer", i);

        ScreenQuad::draw();
    }
    _programs.at("quality")->release();
    TimerDatabase::instance().end("quality");

    _fbo->unbind();

    glViewport((GLsizei)old_vp_params[0], (GLsizei)old_vp_params[1], (GLsizei)old_vp_params[2], (GLsizei)old_vp_params[3]);

    TimerDatabase::instance().end("1preprocess");
}

void LocalKinectArray::setStartTextureUnit(unsigned start_texture_unit)
{
    _start_texture_unit = start_texture_unit;
    _texture_unit_offsets["color"] = _start_texture_unit;
    _texture_unit_offsets["depth"] = _start_texture_unit + 1;
    _texture_unit_offsets["quality"] = _start_texture_unit + 2;
    _texture_unit_offsets["normal"] = _start_texture_unit + 3;
    _texture_unit_offsets["silhouette"] = _start_texture_unit + 4;
    _texture_unit_offsets["morph_depth"] = _start_texture_unit + 5;
    _texture_unit_offsets["color_lab"] = _start_texture_unit + 6;
    _texture_unit_offsets["flow"] = _start_texture_unit + 7;

    bindToTextureUnits();

    _programs.at("filter")->setUniform("kinect_colors", getTextureUnit("color"));
    _programs.at("normal")->setUniform("kinect_depths", getTextureUnit("depth"));
    _programs.at("quality")->setUniform("kinect_depths", getTextureUnit("depth"));
    _programs.at("quality")->setUniform("kinect_normals", getTextureUnit("normal"));
    _programs.at("quality")->setUniform("kinect_colors_lab", getTextureUnit("color_lab"));
    _programs.at("boundary")->setUniform("kinect_colors_lab", getTextureUnit("color_lab"));
    _programs.at("boundary")->setUniform("kinect_depths", getTextureUnit("depth"));
    _programs.at("boundary")->setUniform("kinect_colors", getTextureUnit("color"));
}

void LocalKinectArray::bindToTextureUnits() const
{
    glActiveTexture(GL_TEXTURE0 + getTextureUnit("color"));
    _colorArray->bind();

    glActiveTexture(GL_TEXTURE0 + getTextureUnit("flow"));
    _flowArray->bind();

    _textures_quality->bindActive(getTextureUnit("quality"));
    _textures_normal->bindActive(getTextureUnit("normal"));
    _textures_silhouette->bindActive(getTextureUnit("silhouette"));
    _textures_depth2.front->bindActive(getTextureUnit("morph_depth"));
    _textures_color->bindActive(getTextureUnit("color_lab"));

    glActiveTexture(GL_TEXTURE0 + getTextureUnit("raw_depth"));
    _depthArray_raw->bind();
}

unsigned LocalKinectArray::getStartTextureUnit() const { return _start_texture_unit; }

void LocalKinectArray::filterTextures(bool filter)
{
    _filter_textures = filter;
    // process with new settings
    processTextures();
}
void LocalKinectArray::useProcessedDepths(bool filter)
{
    _use_processed_depth = filter;
    processTextures();
}
void LocalKinectArray::refineBoundary(bool filter)
{
    _refine_bound = filter;
    processTextures();
}
void LocalKinectArray::writeCurrentTexture(std::string prefix)
{
    // depths
    if(_calib_files->isCompressedDepth())
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _depthArray_raw->getGLHandle());
        int width, height, depth;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &depth);

        std::vector<std::uint8_t> depths;
        depths.resize(width * height * depth);

        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED, GL_UNSIGNED_BYTE, (void *)&depths[0]);

        int offset = 0;

        for(int k = 0; k < depth; ++k)
        {
            std::stringstream sstr;
            sstr << "output/" << prefix << "_d_" << k << ".bmp";
            std::string filename(sstr.str());
            std::cout << "writing depth texture for kinect " << k << " to file " << filename << std::endl;

            offset += width * height;

            writeBMP(filename, depths, offset, 1);
            offset += width * height;
        }
    }
    else
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _depthArray_raw->getGLHandle());
        int width, height, depth;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &depth);

        std::vector<float> depthsTmp;
        depthsTmp.resize(width * height * depth);

        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED, GL_FLOAT, (void *)&depthsTmp[0]);

        std::vector<std::uint8_t> depths;
        depths.resize(depthsTmp.size());

        for(int i = 0; i < width * height * depth; ++i)
        {
            depths[i] = (std::uint8_t)depthsTmp[i] * 255.0f;
        }

        int offset = 0;

        for(int k = 0; k < depth; ++k)
        {
            std::stringstream sstr;
            sstr << "output/" << prefix << "_d_" << k << ".bmp";
            std::string filename(sstr.str());
            std::cout << "writing depth texture for kinect " << k << " to file " << filename << " (values are compressed to 8bit)" << std::endl;

            writeBMP(filename, depths, offset, 1);
            offset += width * height;
        }
    }

    // color
    if(_calib_files->isCompressedRGB() == 1)
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _colorArray->getGLHandle());
        int size;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);

        std::vector<std::uint8_t> data;
        data.resize(size);

        glGetCompressedTexImage(GL_TEXTURE_2D_ARRAY, 0, (void *)&data[0]);

        std::vector<std::uint8_t> colors;
        colors.resize(4 * _resolution_color.x * _resolution_color.y);

        for(unsigned k = 0; k < _numLayers; ++k)
        {
            squish::DecompressImage(&colors[0], _resolution_color.x, _resolution_color.y, &data[k * _colorsize], squish::kDxt1);

            std::stringstream sstr;
            sstr << "output/" << prefix << "_col_" << k << ".bmp";
            std::string filename(sstr.str());
            std::cout << "writing color texture for kinect " << k << " to file " << filename << std::endl;

            writeBMP(filename, colors, 0, 4);
        }
    }
    else
    {
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        glBindTexture(GL_TEXTURE_2D_ARRAY, _colorArray->getGLHandle());
        int width, height, depth;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &height);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH, &depth);

        std::vector<std::uint8_t> colors;
        colors.resize(3 * width * height * depth);

        glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGB, GL_UNSIGNED_BYTE, (void *)&colors[0]);

        int offset = 0;

        for(int k = 0; k < depth; ++k)
        {
            std::stringstream sstr;
            sstr << "output/" << prefix << "_col_" << k << ".bmp";
            std::string filename(sstr.str());
            std::cout << "writing color texture for kinect " << k << " to file " << filename << std::endl;

            writeBMP(filename, colors, offset, 3);
            offset += 3 * width * height;
        }
    }
}

// no universal use! very unflexible, resolution depth = resolution color, no row padding
void LocalKinectArray::writeBMP(std::string filename, std::vector<std::uint8_t> const &data, unsigned int offset, unsigned int bytesPerPixel)
{
    std::ofstream file(filename, std::ofstream::binary);
    char c;
    short s;
    int i;

    c = 'B';
    file.write(&c, 1);
    c = 'M';
    file.write(&c, 1);
    i = _resolution_color.x * _resolution_color.y * 3 + 54;
    file.write((char const *)&i, 4);
    i = 0;
    file.write((char const *)&i, 4);
    i = 54;
    file.write((char const *)&i, 4);
    i = 40;
    file.write((char const *)&i, 4);
    i = _resolution_color.x;
    file.write((char const *)&i, 4);
    i = _resolution_color.y;
    file.write((char const *)&i, 4);
    s = 1;
    file.write((char const *)&s, 2);
    s = 24;
    file.write((char const *)&s, 2);
    i = 0;
    file.write((char const *)&i, 4);
    i = _resolution_color.x * _resolution_color.y * 3;
    file.write((char const *)&i, 4);
    i = 0;
    file.write((char const *)&i, 4);
    i = 0;
    file.write((char const *)&i, 4);
    i = 0;
    file.write((char const *)&i, 4);
    i = 0;
    file.write((char const *)&i, 4);

    for(unsigned int h = _resolution_color.y; h > 0; --h)
    {
        for(unsigned int w = 0; w < _resolution_color.x * bytesPerPixel; w += bytesPerPixel)
        {
            if(bytesPerPixel == 1)
            {
                file.write((char const *)&data[offset + w + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
                file.write((char const *)&data[offset + w + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
                file.write((char const *)&data[offset + w + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
            }
            else if(bytesPerPixel == 3 || bytesPerPixel == 4)
            {
                file.write((char const *)&data[offset + w + 2 + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
                file.write((char const *)&data[offset + w + 1 + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
                file.write((char const *)&data[offset + w + 0 + (h - 1) * _resolution_color.x * bytesPerPixel], 1);
            }
        }
    }

    file.close();
}

void LocalKinectArray::readFromFiles(int frame_number)
{
    const size_t frame_size_bytes((_colorsize + _depthsize) * _numLayers);
    const size_t flow_size_bytes = 2048 * 1696 * 2 * sizeof(float);
    const size_t flow_frame_size_bytes = flow_size_bytes * _numLayers;

    _file_buffer->gotoByte(frame_size_bytes * frame_number);
    _file_buffer_flow->gotoByte(flow_frame_size_bytes * frame_number);

    for(unsigned i = 0; i < _numLayers; ++i)
    {
        _file_buffer->read((byte *)_pbo_colors.pointer() + i * _colorsize, _colorsize);
        _file_buffer->read((byte *)_pbo_depths.pointer() + i * _depthsize, _depthsize);
        _file_buffer_flow->read((byte *)_pbo_flow.pointer() + i * flow_size_bytes, flow_size_bytes);
    }

    _pbo_colors.dirty = true;
    _pbo_depths.dirty = true;
    _pbo_flow.dirty = true;
}
void LocalKinectArray::write_out_pbos()
{
    globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures_color->id());
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _out_pbo_colors->id());
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures_normal->id());
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _out_pbo_normals->id());
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures_depth->id());
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _out_pbo_depths->id());
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RG, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _textures_silhouette->id());
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _out_pbo_silhouettes->id());
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RED, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    glBindTexture(GL_TEXTURE_2D_ARRAY, _flowArray->getGLHandle());
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, _out_pbo_flows->id());
    glGetTexImage(GL_TEXTURE_2D_ARRAY, 0, GL_RG, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

    globjects::Sync::fence(GL_SYNC_GPU_COMMANDS_COMPLETE);
}
const unsigned int LocalKinectArray::getNormalsHandle(bool texture)
{
    if(texture)
    {
        return _textures_normal->id();
    }
    return _out_pbo_normals->id();
}
const unsigned int LocalKinectArray::getColorHandle(bool texture)
{
    if(texture)
    {
        return _textures_color->id();
    }
    return _out_pbo_colors->id();
}
const unsigned int LocalKinectArray::getDepthHandle(bool texture)
{
    if(texture)
    {
        return _textures_depth->id();
    }
    return _out_pbo_depths->id();
}
const unsigned int LocalKinectArray::getSilhouetteHandle(bool texture)
{
    if(texture)
    {
        return _textures_silhouette->id();
    }
    _out_pbo_silhouettes->id();
}
const unsigned int LocalKinectArray::getFlowTextureHandle(bool texture)
{
    if(texture)
    {
        return _flowArray->getGLHandle();
    }
    return _out_pbo_flows->id();
}
} // namespace kinect
