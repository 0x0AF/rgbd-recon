#ifndef KINECT_NETKINECTARRAY_H
#define KINECT_NETKINECTARRAY_H

#include "DataTypes.h"
#include "FileBuffer.h"
#include "double_buffer.hpp"
#include "double_pixel_buffer.hpp"
#include "timevalue.h"

#include <glm/gtc/type_precision.hpp>

#include <globjects/base/ref_ptr.h>
namespace globjects
{
class Buffer;
class Program;
class Framebuffer;
class Texture;
class Query;
} // namespace globjects

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace std
{
class thread;
}

namespace kinect
{
class TextureArray;
class KinectCalibrationFile;
class CalibrationFiles;
class CalibVolumes;

class LocalKinectArray
{
  public:
    LocalKinectArray(std::string &file_name, std::string &file_name_flow, CalibrationFiles const *calibs, CalibVolumes const *vols, bool readfromfile = false);
    LocalKinectArray(std::vector<KinectCalibrationFile *> &calibs);

    ~LocalKinectArray();

    bool update(int frame_number);

    void processTextures();
    void setStartTextureUnit(unsigned _start_texture_unit);

    unsigned getStartTextureUnit() const;

    std::vector<KinectCalibrationFile *> const &getCalibs() const;

    void writeCurrentTexture(std::string prefix);
    void writeBMP(std::string, std::vector<std::uint8_t> const &, unsigned int offset, unsigned int bytesPerPixel);

    void filterTextures(bool filter);
    void useProcessedDepths(bool filter);
    void refineBoundary(bool filter);

    glm::uvec2 getDepthResolution() const;
    glm::uvec2 getColorResolution() const;

    int getTextureUnit(std::string const &name) const;

    const unsigned int getColorHandle(bool textures = false);
    const unsigned int getDepthHandle(bool textures = false);
    const unsigned int getSilhouetteHandle(bool textures = false);
    const unsigned int getFlowTextureHandle(bool textures = false);
    std::mutex &getPBOMutex();

    void readFromFiles(int frame_number);

  private:
    void bindToTextureUnits() const;
    void processBackground();
    void processDepth();

    bool init();

    glm::uvec2 _resolution_color;
    glm::uvec2 _resolution_depth;

    unsigned _numLayers;

    globjects::ref_ptr<globjects::Buffer> _out_pbo_colors;
    globjects::ref_ptr<globjects::Buffer> _out_pbo_depths;
    globjects::ref_ptr<globjects::Buffer> _out_pbo_silhouettes;
    globjects::ref_ptr<globjects::Buffer> _out_pbo_flows;

    std::unique_ptr<TextureArray> _colorArray;
    std::unique_ptr<TextureArray> _depthArray_raw;
    std::unique_ptr<TextureArray> _flowArray;
    globjects::ref_ptr<globjects::Texture> _textures_depth;
    globjects::ref_ptr<globjects::Texture> _textures_depth_b;
    double_buffer<globjects::ref_ptr<globjects::Texture>> _textures_depth2;
    globjects::ref_ptr<globjects::Texture> _textures_quality;
    globjects::ref_ptr<globjects::Texture> _textures_normal;
    globjects::ref_ptr<globjects::Texture> _textures_color;
    double_buffer<globjects::ref_ptr<globjects::Texture>> _textures_bg;
    globjects::ref_ptr<globjects::Texture> _textures_silhouette;
    globjects::ref_ptr<globjects::Framebuffer> _fbo;
    std::unique_ptr<TextureArray> _colorArray_back;

    std::map<std::string, globjects::ref_ptr<globjects::Program>> _programs;
    std::map<std::string, unsigned> _texture_unit_offsets;

    unsigned _colorsize; // per frame
    unsigned _depthsize; // per frame
    double_pbo _pbo_colors;
    double_pbo _pbo_depths;
    double_pbo _pbo_flow;

    std::mutex _mutex_pbo;
    bool _running;
    bool _filter_textures;
    bool _refine_bound;
    double _curr_frametime;
    bool _use_processed_depth;
    unsigned _start_texture_unit;

    CalibrationFiles const *_calib_files;
    CalibVolumes const *_calib_vols;
    sys::FileBuffer *_file_buffer;
    sys::FileBuffer *_file_buffer_flow;
};

} // namespace kinect

#endif // #ifndef KINECT_NETKINECTARRAY_H