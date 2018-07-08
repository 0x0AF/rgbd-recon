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

#include <cuda_runtime.h>
#include <vector_types.h>

extern "C" void init_cuda(glm::uvec3 &volume_res, struct_measures &measures, struct_native_handles &native_handles);
extern "C" void copy_reference();
extern "C" void sample_ed_nodes();
extern "C" void estimate_correspondence_field();
extern "C" void pcg_solve();
extern "C" void fuse_data();
extern "C" void deinit_cuda();

#define PASS_NORMALS
// #define WIPE_DATA

#define PIPELINE_SAMPLE
// #define PIPELINE_CORRESPONDENCE
#define PIPELINE_ALIGN
#define PIPELINE_FUSE

namespace kinect
{
using namespace globjects;

int ReconPerformanceCapture::TRI_TABLE[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,
    8,  3,  9,  8,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  2,
    10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, -1, 3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  11, 2,
    8,  11, 0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  9,  0,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1, -1, -1, -1, 3,  10, 1,  11,
    10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, -1, 3,  9,  0,  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, -1, 9,  8,  10, 10, 8,
    11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  9,  8,  4,  7,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  4,  7,  3,  0,  4,  1,
    2,  10, -1, -1, -1, -1, -1, -1, -1, 9,  2,  10, 9,  0,  2,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, 2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,  4,  -1, -1, -1, -1, 8,  4,  7,  3,  11, 2,  -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, 11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, -1, 9,  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, 4,  7,  11, 9,  4,  11, 9,  11, 2,
    9,  2,  1,  -1, -1, -1, -1, 3,  10, 1,  3,  11, 10, 7,  8,  4,  -1, -1, -1, -1, -1, -1, -1, 1,  11, 10, 1,  4,  11, 1,  0,  4,  7,  11, 4,  -1, -1, -1, -1, 4,  7,  8,  9,  0,  11, 9,  11, 10, 11,
    0,  3,  -1, -1, -1, -1, 4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, -1, 9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  5,  4,  0,  8,  3,  -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, 0,  5,  4,  1,  5,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8,  5,  4,  8,  3,  5,  3,  1,  5,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, 3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1, 5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, -1, 2,  10, 5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1,
    -1, -1, -1, 9,  5,  4,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  11, 2,  0,  8,  11, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1, 0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1, -1, -1, -1,
    -1, -1, 2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1, -1, -1, 10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, 4,  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1,
    -1, 5,  4,  0,  5,  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, -1, 5,  4,  8,  5,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1, 9,  7,  8,  5,  7,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1, -1, -1, -1, 0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, -1, 1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,
    7,  8,  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, -1, 10, 1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3,  -1, -1, -1, -1, 8,  0,  2,  8,  2,  5,  8,  5,  7,  10, 5,  2,  -1, -1, -1, -1, 2,  10,
    5,  2,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, 7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, -1, 2,  3,  11,
    0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, -1, 11, 2,  1,  11, 1,  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, -1, 9,  5,  8,  8,  5,  7,  10, 1,  3,  10, 3,  11, -1, -1, -1, -1, 5,  7,  0,  5,
    0,  9,  7,  11, 0,  1,  0,  10, 11, 10, 0,  -1, 11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,  0,  -1, 11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 6,  5,  -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  0,  1,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  8,  3,  1,  9,  8,
    5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, 1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, -1, 9,  6,  5,  9,  0,  6,  0,
    2,  6,  -1, -1, -1, -1, -1, -1, -1, 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, -1, 2,  3,  11, 10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 0,  8,  11, 2,  0,  10, 6,
    5,  -1, -1, -1, -1, -1, -1, -1, 0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, 5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1, -1, 6,  3,  11, 6,  5,  3,  5,  1,  3,
    -1, -1, -1, -1, -1, -1, -1, 0,  8,  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, -1, 3,  11, 6,  0,  3,  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, -1, 6,  5,  9,  6,  9,  11, 11, 9,  8,  -1,
    -1, -1, -1, -1, -1, -1, 5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1, -1, -1, 1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1,
    -1, -1, -1, -1, -1, 10, 6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, -1, 6,  1,  2,  6,  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7,
    -1, -1, -1, -1, 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6,  -1, -1, -1, -1, 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,  -1, 3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1,
    -1, -1, -1, 5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, -1, 0,  1,  9,  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1, 9,  2,  1,  9,  11, 2,  9,  4,  11, 7,  11, 4,  5,  10,
    6,  -1, 8,  4,  7,  3,  11, 5,  3,  5,  1,  5,  11, 6,  -1, -1, -1, -1, 5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,  0,  4,  11, -1, 0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,
    -1, 6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, -1, 10, 4,  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  10, 6,  4,  9,  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, -1,
    10, 0,  1,  10, 6,  0,  6,  4,  0,  -1, -1, -1, -1, -1, -1, -1, 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,  10, -1, -1, -1, -1, 1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1, -1, -1, 3,
    0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, -1, 0,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8,  3,  2,  8,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, 10, 4,
    9,  10, 6,  4,  11, 2,  3,  -1, -1, -1, -1, -1, -1, -1, 0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  -1, -1, -1, -1, 3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1, -1, -1, -1, 6,  4,  1,
    6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  -1, 9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, -1, 8,  11, 1,  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  -1, 3,  11, 6,  3,
    6,  0,  0,  6,  4,  -1, -1, -1, -1, -1, -1, -1, 6,  4,  8,  11, 6,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1, -1, -1, -1, -1, 0,  7,  3,  0,  10,
    7,  0,  9,  10, 6,  7,  10, -1, -1, -1, -1, 10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, -1, 10, 6,  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  6,  1,  6,  8,
    1,  8,  9,  8,  6,  7,  -1, -1, -1, -1, 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9,  -1, 7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1, -1, -1, -1, -1, -1, 7,  3,  2,  6,  7,  2,  -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, -1, 2,  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  -1, 1,  8,  0,  1,  7,  8,  1,  10,
    7,  6,  7,  10, 2,  3,  11, -1, 11, 2,  1,  11, 1,  7,  10, 6,  1,  6,  7,  1,  -1, -1, -1, -1, 8,  9,  6,  8,  6,  7,  9,  1,  6,  11, 6,  3,  1,  3,  6,  -1, 0,  9,  1,  11, 6,  7,  -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, -1, 7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7,  6,  11, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, 3,  0,  8,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  1,  9,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1,
    -1, -1, -1, -1, -1, 10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, 2,  9,  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1,
    -1, -1, -1, -1, 6,  11, 7,  2,  10, 3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, -1, 7,  2,  3,  6,  2,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1, -1, -1,
    -1, -1, -1, 2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1, -1, -1, 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, -1, 10, 7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1,
    -1, -1, 10, 7,  6,  1,  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, -1, 0,  3,  7,  0,  7,  10, 0,  10, 9,  6,  10, 7,  -1, -1, -1, -1, 7,  6,  10, 7,  10, 8,  8,  10, 9,  -1, -1, -1, -1, -1, -1,
    -1, 6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, -1, 8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, -1,
    9,  4,  6,  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, -1, 6,  8,  4,  6,  11, 8,  2,  10, 1,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, 3,  0,  11, 0,  6,  11, 0,  4,  6,  -1, -1, -1, -1, 4,
    11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,  -1, -1, -1, -1, 10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,  -1, 8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, 0,  4,
    2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, -1, 1,  9,  4,  1,  4,  2,  2,  4,  6,  -1, -1, -1, -1, -1, -1, -1, 8,  1,  3,
    8,  6,  1,  8,  4,  6,  6,  10, 1,  -1, -1, -1, -1, 10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1, -1, -1, 4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  -1, 10, 9,  4,  6,
    10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4,  9,  5,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  4,  9,  5,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, 5,  0,  1,  5,  4,
    0,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, 11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1, -1, -1, -1, 9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, 6,  11, 7,  1,  2,  10,
    0,  8,  3,  4,  9,  5,  -1, -1, -1, -1, 7,  6,  11, 5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, -1, 3,  4,  8,  3,  5,  4,  3,  2,  5,  10, 5,  2,  11, 7,  6,  -1, 7,  2,  3,  7,  6,  2,  5,
    4,  9,  -1, -1, -1, -1, -1, -1, -1, 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,  -1, -1, -1, -1, 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1, -1, 6,  2,  8,  6,  8,  7,  2,  1,
    8,  4,  8,  5,  1,  5,  8,  -1, 9,  5,  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, -1, 1,  6,  10, 1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  -1, 4,  0,  10, 4,  10, 5,  0,  3,  10,
    6,  10, 7,  3,  7,  10, -1, 7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,  10, -1, -1, -1, -1, 6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1, -1, -1, 3,  6,  11, 0,  6,  3,  0,  5,  6,  0,
    9,  5,  -1, -1, -1, -1, 0,  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, -1, 6,  11, 3,  6,  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, -1, 1,  2,  10, 9,  5,  11, 9,  11, 8,  11, 5,
    6,  -1, -1, -1, -1, 0,  11, 3,  0,  6,  11, 0,  9,  6,  5,  6,  9,  1,  2,  10, -1, 11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,  2,  5,  -1, 6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,
    -1, -1, -1, -1, 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, -1, 9,  5,  6,  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, -1, 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,
    2,  8,  -1, 1,  5,  6,  2,  1,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,  8,  9,  6,  -1, 10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1,
    -1, -1, 0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 5,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 11, 5,  10, 7,  5,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 11, 5,  10, 11, 7,  5,  8,  3,  0,  -1, -1, -1, -1, -1, -1, -1, 5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1, -1, -1, -1, -1, -1, 10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1, -1, -1,
    11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, -1, 9,  7,  5,  9,  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, -1, 7,
    5,  2,  7,  2,  11, 5,  9,  2,  3,  2,  8,  9,  8,  2,  -1, 2,  5,  10, 2,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, 8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1, -1, -1, -1, 9,  0,
    1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, -1, 9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  -1, 1,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  8,  7,
    0,  7,  1,  1,  7,  5,  -1, -1, -1, -1, -1, -1, -1, 9,  0,  3,  9,  3,  5,  5,  3,  7,  -1, -1, -1, -1, -1, -1, -1, 9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 5,  8,  4,  5,
    10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, 5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, -1, 0,  1,  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, -1, 10, 11, 4,  10, 4,
    5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  -1, 2,  5,  1,  2,  8,  5,  2,  11, 8,  4,  5,  8,  -1, -1, -1, -1, 0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11, 1,  5,  1,  11, -1, 0,  2,  5,  0,  5,  9,
    2,  11, 5,  4,  5,  8,  11, 8,  5,  -1, 9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, -1, 5,  10, 2,  5,  2,  4,  4,
    2,  0,  -1, -1, -1, -1, -1, -1, -1, 3,  10, 2,  3,  5,  10, 3,  8,  5,  4,  5,  8,  0,  1,  9,  -1, 5,  10, 2,  5,  2,  4,  1,  9,  2,  9,  4,  2,  -1, -1, -1, -1, 8,  4,  5,  8,  5,  3,  3,  5,
    1,  -1, -1, -1, -1, -1, -1, -1, 0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, -1, 9,  4,  5,  -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 4,  11, 7,  4,  9,  11, 9,  10, 11, -1, -1, -1, -1, -1, -1, -1, 0,  8,  3,  4,  9,  7,  9,  11, 7,  9,  10, 11, -1, -1, -1, -1, 1,  10, 11, 1,  11, 4,  1,  4,  0,  7,
    4,  11, -1, -1, -1, -1, 3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,  -1, 4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, -1, 9,  7,  4,  9,  11, 7,  9,  1,  11, 2,  11,
    1,  0,  8,  3,  -1, 11, 7,  4,  11, 4,  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, -1, 11, 7,  4,  11, 4,  2,  8,  3,  4,  3,  2,  4,  -1, -1, -1, -1, 2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,  9,
    -1, -1, -1, -1, 9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,  7,  -1, 3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, -1, 1,  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, 4,  9,  1,  4,  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, -1, 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1,  -1, -1, -1, -1, 4,  0,  3,  7,  4,  3,  -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, 4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1,
    -1, 0,  1,  10, 0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, -1, 3,  1,  10, 11, 3,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  2,  11, 1,  11, 9,  9,  11, 8,  -1, -1, -1, -1, -1, -1, -1,
    3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,  -1, -1, -1, -1, 0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,
    3,  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, -1, 9,  10, 2,  0,  9,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,  3,  8,  2,  8,  10, 0,  1,  8,  1,  10, 8,  -1, -1, -1, -1, 1,  10,
    2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0,  3,  8,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

std::string ReconPerformanceCapture::TIMER_DATA_VOLUME_INTEGRATION = "TIMER_DATA_VOLUME_INTEGRATION";
std::string ReconPerformanceCapture::TIMER_REFERENCE_MESH_EXTRACTION = "TIMER_REFERENCE_MESH_EXTRACTION";
std::string ReconPerformanceCapture::TIMER_DATA_MESH_DRAW = "TIMER_DATA_MESH_DRAW";
std::string ReconPerformanceCapture::TIMER_CORRESPONDENCE = "TIMER_CORRESPONDENCE";
std::string ReconPerformanceCapture::TIMER_NON_RIGID_ALIGNMENT = "TIMER_NON_RIGID_ALIGNMENT";
std::string ReconPerformanceCapture::TIMER_FUSION = "TIMER_FUSION";
static int start_image_unit = 3;

ReconPerformanceCapture::ReconPerformanceCapture(NetKinectArray &nka, CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size, float ed_cell_size)
    : Reconstruction(cfs, cv, bbox)
{
    _nka = &nka;

    init(limit, size, ed_cell_size);

    init_shaders();

    _tri_table_buffer->setData(sizeof(GLint) * 4096, TRI_TABLE, GL_STATIC_COPY);
    _tri_table_buffer->bindRange(GL_SHADER_STORAGE_BUFFER, 5, 0, sizeof(GLint) * 4096);

    _buffer_vertex_counter->bind(GL_ATOMIC_COUNTER_BUFFER);
    _buffer_vertex_counter->setData(sizeof(GLuint), nullptr, GL_STREAM_DRAW);
    Buffer::unbind(GL_ATOMIC_COUNTER_BUFFER);
    _buffer_vertex_counter->bindBase(GL_ATOMIC_COUNTER_BUFFER, 6);

    _buffer_reference_mesh_vertices->setData(32 * 524288, nullptr, GL_STREAM_COPY);
    _buffer_reference_mesh_vertices->bindBase(GL_SHADER_STORAGE_BUFFER, 7);

    setVoxelSize(_voxel_size);
    setBrickSize(_brick_size);

    printf("\nres volume (%u,%u,%u)\n", _res_volume.x, _res_volume.y, _res_volume.z);
    printf("\nres bricks (%u,%u,%u)\n", _res_bricks.x, _res_bricks.y, _res_bricks.z);

    _native_handles.buffer_bricks = _buffer_bricks->id();
    _native_handles.buffer_occupied = _buffer_occupied->id();

    _native_handles.buffer_vertex_counter = _buffer_vertex_counter->id();
    _native_handles.buffer_reference_vertices = _buffer_reference_mesh_vertices->id();

    _native_handles.volume_tsdf_data = _volume_tsdf_data;

    // TODO: rgbs output
    /*_native_handles.texture_kinect_rgbs = _nka->getColorHandle();*/
    _native_handles.texture_kinect_depths = _nka->getDepthHandle();
    _native_handles.texture_kinect_silhouettes = _nka->getSilhouettes();

    for(uint8_t i = 0; i < m_num_kinects; i++)
    {
        _native_handles.volume_cv_xyz_inv[i] = cv->getVolumesXYZInv().at(i)->id();
        _native_handles.volume_cv_xyz[i] = cv->getVolumesXYZ().at(i)->id();
        _measures.depth_limits[i] = cv->getDepthLimits(i);
    }

    _measures.size_voxel = _voxel_size;
    _measures.size_ed_cell = _ed_cell_size;
    _measures.size_brick = _brick_size;

    _measures.color_res = nka.getColorResolution();
    _measures.depth_res = nka.getDepthResolution();

    _measures.data_volume_res = _res_volume;
    _measures.data_volume_bricked_res = _res_bricks;

    _measures.data_volume_num_bricks = _res_bricks.x * _res_bricks.y * _res_bricks.z;

    // TODO: generalize
    _measures.cv_xyz_res = glm::uvec3(128u, 128u, 128u);
    _measures.cv_xyz_inv_res = glm::uvec3(200u, 200u, 200u);

    init_cuda(_res_volume, _measures, _native_handles);

    TimerDatabase::instance().addTimer(TIMER_DATA_VOLUME_INTEGRATION);
    TimerDatabase::instance().addTimer(TIMER_REFERENCE_MESH_EXTRACTION);
    TimerDatabase::instance().addTimer(TIMER_CORRESPONDENCE);
    TimerDatabase::instance().addTimer(TIMER_NON_RIGID_ALIGNMENT);
    TimerDatabase::instance().addTimer(TIMER_FUSION);
    TimerDatabase::instance().addTimer(TIMER_DATA_MESH_DRAW);
}
void ReconPerformanceCapture::init(float limit, float size, float ed_cell_size)
{
    _buffer_bricks = new Buffer();
    _buffer_occupied = new Buffer();
    _tri_table_buffer = new Buffer();
    _buffer_vertex_counter = new Buffer();
    _buffer_reference_mesh_vertices = new Buffer();

    _program_pc_draw_data = new Program();
    _program_pc_extract_reference = new Program();
    _program_integration = new Program();
    _program_solid = new Program();
    _program_bricks = new Program();

    _res_volume = glm::uvec3(141, 140, 140);
    _res_bricks = glm::uvec3(16, 16, 16);
    _sampler = new VolumeSampler(_res_volume);

    glGenTextures(1, &_volume_tsdf_data);
    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);

    _mat_vol_to_world = glm::fmat4(1.0f);
    _limit = limit;
    _brick_size = 0.09f;
    _voxel_size = size;
    _ed_cell_size = ed_cell_size;
    _use_bricks = true;
    _draw_bricks = false;
    _ratio_occupied = 0.0f;
    _min_voxels_per_brick = 32;

    _frame_number.store(0);

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    _measures.bbox_translation = bbox_translation;
    _measures.bbox_dimensions = bbox_dimensions;

    _mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    _mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * _mat_vol_to_world;
}
void ReconPerformanceCapture::init_shaders()
{
    NamedString::create("/mc.glsl", new File("glsl/inc_mc.glsl"));

    _program_pc_draw_data->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_draw_data.gs"),
#ifdef PASS_NORMALS
                                  Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_vertex_normals.fs"));
#else
                                  Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_texture_blending.fs"));
#endif
    _program_pc_draw_data->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_draw_data->setUniform("size_voxel", _voxel_size);
    _program_pc_draw_data->setUniform("volume_tsdf", 29);
    _program_pc_draw_data->setUniform("kinect_colors", 1);
    _program_pc_draw_data->setUniform("kinect_depths", 2);
    _program_pc_draw_data->setUniform("kinect_qualities", 3);
    _program_pc_draw_data->setUniform("kinect_normals", 4);
    _program_pc_draw_data->setUniform("kinect_silhouettes", 5);
    _program_pc_draw_data->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    _program_pc_draw_data->setUniform("cv_uv", m_cv->getUVVolumeUnits());
    _program_pc_draw_data->setUniform("num_kinects", m_num_kinects);
    _program_pc_draw_data->setUniform("limit", _limit);

    _program_pc_extract_reference->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_extract_reference.gs"));
    _program_pc_extract_reference->setUniform("vol_to_world", _mat_vol_to_world);
    _program_pc_extract_reference->setUniform("size_voxel", _voxel_size);
    _program_pc_extract_reference->setUniform("volume_tsdf", 29);

    _program_integration->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/tsdf_integration.vs"));
    _program_integration->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    _program_integration->setUniform("volume_tsdf", start_image_unit);
    _program_integration->setUniform("kinect_colors", 1);
    _program_integration->setUniform("kinect_depths", 2);
    _program_integration->setUniform("kinect_qualities", 3);
    _program_integration->setUniform("kinect_normals", 4);
    _program_integration->setUniform("kinect_silhouettes", 5);
    _program_integration->setUniform("num_kinects", m_num_kinects);
    _program_integration->setUniform("res_depth", glm::uvec2{m_cf->getWidth(), m_cf->getHeight()});
    _program_integration->setUniform("limit", _limit);

    _program_solid->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/solid.fs"));
    _program_bricks->attach(Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/bricks.gs"));
}
ReconPerformanceCapture::~ReconPerformanceCapture()
{
    deinit_cuda();

    _tri_table_buffer->destroy();

    _buffer_reference_mesh_vertices->destroy();
    _buffer_vertex_counter->destroy();
    _buffer_bricks->destroy();
    _buffer_occupied->destroy();
}

void ReconPerformanceCapture::drawF()
{
    Reconstruction::drawF();
    if(_draw_bricks)
    {
        drawOccupiedBricks();
    }
}

void ReconPerformanceCapture::draw()
{
    _nka->getPBOMutex().lock();

    integrate_data_frame();

#ifdef PIPELINE_SAMPLE

    if(_frame_number.load() % 256 == 0)
    {
        TimerDatabase::instance().begin(TIMER_REFERENCE_MESH_EXTRACTION);

        copy_reference();
        extract_reference_mesh();
        sample_ed_nodes();

        TimerDatabase::instance().end(TIMER_REFERENCE_MESH_EXTRACTION);
    }

#endif

#ifdef PIPELINE_ALIGN

#ifdef PIPELINE_CORRESPONDENCE

    TimerDatabase::instance().begin(TIMER_CORRESPONDENCE);

    estimate_correspondence_field();

    TimerDatabase::instance().end(TIMER_CORRESPONDENCE);

#endif

    // TODO: estimate ICP rigid body fit

    TimerDatabase::instance().begin(TIMER_NON_RIGID_ALIGNMENT);

    pcg_solve();

    TimerDatabase::instance().end(TIMER_NON_RIGID_ALIGNMENT);

#endif

#ifdef WIPE_DATA

    float2 negative{-_limit, 0.f};
    glClearTexImage(_volume_tsdf_data, 0, GL_RG, GL_FLOAT, &negative);
    glBindImageTexture(start_image_unit, _volume_tsdf_data, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RG32F);

#endif

#ifdef PIPELINE_FUSE

    TimerDatabase::instance().begin(TIMER_FUSION);

    fuse_data();

    TimerDatabase::instance().end(TIMER_FUSION);

#endif

    draw_data();

    _frame_number.store(_frame_number.load() + 1);

    _nka->getPBOMutex().unlock();
}
void ReconPerformanceCapture::extract_reference_mesh()
{
    glEnable(GL_RASTERIZER_DISCARD);

    _buffer_vertex_counter->bind(GL_ATOMIC_COUNTER_BUFFER);
    GLuint *vx_ptr = (GLuint *)_buffer_vertex_counter->mapRange(0, sizeof(GLuint), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    vx_ptr[0] = 0;
    _buffer_vertex_counter->unmap();
    Buffer::unbind(GL_ATOMIC_COUNTER_BUFFER);

    _program_pc_extract_reference->use();

    glMemoryBarrier(GL_ALL_BARRIER_BITS);

    if(_use_bricks)
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

    TimerDatabase::instance().end(TIMER_REFERENCE_MESH_EXTRACTION);
}
void ReconPerformanceCapture::draw_data()
{
    TimerDatabase::instance().begin(TIMER_DATA_MESH_DRAW);

    _program_pc_draw_data->use();

    glBindTextureUnit(29, _volume_tsdf_data);

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
    _program_pc_draw_data->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_pc_draw_data->setUniform("NormalMatrix", normal_matrix);

    if(_use_bricks)
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

    Program::release();

    TimerDatabase::instance().end(TIMER_DATA_MESH_DRAW);
}
void ReconPerformanceCapture::setVoxelSize(float size)
{
    _voxel_size = size;
    _res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / _voxel_size);

    _sampler->resize(_res_volume);

    _program_pc_draw_data->setUniform("res_tsdf", _res_volume);
    _program_pc_draw_data->setUniform("size_voxel", _voxel_size * 0.75f);

    _program_pc_extract_reference->setUniform("res_tsdf", _res_volume);
    _program_pc_extract_reference->setUniform("size_voxel", _voxel_size * 0.75f);

    _program_integration->setUniform("res_tsdf", _res_volume);

    glBindTexture(GL_TEXTURE_3D, _volume_tsdf_data);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RG32F, _res_volume.x, _res_volume.y, _res_volume.z, 0, GL_RG, GL_FLOAT, nullptr);
    glBindTextureUnit(29, _volume_tsdf_data);

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
    glBindImageTexture(start_image_unit, _volume_tsdf_data, 0, GL_TRUE, 0, GL_READ_WRITE, GL_RG32F);

    if(_use_bricks)
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

    Program::release();
    glDisable(GL_RASTERIZER_DISCARD);

    TimerDatabase::instance().end(TIMER_DATA_VOLUME_INTEGRATION);
}
void ReconPerformanceCapture::setUseBricks(bool active) { _use_bricks = active; }
void ReconPerformanceCapture::setDrawBricks(bool active) { _draw_bricks = active; }
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
} // namespace kinect
