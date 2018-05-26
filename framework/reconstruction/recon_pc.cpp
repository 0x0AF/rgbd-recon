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
#include <globjects/VertexArray.h>
#include <globjects/VertexAttributeBinding.h>

#include <globjects/NamedString.h>
#include <globjects/Shader.h>
#include <globjects/globjects.h>

namespace kinect
{
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

static int start_image_unit = 3;

ReconPerformanceCapture::ReconPerformanceCapture(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size) : Reconstruction(cfs, cv, bbox)
{
    init(limit, size);

    globjects::NamedString::create("/mc.glsl", new globjects::File("glsl/inc_mc.glsl"));

    _program_marching_cubes->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/pc_bricked_mc.gs"),
                                    globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/pc_texture_blending.fs"));

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    _mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    _mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * _mat_vol_to_world;

    _program_marching_cubes->setUniform("vol_to_world", _mat_vol_to_world);
    _program_marching_cubes->setUniform("size_voxel", _voxel_size);
    _program_marching_cubes->setUniform("volume_tsdf", 29);

    _program_marching_cubes->setUniform("kinect_colors", 1);
    _program_marching_cubes->setUniform("kinect_depths", 2);
    _program_marching_cubes->setUniform("kinect_qualities", 3);
    _program_marching_cubes->setUniform("kinect_normals", 4);
    _program_marching_cubes->setUniform("kinect_silhouettes", 5);

    _program_marching_cubes->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    _program_marching_cubes->setUniform("cv_uv", m_cv->getUVVolumeUnits());
    _program_marching_cubes->setUniform("num_kinects", m_num_kinects);
    _program_marching_cubes->setUniform("limit", _limit);

    _program_integration->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/tsdf_integration.vs"));
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

    _program_solid->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/solid.fs"));

    _program_bricks->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/bricks.fs"),
                            globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/bricks.gs"));

    _tri_table_buffer->setData(sizeof(int) * 4096, TRI_TABLE, GL_DYNAMIC_COPY);
    _tri_table_buffer->bindRange(GL_SHADER_STORAGE_BUFFER, 5, 0, sizeof(int) * 4096);

    _vertex_counter_buffer->bind(GL_ATOMIC_COUNTER_BUFFER);
    _vertex_counter_buffer->setData(sizeof(GLuint), 0, GL_DYNAMIC_DRAW);
    _vertex_counter_buffer->unbind(GL_ATOMIC_COUNTER_BUFFER);
    _vertex_counter_buffer->bindBase(GL_ATOMIC_COUNTER_BUFFER, 6);

    setVoxelSize(_voxel_size);
}

void ReconPerformanceCapture::init(float limit, float size)
{
    _buffer_bricks = new globjects::Buffer();
    _buffer_occupied = new globjects::Buffer();
    _tri_table_buffer = new globjects::Buffer();
    _vertex_counter_buffer = new globjects::Buffer();

    _program_marching_cubes = new globjects::Program();
    _program_integration = new globjects::Program();
    _program_solid = new globjects::Program();
    _program_bricks = new globjects::Program();

    _res_volume = glm::uvec3(0);
    _res_bricks = glm::uvec3(0);
    _sampler = new VolumeSampler(glm::uvec3{0});
    _volume_tsdf = globjects::Texture::createDefault(GL_TEXTURE_3D);

    _mat_vol_to_world = glm::fmat4(1.0f);
    _limit = limit;
    _brick_size = 0.1f;
    _voxel_size = size;
    _use_bricks = true;
    _draw_bricks = false;
    _ratio_occupied = 0.0f;
    _min_voxels_per_brick = 10;

    _frame_number.store(0);
}

ReconPerformanceCapture::~ReconPerformanceCapture()
{
    _tri_table_buffer->destroy();

    _vertex_counter_buffer->destroy();
    _buffer_bricks->destroy();
    _buffer_occupied->destroy();

    _volume_tsdf->destroy();
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
    if(_frame_number.load() % 16 == 0)
    {
        extract_ref_vx();
    }

    // TODO: estimate ICP rigid body fit

    // TODO: calculate JTJ and JTf in CUDA

    // TODO: run linear LMA (GPUfit) in CUDA interface

    draw_data();

    // TODO: fuse back into reference mesh

    _frame_number.store(_frame_number.load() + 1);
}
void ReconPerformanceCapture::extract_ref_vx()
{
    glEnable(GL_RASTERIZER_DISCARD);

    _vertex_counter_buffer->bind(GL_ATOMIC_COUNTER_BUFFER);
    GLuint *vx_ptr = (GLuint *)_vertex_counter_buffer->mapRange(0, sizeof(GLuint), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    vx_ptr[0] = 0;
    _vertex_counter_buffer->unmap();
    _vertex_counter_buffer->unbind(GL_ATOMIC_COUNTER_BUFFER);

    _program_marching_cubes->use();

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
    _program_marching_cubes->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_marching_cubes->setUniform("NormalMatrix", normal_matrix);

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

    _program_marching_cubes->release();

    glDisable(GL_RASTERIZER_DISCARD);

    // TODO: sample ED nodes as low res marching cubes
}
void ReconPerformanceCapture::draw_data()
{
    // TODO: blend with warped reference mesh

    _vertex_counter_buffer->bind(GL_ATOMIC_COUNTER_BUFFER);
    GLuint *vx_ptr = (GLuint *)_vertex_counter_buffer->mapRange(0, sizeof(GLuint), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    vx_ptr[0] = 0;
    _vertex_counter_buffer->unmap();
    _vertex_counter_buffer->unbind(GL_ATOMIC_COUNTER_BUFFER);

    _program_marching_cubes->use();

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
    _program_marching_cubes->setUniform("img_to_eye_curr", image_to_eye);

    gloost::Matrix modelview;
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview.data());
    glm::fmat4 model_view{modelview};
    glm::fmat4 normal_matrix = glm::inverseTranspose(model_view * _mat_vol_to_world);
    _program_marching_cubes->setUniform("NormalMatrix", normal_matrix);

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

    _program_marching_cubes->release();
}
void ReconPerformanceCapture::setVoxelSize(float size)
{
    _voxel_size = size;
    _res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / _voxel_size);

    _sampler->resize(_res_volume);

    _program_marching_cubes->setUniform("res_tsdf", _res_volume);
    _program_marching_cubes->setUniform("size_voxel", _voxel_size / 2.f);

    _program_integration->setUniform("res_tsdf", _res_volume);
    _volume_tsdf->image3D(0, GL_R32F, glm::ivec3{_res_volume}, 0, GL_RED, GL_FLOAT, nullptr);
    _volume_tsdf->bindActive(GL_TEXTURE0 + 29);

    setBrickSize(_brick_size);
}
void ReconPerformanceCapture::setTsdfLimit(float limit) { _limit = limit; }
void ReconPerformanceCapture::integrate()
{
    glEnable(GL_RASTERIZER_DISCARD);
    _program_integration->use();

    // clearing costs 0,4 ms on titan, filling from pbo 9
    float negative = -_limit;
    _volume_tsdf->clearImage(0, GL_RED, GL_FLOAT, &negative);
    _volume_tsdf->bindImageTexture(start_image_unit, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

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

    _program_integration->release();
    glDisable(GL_RASTERIZER_DISCARD);
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

    _program_solid->release();
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
