#include "recon_pc.hpp"

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

ReconPerformanceCapture::ReconPerformanceCapture(CalibrationFiles const &cfs, CalibVolumes const *cv, gloost::BoundingBox const &bbox, float limit, float size)
    : Reconstruction(cfs, cv, bbox), m_buffer_bricks{new globjects::Buffer()}, m_buffer_occupied{new globjects::Buffer()}, m_program{new globjects::Program()},
      m_program_integration{new globjects::Program()}, m_program_solid{new globjects::Program()}, m_program_bricks{new globjects::Program()}, m_res_volume{0}, m_res_bricks{0},
      m_sampler{glm::uvec3{0}}, m_volume_tsdf{globjects::Texture::createDefault(GL_TEXTURE_3D)}, m_mat_vol_to_world{1.0f}, m_limit{limit}, m_brick_size{0.1f},
      m_tri_table_buffer{new globjects::Buffer()}, m_uv_counter_buffer{new globjects::Buffer()}, m_voxel_size{size}, m_use_bricks{true}, m_draw_bricks{false}, m_ratio_occupied{0.0f},
      m_min_voxels_per_brick{10}
{
    m_program->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/mc.gs"),
                      globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/mc.fs"));

    glm::fvec3 bbox_dimensions = glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]};
    glm::fvec3 bbox_translation = glm::fvec3{m_bbox.getPMin()[0], m_bbox.getPMin()[1], m_bbox.getPMin()[2]};

    m_mat_vol_to_world = glm::scale(glm::fmat4{1.0f}, bbox_dimensions);
    m_mat_vol_to_world = glm::translate(glm::fmat4{1.0f}, bbox_translation) * m_mat_vol_to_world;

    m_program->setUniform("vol_to_world", m_mat_vol_to_world);
    m_program->setUniform("size_voxel", m_voxel_size);
    m_program->setUniform("volume_tsdf", 29);

    m_program->setUniform("kinect_colors", 1);
    m_program->setUniform("kinect_depths", 2);
    m_program->setUniform("kinect_qualities", 3);
    m_program->setUniform("kinect_normals", 4);
    m_program->setUniform("kinect_silhouettes", 5);

    m_program->setUniform("cv_xyz_inv", m_cv->getXYZVolumeUnitsInv());
    m_program->setUniform("cv_uv", m_cv->getUVVolumeUnits());
    m_program->setUniform("num_kinects", m_num_kinects);
    m_program->setUniform("limit", m_limit);

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

    m_program_solid->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/solid.fs"));

    m_program_bricks->attach(globjects::Shader::fromFile(GL_VERTEX_SHADER, "glsl/bricks.vs"), globjects::Shader::fromFile(GL_FRAGMENT_SHADER, "glsl/bricks.fs"),
                             globjects::Shader::fromFile(GL_GEOMETRY_SHADER, "glsl/bricks.gs"));

    m_tri_table_buffer->setData(sizeof(int) * 4096, TRI_TABLE, GL_DYNAMIC_COPY);
    m_tri_table_buffer->bindRange(GL_SHADER_STORAGE_BUFFER, 5, 0, sizeof(int) * 4096);

    m_uv_counter_buffer->bind(GL_ATOMIC_COUNTER_BUFFER);
    m_uv_counter_buffer->setData(sizeof(GLuint), 0, GL_DYNAMIC_DRAW);
    m_uv_counter_buffer->unbind(GL_ATOMIC_COUNTER_BUFFER);

    m_uv_counter_buffer->bindBase(GL_ATOMIC_COUNTER_BUFFER, 6);

    setVoxelSize(m_voxel_size);
}

ReconPerformanceCapture::~ReconPerformanceCapture()
{
    m_tri_table_buffer->destroy();

    m_buffer_bricks->destroy();
    m_buffer_occupied->destroy();
}

void ReconPerformanceCapture::drawF()
{
    Reconstruction::drawF();
    if(m_draw_bricks)
    {
        drawOccupiedBricks();
    }
}
void ReconPerformanceCapture::draw()
{
    m_uv_counter_buffer->bind(GL_ATOMIC_COUNTER_BUFFER);
    GLuint *ptr = (GLuint *)m_uv_counter_buffer->mapRange(0, sizeof(GLuint), GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    ptr[0] = 0;
    m_uv_counter_buffer->unmap();
    m_uv_counter_buffer->unbind(GL_ATOMIC_COUNTER_BUFFER);

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

    glDisable(GL_CULL_FACE);

    if(m_use_bricks)
    {
        for(auto const &index : m_bricks_occupied)
        {
            m_sampler.sample(m_bricks[index].indices);
        }
    }
    else
    {
        m_sampler.sample();
    }

    m_program->release();
}

void ReconPerformanceCapture::setVoxelSize(float size)
{
    m_voxel_size = size;
    m_res_volume = glm::ceil(glm::fvec3{m_bbox.getPMax()[0] - m_bbox.getPMin()[0], m_bbox.getPMax()[1] - m_bbox.getPMin()[1], m_bbox.getPMax()[2] - m_bbox.getPMin()[2]} / m_voxel_size);

    m_sampler.resize(m_res_volume);

    m_program->setUniform("res_tsdf", m_res_volume);
    m_program->setUniform("size_voxel", m_voxel_size / 2.f);

    m_program_integration->setUniform("res_tsdf", m_res_volume);
    m_volume_tsdf->image3D(0, GL_R32F, glm::ivec3{m_res_volume}, 0, GL_RED, GL_FLOAT, nullptr);
    m_volume_tsdf->bindActive(GL_TEXTURE0 + 29);

    setBrickSize(m_brick_size);
}

void ReconPerformanceCapture::setTsdfLimit(float limit) { m_limit = limit; }
void ReconPerformanceCapture::integrate()
{
    glEnable(GL_RASTERIZER_DISCARD);
    m_program_integration->use();

    // clearing costs 0,4 ms on titan, filling from pbo 9
    float negative = -m_limit;
    m_volume_tsdf->clearImage(0, GL_RED, GL_FLOAT, &negative);

    m_volume_tsdf->bindImageTexture(start_image_unit, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);

    if(m_use_bricks)
    {
        for(auto const &index : m_bricks_occupied)
        {
            m_sampler.sample(m_bricks[index].indices);
        }
    }
    else
    {
        m_sampler.sample();
    }

    m_program_integration->release();
    glDisable(GL_RASTERIZER_DISCARD);
}
void ReconPerformanceCapture::setUseBricks(bool active)
{
    // TODO: incorporate toggle
    m_use_bricks = active;
}
void ReconPerformanceCapture::setDrawBricks(bool active) { m_draw_bricks = active; }
void ReconPerformanceCapture::setBrickSize(float size)
{
    m_brick_size = m_voxel_size * glm::round(size / m_voxel_size);
    std::cout << "adjusted bricksize from " << size << " to " << m_brick_size << std::endl;
    divideBox();
}
float ReconPerformanceCapture::occupiedRatio() const { return m_ratio_occupied; }
float ReconPerformanceCapture::getBrickSize() const { return m_brick_size; }
void ReconPerformanceCapture::clearOccupiedBricks() const
{
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
    // clear active bricks
    static unsigned zerou = 0;
    m_buffer_bricks->clearSubData(GL_R32UI, sizeof(unsigned) * 8, m_bricks.size() * sizeof(unsigned), GL_RED_INTEGER, GL_UNSIGNED_INT, &zerou);
    glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
void ReconPerformanceCapture::updateOccupiedBricks()
{
    // load occupied brick info
    m_buffer_bricks->getSubData(sizeof(unsigned) * 8, m_active_bricks.size() * sizeof(unsigned), m_active_bricks.data());
    m_bricks_occupied.clear();

    for(unsigned i = 0; i < m_active_bricks.size(); ++i)
    {
        if(m_active_bricks[i] >= m_min_voxels_per_brick)
        {
            m_bricks_occupied.emplace_back(i);
        }
    }
    m_ratio_occupied = float(m_bricks_occupied.size()) / float(m_active_bricks.size());
    m_buffer_occupied->setSubData(0, sizeof(unsigned) * m_bricks_occupied.size(), m_bricks_occupied.data());
    if(!m_bricks_occupied.empty())
    {
        m_buffer_occupied->bindRange(GL_SHADER_STORAGE_BUFFER, 4, 0, sizeof(unsigned) * m_bricks_occupied.size());
    }
}
void ReconPerformanceCapture::setMinVoxelsPerBrick(unsigned num) { m_min_voxels_per_brick = num; }
void ReconPerformanceCapture::drawOccupiedBricks() const
{
    m_program_solid->use();
    m_program_solid->setUniform("Color", glm::fvec3{1.0f, 0.0f, 0.0f});

    UnitCube::drawWireInstanced(m_bricks_occupied.size());

    m_program_solid->release();
}
void ReconPerformanceCapture::divideBox()
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
    m_buffer_bricks->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_COPY);
    m_buffer_bricks->bindRange(GL_SHADER_STORAGE_BUFFER, 3, 0, sizeof(unsigned) * bricks.size());
    m_active_bricks.resize(m_bricks.size());

    m_buffer_occupied->setData(sizeof(unsigned) * bricks.size(), bricks.data(), GL_DYNAMIC_DRAW);
    std::cout << "brick res " << m_res_bricks.x << ", " << m_res_bricks.y << ", " << m_res_bricks.z << " - " << m_bricks.front().indices.size() << " voxels per brick" << std::endl;
}
}
