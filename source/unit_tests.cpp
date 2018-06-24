#include <cstdio>
#include <gtest/gtest.h>

#include <reconstruction/recon_pc.hpp>

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
#include <glm/gtx/string_cast.hpp>
#include <vector_types.h>

extern "C" glm::uvec3 test_index_3d(unsigned int brick_id);
extern "C" glm::uvec3 test_position_3d(unsigned int position_id);
extern "C" glm::uvec3 test_ed_cell_3d(unsigned int ed_cell_id);
extern "C" unsigned int test_ed_cell_id(glm::uvec3 ed_cell_3d);
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d);
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d);
extern "C" glm::vec3 test_warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight);
extern "C" glm::vec3 test_warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight);
extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node);

namespace
{
const float ACCEPTED_FLOAT_TOLERANCE = 0.0000001f;

TEST(UtilTest, Index3D)
{
    unsigned int brick_id = 256;
    glm::uvec3 brick_index_3d = test_index_3d(brick_id);

    EXPECT_EQ(brick_index_3d.x, 0);
    EXPECT_EQ(brick_index_3d.y, 0);
    EXPECT_EQ(brick_index_3d.z, 4);
}
TEST(UtilTest, Position3D)
{
    unsigned int position_id = 4717;
    glm::uvec3 position_3d = test_position_3d(position_id);

    EXPECT_EQ(position_3d.x, 1);
    EXPECT_EQ(position_3d.y, 10);
    EXPECT_EQ(position_3d.z, 14);
}
TEST(UtilTest, EDCell3D)
{
    unsigned int ed_cell_id = 486;
    glm::uvec3 ed_cell_3d = test_ed_cell_3d(ed_cell_id);

    EXPECT_EQ(ed_cell_3d.x, 0);
    EXPECT_EQ(ed_cell_3d.y, 0);
    EXPECT_EQ(ed_cell_3d.z, 6);
}
TEST(UtilTest, EDCellID)
{
    glm::uvec3 ed_cell_3d{0, 0, 6};
    unsigned int ed_cell_id = test_ed_cell_id(ed_cell_3d);

    EXPECT_EQ(ed_cell_id, 486);
}
TEST(UtilTest, EDCellVoxelID)
{
    glm::uvec3 ed_cell_voxel_3d{1, 0, 1};
    unsigned int ed_cell_voxel_id = test_ed_cell_voxel_id(ed_cell_voxel_3d);

    EXPECT_EQ(ed_cell_voxel_id, 5);
}
TEST(UtilTest, ResidualEDNodeRotation)
{
    glm::mat4 rotation(1.f);
    rotation = glm::rotate(rotation, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(rotation);
    ed_node.translation = glm::vec3(0.0f);

    float res_ed_node_rot = test_evaluate_ed_node_residual(ed_node);
    EXPECT_NEAR(res_ed_node_rot, 0.0f, ACCEPTED_FLOAT_TOLERANCE);
}
TEST(UtilTest, ResidualEDNodeAffine)
{
    glm::mat4 affine(1.f);
    affine = glm::rotate(affine, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    affine *= glm::translate(affine, glm::vec3(1.0f, -1.0f, 0.f));
    affine *= glm::scale(affine, glm::vec3(3.0f));

    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat_cast(affine);
    ed_node.translation = glm::vec3(0.0f);

    float res_ed_node_rot = test_evaluate_ed_node_residual(ed_node);
    EXPECT_NEAR(res_ed_node_rot, 0.0f, ACCEPTED_FLOAT_TOLERANCE);
}
TEST(UtilTest, WarpPositionNoImpact)
{
    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(glm::mat3());
    ed_node.translation = glm::vec3(0.0f);

    glm::vec3 position = glm::vec3(1.0f);
    const float skinning_weight = 1.f;

    glm::vec3 dist = position - ed_node.position;
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight);

    EXPECT_FLOAT_EQ(warped_position.x, position.x);
    EXPECT_FLOAT_EQ(warped_position.y, position.y);
    EXPECT_FLOAT_EQ(warped_position.z, position.z);
}
TEST(UtilTest, WarpPositionTranslation)
{
    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(glm::mat3());
    ed_node.translation = glm::vec3(1.0f);

    glm::vec3 position = glm::vec3(1.0f);
    const float skinning_weight = 1.f;

    glm::vec3 dist = position - ed_node.position;
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight);

    EXPECT_FLOAT_EQ(warped_position.x, position.x + 1.0f);
    EXPECT_FLOAT_EQ(warped_position.y, position.y + 1.0f);
    EXPECT_FLOAT_EQ(warped_position.z, position.z + 1.0f);
}
TEST(UtilTest, WarpPositionRotation)
{
    glm::mat4 rotation(1.f);
    rotation = glm::rotate(rotation, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(rotation);
    ed_node.translation = glm::vec3(0.0f);

    glm::vec3 position = glm::vec3(0.f, 1.f, 0.f);
    const float skinning_weight = 1.f;

    glm::vec3 dist = position - ed_node.position;
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight);

    EXPECT_NEAR(warped_position.x, position.x, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_position.y, position.z, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_position.z, position.y, ACCEPTED_FLOAT_TOLERANCE);
}
TEST(UtilTest, WarpNormalNoImpact)
{
    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(glm::mat3());
    ed_node.translation = glm::vec3(0.0f);

    glm::vec3 normal = glm::vec3(-0.015034f, 0.000000f, -0.015034f);
    const float skinning_weight = 1.f;

    glm::vec3 warped_normal = test_warp_normal(normal, ed_node, skinning_weight);

    EXPECT_FLOAT_EQ(warped_normal.x, normal.x);
    EXPECT_FLOAT_EQ(warped_normal.y, normal.y);
    EXPECT_FLOAT_EQ(warped_normal.z, normal.z);
}
TEST(UtilTest, WarpNormalRotation)
{
    glm::mat4 rotation(1.f);
    rotation = glm::rotate(rotation, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(rotation);
    ed_node.translation = glm::vec3(0.0f);

    glm::vec3 normal = glm::vec3(0.f, 1.f, 0.f);
    const float skinning_weight = 1.f;

    glm::vec3 warped_normal = test_warp_normal(normal, ed_node, skinning_weight);

    EXPECT_NEAR(warped_normal.x, normal.x, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_normal.y, normal.z, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_normal.z, normal.y, ACCEPTED_FLOAT_TOLERANCE);
}
} // namespace

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}