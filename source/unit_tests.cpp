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

extern "C" unsigned int test_identify_brick_id(const glm::vec3 position, struct_measures &measures);
extern "C" unsigned int test_identify_ed_cell_id(const glm::vec3 position, struct_measures &measures);
extern "C" glm::uvec3 test_index_3d(unsigned int brick_id, struct_measures &measures);
extern "C" glm::uvec3 test_position_3d(unsigned int position_id, struct_measures &measures);
extern "C" glm::uvec3 test_ed_cell_3d(unsigned int ed_cell_id, struct_measures &measures);
extern "C" unsigned int test_ed_cell_id(glm::uvec3 ed_cell_3d, struct_measures &measures);
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures);
extern "C" unsigned int test_ed_cell_voxel_id(glm::uvec3 ed_cell_voxel_3d, struct_measures &measures);
extern "C" glm::vec3 test_warp_position(glm::vec3 &dist, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures);
extern "C" glm::vec3 test_warp_normal(glm::vec3 &normal, struct_ed_node &ed_node, const float &skinning_weight, struct_measures &measures);
// extern "C" float test_evaluate_ed_node_residual(struct_ed_node &ed_node, struct_ed_meta_entry &ed_entry, struct_ed_node *ed_neighborhood);

namespace
{
const float ACCEPTED_FLOAT_TOLERANCE = 0.0000001f;
struct_measures mock_measures;

TEST(UtilTest, StructEDSize) { EXPECT_EQ(sizeof(struct_ed_node), 40); }
TEST(UtilTest, IdentifyBrickId)
{
    glm::vec3 pos{0.436170, 0.610714, 0.648405};
    unsigned int brick_id = test_identify_brick_id(pos, mock_measures);

    EXPECT_EQ(brick_id, 1892);
}
TEST(UtilTest, IdentifyEDCellId)
{
    glm::vec3 pos{0.436170, 0.610714, 0.648405};
    unsigned int ed_cell_id = test_identify_ed_cell_id(pos, mock_measures);

    EXPECT_EQ(ed_cell_id, 8);
}
TEST(UtilTest, Index3D)
{
    unsigned int brick_id = 256;
    glm::uvec3 brick_index_3d = test_index_3d(brick_id, mock_measures);

    EXPECT_EQ(brick_index_3d.x, 0);
    EXPECT_EQ(brick_index_3d.y, 0);
    EXPECT_EQ(brick_index_3d.z, 1);
}
TEST(UtilTest, Position3D)
{
    unsigned int position_id = 256;
    glm::uvec3 position_3d = test_position_3d(position_id, mock_measures);

    EXPECT_EQ(position_3d.x, 4);
    EXPECT_EQ(position_3d.y, 1);
    EXPECT_EQ(position_3d.z, 3);
}
TEST(UtilTest, EDCell3D)
{
    unsigned int ed_cell_id = 17;
    glm::uvec3 ed_cell_3d = test_ed_cell_3d(ed_cell_id, mock_measures);

    EXPECT_EQ(ed_cell_3d.x, 2);
    EXPECT_EQ(ed_cell_3d.y, 2);
    EXPECT_EQ(ed_cell_3d.z, 1);
}
TEST(UtilTest, EDCellID)
{
    glm::uvec3 ed_cell_3d{2, 2, 1};
    unsigned int ed_cell_id = test_ed_cell_id(ed_cell_3d, mock_measures);

    EXPECT_EQ(ed_cell_id, 17);
}
TEST(UtilTest, EDCellVoxelID)
{
    glm::uvec3 ed_cell_voxel_3d{1, 0, 1};
    unsigned int ed_cell_voxel_id = test_ed_cell_voxel_id(ed_cell_voxel_3d, mock_measures);

    EXPECT_EQ(ed_cell_voxel_id, 10);
}
// TEST(UtilTest, ResidualEDNodeAffineOutlier)
//{
//    glm::mat4 affine(1.f);
//    affine = glm::rotate(affine, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
//    affine *= glm::translate(affine, glm::vec3(1.0f, -1.0f, 0.f));
//    affine *= glm::scale(affine, glm::vec3(3.0f));
//
//    struct_ed_node ed_node;
//    ed_node.position = glm::vec3(0.28f, 0.f, 0.0f);
//    ed_node.affine = glm::quat_cast(affine);
//    ed_node.translation = glm::vec3(0.0f);
//
//    struct_ed_meta_entry ed_entry;
//    ed_entry.ed_cell_id = 0;
//
//    struct_ed_node ed_neighborhood[27];
//
//    for(unsigned int i = 0; i < 27; i++)
//    {
//        ed_entry.neighbors[i] = i;
//
//        ed_neighborhood[i].position = glm::vec3(0.1f * (float)i, 0.f, 0.f);
//        ed_neighborhood[i].affine = glm::quat();
//        ed_neighborhood[i].translation = glm::vec3(0.0f);
//    }
//
//    float res_ed_node_rot = test_evaluate_ed_node_residual(ed_node, ed_entry, ed_neighborhood);
//    EXPECT_NEAR(res_ed_node_rot, 0.0f, ACCEPTED_FLOAT_TOLERANCE);
//}
// TEST(UtilTest, ResidualEDNodeAffineCoherent)
//{
//    glm::mat4 affine(1.f);
//    affine = glm::rotate(affine, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
//    affine *= glm::translate(affine, glm::vec3(1.0f, -1.0f, 0.f));
//    affine *= glm::scale(affine, glm::vec3(3.0f));
//
//    struct_ed_node ed_node;
//    ed_node.position = glm::vec3(0.28f, 0.f, 0.0f);
//    ed_node.affine = glm::quat_cast(affine);
//    ed_node.translation = glm::vec3(0.0f);
//
//    struct_ed_meta_entry ed_entry;
//    ed_entry.ed_cell_id = 0;
//
//    struct_ed_node ed_neighborhood[27];
//
//    for(unsigned int i = 0; i < 27; i++)
//    {
//        ed_entry.neighbors[i] = i;
//
//        ed_neighborhood[i] = ed_node;
//        ed_neighborhood[i].position = glm::vec3(0.1f * (float)i, 0.f, 0.f);
//    }
//
//    float res_ed_node_rot = test_evaluate_ed_node_residual(ed_node, ed_entry, ed_neighborhood);
//    EXPECT_NEAR(res_ed_node_rot, 0.0f, ACCEPTED_FLOAT_TOLERANCE);
//}
TEST(UtilTest, WarpPositionNoImpact)
{
    struct_ed_node ed_node;
    ed_node.position = glm::vec3(0.0f);
    ed_node.affine = glm::quat(glm::mat3());
    ed_node.translation = glm::vec3(0.0f);

    glm::vec3 position = glm::vec3(1.0f);
    const float skinning_weight = 1.f;

    glm::vec3 dist = position - ed_node.position;
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight, mock_measures);

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
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight, mock_measures);

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
    glm::vec3 warped_position = test_warp_position(dist, ed_node, skinning_weight, mock_measures);

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

    glm::vec3 warped_normal = test_warp_normal(normal, ed_node, skinning_weight, mock_measures);

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

    glm::vec3 warped_normal = test_warp_normal(normal, ed_node, skinning_weight, mock_measures);

    EXPECT_NEAR(warped_normal.x, normal.x, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_normal.y, normal.z, ACCEPTED_FLOAT_TOLERANCE);
    EXPECT_NEAR(warped_normal.z, normal.y, ACCEPTED_FLOAT_TOLERANCE);
}
} // namespace

GTEST_API_ int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    mock_measures.size_voxel = 0.01f;
    mock_measures.size_ed_cell = 0.03f;
    mock_measures.size_brick = 0.09f;

    // mock_measures.color_res = nka.getColorResolution();
    // mock_measures.depth_res = nka.getDepthResolution();

    mock_measures.data_volume_res = glm::uvec3(141, 140, 140);
    mock_measures.data_volume_bricked_res = glm::uvec3(16, 16, 16);
    mock_measures.data_volume_num_bricks = 16 * 16 * 16;

    mock_measures.cv_xyz_res = glm::uvec3(128u, 128u, 128u);
    mock_measures.cv_xyz_inv_res = glm::uvec3(200u, 200u, 200u);

    return RUN_ALL_TESTS();
}