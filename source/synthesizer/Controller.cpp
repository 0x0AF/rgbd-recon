#include "Controller.h"
#include "Frustum.h"
#include "assimp_utilities.h"
#include <calibration/KinectCalibrationFile.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

const aiScene *Controller::import_file(const std::string &pFile, Assimp::Importer &importer)
{
    importer.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, 1);
    const aiScene *scene = importer.ReadFile(pFile, aiProcess_PreTransformVertices | aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

    if(!scene)
    {
        std::cerr << importer.GetErrorString() << std::endl;

        return nullptr;
    }

    return scene;
}
Controller::~Controller()
{
    delete _bbox;
    delete _calib_files;
}
Controller::Controller(std::string &filename_poi, std::string &filename_environment)
{
    // _importer_environment.SetExtraVerbose(true);
    // _importer_poi.SetExtraVerbose(true);

    //_scene_environment = Controller::import_file(filename_environment, _importer_environment);
    _scene_poi = Controller::import_file(filename_poi, _importer_poi);

    if(_scene_poi == nullptr /*|| _scene_environment == nullptr*/)
    {
        std::cerr << "Imports failed" << std::endl;
        return;
    }

    // printAiSceneInfo(_scene_environment);
    // printAiSceneInfo(_scene_poi);

    std::vector<std::string> calib_filenames(4, "");
    calib_filenames[0] = "/home/fusion_4d/Desktop/data/synthetic_rgbd/50.yml";
    calib_filenames[1] = "/home/fusion_4d/Desktop/data/synthetic_rgbd/51.yml";
    calib_filenames[2] = "/home/fusion_4d/Desktop/data/synthetic_rgbd/56.yml";
    calib_filenames[3] = "/home/fusion_4d/Desktop/data/synthetic_rgbd/57.yml";

    gloost::Point3 bbox_min{-0.1999f, 0.f, -1.f};
    gloost::Point3 bbox_max{1.2f, 1.4f, 0.4f};

    _bbox = new gloost::BoundingBox();
    _bbox->setPMin(bbox_min);
    _bbox->setPMax(bbox_max);
    _cv = new kinect::CalibVolumes(calib_filenames, *_bbox);
    _calib_files = new kinect::CalibrationFiles(calib_filenames);

    for(unsigned int i = 0; i < 4; i++)
    {
        std::cout << "Cam: " << glm::to_string(_cv->getCameraPositions()[i]) << std::endl;

        auto frustum = _cv->getFrustum(i);
        auto corners = frustum.m_corners;
        glm::fvec3 center_near((corners[0] + corners[1] + corners[2] + corners[3]) / 4.0f);
        glm::fvec3 center_far((corners[4] + corners[5] + corners[6] + corners[7]) / 4.0f);
        glm::fvec3 view_dir{center_far - center_near};

        std::cout << "Look at: " << glm::to_string(_cv->getCameraPositions()[i] + view_dir) << std::endl;

        std::cout << "Up: " << glm::to_string((corners[5] + corners[6]) / 2.f - center_far) << std::endl;
    }

    /*for(auto calib : _calib_files->getCalibs())
    {
        std::cout << calib.getNear() << std::endl;
        std::cout << calib.getFar() << std::endl;
        calib.printInfo();
    }*/

    _cv->setStartTextureUnit(12);
    _cv->loadInverseCalibs("/home/fusion_4d/Desktop/data/synthetic_rgbd/");
    _cv->setStartTextureUnitInv(20);
}
const aiScene *Controller::get_environment() { return _scene_environment; }
const aiScene *Controller::get_poi() { return _scene_poi; }
kinect::CalibVolumes *Controller::get_cv() { return _cv; }
