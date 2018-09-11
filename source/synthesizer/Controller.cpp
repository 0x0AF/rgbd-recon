#include "Controller.h"
#include "assimp_utilities.h"

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
    _bbox.reset();
    _cv.reset();
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

    std::vector<std::string> calib_filenames;
    calib_filenames.emplace_back("/home/xaf/Desktop/MSc/data/synthetic_rgbd/50.yml");
    calib_filenames.emplace_back("/home/xaf/Desktop/MSc/data/synthetic_rgbd/51.yml");
    calib_filenames.emplace_back("/home/xaf/Desktop/MSc/data/synthetic_rgbd/56.yml");
    calib_filenames.emplace_back("/home/xaf/Desktop/MSc/data/synthetic_rgbd/57.yml");

    gloost::Point3 bbox_min{-0.1999f, 0.f, -1.f};
    gloost::Point3 bbox_max{1.2f, 1.4f, 0.4f};

    _bbox = std::make_shared<gloost::BoundingBox>();
    _bbox->setPMin(bbox_min);
    _bbox->setPMax(bbox_max);
    _cv = std::make_shared<kinect::CalibVolumes>(calib_filenames, *_bbox);
    _cv->setStartTextureUnit(12);
    _cv->loadInverseCalibs("/home/xaf/Desktop/MSc/data/synthetic_rgbd/");
    _cv->setStartTextureUnitInv(20);
}
const aiScene *Controller::get_environment() { return _scene_environment; }
const aiScene *Controller::get_poi() { return _scene_poi; }
