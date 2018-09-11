#ifndef SYNTHETIC_RGBD_CONTROLLER_H
#define SYNTHETIC_RGBD_CONTROLLER_H

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <BoundingBox.h>
#include <memory>
#include <calibration/CalibVolumes.hpp>

class Controller
{
  public:
    Controller(std::string &filename_poi, std::string &filename_environment);
    ~Controller();

    const aiScene *get_environment();
    const aiScene *get_poi();

  private:
    const aiScene *import_file(const std::string &pFile, Assimp::Importer&importer);

    Assimp::Importer _importer_environment, _importer_poi;
    const aiScene *_scene_environment, *_scene_poi;

    std::shared_ptr<gloost::BoundingBox> _bbox;
    std::shared_ptr<kinect::CalibVolumes> _cv;
};

#endif // SYNTHETIC_RGBD_CONTROLLER_H
