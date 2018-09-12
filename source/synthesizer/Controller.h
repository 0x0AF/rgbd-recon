#ifndef SYNTHETIC_RGBD_CONTROLLER_H
#define SYNTHETIC_RGBD_CONTROLLER_H

#include <iostream>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <BoundingBox.h>
#include <memory>
#include <calibration/CalibVolumes.hpp>
#include <calibration/calibration_files.hpp>

class Controller
{
  public:
    Controller(std::string &filename_poi, std::string &filename_environment);
    ~Controller();

    const aiScene *get_environment();
    const aiScene *get_poi();
    kinect::CalibVolumes *get_cv();

  private:
    const aiScene *import_file(const std::string &pFile, Assimp::Importer&importer);

    Assimp::Importer _importer_environment, _importer_poi;
    const aiScene *_scene_environment, *_scene_poi;

    gloost::BoundingBox *_bbox;
    kinect::CalibVolumes *_cv;
    kinect::CalibrationFiles *_calib_files;
};

#endif // SYNTHETIC_RGBD_CONTROLLER_H
