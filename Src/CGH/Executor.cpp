//---------------------------------------------------------------------
// MIT License
// 
// Copyright (c) 2024 TLBurnett3
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//---------------------------------------------------------------------

// Executor.cpp 
// Thomas Burnett


//---------------------------------------------------------------------
// Includes
// System

#include <cmath>


// 3rdPartyLibs
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <pcl/io/pcd_io.h>
#include "opencv2/imgproc/imgproc.hpp"

// SCS
#include "CGH/Executor.h"

using namespace CGH;
//---------------------------------------------------------------------




//---------------------------------------------------------------------
// createDirectories
//---------------------------------------------------------------------
int Executor::createDirectories(void)
{
int rc = 0;
std::filesystem::path outPath = _spJob->_outPath;
std::filesystem::path wfOutPath = outPath;

  wfOutPath /= "WaveField";

  if (!std::filesystem::exists(outPath))
    rc |= std::filesystem::create_directories(outPath) ? 0 : -1;

  if (_spJob->_isWaveField && !std::filesystem::exists(wfOutPath))
    rc |= std::filesystem::create_directories(wfOutPath) ? 0 : -1;

  return rc;
}


//---------------------------------------------------------------------
// loadPointCloud
//---------------------------------------------------------------------
int Executor::loadPointCloud(const std::filesystem::path &filePath)
{
int                   rc          = -1;
std::filesystem::path pclFilePath = filePath / _spJob->_pntCld;

  if (std::filesystem::exists(pclFilePath))
  { 
    if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(pclFilePath.string(),_pntCld) == 0)
    {
    glm::vec3  vMin(FLT_MAX);
    glm::vec3  vMax(-FLT_MAX);

      std::cout << "Point Cloud: " << filePath << " - " << _pntCld.size() << " points." << std::endl;

      std::cout << "Calculating Luminance " << std::endl;
      {
        for (size_t i = 0; i < _pntCld.size(); i++)
        {
        glm::vec3 clr(_pntCld[i].r,_pntCld[i].g,_pntCld[i].b);
        glm::vec3 v(_pntCld[i].x,_pntCld[i].y,_pntCld[i].z);

          vMin = glm::min(vMin,v);
          vMax = glm::max(vMax,v);

          if (clr == glm::fvec3(0))
            _pntCld[i].a = 1.0f;
          else
          {
            _pntCld[i].a = (_pntCld[i].r * _spJob->_luminance.r) +
                           (_pntCld[i].g * _spJob->_luminance.g) +
                           (_pntCld[i].b * _spJob->_luminance.b);
          }
        }
      }
      std::cout << "Finished Luminance Calculation " << std::endl;  

      std::cout << "Calculating Random Phases " << std::endl;
      {
      double pi2 = 2.0 * glm::pi<double>();

        _phaseLst.resize(_pntCld.size());

        for (size_t i = 0; i < _phaseLst.size(); i++)
        {
          _phaseLst[i] = glm::dvec4(glm::linearRand(0.0,pi2),glm::linearRand(0.0,pi2),glm::linearRand(0.0,pi2),glm::linearRand(0.0,pi2));
          _phaseLst[i] = glm::sin(_phaseLst[i]);
        }
      }
      std::cout << "Finished Random Phases " << std::endl;

      {
      glm::vec3 vDim = vMax - vMin;
      glm::vec3 vCen = (vMax + vMin) / 2.0f;

        std::cout << "Point Cloud Center: [" << vCen.x << "," << vCen.y << "," << vCen.z << "]" << std::endl;
        std::cout << "Point Cloud Min   : [" << vMin.x << "," << vMin.y << "," << vMin.z << "]" << std::endl;
        std::cout << "Point Cloud Max   : [" << vMax.x << "," << vMax.y << "," << vMax.z << "]" << std::endl;
        std::cout << "Point Cloud Dim.  : [" << vDim.x << "," << vDim.y << "," << vDim.z << "]" << std::endl;

        if (_spJob->_mTpc != glm::mat4(1.0f))
        {
        glm::mat4 mT = glm::translate(glm::mat4(1),-vCen);

          mT = _spJob->_mTpc * mT; 

          std::cout << "Transform PointCloud " << std::endl;
          for (size_t i = 0; i < _pntCld.size(); i++)
          {
          glm::vec4 v(_pntCld[i].x,_pntCld[i].y,_pntCld[i].z,1);
          
            v = mT * v;

            _pntCld[i].x = v.x;
            _pntCld[i].y = v.y;
            _pntCld[i].z = v.z;
          }

          {
          glm::vec4 v(vMin,1.0);

            v = mT * v;

            vMin = glm::vec3(v);
          }

          {
          glm::vec4 v(vMax,1.0);

            v = mT * v;

            vMax = glm::vec3(v);
          }

          vDim = vMax - vMin;
          vCen = (vMax + vMin) / 2.0f;

          std::cout << "New Point Cloud Center: [" << vCen.x << "," << vCen.y << "," << vCen.z << "]" << std::endl;
          std::cout << "New Point Cloud Min   : [" << vMin.x << "," << vMin.y << "," << vMin.z << "]" << std::endl;
          std::cout << "New Point Cloud Max   : [" << vMax.x << "," << vMax.y << "," << vMax.z << "]" << std::endl;
          std::cout << "New Point Cloud Dim.  : [" << vDim.x << "," << vDim.y << "," << vDim.z << "]" << std::endl;

          std::cout << "Finished Transform PointCloud " << std::endl;  
        }
      }

      rc = 0;
    }
    else
      std::cout << "Failed to load: " << pclFilePath << std::endl;
  }
  else
    std::cout << "File Not Found: " << pclFilePath << std::endl;

  return rc;
}


//---------------------------------------------------------------------
// init
//---------------------------------------------------------------------
int Executor::init(const std::filesystem::path &filePath,Core::SpJob &spJob)
{
int rc = 0;

  _spJob = spJob;

  if (rc == 0)
    rc = createDirectories();

  if (rc == 0)
    rc = loadPointCloud(filePath);

  _proofSize  = glm::min(_proofSize, _spJob->_numPixels);
  _proofStp   = _spJob->_numPixels / _proofSize;
  _proofSize  = _spJob->_numPixels / _proofStp;

  _proofImgDbl.create(_proofSize.y, _proofSize.x, CV_64FC1);
  _proofImgDbl.setTo(0);
  _proofImg.create(_proofImgDbl.rows, _proofImgDbl.cols, CV_16UC1);
  _proofImg.setTo(0);

  std::cout << "Proof Image [" << _proofImgDbl.cols << "," << _proofImgDbl.rows << "]: " << std::endl;

  return rc;
}


//---------------------------------------------------------------------
// start
//---------------------------------------------------------------------
void Executor::start(void)
{
  _runTimer.start();
}


//---------------------------------------------------------------------
// stop
//---------------------------------------------------------------------
void Executor::stop(void)
{
  _run      = false;
}


//---------------------------------------------------------------------
// Executor
//---------------------------------------------------------------------
Executor::Executor(void) :  _spJob(),
                            _run(true),
                            _pntCld(),
                            _phaseLst(),
                            _proofImgDbl(),
                            _proofImg(),
                            _proofSize(1024),
                            _proofStp(0),
                            _proofMinDbl(FLT_MAX),
                            _proofMaxDbl(-FLT_MAX),
                            _proofUpdate(false),
                            _runTimer()
{
}


//---------------------------------------------------------------------
// ~Executor
//---------------------------------------------------------------------
Executor::~Executor()
{
  stop();
}

