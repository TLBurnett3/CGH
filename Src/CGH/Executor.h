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

// Executor.h
// Thomas Burnett

#pragma once

//---------------------------------------------------------------------
// Includes
// System
#include <thread>
#include <condition_variable>
#include <atomic>
#include <filesystem>

// 3rdPartyLibs
#include <opencv2/highgui/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// CGH
#include "Common/JSon.h"
#include "Core/Job.h"
#include "Common/Timer.h"
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// Classes
namespace CGH
{
  class Executor
  {
    // Defines
    private:
    protected:
      typedef pcl::PointCloud<pcl::PointXYZRGBA> PntCld;
      
    public:

    // Members
    private:
    protected:     
       Core::SpJob                _spJob;
       std::atomic<bool>          _run;

       PntCld                     _pntCld; 
       std::vector<glm::dvec4>    _phaseLst;

       cv::Mat                    _proofImgDbl;
       cv::Mat                    _proofImg;
       glm::ivec2                 _proofSize;
       glm::ivec2                 _proofStp;

       double                     _proofMinDbl;
       double                     _proofMaxDbl;
       bool                       _proofUpdate;

       Common::Timer              _runTimer;


  public:   

    // Methods
    private:
 
    protected:
     inline double map(double x, double in_min, double in_max, double out_min, double out_max)
      { return ((x - in_min) * (out_max - out_min)) / ((in_max - in_min) + out_min); }    

      int  createDirectories(void);
      int  loadPointCloud   (const std::filesystem::path& filePath);


    public:
      bool run(void)
      { return _run; }      

      virtual void printStatus(void)        = 0;
      virtual void updateProofImg(void)     = 0;

      virtual void start(void);
      virtual void exec(void)               = 0;
      virtual void stop(void);
  
      virtual int  init(const std::filesystem::path &filePath,Core::SpJob &spJob);

      Executor(void);
      virtual ~Executor();
  };
};
//---------------------------------------------------------------------

