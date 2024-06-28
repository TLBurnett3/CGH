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

       std::mutex                 _tAccess;
       std::mutex                 _wAccess;
       std::atomic<bool>          _run;
       std::atomic<bool>          _dispatch;
       std::condition_variable    _workerCondition;
       std::condition_variable    _waitCondition;
       std::vector<std::thread>   _workers;

       std::vector<uint8_t *>     _pMemLst;
       size_t                     _memSize;

       std::vector<uint8_t *>     _pRBinMemLst;
       std::vector<uint8_t *>     _pGBinMemLst;
       std::vector<uint8_t *>     _pBBinMemLst;
       std::vector<uint8_t *>     _pABinMemLst;
       size_t                     _binMemSize;

       std::vector<Common::Timer> _timerLst;
       std::vector<int>           _procLst;  

       PntCld                     _pntCld; 
       std::vector<glm::dvec4>    _phaseLst;

       cv::Mat                    _qSImgAss;
       cv::Mat                    _qSImgFin;
       cv::Mat                    _proofImgDbl;
       cv::Mat                    _proofImg;
       glm::ivec2                 _proofSize;
       glm::ivec2                 _proofStp;

       double                     _proofMinDbl;
       double                     _proofMaxDbl;
       bool                       _proofUpdate;

       double                     _accTime;
       int                        _numProcessed;

       Common::Timer              _runTimer;
       Common::Timer              _printTimer;

  public:   

    // Methods
    private:
    protected:

    //  void createSWaveRow      (const uint32_t wId,const int row);
    //  void writeSWaveRow       (const uint32_t wId,const int row);

      void processWaveFrontRow (const uint32_t wId,const int row);
      void writeWaveFrontRow   (const uint32_t wId,const int row);
      void proofRow            (const uint32_t wId,const int row);

      int  findWaveFrontRow    (cv::Point *pIdx);

      void updateQStat(const uint32_t wId,const cv::Point &idx);

      void worker(void);

      int  createJob(void);   
      int  loadPointCloud(const std::filesystem::path& filePath);


    public:
      void printTable(void);
      void updateProofImg(void);

      bool run(void)
      { return _run; }

      void start(void);
      void stop(void);

      void exec(void);
  
      int  init(const std::filesystem::path &filePath,Core::SpJob &spJob);

      Executor(void);
      ~Executor();
  };
};
//---------------------------------------------------------------------

