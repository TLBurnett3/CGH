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
#include <filesystem>
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
// printTable
//---------------------------------------------------------------------
void Executor::printTable(void) 
{
int    rMin  = _job._numPixels.y;
int    rMax  = 0;

  std::cout << std::endl;

  for (size_t i = 0; i < _procLst.size(); i++)
  {
  double tS = _timerLst[i].seconds();
  double tM = tS / 60.0;

    rMin = glm::min(rMin,_procLst[i]);
    rMax = glm::max(rMax,_procLst[i]);

    std::cout << "Worker:" << i << "  ";
    std::cout << "Row:" << _procLst[i] << "  ";
    std::cout << "tS: " << tS << "s  (" << tM << "m)  ";
    std::cout << std::endl;
  }

  {
  float c        = (float)rMin / (float)_job._numPixels.y * 100.0f;
  int    remain  = _job._numPixels.y - rMin;

    std::cout << "Complete: %" << c  << "  ";
    std::cout << "Remaining: " << remain  << "  ";
    std::cout << std::endl;

    if (rMin)
    {
    double rTime   = _runTimer.seconds();
    double avgTime = _numProcessed ? (_accTime / (double)_numProcessed) : 0;
    double estTime = ((double)remain * avgTime) / (double)_procLst.size();

      std::cout << "Min Value: " << _proofMinDbl << "  Max Value: " << _proofMaxDbl << std::endl;
      std::cout << "Run Time: " <<  rTime  << "s (" << rTime / (60.0 * 60.0) << "h)  " << std::endl;
      std::cout << "Avg. Row Time: " << avgTime << "s (" << avgTime / 60.0 << "m)  " << std::endl;
      std::cout << "Est. Time Remaining: " <<  estTime  << "s (" << estTime / (60.0 * 60.0) << "h)  " << std::endl;
    }
  }

  std::cout << std::endl;
}

//---------------------------------------------------------------------
// updateProofImg
//---------------------------------------------------------------------
#define map(x,in_min,in_max,out_min,out_max) (((x - in_min) * (out_max - out_min)) / ((in_max - in_min) + out_min))

void Executor::updateProofImg(void) 
{
size_t    n        = _proofImgDbl.rows * _proofImgDbl.cols;
double    *pD      = (double *)_proofImgDbl.ptr(0);
double    *pDEnd   = pD + n;
uint16_t *pV       = (uint16_t *)_proofImg.ptr(0);
double    dMin     = _proofMinDbl;
double    dMax     = _proofMaxDbl;

  while (pD < pDEnd)
  {
  uint16_t c = (uint16_t)map(*pD,dMin,dMax,((double)0),((double)0xffff));

    *pV = c;
   
    pD++;
    pV++;
  }

  cv::imshow("ProofImg",_proofImg);

  {
  std::filesystem::path fPath = _job._outPath;

    fPath /= "ProofImg.png";

    cv::imwrite(fPath.string(),_proofImg);
  }

  {
  std::filesystem::path fPath = _job._outPath;
  FILE *fp = 0;

    fPath /= "ProofImg.raw";

    fp = fopen(fPath.string().c_str(),"wb");
    if (fp)
    {
    size_t n   = _proofImgDbl.rows * _proofImgDbl.cols;

      fwrite(_proofImgDbl.data,sizeof(double),n,fp);
      fclose(fp);
    }
  }

  std::unique_lock<std::mutex> lock(_wAccess);
  {
    _proofUpdate = false;
  }
}


//---------------------------------------------------------------------
// exec
//---------------------------------------------------------------------
void Executor::exec(void) 
{
  if (_printTimer.seconds() > 5)
  {
    printTable();

    if (_proofUpdate)
      updateProofImg();

//    cv::imshow("QStat",_qSImgFin);

    _printTimer.start();
  }
}



//---------------------------------------------------------------------
// processWaveFrontRow
//---------------------------------------------------------------------
void Executor::processWaveFrontRow(const uint32_t wId,const int row)
{
glm::ivec2        hIdx(0,row);
glm::dvec3        vS(_job._pixelSize.x,_job._pixelSize.y,1.0);
glm::dvec3        vT(_job._numPixels.x >> 1,_job._numPixels.y >> 1,0.0);
glm::dvec4        *pM = (glm::dvec4 *)_pMemLst[wId];
pcl::PointXYZRGBA *pCld = &_pntCld[0];
pcl::PointXYZRGBA *pEnd = pCld + _pntCld.size();
glm::dvec4        k = glm::dvec4(2.0 * glm::pi<double>()) / _job._waveLengths;
float             hA = glm::radians(_job._fov / 2.0);

  while ((_run) && (hIdx.x < _job._numPixels.x))
  {
  glm::dvec3        vC(hIdx.x,hIdx.y,0);
  pcl::PointXYZRGBA *pS(pCld);
  glm::dvec4        *pPh(&_phaseLst[0]);

    vC -= vT;  
    vC *= vS;  

    *pM = glm::dvec4(0);

    while (pS < pEnd)
    {
    glm::dvec3 vP(pS->x,pS->y,pS->z);
    glm::dvec3 vD(vC.x,vC.y,pS->z);
    double     o  = glm::distance(vD,vP);
    double     a  = glm::distance(vD,vC);
    double     th = glm::atan(o,a);

      if (th <= hA)
      {
      glm::dvec4 ap(pS->r,pS->g,pS->b,pS->a);
      double     d = glm::distance(vC,vP);
      glm::dvec4 v = (ap * glm::cos((k * d) + *pPh)) / d;

        *pM += v;
      }

      pS++;
      pPh++;
    }

    pM++;
    hIdx.x++;
  }
}


//---------------------------------------------------------------------
// updateQStat
//---------------------------------------------------------------------
void Executor::updateQStat(const uint32_t wId,const cv::Point &idx)
{
std::filesystem::path fPath = _job._outPath;

  fPath /= "QStat.png";

  {
  std::unique_lock<std::mutex> lock(_wAccess);

    _accTime += _timerLst[wId].seconds();
    _numProcessed++;

    _qSImgFin.at<uint8_t>(idx) = 0xff;

    cv::imwrite(fPath.string(),_qSImgFin);

    if ((idx.x == (_qSImgFin.cols - 1)) && (idx.y == (_qSImgFin.rows - 1)))
      _run = false;
  }
}


//---------------------------------------------------------------------
// writeWaveFrontRow
//---------------------------------------------------------------------
void Executor::writeWaveFrontRow(const uint32_t wId,const int row)
{
std::filesystem::path fPath = _job._outPath;

  fPath /= "WaveField";
  fPath /= "CGHRow_";
  fPath += std::to_string(row);
  fPath += ".wf";

  {
  FILE *fp = fopen(fPath.string().c_str(),"wb");

    if (fp)
    {
      fwrite(_pMemLst[wId],1,_memSize,fp);

      fclose(fp);
    }
  }
}


//---------------------------------------------------------------------
// proofRow
//---------------------------------------------------------------------
void Executor::proofRow(const uint32_t wId,const int row)
{
int         y  = row / _proofStp.y;
glm::dvec4 *pS = (glm::dvec4 *)_pMemLst[wId];
double     dMin(FLT_MAX);
double     dMax(-FLT_MAX);

  for (int x = 0;x < _proofImgDbl.cols;x++)
  {
    _proofImgDbl.at<double>(y,x) = (*pS).w;

    dMin = glm::min(dMin,(*pS).w);
    dMax = glm::max(dMax,(*pS).w);

    pS += _proofStp.x;
  }

  std::unique_lock<std::mutex> lock(_wAccess);
  {
    _proofMinDbl = glm::min(_proofMinDbl,dMin);
    _proofMaxDbl = glm::max(_proofMaxDbl,dMax);

    _proofUpdate = true;
  }
}


//---------------------------------------------------------------------
// findWaveFrontRow
//---------------------------------------------------------------------
int Executor::findWaveFrontRow(cv::Point *pIdx)
{
int        row = 0;
uint8_t    v;
cv::Point  idx;

  for (idx.y = 0;idx.y < _qSImgAss.rows;idx.y++)
  {
    for (idx.x = 0;idx.x < _qSImgAss.cols;idx.x++)
    {
      v = _qSImgAss.at<uint8_t>(idx);

      if (v == 0)
      {
        _qSImgAss.at<uint8_t>(idx) = 0xff;

        *pIdx = idx;

        return row;
      }

      row++;
    }
  }

  return -1;
}



//---------------------------------------------------------------------
// worker
//---------------------------------------------------------------------
static uint32_t workerCount = 0;
void Executor::worker(void) 
{
uint32_t    wId = 0;

  {
  std::unique_lock<std::mutex> lock(_tAccess);

    wId = workerCount++;
  }

  while (_run)
  {
  int       row = -1;
  cv::Point idx;

    {
    std::unique_lock<std::mutex> lock(_tAccess);

      _workerCondition.wait(lock,[this]{ return ((_run == false) | (_dispatch == true)); });  

      if (_run && _dispatch)
      {        
        row = findWaveFrontRow(&idx);

        if (row == -1)
          _dispatch = false;
      }
    }

    if (row != -1)
    {
      _procLst[wId] = row;
      _timerLst[wId].start();

      processWaveFrontRow(wId,row);

      if (_run)
      { 
        if ((row % _proofStp.y) == 0)
          proofRow(wId,row);

        if (_job._isWaveField)
          writeWaveFrontRow(wId,row);

        updateQStat(wId,idx);
      }
    }
    else
      _procLst[wId] = -1;
  }
}


//---------------------------------------------------------------------
// createJob
//---------------------------------------------------------------------
int Executor::createJob(void)
{
int rc = 0;
std::filesystem::path outPath   = _job._outPath;
std::filesystem::path wfOutPath = outPath;
std::filesystem::path swOutPath = outPath;

  wfOutPath /= "WaveField";
  swOutPath /= "SWave";

  if (!std::filesystem::exists(outPath))
    rc |= std::filesystem::create_directories(outPath) ? 0 : -1;

  if (_job._isWaveField && !std::filesystem::exists(wfOutPath))
    rc |= std::filesystem::create_directories(wfOutPath) ? 0 : -1;

  if (rc == 0)
  {
  std::filesystem::path fPath  = outPath;
  glm::ivec2 qSDim(sqrt(_job._numPixels.x));
  bool loadProof = false;

    fPath  /= "QStat.png";

    while ((qSDim.x * qSDim.y) != _job._numPixels.x)
    {
      qSDim.y--;
      qSDim.x = _job._numPixels.x / qSDim.y;
    }

    if (!std::filesystem::exists(fPath))
    {
      _qSImgAss.create(qSDim.y,qSDim.x,CV_8UC1);
      _qSImgFin.create(qSDim.y,qSDim.x,CV_8UC1);

      _qSImgAss.setTo(0);
      _qSImgFin.setTo(0);

       cv::imwrite(fPath.string(),_qSImgFin);
    }
    else
    {
      _qSImgFin = cv::imread(fPath.string());
      _qSImgAss = _qSImgFin.clone();

      if ((_qSImgFin.cols != qSDim.x) || (_qSImgFin.rows != qSDim.y))
      {
        std::cout << "QStat Img Dim mismatch" << std::endl;
        std::cout << "!QSTat Dim!: "     << "[" << qSDim.x << "," << qSDim.y << "]" << std::endl;

        rc = -1;
      }
      else
        loadProof = true;

      {
      uint8_t v = _qSImgFin.at<uint8_t>(_qSImgFin.rows -1,_qSImgFin.cols - 1);

        if (v == 0xff)
        {
          std::cout << "Job Already Complete." << std::endl;
          _run = false;
          rc = -1;
        }
      }
    }

    _proofSize = glm::min(_proofSize,_job._numPixels);
    _proofStp  = _job._numPixels / _proofSize;
    _proofSize = _job._numPixels / _proofStp;

    _proofImgDbl.create(_proofSize.y,_proofSize.x,CV_64FC1);
    _proofImgDbl.setTo(0);
    _proofImg.create(_proofImgDbl.rows,_proofImgDbl.cols,CV_16UC1);
    _proofImg.setTo(0);

    std::cout << "Proof Image [" << _proofImgDbl.cols << "," << _proofImgDbl.rows << "]: " << std::endl;
    std::cout << "QSTat Dim: "     << "[" << _qSImgFin.cols << "," << _qSImgFin.rows << "]" << std::endl;

    if (0)
    {
    std::filesystem::path fPathP = outPath;

      fPathP /= "ProofImg.png";

      if (std::filesystem::exists(fPathP))
      {
        std::cout << "Loading " << fPathP.string() << std::endl;

        _proofImg = cv::imread(fPathP.string());

        std::cout << "Finished Loading " << fPathP.string() << std::endl;
      }
    }

    if (loadProof)
    {
    std::filesystem::path fPathR = _job._outPath;

      fPathR /= "ProofImg.raw";

      if (std::filesystem::exists(fPathR))
      {
      size_t n   = _proofImgDbl.rows * _proofImgDbl.cols;
      FILE   *fp = fopen(fPathR.string().c_str(),"rb");

        std::cout << "Loading " << fPathR.string() << std::endl;

        if (fp)
        {
        size_t m = fread(_proofImgDbl.data,sizeof(double),n,fp);

          if (m == n)
          {
          double *p    = (double *)_proofImgDbl.data;
          double *pEnd = p + n;

            while (p < pEnd)
            {
              if (*p != 0)
              {
                _proofMinDbl = glm::min(*p,_proofMinDbl);
                _proofMaxDbl = glm::max(*p,_proofMaxDbl);
              }
              else
                p = pEnd;

              p++;
            }
          }

          fclose(fp);

          std::cout << "Min: " << _proofMinDbl << "  Max:" << _proofMinDbl << std::endl;
          std::cout << "Finished Loading " << fPathR.string() << std::endl;
        }
      }
    }
  }

  return rc;
}


//---------------------------------------------------------------------
// loadPointCloud
//---------------------------------------------------------------------
int Executor::loadPointCloud(const char *fName)
{
int rc = -1;
std::filesystem::path filePath = fName;

  filePath = filePath.remove_filename();
  filePath /= _job._pntCld;

  if (pcl::io::loadPCDFile<pcl::PointXYZRGBA>(filePath.string(),_pntCld) == 0)
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
          _pntCld[i].a = (_pntCld[i].r * _job._luminance.r) +
                         (_pntCld[i].g * _job._luminance.g) +
                         (_pntCld[i].b * _job._luminance.b);
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

      if (_job._mTpc != glm::mat4(1.0f))
      {
      glm::mat4 mT = glm::translate(glm::mat4(1),-vCen);

        mT = _job._mTpc * mT; 

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
    std::cout << "Failed to load: " << filePath << std::endl;

  return rc;
}


//---------------------------------------------------------------------
// init
//---------------------------------------------------------------------
int Executor::init(const char *fName)
{
int rc = 0;
Common::JSon::Value doc;
uint32_t numThreads = 2;

  rc = Common::JSon::readFile(doc,fName);

  rc |= Common::JSon::parse(doc,"Debug",       _debug,       false);
  rc |= Common::JSon::parse(doc,"NumThreads",  numThreads,   false);

  if (rc == 0)
    rc = _job.init(doc);

  if (rc == 0)
    rc = createJob();

  if (rc == 0)
    rc = loadPointCloud(fName);

  if (rc == 0)
  {
    _memSize = sizeof(glm::dvec4) * _job._numPixels.x;
    _binMemSize = _memSize / 8;

    if ((_binMemSize * 8) != _memSize)
      _binMemSize++;

    _timerLst.resize(numThreads);

    for (uint32_t i = 0;i < numThreads;i++)
    {
      _workers.emplace_back(&Executor::worker,this);
      _procLst.emplace_back(-1);
    }

    for (uint32_t i = 0;i < numThreads;i++)
    {
      std::cout << "Allocating WaveFront Buffer [" << i << "]: " << _memSize << " bytes" << std::endl;
      _pMemLst.emplace_back(new uint8_t[_memSize]);
    }

    for (uint32_t i = 0;i < numThreads;i++)
    {
      std::cout << "Allocating Bin Buffers [" << i << "]: " << _binMemSize << " * 4 bytes" << std::endl;
      _pRBinMemLst.emplace_back(new uint8_t[_binMemSize]);
      _pGBinMemLst.emplace_back(new uint8_t[_binMemSize]);
      _pBBinMemLst.emplace_back(new uint8_t[_binMemSize]);
      _pABinMemLst.emplace_back(new uint8_t[_binMemSize]);
    }
  }

  return rc;
}


//---------------------------------------------------------------------
// start
//---------------------------------------------------------------------
void Executor::start(void)
{
  _dispatch   = true;
  _workerCondition.notify_all();

  _printTimer.start();
  _runTimer.start();
}


//---------------------------------------------------------------------
// stop
//---------------------------------------------------------------------
void Executor::stop(void)
{
  _run      = false;
  _workerCondition.notify_all();
}


//---------------------------------------------------------------------
// Executor
//---------------------------------------------------------------------
Executor::Executor(void) :  _debug(false),
                            _job(),
                            _tAccess(),
                            _wAccess(),
                            _run(true),
                            _dispatch(false),
                            _workerCondition(),
                            _waitCondition(),
                            _workers(),
                            _pMemLst(),
                            _memSize(0),
                            _pRBinMemLst(),
                            _pGBinMemLst(),
                            _pBBinMemLst(),
                            _pABinMemLst(),
                            _timerLst(),
                            _procLst(),
                            _pntCld(),
                            _phaseLst(),
                            _qSImgAss(),
                            _qSImgFin(),
                            _proofImgDbl(),
                            _proofImg(),
                            _proofSize(1024),
                            _proofStp(0),
                            _proofMinDbl(FLT_MAX),
                            _proofMaxDbl(-FLT_MAX),
                            _proofUpdate(false),
                            _accTime(0),
                            _numProcessed(0),
                            _runTimer(),
                            _printTimer()
{
}


//---------------------------------------------------------------------
// ~Executor
//---------------------------------------------------------------------
Executor::~Executor()
{
double rTime = _runTimer.seconds();

  stop();

  std::cout << "Run Time: " <<  rTime  << "s (" << rTime / (60.0 * 60.0) << "h)  " << std::endl;

  // wait for all threads to finish and join
  {
  size_t n = _workers.size();

    for (size_t i = 0;i < n;i++)
      _workers[i].join();
  }

  {
  size_t n = _pMemLst.size();

    for (size_t i = 0;i < n;i++)
    {
      delete _pMemLst[i];
      delete _pRBinMemLst[i];
      delete _pGBinMemLst[i];
      delete _pBBinMemLst[i];
      delete _pABinMemLst[i];
    }
  }
}

