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
#include "CGH/ExecutorCpp.h"

using namespace CGH;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// printStatus
//---------------------------------------------------------------------
void ExecutorCpp::printStatus(void)
{
int    rMin  = _spJob->_numPixels.y;
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
  float c        = (float)rMin / (float)_spJob->_numPixels.y * 100.0f;
  int    remain  = _spJob->_numPixels.y - rMin;

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
void ExecutorCpp::updateProofImg(void)
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
  std::filesystem::path fPath = _spJob->_outPath;

    fPath /= "ProofImg.png";

    cv::imwrite(fPath.string(),_proofImg);
  }

  {
  std::filesystem::path fPath = _spJob->_outPath;
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
// processWaveFrontRow
//---------------------------------------------------------------------
void ExecutorCpp::processWaveFrontRow(const uint32_t wId,const int row)
{
glm::ivec2        hIdx(0,row);
glm::dvec3        vS(_spJob->_pixelSize.x,_spJob->_pixelSize.y,1.0);
glm::dvec3        vT(_spJob->_numPixels.x >> 1,_spJob->_numPixels.y >> 1,0.0);
glm::dvec4        *pM = (glm::dvec4 *)_pMemLst[wId];
pcl::PointXYZRGBA *pCld = &_pntCld[0];
pcl::PointXYZRGBA *pEnd = pCld + _pntCld.size();
glm::dvec4        k = glm::dvec4(2.0 * glm::pi<double>()) / _spJob->_waveLengths;
float             hA = glm::radians(_spJob->_fov / 2.0);

  while ((_run) && (hIdx.x < _spJob->_numPixels.x))
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
void ExecutorCpp::updateQStat(const uint32_t wId,const cv::Point &idx)
{
std::filesystem::path fPath = _spJob->_outPath;

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
void ExecutorCpp::writeWaveFrontRow(const uint32_t wId,const int row)
{
std::filesystem::path fPath = _spJob->_outPath;

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
void ExecutorCpp::proofRow(const uint32_t wId,const int row)
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
int ExecutorCpp::findWaveFrontRow(cv::Point *pIdx)
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
void ExecutorCpp::worker(void)
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

        if (_spJob->_isWaveField)
          writeWaveFrontRow(wId,row);

        updateQStat(wId,idx);
      }
    }
    else
      _procLst[wId] = -1;
  }
}



//---------------------------------------------------------------------
// loadProof
//---------------------------------------------------------------------
int ExecutorCpp::loadProof(void)
{
std::filesystem::path fPathR = _spJob->_outPath;
int                   rc     = -1;

  fPathR /= "ProofImg.raw";

  if (std::filesystem::exists(fPathR))
  {
    size_t n = _proofImgDbl.rows * _proofImgDbl.cols;
    FILE* fp = fopen(fPathR.string().c_str(), "rb");

    std::cout << "Loading " << fPathR.string() << std::endl;

    if (fp)
    {
      size_t m = fread(_proofImgDbl.data, sizeof(double), n, fp);

      if (m == n)
      {
        double* p = (double*)_proofImgDbl.data;
        double* pEnd = p + n;

        while (p < pEnd)
        {
          if (*p != 0)
          {
            _proofMinDbl = glm::min(*p, _proofMinDbl);
            _proofMaxDbl = glm::max(*p, _proofMaxDbl);
          }
          else
            p = pEnd;

          p++;
        }
      }

      fclose(fp);

      rc = 0;

      std::cout << "Min: " << _proofMinDbl << "  Max:" << _proofMinDbl << std::endl;
      std::cout << "Finished Loading " << fPathR.string() << std::endl;
    }
  }
  
  return rc;
}


//---------------------------------------------------------------------
// readyQStat
//---------------------------------------------------------------------
int ExecutorCpp::readyQStat(bool &bLP)
{
int                   rc        = 0;
std::filesystem::path fPath     = _spJob->_outPath;
bool                  loadProof = false;
glm::ivec2            qSDim(sqrt(_spJob->_numPixels.x));

  fPath /= "QStat.png";

  while ((qSDim.x * qSDim.y) != _spJob->_numPixels.x)
  {
    qSDim.y--;
    qSDim.x = _spJob->_numPixels.x / qSDim.y;
  }

  if (!std::filesystem::exists(fPath))
  {
    _qSImgAss.create(qSDim.y, qSDim.x, CV_8UC1);
    _qSImgFin.create(qSDim.y, qSDim.x, CV_8UC1);

    _qSImgAss.setTo(0);
    _qSImgFin.setTo(0);

    cv::imwrite(fPath.string(), _qSImgFin);
  }
  else
  {
    _qSImgFin = cv::imread(fPath.string());
    _qSImgAss = _qSImgFin.clone();

    if ((_qSImgFin.cols != qSDim.x) || (_qSImgFin.rows != qSDim.y))
    {
      std::cout << "QStat Img Dim mismatch" << std::endl;
      std::cout << "!QSTat Dim!: " << "[" << qSDim.x << "," << qSDim.y << "]" << std::endl;

      rc = -1;
    }
    else
      bLP = true;

    {
    uint8_t v = _qSImgFin.at<uint8_t>(_qSImgFin.rows - 1, _qSImgFin.cols - 1);

      if (v == 0xff)
      {
        std::cout << "Job Already Complete." << std::endl;
        _run = false;

        rc = -1;
      }
    }
  }

  return rc;
}


//---------------------------------------------------------------------
// exec
//---------------------------------------------------------------------
void ExecutorCpp::exec(void)
{
  if (_printTimer.seconds() > 5)
  {
    printStatus();

    if (_proofUpdate)
      updateProofImg();

    _printTimer.start();
  }
}



//---------------------------------------------------------------------
// init
//---------------------------------------------------------------------
int ExecutorCpp::init(const std::filesystem::path &filePath,Core::SpJob &spJob)
{
int   rc  = Executor::init(filePath,spJob);
bool  bLP = false;

  if (rc == 0)
    rc = readyQStat(bLP);

  if ((rc == 0) && bLP)
    rc = loadProof();

  if (rc == 0)
  {
    _memSize = sizeof(glm::dvec4) * _spJob->_numPixels.x;
    _binMemSize = _memSize / 8;

    if ((_binMemSize * 8) != _memSize)
      _binMemSize++;

    _timerLst.resize(_spJob->_numThreads);

    for (uint32_t i = 0;i < _spJob->_numThreads;i++)
    {
      _workers.emplace_back(&ExecutorCpp::worker,this);
      _procLst.emplace_back(-1);
    }

    for (uint32_t i = 0;i < _spJob->_numThreads;i++)
    {
      std::cout << "Allocating WaveFront Buffer [" << i << "]: " << _memSize << " bytes" << std::endl;
      _pMemLst.emplace_back(new uint8_t[_memSize]);
    }

    for (uint32_t i = 0;i < _spJob->_numThreads;i++)
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
void ExecutorCpp::start(void)
{
  Executor::start(); 

   _printTimer.start();

   _dispatch   = true;
  _workerCondition.notify_all();
}


//---------------------------------------------------------------------
// stop
//---------------------------------------------------------------------
void ExecutorCpp::stop(void)
{
  Executor::stop();

  _workerCondition.notify_all();
}


//---------------------------------------------------------------------
// ExecutorCpp
//---------------------------------------------------------------------
ExecutorCpp::ExecutorCpp(void) : Executor(),
                            _tAccess(),
                            _wAccess(),
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
                            _qSImgAss(),
                            _qSImgFin(),
                            _accTime(0),
                            _numProcessed(0),
                            _printTimer()
{
}


//---------------------------------------------------------------------
// ~ExecutorCpp
//---------------------------------------------------------------------
ExecutorCpp::~ExecutorCpp()
{
double rTime = _runTimer.seconds();

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

