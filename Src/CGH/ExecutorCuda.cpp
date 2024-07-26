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


// 3rdPartyLibs
#include "opencv2/imgproc/imgproc.hpp"

// CGH
#include "CGH/ExecutorCuda.h"
#include "WaveFrontAccum.cuh"

using namespace CGH;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// printStatus
//---------------------------------------------------------------------
void ExecutorCuda::printStatus(void)
{
  std::cout << std::endl;
}


//---------------------------------------------------------------------
// updateProofImg
//---------------------------------------------------------------------
void ExecutorCuda::updateProofImg(void)
{

}


//---------------------------------------------------------------------
// query
//---------------------------------------------------------------------
int ExecutorCuda::query(void)
{
int rc        = -1;
int dId       = 0;
int nDevices  = 0;

  cudaGetDeviceCount(&nDevices);

  if (nDevices && cudaGetDevice(&dId) == cudaSuccess)
  {
  cudaDeviceProp  prop;

    cudaGetDeviceProperties(&prop,dId);

    std::cout << "------------------------" << std::endl;
    std::cout << "Cuda Device Number: " << dId << std::endl;
    std::cout << "                                  Name: " << prop.name << std::endl;
    std::cout << "               Memory Clock Rate (MHz): " << prop.memoryClockRate / 1024 << std::endl;
    std::cout << "               Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "          Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
    std::cout << "          Total global memory (Gbytes): " << prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << std::endl;
    std::cout << "      Shared memory per block (Kbytes): " << prop.sharedMemPerBlock / 1024.0 << std::endl;
    std::cout << "                           Minor-Major: " << prop.minor << "-" << prop.major << std::endl;
    std::cout << "                             Warp-size: " << prop.warpSize << std::endl;
    std::cout << "                    Concurrent kernels: " << prop.concurrentKernels << std::endl;
    std::cout << "  Concurrent computation/communication: " << prop.deviceOverlap << std::endl;
    std::cout << "------------------------" << std::endl;


    rc = 0;
  }
  return rc;
}


//---------------------------------------------------------------------
// determineMinMax
//---------------------------------------------------------------------
void ExecutorCuda::determineMinMax(double &dMin,double &dMax)
{
double4 *pS     = (double4*)_pAccMem;
double4*pSEnd  = pS + (_spJob->_numPixels.x * _spJob->_numPixels.y);

  dMin =  FLT_MAX;
  dMax = -FLT_MAX;

  while (pS < pSEnd)
  {
    dMin = glm::min(dMin,pS->w);
    dMax = glm::max(dMax,pS->w);

    pS++;
  }
}


//---------------------------------------------------------------------
// createProofImage
//---------------------------------------------------------------------
void ExecutorCuda::createProofImage(double dMin,double dMax)
{
double4   *pS    = (double4  *)_pAccMem;
uint16_t  *pD    = (uint16_t *)_proofImg.data;

  for (uint16_t i = 0; i < _proofImg.rows; i++)
  {
  double4* pR = pS + (i * (_spJob->_numPixels.x * _proofStp.y));

    for (uint16_t j = 0; j < _proofImg.cols; j++)
    {
    uint16_t c = (uint16_t)map(pR->w, dMin, dMax, ((double)0), ((double)0xffff));

      *pD = c;

      pD++;
      pR += _proofStp.x;
    }
  }
}


//---------------------------------------------------------------------
// start
//---------------------------------------------------------------------
void ExecutorCuda::start(void)
{
  Executor::start(); 

}


//---------------------------------------------------------------------
// exec
//---------------------------------------------------------------------
void ExecutorCuda::exec(void)
{
void*                 pdWFAP = 0;
WaveFrontAccumParams  wfap;

  if (cudaMalloc(&pdWFAP,sizeof(WaveFrontAccumParams)) == cudaSuccess)
  {
    wfap._nRow = _spJob->_numPixels.y;
    wfap._nCol = _spJob->_numPixels.x;
    wfap._nPntCld = (int)_pntCld.size();

    wfap._vS      = make_double3(_spJob->_pixelSize.x,_spJob->_pixelSize.y,1.0);
    wfap._vT      = make_double3(_spJob->_numPixels.x >> 1, _spJob->_numPixels.y >> 1, 0.0);

    wfap._waveLengths = glmmake_double4(_spJob->_waveLengths);

    wfap._fov     = (double)_spJob->_fov;

    cudaMemcpy(pdWFAP,&wfap,sizeof(WaveFrontAccumParams),cudaMemcpyHostToDevice);

    {
    dim3 numThreads(128, 1, 1);
    dim3 numBlocks(_spJob->_numPixels.x / numThreads.x, 1, 1);
    Common::Timer  rT;

      launchWaveFrontAccum(numThreads,numBlocks,_pdAccMem,_pdPhaseLst,_pdPointCld,pdWFAP);
      cudaDeviceSynchronize();

      std::cout << "Wavefront Accumulation: " << rT.seconds() << "s" << std::endl;

      cudaMemcpy(_pAccMem, _pdAccMem, _nAccBytes, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();    
    }


    if (1)
    {
    double4        mm[2];
    Common::Timer  rT;

      // Gather min/max by row
      {
      dim3 numThreads(128, 1, 1);
      dim3 numBlocks(_spJob->_numPixels.x / numThreads.x, 1, 1);

        launchDetermineRowMinMax(numThreads, numBlocks, _pdAccMem, pdWFAP);
        cudaDeviceSynchronize();
      }

      // Gather all the row min/max, use just 1 thread
      {
      dim3 numThreads(1, 1, 1);
      dim3 numBlocks(1, 1, 1);

        launchDetermineFinalMinMax(numThreads, numBlocks, _pdAccMem, pdWFAP);
        cudaDeviceSynchronize();
      }

      std::cout << "Min/Max Determination: " << rT.seconds() << "s" << std::endl;

      cudaMemcpy(mm, _pdAccMem, sizeof(double4) * 2, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      std::cout << "Cuda Min: " << mm[0].x << ", " << mm[0].y << ", " << mm[0].z << ", " << mm[0].w << std::endl;
      std::cout << "Cuda Max: " << mm[1].x << ", " << mm[1].y << ", " << mm[1].z << ", " << mm[1].w << std::endl;

      _proofMinDbl = mm[0].w;
      _proofMaxDbl = mm[1].w;
    }

    cudaFree(pdWFAP);
  }
  _run = false;
}


//---------------------------------------------------------------------
// stop
//---------------------------------------------------------------------
void ExecutorCuda::stop(void)
{
  Executor::stop();

  {
  double rTime = _runTimer.seconds();
  double avgTime = rTime / (double)_spJob->_numPixels.y;

    std::cout << "Min Value: " << _proofMinDbl << "  Max Value: " << _proofMaxDbl << std::endl;
    std::cout << "Run Time: " << rTime << "s (" << rTime / (60.0 * 60.0) << "h)  " << std::endl;
    std::cout << "Avg. Row Time: " << avgTime << "s (" << avgTime / 60.0 << "m)  " << std::endl;
  }

  {
    std::cout << "Creating Proof Image" << std::endl;
    createProofImage(_proofMinDbl,_proofMaxDbl);
  }

  cv::imshow("ProofImg", _proofImg);

  {
  std::filesystem::path fPath = _spJob->_outPath;

    fPath /= "ProofImg.png";

    cv::imwrite(fPath.string(), _proofImg);
  }

}


//---------------------------------------------------------------------
// init
//---------------------------------------------------------------------
int ExecutorCuda::init(const std::filesystem::path& filePath, Core::SpJob& spJob)
{
  int       rc = Executor::init(filePath, spJob);

  if (rc == 0)
    rc = query();

  if (rc == 0)
  {
    _nAccBytes = sizeof(glm::dvec4) * _spJob->_numPixels.x * _spJob->_numPixels.y;
    _nPLstBytes = sizeof(glm::dvec4) * _phaseLst.size();
    _nPCldBytes = sizeof(Point) * _pntCld.size();

    std::cout << "Requested CPU Memory: " << _nAccBytes << std::endl;

    _pAccMem = new uchar[_nAccBytes];
    if (!_pAccMem)
    {
      std::cout << "Could not allocate CPU memory, fatel error" << std::endl;
      rc = -1;
    }

    std::cout << "Requested GPU Accumenlation Memory: " << _nAccBytes << std::endl;

    if (cudaMalloc(&_pdAccMem, _nAccBytes) == cudaSuccess)
      cudaMemset(_pdAccMem, 0x00, _nAccBytes);
    else
    {
      std::cout << "Could not allocate GPU memory, fatel error" << std::endl;
      rc = -1;
    }

    std::cout << "Requested GPU PhaseLst Memory: " << _nPLstBytes << std::endl;

    if (cudaMalloc(&_pdPhaseLst, _nPLstBytes) == cudaSuccess)
      cudaMemcpy(_pdPhaseLst, _phaseLst.data(), _nPLstBytes, cudaMemcpyHostToDevice);
    else
    {
      std::cout << "Could not allocate GPU memory, fatel error" << std::endl;
      rc = -1;
    }

    std::cout << "Requested GPU Point Cloud Memory: " << _nPCldBytes << std::endl;

    if (cudaMalloc(&_pdPointCld, _nPCldBytes) == cudaSuccess)
    {
    size_t            n       = _pntCld.size();
    Point             *pMem   = new Point[n];
    Point             *pD     = pMem;
    Point             *pDEnd  = pD + n;
    pcl::PointXYZRGBA *pCld   = &_pntCld[0];

      
      while (pD < pDEnd)
      {
        pD->_clr  = make_double4((double)pCld->r, (double)pCld->g, (double)pCld->b, (double)pCld->a);
        pD->_pos  = make_double3((double)pCld->x, (double)pCld->y, (double)pCld->z);

        pD++;
        pCld++;
      }

      cudaMemcpy(_pdPointCld,pMem,_nPCldBytes,cudaMemcpyHostToDevice);

      delete pMem;
    }
    else
    {
      std::cout << "Could not allocate GPU memory, fatel error" << std::endl;
      rc = -1;
    }

    cudaDeviceSynchronize();
  }

  return rc;
}


//---------------------------------------------------------------------
// ExecutorCuda
//---------------------------------------------------------------------
ExecutorCuda::ExecutorCuda(void) : Executor(),
                                   _pAccMem(0),
                                   _pdAccMem(0),
                                   _nAccBytes(0),
                                   _pdPhaseLst(0),
                                   _nPLstBytes(0),
                                   _pdPointCld(0),
                                   _nPCldBytes(0)
{
}


//---------------------------------------------------------------------
// ~ExecutorCuda
//---------------------------------------------------------------------
ExecutorCuda::~ExecutorCuda()
{
  delete _pAccMem;

  if (_pdPointCld)
    cudaFree(_pdPointCld);

  if (_pdPhaseLst)
    cudaFree(_pdPhaseLst);

  if (_pdAccMem)
    cudaFree(_pdAccMem);
}

