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
#include "cuda_runtime_api.h"

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
dim3  numThreads(1,256,1);
dim3  numBlocks(1,_spJob->_numPixels.y / numThreads.y,1);

  //accWaveFront<<<numBlocks,numThreads>>>(_pdAccMem,_pdPhaseLst,_pdPointCld,
  //                                       _spJob->_numPixels.x,(int)_pntCld.size(),
  //                                       _spJob->_waveLengths.r, _spJob->_waveLengths.g, 
  //                                       _spJob->_waveLengths.b, _spJob->_waveLengths.a);

  launch();

  _run = false;
}


//---------------------------------------------------------------------
// stop
//---------------------------------------------------------------------
void ExecutorCuda::stop(void)
{
  Executor::stop();

  cudaMemcpy(_pAccMem,_pdAccMem,_nAccBytes,cudaMemcpyDeviceToHost);
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
    _nPCldBytes = sizeof(pcl::PointXYZRGBA) * _pntCld.size();

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
      cudaMemcpy(_pdPointCld, _pntCld.data(), _nPCldBytes, cudaMemcpyHostToDevice);
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

