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

// SCS
#include "CGH/ExecutorCuda.h"

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
// exec
//---------------------------------------------------------------------
void ExecutorCuda::exec(void)
{

  _run = false;
}



//---------------------------------------------------------------------
// init
//---------------------------------------------------------------------
int ExecutorCuda::init(const std::filesystem::path &filePath,Core::SpJob &spJob)
{
int   rc  = Executor::init(filePath,spJob);

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
// stop
//---------------------------------------------------------------------
void ExecutorCuda::stop(void)
{
  Executor::stop();
}


//---------------------------------------------------------------------
// ExecutorCuda
//---------------------------------------------------------------------
ExecutorCuda::ExecutorCuda(void) : Executor()
{
}


//---------------------------------------------------------------------
// ~ExecutorCuda
//---------------------------------------------------------------------
ExecutorCuda::~ExecutorCuda()
{

  stop();
}

