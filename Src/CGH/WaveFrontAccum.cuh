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

#pragma once


//---------------------------------------------------------------------
// Includes
// System

// 3rdPartyLibs

// CUDA
#include "vector_functions.h"

// CGH

//---------------------------------------------------------------------


//---------------------------------------------------------------------
// Definitions
typedef struct WaveFrontAccumParams_Def
{
  int _nRow;
  int _nCol;
  int _nPntCld;

  double3 _vS;
  double3 _vT;

  double4 _waveLengths;

  double  _fov;
} WaveFrontAccumParams;


typedef struct Point_DEF
{
  double3 _pos;
  double4 _clr;
} Point;

//---------------------------------------------------------------------


//---------------------------------------------------------------------
// Protoypes

extern "C"
{
  void launchWaveFrontAccum(dim3& nT, dim3& nB,
                            void* pdImg, void* pdPhaseLst, void* pdPointCld, void* pdWFAP);
  void launchDetermineRowMinMax(dim3& nT, dim3& nB,
                                void* pdImg, void* pdWFAP);
  void launchDetermineFinalMinMax(dim3& nT, dim3& nB,
                                  void* pdImg, void* pdWFAP);
};

//---------------------------------------------------------------------
