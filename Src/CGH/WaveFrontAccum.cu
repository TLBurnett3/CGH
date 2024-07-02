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


//---------------------------------------------------------------------
// Includes
// System
#include <stdio.h>


// 3rdPartyLibs
// 
// Cuda
#include "math_constants.h"
#include "cuda_runtime_api.h"
#include "vector_functions.h"

// CGH
#include "WaveFrontAccum.cuh"
//---------------------------------------------------------------------


// https://github.com/openmm/openmm/blob/master/platforms/cuda/src/kernels/vectorOps.cu


inline __device__ double3 make_double3(double a) {
  return make_double3(a, a, a);
}

inline __device__ double3 operator-(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator-=(double3& a, double3 b) {
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator*=(double3& a, double3 b) {
  a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ double dot(double3 a, double3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ double length(double3 a) {
  return sqrt(dot(a,a));
}

inline __device__ double distance(double3 a, double3 b) {
  return length(b-a);
}

inline __device__ double4 make_double4(double a) {
  return make_double4(a, a, a, a);
}

inline __device__ double4 operator-(double4 a) {
  return make_double4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ double4 operator+(double4 a, double4 b) {
  return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ double4 operator*(double4 a, double4 b) {
  return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ double4 operator/(double4 a, double4 b) {
  return make_double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __device__ double4 cos(double4 a) {
  return make_double4(cos(a.x),cos(a.y),cos(a.z),cos(a.w));
}

inline __device__ double4 operator*(double4 a, double b) {
  return make_double4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ double4 operator/(double4 a, double b) {
  double scale = 1.0 / b;
  return a * scale;
}

inline __device__ void operator+=(double4 &a, double4 b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}


__global__
void accWaveFront(void *pdImg,void *pdPhaseLst,void *pdPointCld,void* pdWFAP)
{
unsigned int          row     = blockIdx.y * blockDim.y + threadIdx.y;
double2               hIdx    = make_double2(0,(double)row);
WaveFrontAccumParams  *pdW    = reinterpret_cast<WaveFrontAccumParams*>(pdWFAP);
double4               *pdD    = reinterpret_cast<double4 *>(pdImg) + (row * pdW->_nCol);
double4               *pdDEnd = pdD + pdW->_nCol;
double4               k       = make_double4(2.0 * CUDART_PI) / pdW->_waveLengths;
double3               vS      = pdW->_vS;
double3               vT      = pdW->_vT;
double                hA      = (pdW->_fov / 2.0) * 0.01745329251994329576923690768489;

  while (pdD < pdDEnd)
  {
  Point   *pdS    = reinterpret_cast<Point*>(pdPointCld);
  Point   *pdSEnd = pdS + pdW->_nPntCld;
  double4 *pdP    = reinterpret_cast<double4 *>(pdPhaseLst);
  double3 vC      = make_double3(hIdx.x,hIdx.y,0);

    vC -= vT;
    vC *= vS;

    *pdD = make_double4(0);

    while (pdS < pdSEnd)
    {
    double3   vD = make_double3(vC.x,vC.y,pdS->_pos.z);
    double    o  = distance(vD,pdS->_pos);
    double    a  = distance(vD,vC);
    double    th = atan(o/a);

      if (th <= hA)
      {
      double    d = distance(vC,pdS->_pos);
      double4   v = (pdS->_clr * cos(make_double4(d) * k) + *pdP) / d;

         *pdD += v;
      }

      pdS++;
      pdP++;
    }

    pdD++;
    hIdx.x++;
  }
}


__global__
void fillmem(void* pdImg, void* pdPhaseLst, void* pdPointCld, void* pdWFAP)
{
unsigned int          row     = blockIdx.y * blockDim.y + threadIdx.y;
WaveFrontAccumParams* pdW     = reinterpret_cast<WaveFrontAccumParams*>(pdWFAP);
double4*              pdD     = reinterpret_cast<double4*>(pdImg) + (row * pdW->_nCol);
double4*              pdDEnd  = pdD + pdW->_nCol;

  while (pdD < pdDEnd)
  {
    *pdD = make_double4((double)row);
    pdD++;
  }
}

__host__ 
void launchWaveFrontAccum(dim3 &nT,dim3 &nB,
                          void* pdImg, void* pdPhaseLst, void* pdPointCld,void* pdWFAP)
{
  accWaveFront<<<nB,nT>>>(pdImg,pdPhaseLst,pdPointCld,pdWFAP);
  //fillmem<<<nB,nT>>>(pdImg, pdPhaseLst, pdPointCld, pdWFAP);

}