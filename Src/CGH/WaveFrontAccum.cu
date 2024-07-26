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
#include <float.h>

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

inline __device__ double3 operator-(double3& a, double3& b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ void operator-=(double3& a, double3& b) {
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __device__ void operator*=(double3& a, double3& b) {
  a.x *= b.x; a.y *= b.y; a.z *= b.z;
}

inline __device__ double dot(double3 const &a, double3 const &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ double length(double3 const &a) {
  return sqrt(dot(a,a));
}

inline __device__ double distance(double3& a, double3& b) {
  return length(b-a);
}

inline __device__ double4 min4(double4& a, double4& b) {
double4 t;

  t.x = min(a.x, b.x);
  t.y = min(a.y, b.y);
  t.z = min(a.z, b.z);
  t.w = min(a.w, b.w);

  return t;
}

inline __device__ double4 max4(double4& a, double4& b) {
double4 t;

  t.x = max(a.x, b.x);
  t.y = max(a.y, b.y);
  t.z = max(a.z, b.z);
  t.w = max(a.w, b.w);

  return t;
}

inline __device__ double4 make_double4(double a) {
  return make_double4(a, a, a, a);
}

inline __device__ double4 operator-(double4& a) {
  return make_double4(-a.x, -a.y, -a.z, -a.w);
}

inline __device__ double4 operator+(double4& a, double4& b) {
  return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ double4 operator*(double4& a, double4& b) {
  return make_double4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ double4 operator/(double4& a, double4& b) {
  return make_double4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __device__ double4 cos(double4& a) {
  return make_double4(cos(a.x),cos(a.y),cos(a.z),cos(a.w));
}

inline __device__ double4 operator*(double4& a, double b) {
  return make_double4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ double4 operator/(double4& a, double b) {
  double scale = 1.0 / b;
  return a * scale;
}

inline __device__ void operator+=(double4& a, double4& b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__device__
int getGlobalIdx_1D_1D() {
  return (blockIdx.x * blockDim.x) + threadIdx.x;

}

__global__
void accWaveFront(void *pdImg,void *pdPhaseLst,void *pdPointCld,void* pdWFAP)
{
unsigned int          row     = getGlobalIdx_1D_1D();;
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
unsigned int          row     = getGlobalIdx_1D_1D();
WaveFrontAccumParams* pdW     = reinterpret_cast<WaveFrontAccumParams*>(pdWFAP);
double4*              pdD     = reinterpret_cast<double4*>(pdImg) + (row * pdW->_nCol);
double4*              pdDEnd  = pdD + pdW->_nCol;

  while (pdD < pdDEnd)
  {
    *pdD = make_double4((double)row);
    pdD++;
  }
}



__global__
void determineRowMinMax(void* pdImg, void* pdWFAP)
{
unsigned int          row     = getGlobalIdx_1D_1D();
WaveFrontAccumParams* pdW     = reinterpret_cast<WaveFrontAccumParams*>(pdWFAP);
double4*              pdSBeg  = reinterpret_cast<double4*>(pdImg) + (row * pdW->_nCol);
double4*              pdS     = pdSBeg;
double4*              pdSEnd  = pdS + pdW->_nCol;
double4*              pdSStp  = pdSEnd - 8;
double4               dMin    = make_double4(DBL_MAX);
double4               dMax    = make_double4(-DBL_MAX);

  while (pdS < pdSStp)
  {
    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;

    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;
  }

  while (pdS < pdSEnd)
  {
    dMin = min4(dMin, *pdS);
    dMax = max4(dMax, *pdS);
    pdS++;
  }

  *(pdSBeg + 0) = dMin;
  *(pdSBeg + 1) = dMax;
}


/* this is intented to executed on only 1 thread */
__global__
void determineFinalMinMax(void* pdImg, void* pdWFAP)
{
WaveFrontAccumParams* pdW     = reinterpret_cast<WaveFrontAccumParams*>(pdWFAP);
double4*              pdSBeg  = reinterpret_cast<double4*>(pdImg);
double4*              pdS     = pdSBeg;
double4*              pdSEnd  = pdS + pdW->_nCol * pdW->_nRow;
double4*              pdSStp  = pdSEnd - (8 * pdW->_nRow);
double4               dMin    = make_double4(DBL_MAX);
double4               dMax    = make_double4(-DBL_MAX);

  while (pdS < pdSStp)
  {
    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;

    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;
  }

  while (pdS < pdSEnd)
  {
    dMin = min4(dMin, *(pdS + 0));
    dMax = max4(dMax, *(pdS + 1));
    pdS += pdW->_nRow;
  }

  *(pdSBeg + 0) = dMin;
  *(pdSBeg + 1) = dMax;
}

__host__ 
void launchWaveFrontAccum(dim3 &nT,dim3 &nB,
                          void* pdImg, void* pdPhaseLst, void* pdPointCld,void* pdWFAP)
{

  accWaveFront<<<nB,nT>>>(pdImg,pdPhaseLst,pdPointCld,pdWFAP);
}

__host__
void launchDetermineRowMinMax(dim3& nT, dim3& nB, void* pdImg, void* pdWFAP)
{
  determineRowMinMax << <nB, nT >> > (pdImg, pdWFAP);
}

__host__
void launchDetermineFinalMinMax(dim3& nT, dim3& nB, void* pdImg, void* pdWFAP)
{
  determineFinalMinMax << <nB, nT >> > (pdImg, pdWFAP);
}