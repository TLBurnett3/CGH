#include "math_constants.h"
#include "cuda_runtime_api.h"
#include "vector_functions.h"
#include "stdio.h"

#include "WaveFrontAccum.cuh"

// https://github.com/openmm/openmm/blob/master/platforms/cuda/src/kernels/vectorOps.cu

typedef struct POINT_DEF
{
  double3 _pos;
  double4 _clr;
} POINT;

inline __device__ double3 make_double3(double a) {
  return make_double3(a, a, a);
}

inline __device__ double3 operator-(double3 a, double3 b) {
  return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
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
void accWaveFront(void *pdImg,void *pdPhaseLst,void *pdPointCld,
                  int nCol,int nPntCld,
                  double wlr,double wlg,double wlb,double wla)
{
double4 *pdD    = reinterpret_cast<double4 *>(pdImg);
double4 *pdDEnd = pdD + nCol;
double4 k       = make_double4(2.0 * CUDART_PI) / make_double4(wlr,wlg,wlb,wla);

  while (pdD < pdDEnd)
  {
  POINT   *pdS    = reinterpret_cast<POINT*>(pdPointCld);
  POINT   *pdSEnd = pdS + nPntCld;
  double4 *pdP    = reinterpret_cast<double4 *>(pdPhaseLst);

    while (pdS < pdSEnd)
    {
    double3   vC = make_double3(0);
    double    d  = distance(vC,pdS->_pos);
    double4   v = (pdS->_clr * cos(make_double4(d) * k) + *pdP) / d;

      *pdD += v;

      pdS++;
      pdP++;
    }

    pdD++;
  }
}

__host__ 
void launch(void)
{
  printf("launching Kernel");
}