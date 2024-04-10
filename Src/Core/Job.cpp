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

// Job.cpp 
// Thomas Burnett


//---------------------------------------------------------------------
// Includes
// System

// 3rdPartyLibs

// SCS
#include "Core/Job.h"

using namespace CGH;
using namespace Core;
//---------------------------------------------------------------------



//---------------------------------------------------------------------
// Job
//---------------------------------------------------------------------
int Job::init(Common::JSon::Value &doc)
{
int rc = 0;

  rc |= Common::JSon::parse(doc,"JobName",      _jobName,     true);
  rc |= Common::JSon::parse(doc,"OutPath",      _outPath,     true);
  rc |= Common::JSon::parse(doc,"Dim",          _dim,         true);
  rc |= Common::JSon::parse(doc,"FoV",          _fov,         true);
  rc |= Common::JSon::parse(doc,"PixelSize",    _pixelSize,   false);
  rc |= Common::JSon::parse(doc,"WaveField",    _isWaveField, false);
   rc |= Common::JSon::parse(doc,"PixelSize",    _pixelSize,   false);
  rc |= Common::JSon::parse(doc,"PointCloud",   _pntCld,      true);

  rc |= Common::JSon::parse(doc,"Luminance",    _luminance,   false);

  rc |= Common::JSon::parseTransform(doc,"PCTransform",_mTpc, false);

  rc |= Common::JSon::parse(doc,"WaveLengths", _waveLengths, false);

  _waveLengths.w = (_waveLengths.r * _luminance.r) +
                   (_waveLengths.g * _luminance.g) +
                   (_waveLengths.b * _luminance.b);

  _numPixels = glm::ivec2(glm::dvec2(_dim) / _pixelSize);

  std::cout << "JobName: "      << _jobName << std::endl;
  std::cout << "OutPath: "      << _outPath << std::endl;
  std::cout << "Dim: "          << "[" << _dim.x << "," << _dim.y << "]" << std::endl;
  std::cout << "PixelSize: "    << "[" << _pixelSize.x << "," << _pixelSize.y << "]" << std::endl;
  std::cout << "NumPixels: "    << "[" << _numPixels.x << "," << _numPixels.y << "]" << std::endl;
  std::cout << "FoV: "          << _fov     << std::endl;
  std::cout << "WaveField: "    << (_isWaveField ? "true" : "false")     << std::endl;
   std::cout << "Point Cloud: "  << _pntCld     << std::endl;
  std::cout << "Luminance: ["   << _luminance.r << "," << _luminance.g << "," << _luminance.b << "]" << std::endl;
  std::cout << "WaveLengths: [" << _waveLengths.r << "," << _waveLengths.g << "," 
                                << _waveLengths.b << "," << _waveLengths.w << "]" << std::endl;

  return rc;
}


//---------------------------------------------------------------------
// Job
//---------------------------------------------------------------------
Job::Job(void) : _jobName(),
                 _outPath(),
                 _dim(0),
                 _fov(),
                 _pixelSize(1.0),
                 _numPixels(0),
                 _pntCld(),
                 _isWaveField(false),
                 _waveLengths(700.0,546.1,435.8,0),
                 _luminance(0.2,0.7,0.1),
                 _mTpc(1.0f)
{
}

//---------------------------------------------------------------------
// ~Job
//---------------------------------------------------------------------
Job::~Job()
{

}

