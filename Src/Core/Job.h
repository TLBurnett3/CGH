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

// Job.h
// Thomas Burnett

#pragma once

//---------------------------------------------------------------------
// Includes
// System
#include <string>

// 3rdPartyLibs
#include <glm/glm.hpp>

// CGH
#include "Common/JSon.h"
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// Classes
namespace CGH
{
  namespace Core
  {
    class Job
    {
      // Defines
      private:
      protected:
      public:

      // Members
      private:
      protected:
      public:   
        std::string _jobName;
        std::string _outPath;
        glm::ivec2  _dim;
        float       _fov;
        glm::dvec2  _pixelSize;
        glm::ivec2  _numPixels;
        std::string _pntCld;
        bool        _isWaveField;

        glm::dvec4  _waveLengths;
        glm::dvec3  _luminance;

        glm::mat4   _mTpc;


      // Methods
      private:
      protected:
      public:
        EXPORT int init(Common::JSon::Value &doc);

        EXPORT Job(void);
        EXPORT ~Job();
    };
  };
};
//---------------------------------------------------------------------

