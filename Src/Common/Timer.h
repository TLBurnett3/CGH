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

// Timer.h
// Thomas Burnett



#pragma once

//---------------------------------------------------------------------
// Includes
// System
#include <chrono>

// 3rdPartyLibs

// SCS
#include "Common/Export.h"
//---------------------------------------------------------------------

//---------------------------------------------------------------------
namespace CGH
{
  namespace Common
  {
    class Timer
    {
      // Definition
      private:
      protected:
      public:

      // Members
      private:
      protected:
        std::chrono::high_resolution_clock::time_point   _sT;
        std::chrono::high_resolution_clock::time_point   _eT;
        bool                                             _isTiming;

      public:

      // Methods
      private:
      protected:
      public:
        EXPORT bool  isTiming(void)
        { return _isTiming; }

        EXPORT virtual void  start(void)
        {
          _sT         = std::chrono::high_resolution_clock::now(); 
          _isTiming  = true;
        }

        EXPORT virtual void  stop(void)
        { 
          _eT        = std::chrono::high_resolution_clock::now(); 
          _isTiming = false;
        }

        EXPORT void reset(void)
        {
          _sT       = std::chrono::high_resolution_clock::now(); 
          _eT       = _sT;
          _isTiming = false;
        }

        EXPORT double seconds(void)
        {
        std::chrono::high_resolution_clock::time_point    eT(_isTiming ? std::chrono::high_resolution_clock::now() : _eT);
        std::chrono::duration<double>                     tD(std::chrono::duration<double>(eT - _sT));

          return tD.count();
        }

        EXPORT Timer(void) : _sT(std::chrono::high_resolution_clock::now()),
                             _eT(_sT),
                             _isTiming(true)
        {}   
    };  
  };
};
//---------------------------------------------------------------------


