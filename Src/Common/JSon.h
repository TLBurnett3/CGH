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

// JSon.h
// Thomas Burnett

#pragma once

//---------------------------------------------------------------------
// Includes
// System
#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <cstdlib>

// 3rdPartyLibs
#include <glm/glm.hpp>

#include "Common/JSon.hpp"
#include "Common/Export.h"
//---------------------------------------------------------------------
// Classes
namespace CGH
{
  namespace Common
  {
	  class JSon
	  {
      public:
				typedef nlohmann::json Value;
	
      private:
				void    static printParseError(const char *pT);

      public:

				EXPORT static int parse(Value &v, const char *pT, std::string              &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, float                    &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, double                   &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, int32_t                  &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, int64_t                  &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, uint8_t                  &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, uint16_t                 &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, uint32_t                 &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, uint64_t                 &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, bool                     &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, std::vector<int>         &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, std::vector<float>       &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, std::vector<std::string> &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::ivec2               &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::vec2                &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::vec3                &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::dvec2               &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::dvec3               &val, bool req = false, bool quiet = false);
				EXPORT static int parse(Value &v, const char *pT, glm::dvec4               &val, bool req = false, bool quiet = false);
		    EXPORT static int parse(Value &v, const char *pT, glm::mat4                &val, bool req = false, bool quiet = false);

				EXPORT static int parseTransform(Value &v, const char *pT, glm::mat4       &val, bool req = false, bool quiet = false);

				EXPORT static int readFile (Value &v,const char *pFP);
				EXPORT static int readMem	 (Value &v,const char *pMem);

				EXPORT static void insert(Value &v, const char* pT, const std::string                 &val);
				EXPORT static void insert(Value &v, const char* pT, const float                       &val);
				EXPORT static void insert(Value &v, const char* pT, const double                      &val);
				EXPORT static void insert(Value &v, const char* pT, const int32_t                     &val);
				EXPORT static void insert(Value &v, const char* pT, const int64_t                     &val);
				EXPORT static void insert(Value &v, const char* pT, const uint16_t                    &val);
				EXPORT static void insert(Value &v, const char* pT, const uint32_t                    &val);
				EXPORT static void insert(Value &v, const char* pT, const uint64_t                    &val);
				EXPORT static void insert(Value &v, const char* pT, const bool                        &val);
				EXPORT static void insert(Value &v, const char* pT, const std::vector<int>					  &val);
				EXPORT static void insert(Value &v, const char* pT, const std::vector<float>					&val);
				EXPORT static void insert(Value &v, const char* pT, const glm::vec3                   &val);
				EXPORT static void insert(Value &v, const char* pT, const glm::vec2                   &val);
	  };
  };
};
//---------------------------------------------------------------------

