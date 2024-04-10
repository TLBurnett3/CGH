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

// JSon.cpp 
// Thomas Burnett


//---------------------------------------------------------------------
// Includes
// System
#include <iomanip>

// 3rdPartyLibs
#include <glm/ext.hpp>
#include <glm/gtc/type_ptr.hpp>

// SCS
#include "Common/JSon.h"

using namespace CGH;
using namespace Common;
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
void JSon::printParseError(const char *pT)
{
  printf("Fatal config parse error: %s not found\n",pT);
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, std::string &val, bool req, bool quiet)
{
int rc = -1;

  if (!v[pT].is_null())
  {
    if (v[pT].is_string())
  	{
	    val = v[pT].get<std::string>();
  	  rc = 0;
  	}
  }

  if ((rc == -1) && (!req))
	rc = 0;
 
  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, float &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_float())
		{
			val = v[pT].get<float>();
			rc = 0;
		}
		else if (v[pT].is_number_integer())
		{
			val = (float)v[pT].get<int64_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, double &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_float())
		{
			val = v[pT].get<double>();
			rc = 0;
		}
		else if (v[pT].is_number_integer())
		{
			val = (double)v[pT].get<int64_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, int32_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_integer())
		{
			val = v[pT].get<int32_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, int64_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_integer())
		{
			val = v[pT].get<int64_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, uint8_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_unsigned())
		{
			val = v[pT].get<uint8_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, uint16_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_unsigned())
		{
			val = v[pT].get<uint16_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, uint32_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_unsigned())
		{
			val = v[pT].get<uint32_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, uint64_t &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_number_unsigned())
		{
			val = v[pT].get<uint64_t>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, bool &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_boolean())
		{
			val = v[pT].get<bool>();
			rc = 0;
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, std::vector<int> &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			rc = 0;
			val.clear();
			for (Value::iterator it = v[pT].begin(); it != v[pT].end(); it++) {
				val.push_back((*it).get<int>());
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT, std::vector<float> &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			rc = 0;

			val.clear();

			for (Value::iterator it = v[pT].begin(); it != v[pT].end(); it++) 
      {
				val.push_back((*it).get<float>());
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v, const char *pT,std::vector<std::string> &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			rc = 0;

			val.clear();

			for (Value::iterator it = v[pT].begin(); it != v[pT].end(); it++) 
      {
				val.push_back((*it).get<std::string>());
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::ivec2 &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 2)
			{
				Value::iterator it = v[pT].begin();

				val.x = (*it++).get<int>();
				val.y = (*it++).get<int>();
				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}



//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::vec2 &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 2)
			{
				Value::iterator it = v[pT].begin();

				val.x = (*it++).get<float>();
				val.y = (*it++).get<float>();
				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::vec3 &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 3)
			{
				Value::iterator it = v[pT].begin();

				val.x = (*it++).get<float>();
				val.y = (*it++).get<float>();
				val.z = (*it++).get<float>();
				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

  if (rc && !quiet)
    printParseError(pT);

	return rc;
}

//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::dvec2 &val, bool req, bool quiet)
{
	int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 2)
			{
				Value::iterator it = v[pT].begin();

				val.x = (*it++).get<double>();
				val.y = (*it++).get<double>();

				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

	if (rc && !quiet)
		printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::dvec3 &val, bool req, bool quiet)
{
	int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 3)
			{
				Value::iterator it = v[pT].begin();

				val.x = (*it++).get<double>();
				val.y = (*it++).get<double>();
				val.z = (*it++).get<double>();
				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

	if (rc && !quiet)
		printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::dvec4 &val, bool req, bool quiet)
{
	int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 4)
			{
			Value::iterator it = v[pT].begin();

				val.x = (*it++).get<double>();
				val.y = (*it++).get<double>();
				val.z = (*it++).get<double>();
				val.w = (*it++).get<double>();
				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

	if (rc && !quiet)
		printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parse
//---------------------------------------------------------------------
int JSon::parse(Value &v,const char *pT, glm::mat4 &val, bool req, bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_array())
		{
			if (v[pT].size() == 16)
			{
			Value::iterator it = v[pT].begin();

				val[0][0] = (*it++).get<float>();
				val[0][1] = (*it++).get<float>();
				val[0][2] = (*it++).get<float>();
				val[0][3] = (*it++).get<float>();

				val[1][0] = (*it++).get<float>();
				val[1][1] = (*it++).get<float>();
				val[1][2] = (*it++).get<float>();
				val[1][3] = (*it++).get<float>();

				val[2][0] = (*it++).get<float>();
				val[2][1] = (*it++).get<float>();
				val[2][2] = (*it++).get<float>();
				val[2][3] = (*it++).get<float>();

				val[3][0] = (*it++).get<float>();
				val[3][1] = (*it++).get<float>();
				val[3][2] = (*it++).get<float>();
				val[3][3] = (*it++).get<float>();

				rc = 0;
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

	if (rc && !quiet)
		printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// parseTransform
//---------------------------------------------------------------------
int JSon::parseTransform(Value &v,const char *pT,glm::mat4 &val,bool req,bool quiet)
{
int rc = -1;

	if (!v[pT].is_null())
	{
		if (v[pT].is_object())
		{
			rc = JSon::parse(v[pT],"Transform",val,true,false);

			if (rc < 0)
			{
			int				rc2  = 0;
			glm::vec3 vT(0);
			glm::vec3 vS(1);
			glm::vec3 vR(0);

				rc2 = JSon::parse(v[pT],"Translate",vT,true,true);

				if (rc2 >= 0)
					val = glm::translate(val,vT);

				rc2 = JSon::parse(v[pT],"Rotate",vR,true,true);

				if (rc2 >= 0)
			    val = val * glm::mat4(glm::quat(glm::radians(glm::vec3(vR.x,vR.y,vR.z))));
	
				if (rc2 == 0)
					rc = 0;

				rc2 = JSon::parse(v[pT],"Scale",vS,true,true);

				if (rc2 >= 0)
  				val = glm::scale(val,vS);
			}
		}
	}

	if ((rc == -1) && !req)
		rc = 0;

	if (rc && !quiet)
		printParseError(pT);

	return rc;
}


//---------------------------------------------------------------------
// readFile
//---------------------------------------------------------------------
int JSon::readFile(Value &v,const char *pFP)
{
int rc = -1;
std::ifstream file(pFP);

	if (file.is_open())
	{
		v = Value::parse(file);

    rc = 0;
	}

	return rc;
}



//---------------------------------------------------------------------
// readMem
//---------------------------------------------------------------------
int JSon::readMem(Value &v,const char *pMem)
{
int rc = -1;

  {
		v = Value::parse(pMem);

    rc = 0;
	}

	return rc;
}



//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const std::string &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const float &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const double &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const int32_t &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const int64_t &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const uint16_t &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const uint32_t &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const uint64_t &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const bool &val)
{
	if (v.is_array())
		v.push_back(val);
	else
		v[pT] = val;
}

//---------------------------


//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const std::vector<int> &val)
{
	v[pT] = val;
}


//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v, const char* pT, const std::vector<float> &val)
{
	v[pT] = val;
}


//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v,const char* pT,const glm::vec3 &val)
{
	v[pT] = { val.x, val.y , val.z };
}


//---------------------------------------------------------------------
// insert
//---------------------------------------------------------------------
void JSon::insert(Value &v,const char* pT,const glm::vec2 &val)
{
	v[pT] = { val.x, val.y };
}
