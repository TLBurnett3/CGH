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

// Thomas Burnett
// main.cpp 

//---------------------------------------------------------------------
// Includes
#include <string>
#include <iostream>
#include <filesystem>

// 3rd Party Libs

// CGH
#include "CGH/Executor.h"
#include "CGH/ExecutorCpp.h"
#include "Common/Timer.h"
//---------------------------------------------------------------------


//---------------------------------------------------------------------
// Globals

//---------------------------------------------------------------------

#ifdef WIN32
#include <conio.h>
#else
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

bool _kbhit()
{
  termios term;
  tcgetattr(0, &term);

  termios term2 = term;
  term2.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &term2);

  int byteswaiting;
  ioctl(0, FIONREAD, &byteswaiting);

  tcsetattr(0, TCSANOW, &term);

  return byteswaiting > 0;
}
char _getch(void)
{
  char buf = 0;
  struct termios old = {0};
  fflush(stdout);
  if(tcgetattr(0, &old) < 0)
    perror("tcsetattr()");
  old.c_lflag &= ~ICANON;
  old.c_lflag &= ~ECHO;
  old.c_cc[VMIN] = 1;
  old.c_cc[VTIME] = 0;
  if(tcsetattr(0, TCSANOW, &old) < 0)
    perror("tcsetattr ICANON");
  if(read(0, &buf, 1) < 0)
    perror("read()");
  old.c_lflag |= ICANON;
  old.c_lflag |= ECHO;
  if(tcsetattr(0, TCSADRAIN, &old) < 0)
    perror("tcsetattr ~ICANON");
  return buf;
}
#endif


//---------------------------------------------------------------------
// main
//---------------------------------------------------------------------
int main(int argc,char *argv[])
{
int               rc = -1;
CGH::Core::SpJob  spJob;
char              *pFName = "Cfg/Default.json";

  if (argc > 1)
    pFName = argv[1];
    
  std::cout << "CGH: Initialization\n";

  {
  CGH::Common::JSon::Value doc;

    rc = CGH::Common::JSon::readFile(doc,pFName);

    if (rc == 0)
    {
      spJob = std::make_shared<CGH::Core::Job>();
      rc = spJob->init(doc);
    }
  }

  if (rc == 0)
  {
  CGH::Executor     *pE   = new CGH::ExecutorCpp();

    if (pE)
    {
    std::filesystem::path filePath = pFName;

      filePath = filePath.remove_filename();

      rc = pE->init(filePath,spJob);

      if (rc == 0)
      {
      bool               run = true;
      CGH::Common::Timer tS;

        std::cout << "CGH: Execution\n";

        pE->start();

        while (run && pE->run())
        {
          pE->exec();

          {
          int key = 0;

            key = cv::waitKey((int)(1.0f / 30.f * 1000.0f));

            if ((key == 'Q') || (key == 'q'))
              run = false;
          }

          if (_kbhit())
          {
            char ch = _getch();

            if ((ch == 'q') || (ch == 'Q'))
              run = false;
          }
        }

        pE->stop();
        pE->printStatus();
        pE->updateProofImg();     
      }

      delete pE;
    }
  }

  std::cout << "CGH: Exit\n";

  return rc;
}
