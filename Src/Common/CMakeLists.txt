#----COPYRIGHT NOTICE--------------------------------------------------
#
#	Copyright (C) FoVI3D
#
#----COPYRIGHT NOTICE--------------------------------------------------

cmake_minimum_required (VERSION 3.6)

project(cghCommon)

include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CGH_SRC_DIR})

link_directories(${CMAKE_LIBRARY_PATH})

set (HEADERS
  Export.h
  Template.h
  JSon.h
  Timer.h
)

set(SOURCE
  Template.cpp
  JSon.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADERS} ${SOURCE})

install (TARGETS ${PROJECT_NAME})
