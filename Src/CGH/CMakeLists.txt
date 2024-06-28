#----COPYRIGHT NOTICE--------------------------------------------------
#
#	Copyright (C) FoVI3D
#
#----COPYRIGHT NOTICE--------------------------------------------------

cmake_minimum_required (VERSION 3.6)

project(CGH)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CGH_SRC_DIR})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${3RDPARTY_INC_DIR}/eigen3)

link_directories(${CMAKE_LIBRARY_PATH})

if (UNIX)
  link_directories(/usr/local/lib)
endif()

set(HEADERS
  Executor.h
  ExecutorCpp.h

)

set(SOURCE
  Executor.cpp
  ExecutorCpp.cpp
  Main.cpp
)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCE})

target_link_libraries(${PROJECT_NAME} ${OPENCV_DEBUG_LIB} ${OPENCV_RELEASE_LIB})
target_link_libraries(${PROJECT_NAME} ${PCL_DEBUG_LIB} ${PCL_RELEASE_LIB})
target_link_libraries(${PROJECT_NAME} cghCommon)
target_link_libraries(${PROJECT_NAME} cghCore)

install (TARGETS ${PROJECT_NAME})
