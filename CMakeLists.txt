cmake_minimum_required (VERSION 3.17)

project (CGH VERSION 0.0.1)

set (CMAKE_VERBOSE_MAKEFILE 1)

##############################################################################################
# Variables

set (CGH_VERSION_MAJOR      1)
set (CGH_VERSION_MINOR      0)
set (CGH_VERSION_PATCH      2)

if (WIN32)
set (3RDPARTY_BASE_DIR          /3rdPartyLibs)
else (WIN32)
set (3RDPARTY_BASE_DIR          /usr/local)
endif (WIN32)

set (3RDPARTY_INC_DIR          ${3RDPARTY_BASE_DIR}/include)
set (3RDPARTY_BIN_DIR          ${3RDPARTY_BASE_DIR}/bin)
set (3RDPARTY_LIB_DIR          ${3RDPARTY_BASE_DIR}/lib)

message("           3RdParty Variables")
message("             3RDPARTY_BASE_DIR:    " ${3RDPARTY_BASE_DIR})
message("              3RDPARTY_INC_DIR:    " ${3RDPARTY_INC_DIR})
message("              3RDPARTY_LIB_DIR:    " ${3RDPARTY_LIB_DIR})
message("              3RDPARTY_BIN_DIR:    " ${3RDPARTY_BIN_DIR})
										    
set(CMAKE_INCLUDE_PATH                      ${3RDPARTY_INC_DIR})
set(CMAKE_LIBRARY_PATH                      ${3RDPARTY_LIB_DIR})
set(CMAKE_MODULE_PATH                       ${3RDPARTY_LIB_DIR}/CMake;${3RDPARTY_LIB_DIR}/cmake)
set(CMAKE_PREFIX_PATH                       ${3RDPARTY_BASE_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY          ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY          ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY          ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIR_RELEASE        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
set(CMAKE_RUNTIME_OUTPUT_DIR_DEBUG          ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
set(CMAKE_CXX_STANDARD                      17)
set(CMAKE_DEBUG_POSTFIX                     d)

set(CMAKE_INSTALL_PREFIX                    /TLB3)

if (UNIX)
  set(CMAKE_CXX_FLAGS                       "${CMAKE_CXX_FLAGS} -pthread")
  set(CMAKE_C_COMPILER                      /usr/bin/gcc-8 CACHE PATH "" FORCE)
  set(CMAKE_CXX_COMPILER                    /usr/bin/g++-8 CACHE PATH "" FORCE)
#  set(CMAKE_CXX_FLAGS                       "-fPIC")
  set(CMAKE_CXX_FLAGS_DEBUG                 "-g")
  set(CMAKE_CXX_FLAGS_RELEASE               "-O3")  
  set(CMAKE_CXX_STANDARD_LIBRARIES          "-lstdc++fs -ldl -lpthread")

  include_directories(/usr/local/include)
  include_directories(/usr/include)
endif()

message("           CMAKE Variables")
message("               CMAKE_BINARY_DIR:   " ${CMAKE_BINARY_DIR})
message("              CMAKE_MODULE_PATH:   " ${CMAKE_MODULE_PATH})
message("             CMAKE_INCLUDE_PATH:   " ${CMAKE_INCLUDE_PATH})
message("             CMAKE_LIBRARY_PATH:   " ${CMAKE_LIBRARY_PATH})
message("              CMAKE_PREFIX_PATH:   " ${CMAKE_PREFIX_PATH})
message("           CMAKE_INSTALL_PREFIX:   " ${CMAKE_INSTALL_PREFIX})
message(" CMAKE_RUNTIME_OUTPUT_DIRECTORY:   " ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
message(" CMAKE_LIBRARY_OUTPUT_DIRECTORY:   " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
message(" CMAKE_ARCHIVE_OUTPUT_DIRECTORY:   " ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
message("           CMAKE_INSTALL_PREFIX:   " ${CMAKE_INSTALL_PREFIX})
message("CPACK_PACKAGE_INSTALL_DIRECTORY:   " ${CPACK_PACKAGE_INSTALL_DIRECTORY})

message("")

set(CGH_CFG_DIR                        ${CMAKE_CURRENT_SOURCE_DIR}/Cfg)
set(CGH_SRC_DIR                        ${CMAKE_CURRENT_SOURCE_DIR}/Src)

message("             CGH_SRC_DIR:   " ${CGH_SRC_DIR})


##############################################################################################
# Find Packages
find_package(OpenCV		  REQUIRED)
find_package(PCL		    REQUIRED)

set(FLANN_ROOT    ${3RDPARTY_INC_DIR})

message("")
message("                OpenCV Variables")
set(OpenCV_VERSION_S                      490)
message("                 OpenCV_VERSION:   " ${OpenCV_VERSION})
message("            OpenCV_INCLUDE_DIRS:   " ${OpenCV_INCLUDE_DIRS})
message("                     OpenCV_DIR:   " ${OpenCV_DIR})
message("                    OpenCV_LIBS:   " ${OpenCV_LIBS})
message("                 OpenCV_LIB_DIR:   " ${OpenCV_LIB_DIR})
message("")

message("")
message("                 Eigen Variables")
message("                    Eigen_DIR:   " ${EIGEN_DIR})
message("                Eigen_VERSION:   " ${})
message("           Eigen_INCLUDE_DIRS:   " ${EIGEN_INCLUDE_DIRS})
message("           Eigen_LIBRARY_DIRS:   " ${EIGEN_LIBRARY_DIRS})

message("")
message("                 PCL Variables")
message("                      PCL_DIR:   " ${PCL_DIR})
message("                  PCL_VERSION:   " ${PCL_VERSION})
message("             PCL_INCLUDE_DIRS:   " ${PCL_INCLUDE_DIRS})
message("             PCL_LIBRARY_DIRS:   " ${PCL_LIBRARY_DIRS})
message("                PCL_LIBRARIES:   " ${PCL_LIBRARIES})
message("")

##############################################################################################


if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DNOMINMAX)
  add_definitions(-DGLM_FORCE_CTOR_INIT)
endif()

if (UNIX)
  find_package(Threads)
endif()

add_subdirectory(Src)

##########################################################################################################################################
## Installation

if (WIN32)
set (3RDPARTY_INSTALL_FILES_RELEASE 
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_core${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_highgui${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_imgproc${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_video${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_imgcodecs${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_video${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_videoio${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_videoio_ffmpeg${OpenCV_VERSION_S}_64.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_dnn${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_calib3d${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_features2d${OpenCV_VERSION_S}.dll    
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_flann${OpenCV_VERSION_S}.dll
    ${3RDPARTY_BIN_DIR}/pcl_common.dll
    ${3RDPARTY_BIN_DIR}/pcl_filters.dll
    ${3RDPARTY_BIN_DIR}/pcl_features.dll
    ${3RDPARTY_BIN_DIR}/pcl_io.dll
    ${3RDPARTY_BIN_DIR}/pcl_segmentation.dll
    ${3RDPARTY_BIN_DIR}/pcl_sample_consensus.dll
    ${3RDPARTY_BIN_DIR}/pcl_search.dll
    ${3RDPARTY_BIN_DIR}/pcl_kdtree.dll      
    ${3RDPARTY_BIN_DIR}/pcl_octree.dll   
    ${3RDPARTY_BIN_DIR}/pcl_ml.dll
    ${3RDPARTY_BIN_DIR}/pcl_io_ply.dll
    "C:/Program Files/OpenNI2/Redist/OpenNI2.dll"
)

set (3RDPARTY_INSTALL_FILES_DEBUG
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_core${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_highgui${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_imgproc${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_video${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_imgcodecs${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_video${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_videoio${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_videoio_ffmpeg${OpenCV_VERSION_S}_64.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_dnn${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_calib3d${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_features2d${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BASE_DIR}/x64/vc16/bin/opencv_flann${OpenCV_VERSION_S}d.dll
    ${3RDPARTY_BIN_DIR}/pcl_common.dll
    ${3RDPARTY_BIN_DIR}/pcl_filters.dll
    ${3RDPARTY_BIN_DIR}/pcl_features.dll
    ${3RDPARTY_BIN_DIR}/pcl_io.dll
    ${3RDPARTY_BIN_DIR}/pcl_segmentation.dll
    ${3RDPARTY_BIN_DIR}/pcl_sample_consensus.dll
    ${3RDPARTY_BIN_DIR}/pcl_search.dll
    ${3RDPARTY_BIN_DIR}/pcl_kdtree.dll
    ${3RDPARTY_BIN_DIR}/pcl_octree.dll   
    ${3RDPARTY_BIN_DIR}/pcl_ml.dll
    ${3RDPARTY_BIN_DIR}/pcl_io_ply.dll
    "C:/Program Files/OpenNI2/Redist/OpenNI2.dll"
)

install (FILES       ${3RDPARTY_INSTALL_FILES_RELEASE}   CONFIGURATIONS Release  DESTINATION bin)
install (FILES       ${CGH_COMMAND_FILES}                CONFIGURATIONS Release  DESTINATION bin)
install (DIRECTORY   ${CGH_CFG_DIR}                      CONFIGURATIONS Release  DESTINATION bin/..)

endif()



##########################################################################################################################################
## Packaging

set (CPACK_PACKAGE_NAME                       "CGH")
set (CPACK_PACKAGE_VENDOR                     "TLBurnettIII")
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY        "Computer Generated Hologram")
set (CPACK_PACKAGE_VERSION_MAJOR              ${CGH_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR              ${CGH_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH              ${CGH_VERSION_PATCH})
set (CPACK_PACKAGE_VERSION                    ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
set (CPACK_DEBIAN_PACKAGE_MAINTAINER          "Thomas Burnett") 
set (CPACK_PACKAGE_INSTALL_DIRECTORY          "CGH")
 
include (CPack)