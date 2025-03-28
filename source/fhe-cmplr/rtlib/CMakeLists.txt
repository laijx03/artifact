#=============================================================================
#
# Copyright (c) XXXX-XXXX., Ltd
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#=============================================================================

cmake_minimum_required(VERSION 3.13)

project(rtlib C CXX)

set(RTLIB_VERSION_MAJOR 0)
set(RTLIB_VERSION_MINOR 0)
set(RTLIB_VERSION_PATCH 1)
set(RTLIB_VERSION ${RTLIB_VERSION_MAJOR}.${RTLIB_VERSION_MINOR}.${RTLIB_VERSION_PATCH})

set(CMAKE_CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED ON)

if(IS_DIRECTORY "${PACKAGE_BASE_DIR}/cmake/air-infra")
  include(${PACKAGE_BASE_DIR}/cmake/air-infra/function_common.cmake)
  include(${PACKAGE_BASE_DIR}/cmake/air-infra/function_app.cmake)
else()
  include(${PACKAGE_BASE_DIR}/cmake/function_common.cmake)
  include(${PACKAGE_BASE_DIR}/cmake/function_app.cmake)
endif()

include(cmake/function_extralib.cmake)

# Add path for custom CMake modules.
list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#--------------------------------------------------------------------
# Build options
#--------------------------------------------------------------------
option(BUILD_STATIC          "Control whether to build the static libraries"    ON)
option(BUILD_SHARED          "Control whether to build the shared libraries"    OFF)
option(BUILD_UNITTEST        "Control whether to build the unittest"            ON)
option(BUILD_BENCH           "Control whether to build the benmark"             ON)
option(BUILD_DOC             "Control whether to build API documentation"       OFF)
option(RTLIB_CODE_CHECK      "Control whether to check coding style"            ON)
option(RTLIB_BUILD_TEST      "Control whether to build the tests"               ON)
option(RTLIB_BUILD_EXAMPLE   "Control whether to build the examples"            ON)
option(RTLIB_INSTALL_APP     "Control whether to install the APPs"              ON)
option(RTLIB_ENABLE_SEAL     "Control whether to build with SEAL libraries"     ON)
option(RTLIB_ENABLE_SEAL_BTS "Control whether to build with SEAL_BTS libraries" OFF)
option(RTLIB_ENABLE_OPENFHE  "Control whether to build with OPENFHE libraries"  ON)
option(BUILD_WITH_OPENMP     "Control whether to build with OpenMP libraries"   OFF)
option(RTLIB_ENABLE_LINUX    "Control build rtlib support bare or linux"        ON)

set(RT_INSTALL_PATH)
if(RTLIB_ENABLE_LINUX)
  message(STATUS "Build RTLIB for Linux")
  add_definitions(-DRTLIB_SUPPORT_LINUX)
  set(RTLIB_INSTALL_PATH "rtlib/lib")
else()
  message(STATUS "Build RTLIB for Bare")
  set(RTLIB_INSTALL_PATH "rtlib/lib_bare")
endif()

# Check option : -DCMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Set the default type of build" FORCE)
else()
  check_build_type(${CMAKE_BUILD_TYPE})
endif()

### Print options
message(STATUS "RTLIB_SOURCE_DIR              : ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_BUILD_TYPE              : ${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_STATIC                  : ${BUILD_STATIC}")
message(STATUS "BUILD_SHARED                  : ${BUILD_SHARED}")
message(STATUS "BUILD_UNITTEST                : ${BUILD_UNITTEST}")
message(STATUS "BUILD_BENCH                   : ${BUILD_BENCH}")
message(STATUS "BUILD_DOC                     : ${BUILD_DOC}")
message(STATUS "RTLIB_CODE_CHECK              : ${RTLIB_CODE_CHECK}")
message(STATUS "RTLIB_BUILD_TEST              : ${RTLIB_BUILD_TEST}")
message(STATUS "RTLIB_BUILD_EXAMPLE           : ${RTLIB_BUILD_EXAMPLE}")
message(STATUS "RTLIB_ENABLE_SEAL             : ${RTLIB_ENABLE_SEAL}")
message(STATUS "RTLIB_ENABLE_SEAL_BTS         : ${RTLIB_ENABLE_SEAL_BTS}")
message(STATUS "RTLIB_ENABLE_OPENFHE          : ${RTLIB_ENABLE_OPENFHE}")
message(STATUS "RTLIB_INSTALL_APP             : ${RTLIB_INSTALL_APP}")
message(STATUS "BUILD_WITH_OPENMP             : ${BUILD_WITH_OPENMP}")
message(STATUS "BUILD_TIMESTAMP               : ${BUILD_TIMESTAMP}")
message(STATUS "BUILD_GIT_COMMIT              : ${BUILD_GIT_COMMIT}")

#--------------------------------------------------------------------
# Logic: Configure
#--------------------------------------------------------------------

add_custom_target(rtlib_depend)

# build external dependent libraries
if(BUILD_UNITTEST)
  include(unittest)
endif()

if(BUILD_BENCH)
  include(benchmark)
endif()

#--------------------------------------------------------------------
# Logic: Target
#--------------------------------------------------------------------

# Go back to the list of passed include files
string(REPLACE "|" ";" RTLIB_INCLUDE_DIRS "${PACKAGE_INC_DIR_TF}")
include_directories("${RTLIB_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/ant/include")

# find extra libs : gmp, m
find_extralib_math(MATH_LIBS)
set(FHERT_ANT_LIBS ${FHERT_ANT_LIBS} FHErt_ant_encode FHErt_common ${MATH_LIBS})

# find extra libs : openmp
if(BUILD_WITH_OPENMP)
  find_extralib_openmp("find lib : libopenmp")
endif()

if(RTLIB_BUILD_TEST)
  enable_testing()
endif()

add_subdirectory(common)
add_subdirectory(ant)

if(RTLIB_ENABLE_SEAL OR RTLIB_ENABLE_SEAL_BTS)
  add_subdirectory(seal)
endif()

if(RTLIB_ENABLE_OPENFHE)
  add_subdirectory(openfhe)
endif()

#--------------------------------------------------------------------
# Logic: Package
#--------------------------------------------------------------------
# install header file

#install(TARGETS FHERT_ANT_LIBS EXPORT RTLIBTargets
#        ARCHIVE DESTINATION lib
#        LIBRARY DESTINATION lib
#        RUNTIME DESTINATION bin)

install(DIRECTORY include/ DESTINATION rtlib/include)

#install(EXPORT RTLIBTargets
#        FILE RTLIBTargets.cmake
#        NAMESPACE RTLIB::
#        DESTINATION lib/cmake/RTLIB)

#install(FILES RTLIBConfig.cmake
#        DESTINATION lib)

