cmake_minimum_required(VERSION 3.20)

# Independent real-HIP build preset for the local Windows ROCm SDK.
# Usage from PowerShell:
#   cmake --preset hip-windows
#   cmake --build --preset hip-windows

if(NOT DEFINED HYBRIDAI_ROCM_ROOT OR
   HYBRIDAI_ROCM_ROOT MATCHES "\\$root")
	set(HYBRIDAI_ROCM_ROOT
		"C:/Users/yinxi/.venv/Lib/site-packages/_rocm_sdk_devel"
		CACHE PATH "ROCm SDK root" FORCE)
endif()

# Keep the host C++ compiler selected by the Visual Studio generator. The HIP
# backend uses hipcc through an isolated custom object compilation rule.
set(CMAKE_PREFIX_PATH "${HYBRIDAI_ROCM_ROOT};${HYBRIDAI_ROCM_ROOT}/lib/cmake" CACHE STRING "ROCm package search path")
set(CMAKE_MODULE_PATH "${HYBRIDAI_ROCM_ROOT}/cmake" CACHE STRING "ROCm CMake modules")

set(HYBRIDAI_ENABLE_HIP ON CACHE BOOL "Enable HIP/ROCm backend")
set(HYBRIDAI_ENABLE_CPU ON CACHE BOOL "Enable CPU backend")
set(HYBRIDAI_BUILD_TESTS ON CACHE BOOL "Build unit tests")
set(HYBRIDAI_BUILD_CLI ON CACHE BOOL "Build command line interface")
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
