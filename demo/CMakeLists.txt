cmake_minimum_required(VERSION 3.20)

# This directory supports both:
#   add_subdirectory(demo) from the main HybridAI project; and
#   cmake -S demo -B demo/build against an existing libhybridai.so.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(HybridAIDemos LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    get_filename_component(HYBRIDAI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    set(HYBRIDAI_ROOT "${HYBRIDAI_ROOT}" CACHE PATH
        "HybridAI source directory")
    set(HYBRIDAI_LIBRARY "${HYBRIDAI_ROOT}/build/libhybridai.so" CACHE FILEPATH
        "Path to the prebuilt HybridAI shared library")

    if(NOT EXISTS "${HYBRIDAI_LIBRARY}")
        message(FATAL_ERROR
            "Prebuilt HybridAI shared library not found: ${HYBRIDAI_LIBRARY}\n"
            "Build it first or set -DHYBRIDAI_LIBRARY=/path/to/libhybridai.so")
    endif()
    if(NOT EXISTS "${HYBRIDAI_ROOT}/src/core/tensor.h")
        message(FATAL_ERROR "Invalid HYBRIDAI_ROOT: ${HYBRIDAI_ROOT}")
    endif()

    add_library(hybridai SHARED IMPORTED GLOBAL)
    set_target_properties(hybridai PROPERTIES
        IMPORTED_LOCATION "${HYBRIDAI_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${HYBRIDAI_ROOT}/include;${HYBRIDAI_ROOT}/src;${HYBRIDAI_ROOT}/third_party/nlohmann_json/include"
    )

    # qwen_infer contains optional direct HIP/rocBLAS diagnostics. Locate the
    # same SDK used by libhybridai.so and expose those APIs when available.
    find_program(HYBRIDAI_HIPCC NAMES hipcc)
    if(HYBRIDAI_HIPCC)
        get_filename_component(HYBRIDAI_HIP_BIN_DIR "${HYBRIDAI_HIPCC}" DIRECTORY)
        get_filename_component(HYBRIDAI_ROCM_ROOT "${HYBRIDAI_HIP_BIN_DIR}" DIRECTORY)
        list(PREPEND CMAKE_PREFIX_PATH
            "${HYBRIDAI_ROCM_ROOT}"
            "${HYBRIDAI_ROCM_ROOT}/hip"
            "${HYBRIDAI_ROCM_ROOT}/rocblas"
            "${HYBRIDAI_ROCM_ROOT}/dcc"
            "${HYBRIDAI_ROCM_ROOT}/dcc/comgr"
        )
        set(ROCM_PATH "${HYBRIDAI_ROCM_ROOT}")
    endif()

    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
endif()

add_executable(simple_infer simple_infer.cc)
target_link_libraries(simple_infer PRIVATE hybridai)

add_executable(qwen_infer qwen_infer.cc)
target_link_libraries(qwen_infer PRIVATE hybridai)

add_executable(generate generate.cc)
target_link_libraries(generate PRIVATE hybridai)

if(TARGET hip::host AND TARGET roc::rocblas)
    target_compile_definitions(qwen_infer PRIVATE HYBRIDAI_HAS_HIP)
    target_link_libraries(qwen_infer PRIVATE hip::host roc::rocblas)
endif()

add_executable(test_tokenizer test_tokenizer.cc)
target_link_libraries(test_tokenizer PRIVATE hybridai)

set_target_properties(simple_infer qwen_infer generate test_tokenizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    get_filename_component(HYBRIDAI_LIBRARY_DIR "${HYBRIDAI_LIBRARY}" DIRECTORY)
    set_target_properties(simple_infer qwen_infer generate test_tokenizer PROPERTIES
        BUILD_RPATH "${HYBRIDAI_LIBRARY_DIR}"
    )
endif()
