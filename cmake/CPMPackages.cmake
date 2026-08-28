# Use CPM.cmake for dependency management
set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/third_party/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
    file(DOWNLOAD
        "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()

include(${CPM_DOWNLOAD_LOCATION})

set(HYBRIDAI_DEPS_LOCAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/deps_local")

# Helper to use a local extracted source tree when present,
# otherwise fall back to a GitHub git clone.
function(hybridai_add_local_or_github name version github_repo git_tag source_dir)
    set(_local_extracted "${HYBRIDAI_DEPS_LOCAL_DIR}/${source_dir}")
    if(EXISTS "${_local_extracted}/CMakeLists.txt")
        message(STATUS "CPM: Using local ${name} from ${_local_extracted}")
        CPMAddPackage(
            NAME ${name}
            SOURCE_DIR ${_local_extracted}
            ${ARGN}
        )
    else()
        message(STATUS "CPM: Local source for ${name} not found, falling back to GitHub")
        CPMAddPackage(
            NAME ${name}
            GITHUB_REPOSITORY ${github_repo}
            GIT_TAG ${git_tag}
            ${ARGN}
        )
    endif()
endfunction()

if(HYBRIDAI_BUILD_CLI)
    find_package(fmt CONFIG QUIET)
    if(NOT TARGET fmt::fmt)
        hybridai_add_local_or_github(
            fmt 11.0.2 fmtlib/fmt 11.0.2 "fmt-11.0.2"
        )
    endif()

    find_package(spdlog CONFIG QUIET)
    if(NOT TARGET spdlog::spdlog)
        hybridai_add_local_or_github(
            spdlog 1.14.1 gabime/spdlog v1.14.1 "spdlog-1.14.1"
            OPTIONS
                "SPDLOG_FMT_EXTERNAL ON"
        )
    endif()
endif()

find_package(nlohmann_json CONFIG QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json AND
   HYBRIDAI_NLOHMANN_JSON_INCLUDE_DIR)
    if(NOT EXISTS
       "${HYBRIDAI_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
        message(FATAL_ERROR
            "HYBRIDAI_NLOHMANN_JSON_INCLUDE_DIR does not contain "
            "nlohmann/json.hpp")
    endif()
    add_library(hybridai_nlohmann_json INTERFACE)
    add_library(nlohmann_json::nlohmann_json ALIAS hybridai_nlohmann_json)
    target_include_directories(hybridai_nlohmann_json INTERFACE
        "${HYBRIDAI_NLOHMANN_JSON_INCLUDE_DIR}")
    message(STATUS
        "Using external nlohmann-json headers from "
        "${HYBRIDAI_NLOHMANN_JSON_INCLUDE_DIR}")
elseif(NOT TARGET nlohmann_json::nlohmann_json)
    hybridai_add_local_or_github(
        nlohmann_json 3.11.3 nlohmann/json v3.11.3 "json-3.11.3"
    )
endif()

if(HYBRIDAI_BUILD_TESTS)
    hybridai_add_local_or_github(
        googletest 1.15.2 google/googletest v1.15.2 "googletest-1.15.2"
        OPTIONS
            "INSTALL_GTEST OFF"
            "gtest_force_shared_crt ON"
    )
endif()
