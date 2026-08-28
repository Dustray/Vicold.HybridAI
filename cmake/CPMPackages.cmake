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

# nlohmann_json is vendored in the repository. Do not search the host system
# and do not fall back to a network download: reproducible/offline builds must
# always use this exact source tree.
set(HYBRIDAI_NLOHMANN_JSON_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json")
if(NOT EXISTS
   "${HYBRIDAI_NLOHMANN_JSON_SOURCE_DIR}/include/nlohmann/json.hpp")
    message(FATAL_ERROR
        "Vendored nlohmann_json was not found at "
        "${HYBRIDAI_NLOHMANN_JSON_SOURCE_DIR}/include/nlohmann/json.hpp")
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    message(STATUS
        "CPM: Using vendored nlohmann_json from "
        "${HYBRIDAI_NLOHMANN_JSON_SOURCE_DIR}")
    CPMAddPackage(
        NAME nlohmann_json
        SOURCE_DIR "${HYBRIDAI_NLOHMANN_JSON_SOURCE_DIR}"
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
