# HIP backend. Native HIP/ROCm headers are only included in hip_backend.cc.
# We always compile the HIP backend source files so that the backend factory is
# registered, even on Windows where ROCm is unavailable. When HIP is not found the
# implementation falls back to compile-time stubs.

target_sources(hybridai
    PRIVATE
        hip_backend.h
        hip_backend_factory.cc
)

if(HYBRIDAI_ENABLE_HIP)
    # Prefer ROCm config packages. ROCm-derived Linux distributions may place
    # HIP, rocBLAS and compiler support packages under different prefixes
    # below the SDK root (for example /opt/dtk/{hip,dcc,dcc/comgr}).
    if(NOT HYBRIDAI_ROCM_ROOT)
        if(DEFINED ENV{ROCM_PATH})
            set(HYBRIDAI_ROCM_ROOT "$ENV{ROCM_PATH}")
        elseif(DEFINED ENV{HIP_PATH})
            get_filename_component(HYBRIDAI_ROCM_ROOT "$ENV{HIP_PATH}" DIRECTORY)
        else()
            find_program(HYBRIDAI_HIPCC NAMES hipcc)
            if(HYBRIDAI_HIPCC)
                get_filename_component(HYBRIDAI_HIP_BIN_DIR
                                       "${HYBRIDAI_HIPCC}" DIRECTORY)
                get_filename_component(HYBRIDAI_ROCM_ROOT
                                       "${HYBRIDAI_HIP_BIN_DIR}" DIRECTORY)
            endif()
        endif()
    endif()

    if(HYBRIDAI_ROCM_ROOT)
        get_filename_component(HYBRIDAI_ROCM_ROOT
                               "${HYBRIDAI_ROCM_ROOT}" REALPATH)
        list(PREPEND CMAKE_PREFIX_PATH
            "${HYBRIDAI_ROCM_ROOT}"
            "${HYBRIDAI_ROCM_ROOT}/hip"
            "${HYBRIDAI_ROCM_ROOT}/rocblas"
            "${HYBRIDAI_ROCM_ROOT}/dcc"
            "${HYBRIDAI_ROCM_ROOT}/dcc/comgr"
        )
        if(NOT WIN32)
            set(ROCM_PATH "${HYBRIDAI_ROCM_ROOT}")
        endif()
        message(STATUS "HybridAI ROCm root: ${HYBRIDAI_ROCM_ROOT}")
    endif()

    # The Windows Python SDK needs explicit package directories. Keep these
    # overrides isolated from the Linux package search above.
    if(WIN32 AND HYBRIDAI_ROCM_ROOT)
        if(NOT hip_DIR)
            set(hip_DIR "${HYBRIDAI_ROCM_ROOT}/lib/cmake/hip")
        endif()
        if(NOT rocblas_DIR)
            set(rocblas_DIR "${HYBRIDAI_ROCM_ROOT}/lib/cmake/rocblas")
        endif()
    endif()
    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
     if(hip_FOUND AND rocblas_FOUND AND TARGET hip::host AND
         TARGET roc::rocblas)
        # Make the HIP macro visible to all consumers of the hybridai target
        # (demos, tests, CLI) so they can conditionally include HIP headers.
        target_compile_definitions(hybridai PUBLIC HYBRIDAI_HAS_HIP)
        if(WIN32)
            # The Visual Studio generator does not support HIP as a native
            # language. The HIP runtime API headers on Windows are host-only
            # enough to be compiled directly with MSVC, so we add the ROCm
            # include path and platform macro for the HIP source file and any
            # consumer that needs to probe the device.
            set(HYBRIDAI_HIPCC "${hip_HIPCC_EXECUTABLE}")
            if(NOT HYBRIDAI_HIPCC)
                message(FATAL_ERROR "HIP was found but hipcc.exe is unavailable")
            endif()

            get_filename_component(HYBRIDAI_ROCM_ROOT "${HYBRIDAI_HIPCC}" DIRECTORY)
            get_filename_component(HYBRIDAI_ROCM_ROOT "${HYBRIDAI_ROCM_ROOT}" DIRECTORY)
            set(HYBRIDAI_ROCM_INCLUDE "${HYBRIDAI_ROCM_ROOT}/include")
            if(NOT EXISTS "${HYBRIDAI_ROCM_INCLUDE}/hip/hip_runtime.h")
                message(FATAL_ERROR
                    "ROCm include directory not found at ${HYBRIDAI_ROCM_INCLUDE}")
            endif()

            target_sources(hybridai PRIVATE hip_backend.cc)
            set_source_files_properties(
                hip_backend.cc
                PROPERTIES
                    INCLUDE_DIRECTORIES "${HYBRIDAI_ROCM_INCLUDE}"
                    COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__=1;__HIP_NO_CUDA_MATH__=1"
            )
            # Expose ROCm headers to consumers as well.
            target_include_directories(hybridai PUBLIC "${HYBRIDAI_ROCM_INCLUDE}")
        else()
            # Compile device kernels with hipcc while keeping the project CXX
            # only. This works with normal `cmake ..` builds even when the host
            # compiler is GCC/Clang, and avoids requiring CMake HIP language
            # support from vendor-specific SDK packages.
            if(NOT HYBRIDAI_HIPCC)
                find_program(HYBRIDAI_HIPCC NAMES hipcc
                             HINTS "${HYBRIDAI_ROCM_ROOT}/bin")
            endif()
            if(NOT HYBRIDAI_HIPCC)
                message(FATAL_ERROR "HIP was found but hipcc is unavailable")
            endif()

            set(HYBRIDAI_HIP_KERNEL_SOURCE
                "${CMAKE_CURRENT_SOURCE_DIR}/hip_kernels.hip")
            set(HYBRIDAI_HIP_KERNEL_OBJECT
                "${CMAKE_CURRENT_BINARY_DIR}/hip_kernels.hip.o")
            set(HYBRIDAI_HIP_FUSED_KERNEL_SOURCE
                "${CMAKE_CURRENT_SOURCE_DIR}/hip_fused_kernels.hip")
            set(HYBRIDAI_HIP_FUSED_KERNEL_OBJECT
                "${CMAKE_CURRENT_BINARY_DIR}/hip_fused_kernels.hip.o")
            set(HYBRIDAI_HIP_KERNEL_ARCH_FLAGS)
            set(HYBRIDAI_EFFECTIVE_HIP_ARCHITECTURES
                "${HYBRIDAI_HIP_ARCHITECTURES}")
            if(CMAKE_HIP_ARCHITECTURES)
                set(HYBRIDAI_EFFECTIVE_HIP_ARCHITECTURES
                    "${CMAKE_HIP_ARCHITECTURES}")
            endif()
            foreach(HYBRIDAI_HIP_ARCH IN LISTS
                    HYBRIDAI_EFFECTIVE_HIP_ARCHITECTURES)
                list(APPEND HYBRIDAI_HIP_KERNEL_ARCH_FLAGS
                    "--offload-arch=${HYBRIDAI_HIP_ARCH}")
            endforeach()
            message(STATUS "HybridAI HIP kernel architectures: "
                           "${HYBRIDAI_EFFECTIVE_HIP_ARCHITECTURES}")
            add_custom_command(
                OUTPUT "${HYBRIDAI_HIP_KERNEL_OBJECT}"
                COMMAND "${HYBRIDAI_HIPCC}" -std=c++20 -fPIC
                    ${HYBRIDAI_HIP_KERNEL_ARCH_FLAGS}
                    "-I${PROJECT_SOURCE_DIR}/src"
                        -c "${HYBRIDAI_HIP_KERNEL_SOURCE}"
                        -o "${HYBRIDAI_HIP_KERNEL_OBJECT}"
                DEPENDS
                    "${HYBRIDAI_HIP_KERNEL_SOURCE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/hip_kernels.h"
                VERBATIM
                COMMAND_EXPAND_LISTS
                COMMENT "Compiling HybridAI HIP kernels"
            )
            add_custom_command(
                OUTPUT "${HYBRIDAI_HIP_FUSED_KERNEL_OBJECT}"
                COMMAND "${HYBRIDAI_HIPCC}" -std=c++20 -fPIC
                    ${HYBRIDAI_HIP_KERNEL_ARCH_FLAGS}
                    "-I${PROJECT_SOURCE_DIR}/src"
                    -c "${HYBRIDAI_HIP_FUSED_KERNEL_SOURCE}"
                    -o "${HYBRIDAI_HIP_FUSED_KERNEL_OBJECT}"
                DEPENDS
                    "${HYBRIDAI_HIP_FUSED_KERNEL_SOURCE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/hip_fused_kernels.h"
                VERBATIM
                COMMAND_EXPAND_LISTS
                COMMENT "Compiling HybridAI fused HIP kernels"
            )
            add_custom_target(hybridai_hip_kernels
                DEPENDS "${HYBRIDAI_HIP_KERNEL_OBJECT}"
                        "${HYBRIDAI_HIP_FUSED_KERNEL_OBJECT}")
            add_dependencies(hybridai hybridai_hip_kernels)
            set_source_files_properties(
                "${HYBRIDAI_HIP_KERNEL_OBJECT}"
                PROPERTIES
                    GENERATED TRUE
                    EXTERNAL_OBJECT TRUE
            )
            target_sources(hybridai PRIVATE
                hip_backend.cc
                hip_kernels.h
                hip_fused_kernels.h
                "${HYBRIDAI_HIP_KERNEL_OBJECT}"
                "${HYBRIDAI_HIP_FUSED_KERNEL_OBJECT}"
            )
            target_compile_definitions(hybridai PRIVATE HYBRIDAI_HAS_HIP_KERNELS)
        endif()
        target_link_libraries(hybridai
            PUBLIC
                hip::host
                roc::rocblas
        )
        message(STATUS "HIP backend enabled with hip/rocBLAS")
    else()
        message(WARNING
            "HYBRIDAI_ENABLE_HIP=ON but usable hip/rocBLAS config packages "
            "were not found. "
            "Compiling HIP backend as stubs.")
        if(WIN32 AND HYBRIDAI_ENABLE_HIP)
            message(STATUS "  hip_DIR=${hip_DIR}")
            message(STATUS "  rocblas_DIR=${rocblas_DIR}")
        endif()
    endif()
endif()
