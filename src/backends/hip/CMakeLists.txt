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
    # Prefer ROCm config packages. The SDK's legacy FindHIP module can report
    # HIP correctly but still leave the rocBLAS config package undiscovered.
    if(WIN32 AND DEFINED HYBRIDAI_ROCM_ROOT)
        if(NOT hip_DIR)
            set(hip_DIR "${HYBRIDAI_ROCM_ROOT}/lib/cmake/hip")
        endif()
        if(NOT rocblas_DIR)
            set(rocblas_DIR "${HYBRIDAI_ROCM_ROOT}/lib/cmake/rocblas")
        endif()
    endif()
    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
    if(hip_FOUND AND rocblas_FOUND AND TARGET hip::device AND
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
            target_sources(hybridai PRIVATE hip_backend.cc)
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
