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
    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
    if(hip_FOUND AND rocblas_FOUND AND TARGET hip::device AND
       TARGET roc::rocblas)
        target_compile_definitions(hybridai PRIVATE HYBRIDAI_HAS_HIP)
        if(WIN32)
            # The Visual Studio generator does not support HIP as a native
            # language. Compile the implementation explicitly with hipcc and
            # add the resulting COFF object to the normal MSVC target.
            set(HYBRIDAI_HIPCC "${hip_HIPCC_EXECUTABLE}")
            if(NOT HYBRIDAI_HIPCC)
                message(FATAL_ERROR "HIP was found but hipcc.exe is unavailable")
            endif()

            set(HYBRIDAI_HIP_OBJECT
                "${CMAKE_CURRENT_BINARY_DIR}/hip_backend_hip.obj")
            add_custom_command(
                OUTPUT "${HYBRIDAI_HIP_OBJECT}"
                COMMAND "${HYBRIDAI_HIPCC}"
                    -nogpulib
                    --offload-arch=gfx1150
                    -O2
                    -std=c++20
                    $<$<CONFIG:Debug>:-O0>
                    $<$<CONFIG:Debug>:-g>
                    $<$<CONFIG:Debug>:-D_DEBUG>
                    $<$<CONFIG:Debug>:-D_ITERATOR_DEBUG_LEVEL=2>
                    $<$<CONFIG:Debug>:-fms-runtime-lib=dll_dbg>
                    -I"${CMAKE_SOURCE_DIR}/include"
                    -I"${CMAKE_SOURCE_DIR}/src"
                    -I"${hip_INCLUDE_DIR}"
                    -DHYBRIDAI_HAS_HIP=1
                    -D__HIP_NO_CUDA_MATH__=1
                    -c "${CMAKE_CURRENT_SOURCE_DIR}/hip_backend.cc"
                    -o "${HYBRIDAI_HIP_OBJECT}"
                DEPENDS
                    "${CMAKE_CURRENT_SOURCE_DIR}/hip_backend.cc"
                    "${CMAKE_CURRENT_SOURCE_DIR}/hip_backend.h"
                COMMENT "Compiling HIP backend with hipcc"
                VERBATIM
            )
            add_custom_target(hybridai_hip_backend_compile
                DEPENDS "${HYBRIDAI_HIP_OBJECT}")
            add_dependencies(hybridai hybridai_hip_backend_compile)
            set_source_files_properties(
                "${HYBRIDAI_HIP_OBJECT}" PROPERTIES GENERATED TRUE)
            target_sources(hybridai PRIVATE "${HYBRIDAI_HIP_OBJECT}")
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
    endif()
endif()
