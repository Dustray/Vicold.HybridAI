# HIP backend. Native HIP/ROCm headers are only included in hip_backend.cc.
# We always compile the HIP backend source files so that the backend factory is
# registered, even on Windows where ROCm is unavailable. When HIP is not found the
# implementation falls back to compile-time stubs.

target_sources(hybridai
    PRIVATE
        hip_backend.h
        hip_backend.cc
        hip_backend_factory.cc
)

if(HYBRIDAI_ENABLE_HIP)
    find_package(hip QUIET)
    find_package(rocblas QUIET)
    if(hip_FOUND AND rocblas_FOUND)
        target_compile_definitions(hybridai PRIVATE HYBRIDAI_HAS_HIP)
        target_link_libraries(hybridai
            PUBLIC
                hip::device
                roc::rocblas
        )
        message(STATUS "HIP backend enabled with hip/rocBLAS")
    else()
        message(WARNING
            "HYBRIDAI_ENABLE_HIP=ON but hip/rocBLAS not found. "
            "Compiling HIP backend as stubs.")
    endif()
endif()
