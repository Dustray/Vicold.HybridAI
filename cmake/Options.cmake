# Platform detection
if(WIN32)
    set(HYBRIDAI_PLATFORM_WINDOWS TRUE)
    add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)
else()
    set(HYBRIDAI_PLATFORM_LINUX TRUE)
endif()

# Backend validation
if(HYBRIDAI_ENABLE_HIP AND HYBRIDAI_ENABLE_CUDA)
    message(WARNING "Both HIP and CUDA are enabled. This is supported but make sure only one is active per binary.")
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Position independent code for static library linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
