cmake_minimum_required(VERSION 3.25)

project(lonexgrad VERSION 1.0.7 LANGUAGES CXX)

# Build options
option(LONEXGRAD_BUILD_CUDA "Build lonexgrad with cuda backend support" OFF)
option(LONEXGRAD_BUILD_TESTS "Build lonexgrad tests" OFF)
option(LONEXGRAD_BUILD_EXAMPLES "Build lonexgrad examples" OFF)

# Hide symbols by default
if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET AND
    NOT DEFINED CMAKE_VISIBILITY_INLINES_HIDDEN)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)
endif()

# Let lonexgrad_SHARED_LIBS override BUILD_SHARED_LIBS
if(DEFINED lonexgrad_SHARED_LIBS)
    set(BUILD_SHARED_LIBS "${lonexgrad_SHARED_LIBS}")
endif()

# Library and properties
add_library(lonexgrad)
add_library(lonexgrad::lonexgrad ALIAS lonexgrad)
set_target_properties(lonexgrad PROPERTIES
                      VERSION ${lonexgrad_VERSION}
                      SOVERSION ${lonexgrad_VERSION_MAJOR}
                      POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
    lonexgrad PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_compile_features(lonexgrad PUBLIC cxx_std_20)

# Generate export headers
include(GenerateExportHeader)
generate_export_header(lonexgrad 
    EXPORT_FILE_NAME include/lonexgrad/export.h
)
target_compile_definitions(
    lonexgrad PUBLIC "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:lonexgrad_STATIC_DEFINE>")
target_include_directories(
    lonexgrad PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>")

# Find cuda and set flags to build cuda backend
set(LONEXGRAD_BUILD_CUDA_AND_FOUND OFF)
if(LONEXGRAD_BUILD_CUDA)
    find_package(CUDAToolkit)
    if(${CUDAToolkit_FOUND})
        message("Cuda found, building with cuda backend support")
        set(LONEXGRAD_BUILD_CUDA_AND_FOUND ON)
        enable_language(CUDA)
        set(lonexgrad_CUDA_FLAGS "--expt-relaxed-constexpr")
        target_compile_features(lonexgrad PUBLIC cuda_std_20)
        target_compile_options(lonexgrad PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${lonexgrad_CUDA_FLAGS}>")
        target_link_libraries(lonexgrad PRIVATE CUDA::cudart)
        set_target_properties(lonexgrad PROPERTIES 
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES native
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        # Macro definition so source files can conditionally set CUDA device/backend
        target_compile_definitions(lonexgrad PUBLIC LONEXGRAD_CUDA)

    else()
        message(WARNING "Couldn't find cuda, skipping cuda backend support")
    endif()
endif()

# Enable SSE
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# Library source
add_subdirectory(lonexgrad)

# Include the install rules if the user wanted them (included by default when top-level)
string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" is_top_level)
option(lonexgrad_INCLUDE_PACKAGING "Include packaging rules for lonexgrad" "${is_top_level}")
if (lonexgrad_INCLUDE_PACKAGING)
    add_subdirectory(packaging)
endif ()

# Build tests
if (${LONEXGRAD_BUILD_TESTS})
    enable_testing()
    add_subdirectory(tests)
endif()

# Build examples
if (${LONEXGRAD_BUILD_EXAMPLES})
    add_subdirectory(examples)
endif()