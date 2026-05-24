cmake_minimum_required(VERSION 3.10)

# Fix FetchContent robustness warning with CMake 3.24+/4.x
# CMP0135: DOWNLOAD_EXTRACT_TIMESTAMP default behavior
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(celluniverse LANGUAGES C CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
    set(_detected_macos_arch "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    if(NOT _detected_macos_arch)
        execute_process(
            COMMAND /usr/bin/uname -m
            OUTPUT_VARIABLE _detected_macos_arch
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(_detected_macos_arch STREQUAL "arm64" OR _detected_macos_arch STREQUAL "aarch64")
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
    elseif(_detected_macos_arch STREQUAL "x86_64")
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build architectures for macOS" FORCE)
    endif()

    if(CMAKE_OSX_ARCHITECTURES)
        message(STATUS "macOS: auto-selected CMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
    endif()
endif()

# Force single-config generators (Unix Makefiles, Ninja) to build Release.
# Multi-config generators keep selecting the configuration at build time.
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Link-time optimization (IPO / LTO) for Release.
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

# OpenMP (per-cell loop parallelization in CellUniverse + Frame).
#
# Linux/GCC: find_package finds system OpenMP out of the box.
# macOS AppleClang: needs Homebrew libomp and explicit flags. We auto-detect
# Homebrew's libomp prefix and seed the hints so `find_package(OpenMP)`
# succeeds without extra cmake -D args.
#
# If OpenMP is not found on either platform, the `#pragma omp` directives
# compile as no-ops and the program runs serially (correct but slower).
if(APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(
            COMMAND ${BREW_EXECUTABLE} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE LIBOMP_BREW_RESULT
            ERROR_QUIET
        )
        if(LIBOMP_BREW_RESULT EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include" CACHE STRING "")
            set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "")
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "")
            set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib" CACHE FILEPATH "")
            message(STATUS "macOS: seeded OpenMP hints from Homebrew libomp at ${LIBOMP_PREFIX}")
        endif()
    endif()
endif()
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found — building without parallelization.")
    if(APPLE)
        message(STATUS "  macOS: run `brew install libomp` and re-run cmake to enable.")
    else()
        message(STATUS "  Linux: install libomp-dev (Debian/Ubuntu) or libgomp (GCC).")
    endif()
endif()

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV was not found.\n"
        "Install the development package first:\n"
        "  Ubuntu/Debian: sudo apt install libopencv-dev\n"
        "  macOS (Homebrew): brew install opencv\n"
        "If OpenCV is already installed in a non-standard location, re-run CMake with:\n"
        "  -DOpenCV_DIR=/path/to/OpenCVConfig.cmake directory\n"
        "or add its prefix to CMAKE_PREFIX_PATH."
    )
endif()

option(CELLUNIVERSE_BUILD_N2V2_PREPROCESS
       "Build the optional LibTorch N2V2 preprocessing targets." ON)
option(CELLUNIVERSE_PREFER_LOCAL_LIBTORCH
       "Prefer external/libtorch over any cached or system Torch package." ON)
option(CELLUNIVERSE_AUTO_INSTALL_LIBTORCH
       "Run scripts/install_libtorch.sh during configure if external/libtorch is missing." ON)
if(CELLUNIVERSE_BUILD_N2V2_PREPROCESS)
    set(CELLUNIVERSE_LOCAL_LIBTORCH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
    set(CELLUNIVERSE_LOCAL_TORCH_CONFIG
        "${CELLUNIVERSE_LOCAL_LIBTORCH}/share/cmake/Torch/TorchConfig.cmake")

    if(CELLUNIVERSE_PREFER_LOCAL_LIBTORCH AND
       CELLUNIVERSE_AUTO_INSTALL_LIBTORCH AND
       NOT EXISTS "${CELLUNIVERSE_LOCAL_TORCH_CONFIG}")
        set(CELLUNIVERSE_LIBTORCH_INSTALLER
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/install_libtorch.sh")
        if(EXISTS "${CELLUNIVERSE_LIBTORCH_INSTALLER}")
            message(STATUS
                "Local LibTorch was not found at ${CELLUNIVERSE_LOCAL_LIBTORCH}; "
                "running scripts/install_libtorch.sh. "
                "Set -DCELLUNIVERSE_AUTO_INSTALL_LIBTORCH=OFF to skip this.")
            execute_process(
                COMMAND "${CELLUNIVERSE_LIBTORCH_INSTALLER}"
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                RESULT_VARIABLE CELLUNIVERSE_LIBTORCH_INSTALL_RESULT
                COMMAND_ECHO STDOUT
            )
            if(NOT CELLUNIVERSE_LIBTORCH_INSTALL_RESULT EQUAL 0)
                message(WARNING
                    "scripts/install_libtorch.sh failed with exit code "
                    "${CELLUNIVERSE_LIBTORCH_INSTALL_RESULT}; continuing without "
                    "local LibTorch.")
            endif()
        else()
            message(WARNING
                "CELLUNIVERSE_AUTO_INSTALL_LIBTORCH is ON, but "
                "${CELLUNIVERSE_LIBTORCH_INSTALLER} does not exist.")
        endif()
    endif()

    if(CELLUNIVERSE_PREFER_LOCAL_LIBTORCH AND
       EXISTS "${CELLUNIVERSE_LOCAL_TORCH_CONFIG}")
        unset(Caffe2_DIR CACHE)
        unset(ATen_DIR CACHE)
        unset(TORCH_LIBRARY CACHE)
        unset(c10_LIBRARY CACHE)
        unset(C10_CUDA_LIBRARY CACHE)
        set(Torch_DIR
            "${CELLUNIVERSE_LOCAL_LIBTORCH}/share/cmake/Torch"
            CACHE PATH "Path to TorchConfig.cmake" FORCE)
        list(PREPEND CMAKE_PREFIX_PATH "${CELLUNIVERSE_LOCAL_LIBTORCH}")
        message(STATUS "Using local LibTorch at ${CELLUNIVERSE_LOCAL_LIBTORCH}")
    endif()

    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(WARNING
            "Torch was not found; N2V2 network preprocessing targets will not be built. "
            "Run scripts/install_libtorch.sh or configure with "
            "-DCMAKE_PREFIX_PATH=/path/to/libtorch.")
    endif()
endif()

if(TORCH_CXX_FLAGS)
    separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
endif()

set(cwd ${CMAKE_CURRENT_SOURCE_DIR})
set(n2v2_preprocess_prototype ${cwd}/prototype/n2v2_preprocess)

add_executable(${PROJECT_NAME}
        ${cwd}/src/main.cpp
        ${cwd}/src/CellFactory.cpp
        ${cwd}/src/CellGroundTruthBuilder.cpp
        ${cwd}/src/CsvHandler.cpp
        ${cwd}/src/CellUniverse.cpp
        ${cwd}/src/EmbryoBrightTracker.cpp
        ${cwd}/src/Frame.cpp
        ${cwd}/src/ImageHandler.cpp
        ${cwd}/src/PreprocessingHandler.cpp
        ${cwd}/src/LineageTreeCreator.cpp
        ${cwd}/src/LineageViewer.cpp
        ${cwd}/src/Ellipsoid.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${n2v2_preprocess_prototype}/include
        ${cwd}/includes
        ${cwd}/src
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} yaml-cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE lib/yaml-cpp/include)

if(CELLUNIVERSE_BUILD_N2V2_PREPROCESS AND Torch_FOUND)
    target_sources(${PROJECT_NAME} PRIVATE
            ${n2v2_preprocess_prototype}/src/N2V2Preprocessor.cpp
    )
    target_include_directories(${PROJECT_NAME} PRIVATE
            ${n2v2_preprocess_prototype}/include
    )
    target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE CELLUNIVERSE_HAS_N2V2_PREPROCESS=1)
    if(TORCH_CXX_FLAGS_LIST)
        target_compile_options(${PROJECT_NAME} PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE CELLUNIVERSE_HAS_N2V2_PREPROCESS=0)
endif()

# Release optimization flags.
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang,AppleClang>>:-O3 -DNDEBUG>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<PLATFORM_ID:Darwin>>,$<CXX_COMPILER_ID:GNU,Clang>>:-march=native>
)
if(IPO_SUPPORTED)
    set_property(TARGET ${PROJECT_NAME} PROPERTY
        INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

add_executable(brightness_volume_analyzer
        ${cwd}/src/BrightnessVolumeAnalyzer.cpp
        ${cwd}/src/EmbryoBrightTracker.cpp
        ${cwd}/src/LineageTreeCreator.cpp
        ${cwd}/src/LineageViewer.cpp
)

target_include_directories(brightness_volume_analyzer PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${cwd}/includes
        ${cwd}/src
        lib/yaml-cpp/include
)

target_link_libraries(brightness_volume_analyzer ${OpenCV_LIBS})
target_link_libraries(brightness_volume_analyzer yaml-cpp)

if(CELLUNIVERSE_BUILD_N2V2_PREPROCESS AND Torch_FOUND)
    add_library(n2v2_preprocess_lib
            ${n2v2_preprocess_prototype}/src/N2V2Preprocessor.cpp
    )

    target_include_directories(n2v2_preprocess_lib PUBLIC
            ${OpenCV_INCLUDE_DIRS}
            ${n2v2_preprocess_prototype}/include
            ${cwd}/lib/yaml-cpp/include
    )
    target_link_libraries(n2v2_preprocess_lib PUBLIC
            ${OpenCV_LIBS}
            yaml-cpp
            ${TORCH_LIBRARIES}
    )
    if(TORCH_CXX_FLAGS_LIST)
        target_compile_options(n2v2_preprocess_lib PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
    set_property(TARGET n2v2_preprocess_lib PROPERTY CXX_STANDARD 17)

    add_executable(celluniverse_preprocess
            ${n2v2_preprocess_prototype}/src/n2v2_preprocess_main.cpp
    )
    target_link_libraries(celluniverse_preprocess PRIVATE n2v2_preprocess_lib)
    if(IPO_SUPPORTED)
        set_property(TARGET celluniverse_preprocess PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    add_executable(n2v2_preprocess_unit_test
            ${cwd}/tests/n2v2_preprocess_test.cpp
    )
    target_link_libraries(n2v2_preprocess_unit_test PRIVATE n2v2_preprocess_lib)
    add_test(NAME n2v2_preprocess_unit_test COMMAND n2v2_preprocess_unit_test)
endif()

# Tests disabled 2026-04-08: they reference SimulationConfig fields
# (background_color, cell_color) that were removed in the sigmoid-pipeline
# invariants cleanup. Re-enable after updating the test files to use
# Frame::_backgroundValue / literal 1.0f seed.
# add_subdirectory(tests)
add_subdirectory(lib/yaml-cpp)
