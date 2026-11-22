cmake_minimum_required(VERSION 3.24)
project(cengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Apple Metal detection
# ----------------------------------------------------------------------------
# The native Objective-C++ Metal runtime + compiled shader library are the
# normal development configuration on Apple Silicon (architectural correction
# §2): CENGINE_COMPILE_METAL defaults ON when the Metal toolchain is present.
# It degrades to OFF (with a warning) when `xcrun metal` is unavailable so the
# plain C++/CPU-fallback build still works everywhere.
# ============================================================================
set(CENGINE_METAL_ENABLED OFF)
set(CENGINE_METAL_TOOLCHAIN "" CACHE STRING
    "Optional xcrun toolchain identifier for a separately downloaded Metal toolchain")

if(APPLE)
  find_program(XCRUN xcrun)
  set(_metal_tools_ok FALSE)
  if(XCRUN)
    execute_process(COMMAND ${XCRUN} metal --version
                    RESULT_VARIABLE _metal_rc OUTPUT_QUIET ERROR_QUIET)
    if(_metal_rc EQUAL 0)
      set(_metal_tools_ok TRUE)
    endif()
  endif()
  option(CENGINE_COMPILE_METAL
         "Compile Metal shaders + build the native Objective-C++ Metal runtime"
         ${_metal_tools_ok})
  if(CENGINE_COMPILE_METAL)
    if(_metal_tools_ok)
      enable_language(OBJCXX)
      set(CMAKE_OBJCXX_STANDARD 20)
      set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
      set(CENGINE_METAL_ENABLED ON)
      message(STATUS "cengine: Metal toolchain found; native Obj-C++ runtime ENABLED")
    else()
      message(WARNING
        "CENGINE_COMPILE_METAL=ON but 'xcrun metal' is unavailable. "
        "Install it with: xcodebuild -downloadComponent MetalToolchain. "
        "Building the plain C++/CPU-fallback configuration for now.")
    endif()
  endif()
else()
  option(CENGINE_COMPILE_METAL "Compile Metal shaders + native runtime" OFF)
endif()

# ============================================================================
# Core native library
# ============================================================================
set(CENGINE_CORE_SOURCES
  analytics/features.cpp
  market_depth/market-book.cpp
  orderbook/orderbook.cpp
  risk/risk_engine.cpp
  statArb/stat-arb.cpp
  volatility/volatility.cpp
  opportunity_discovery/cpu_reference.cpp
  opportunity_discovery/kernel_registry.cpp
  opportunity_discovery/discovery_engine.cpp)

if(CENGINE_METAL_ENABLED)
  list(APPEND CENGINE_CORE_SOURCES
    opportunity_discovery/metal_buffers.mm
    opportunity_discovery/metal_runtime.mm)
endif()

add_library(cengine_core STATIC ${CENGINE_CORE_SOURCES})
target_include_directories(cengine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Strict warnings for C++ TUs; the .mm TUs opt out of -Wpedantic (framework idioms).
target_compile_options(cengine_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra>)
if(CENGINE_METAL_ENABLED)
  target_compile_definitions(cengine_core PUBLIC CENGINE_WITH_METAL=1)
endif()

# ============================================================================
# Python module (pybind11)
# ============================================================================
option(CENGINE_BUILD_PYTHON "Build pybind11 Python module" ON)
if(CENGINE_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_cengine_native
    bindings/python_bindings.cpp
    bindings/discovery_bindings.cpp)
  target_link_libraries(_cengine_native PRIVATE cengine_core)
  if(CENGINE_METAL_ENABLED)
    target_compile_definitions(_cengine_native PRIVATE CENGINE_WITH_METAL=1)
    target_link_libraries(_cengine_native PRIVATE
      "-framework Metal" "-framework Foundation")
  endif()
  install(TARGETS _cengine_native LIBRARY DESTINATION .)
endif()

# ============================================================================
# Metal shader library
# ============================================================================
if(CENGINE_METAL_ENABLED)
  if(CENGINE_METAL_TOOLCHAIN)
    set(METAL_XCRUN_ARGS --toolchain ${CENGINE_METAL_TOOLCHAIN} -sdk macosx)
  else()
    set(METAL_XCRUN_ARGS -sdk macosx)
  endif()
  # The 45 "existing" kernels (covariance/imbalance/indicators/monte-carlo/
  # reductions) plus the Phase B opportunity_screening kernel newly added by the
  # native discovery layer (directive §13/§80: new shaders are tracked as
  # additions to the preserved 45-kernel baseline, not a rewrite of it).
  set(METAL_SOURCES covariance imbalance indicators monte-carlo
                    opportunity_screening technical_indicators reductions)
  foreach(NAME IN LISTS METAL_SOURCES)
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
      COMMAND ${XCRUN} ${METAL_XCRUN_ARGS} metal -c
              ${CMAKE_CURRENT_SOURCE_DIR}/metal/${NAME}.metal
              -o ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/${NAME}.metal)
    list(APPEND METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air)
  endforeach()
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib
    COMMAND ${XCRUN} ${METAL_XCRUN_ARGS} metallib ${METAL_AIR}
            -o ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib
    DEPENDS ${METAL_AIR})
  add_custom_target(cengine_metal ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib)
  # Install the metallib BESIDE the module so the runtime loads an authoritative,
  # version-matched artifact instead of discovering a stale build dir (§2).
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib DESTINATION .)
endif()
