cmake_minimum_required(VERSION 3.24)
project(cengine LANGUAGES CXX)

install(FILES run.py order_manager.py strategies.py liquidity.py native_adapters.py
        DESTINATION .)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(cengine_core STATIC
  analytics/features.cpp
  market_depth/market-book.cpp
  orderbook/orderbook.cpp
  risk/risk_engine.cpp
  statArb/stat-arb.cpp
  volatility/volatility.cpp)
target_include_directories(cengine_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(cengine_core PRIVATE -Wall -Wextra -Wpedantic)

option(CENGINE_BUILD_PYTHON "Build pybind11 Python module" ON)
if(CENGINE_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_cengine_native bindings/python_bindings.cpp)
  target_link_libraries(_cengine_native PRIVATE cengine_core)
  install(TARGETS _cengine_native LIBRARY DESTINATION .)
endif()

if(APPLE)
  option(CENGINE_COMPILE_METAL "Compile Metal kernels (requires full Xcode Metal tools)" OFF)
  if(CENGINE_COMPILE_METAL)
    find_program(XCRUN xcrun REQUIRED)
    set(METAL_SOURCES covariance imbalance indicators monte-carlo reductions)
    foreach(NAME IN LISTS METAL_SOURCES)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
        COMMAND ${XCRUN} -sdk macosx metal -c
                ${CMAKE_CURRENT_SOURCE_DIR}/metal/${NAME}.metal
                -o ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/metal/${NAME}.metal)
      list(APPEND METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/${NAME}.air)
    endforeach()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib
      COMMAND ${XCRUN} -sdk macosx metallib ${METAL_AIR}
              -o ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib
      DEPENDS ${METAL_AIR})
    add_custom_target(cengine_metal ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cengine.metallib)
  endif()
endif()
