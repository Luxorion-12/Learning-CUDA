cmake_minimum_required(VERSION 3.24)
project(LuxorionQuantDequant LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LUXORION_ENABLE_CUDA "Build the CUDA baseline and benchmark" ON)
if(LUXORION_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

function(luxorion_configure_target target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif()
endfunction()

# 可独立运行的CPU参考实现，也是GPU benchmark的正确性oracle。
add_library(cpu_reference_lib STATIC
  cpu_reference/src/formats.cpp
  cpu_reference/src/io.cpp
  cpu_reference/src/metrics.cpp
  cpu_reference/src/mxfp8_cpu.cpp
  cpu_reference/src/nvfp4_cpu.cpp
)
target_include_directories(cpu_reference_lib PUBLIC cpu_reference/include)
luxorion_configure_target(cpu_reference_lib)

add_executable(cpu_reference cpu_reference/src/main.cpp)
target_link_libraries(cpu_reference PRIVATE cpu_reference_lib)
luxorion_configure_target(cpu_reference)

if(LUXORION_ENABLE_CUDA)
  add_library(gpu_baseline STATIC
    gpu_baseline/src/mxfp8.cu
    gpu_baseline/src/nvfp4.cu
  )
  target_include_directories(gpu_baseline PUBLIC
    gpu_baseline/include
    cpu_reference/include
  )
  luxorion_configure_target(gpu_baseline)

  add_executable(generate_gpu_baseline_inputs
    gpu_baseline/tools/generate_inputs.cpp)
  target_link_libraries(generate_gpu_baseline_inputs PRIVATE cpu_reference_lib)
  luxorion_configure_target(generate_gpu_baseline_inputs)

  add_executable(gpu_baseline_benchmark
    gpu_baseline/tools/benchmark_main.cpp)
  target_link_libraries(gpu_baseline_benchmark PRIVATE
    gpu_baseline cpu_reference_lib)
  luxorion_configure_target(gpu_baseline_benchmark)
endif()
