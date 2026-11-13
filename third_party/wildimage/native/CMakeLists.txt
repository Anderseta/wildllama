cmake_minimum_required(VERSION 3.24)
project(wildimage_native_tests LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(ggml CONFIG REQUIRED)
set(SD_SOURCE_DIRECTORY "" CACHE PATH "Pinned stable-diffusion.cpp source with WildImage overlays")
if(NOT EXISTS "${SD_SOURCE_DIRECTORY}/src/core/cuda/runtime.h")
    message(FATAL_ERROR "Set SD_SOURCE_DIRECTORY to the prepared pinned source")
endif()
add_executable(test-fp8 test_fp8.cu wildimage_fp8.cu)
target_include_directories(test-fp8 PRIVATE "${SD_SOURCE_DIRECTORY}/src")
target_link_libraries(test-fp8 PRIVATE ggml::ggml CUDA::cudart CUDA::cublasLt)
target_compile_features(test-fp8 PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(test-fp8 PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")
enable_testing()
add_test(NAME fp8-exact-parity COMMAND test-fp8)
add_executable(test-cuda-wait test_cuda_wait.cpp)
target_compile_features(test-cuda-wait PRIVATE cxx_std_17)
add_test(NAME cuda-wait-policy COMMAND test-cuda-wait)

add_executable(test-gguf-shapes test_gguf_shapes.cpp "${SD_SOURCE_DIRECTORY}/src/model_io/gguf_io.cpp")
target_include_directories(test-gguf-shapes PRIVATE "${SD_SOURCE_DIRECTORY}/src" "${SD_SOURCE_DIRECTORY}/include" "${SD_SOURCE_DIRECTORY}/thirdparty")
target_link_libraries(test-gguf-shapes PRIVATE ggml::ggml)
target_compile_features(test-gguf-shapes PRIVATE cxx_std_17)
add_test(NAME gguf-original-shapes COMMAND test-gguf-shapes "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(test-int8-workspace test_int8_workspace.cu
    "${SD_SOURCE_DIRECTORY}/src/core/cuda/i8_gemm.cu")
target_include_directories(test-int8-workspace PRIVATE "${SD_SOURCE_DIRECTORY}/src"
    "${SD_SOURCE_DIRECTORY}/thirdparty/cutlass/include")
target_link_libraries(test-int8-workspace PRIVATE ggml::ggml CUDA::cudart CUDA::cublas)
target_compile_features(test-int8-workspace PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(test-int8-workspace PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--extended-lambda;--expt-relaxed-constexpr>")
if(MSVC)
    target_compile_options(test-int8-workspace PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/Zc:__cplusplus>")
endif()
add_test(NAME int8-workspace-aliases COMMAND test-int8-workspace)

add_executable(benchmark-int8 EXCLUDE_FROM_ALL benchmark_int8.cu)
target_include_directories(benchmark-int8 PRIVATE "${SD_SOURCE_DIRECTORY}/src"
    "${SD_SOURCE_DIRECTORY}/thirdparty/cutlass/include")
target_link_libraries(benchmark-int8 PRIVATE ggml::ggml CUDA::cudart)
target_compile_features(benchmark-int8 PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(benchmark-int8 PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--extended-lambda;--expt-relaxed-constexpr>")
if(MSVC)
    target_compile_options(benchmark-int8 PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor,/Zc:__cplusplus>")
endif()

set(SD_RUNTIME_LIBRARY "" CACHE FILEPATH "Built image runtime library for the optional model staging test")
if(EXISTS "${SD_RUNTIME_LIBRARY}")
    add_executable(test-image-staging EXCLUDE_FROM_ALL test_image_staging.cpp)
    target_include_directories(test-image-staging PRIVATE "${SD_SOURCE_DIRECTORY}/include")
    target_link_libraries(test-image-staging PRIVATE "${SD_RUNTIME_LIBRARY}" ggml::ggml)
    target_compile_features(test-image-staging PRIVATE cxx_std_17)
    target_compile_definitions(test-image-staging PRIVATE SD_BUILD_SHARED_LIB)
endif()
