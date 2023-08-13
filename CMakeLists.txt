cmake_minimum_required(VERSION 3.0.0)

set(CMAKE_CUDA_ARCHITECTURES 86) # 使用するGPUのCompute Capability
# set(CUDAToolkit_ROOT "Your-CUDA-Install-Dir") # PCに複数VerのCUDAがインストールされている場合は明示的に指定する.

project(PracCuSparse VERSION 0.1.0 LANGUAGES C CXX CUDA)

find_package(CUDAToolkit)

add_executable(${PROJECT_NAME}  
    main.cu
    )

target_link_libraries(${PROJECT_NAME} PRIVATE 
    cudart
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")