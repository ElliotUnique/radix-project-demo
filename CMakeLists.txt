cmake_minimum_required(VERSION 3.20)
project(radix LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(demo radix-demo.cu)

target_compile_options(demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xptxas=-v>)

if(MSVC)
    target_compile_options(demo PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4,/permissive-,/WX>)
else()
    target_compile_options(demo PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
endif()
