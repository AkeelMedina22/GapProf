cmake_minimum_required(VERSION 3.17)
project(GapProf LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")

set(USE_TRACY OFF CACHE BOOL "Use Tracy for Profiling")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

include_directories( # Edit path to tracy
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /path/to/tracy
)

file(GLOB GAPPROF_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

if(USE_TRACY) # Edit path to tracy client
    list(APPEND GAPPROF_SOURCES "/path/to/tracy/TracyClient.cpp")
    add_compile_definitions(USE_TRACY)
endif()

add_library(gapprof SHARED ${GAPPROF_SOURCES})

if(USE_TRACY)
    target_compile_definitions(gapprof PRIVATE TRACY_NO_EXIT)
endif()

target_link_libraries(gapprof PRIVATE 
    CUDA::nvml 
    CUDA::cupti
    CUDA::cudart
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

enable_testing()

add_executable(test_prof tests/test_prof.cu)
add_executable(test_prof_async tests/test_prof_async.cu)
add_executable(test_queue tests/test_queue.cpp)
add_executable(bench_queue tests/bench_queue.cpp)
add_executable(test_nvml_refresh tests/test_nvml_refresh.cpp)
add_executable(test_api_latency tests/test_api_latency.cpp)

target_link_libraries(test_queue 
    PRIVATE 
        Threads::Threads
)

target_link_libraries(bench_queue 
    PRIVATE 
        Threads::Threads
)

target_link_libraries(test_nvml_refresh
    PRIVATE
        gapprof
        CUDA::nvml
)

target_link_libraries(test_api_latency
    PRIVATE
        CUDA::nvml
)

target_link_libraries(test_prof PRIVATE gapprof CUDA::cudart)
target_link_libraries(test_prof_async PRIVATE gapprof CUDA::cudart)

add_test(NAME QueuePerformanceTest COMMAND test_queue)

add_executable(test_queue_tsan tests/test_queue.cpp)
target_link_libraries(test_queue_tsan PRIVATE Threads::Threads)
target_compile_definitions(test_queue_tsan PRIVATE GAPPROF_SKIP_BENCHMARK)
target_compile_options(test_queue_tsan PRIVATE -fsanitize=thread -g -O1)
target_link_options(test_queue_tsan PRIVATE -fsanitize=thread)

add_test(NAME QueueThreadSanitizerTest COMMAND test_queue_tsan)