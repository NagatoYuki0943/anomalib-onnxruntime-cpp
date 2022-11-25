cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(../include)

# opencv
set(OpenCV_DIR D:/ai/opencv/build/x64/vc15/lib)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBRARIES: " ${OpenCV_LIBRARIES})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBRARIES})

# onnxruntime include
include_directories(D:/ai/onnxruntime/include)
# onnxruntime lib
link_directories(D:/ai/onnxruntime/lib)
link_libraries(onnxruntime.lib onnxruntime_providers_cuda.lib onnxruntime_providers_shared.lib onnxruntime_providers_tensorrt.lib)

add_executable(main main.cpp utils.cpp opencv_utils.cpp inference.hpp)
add_executable(test test.cpp)