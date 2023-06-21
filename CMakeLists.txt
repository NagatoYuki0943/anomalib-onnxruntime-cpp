cmake_minimum_required(VERSION 3.19)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR D:/ml/opencv/build)
set(ONNXRuntime_DIR D:/ml/onnxruntime)

# rapidjson 为相对目录,可以更改为绝对目录
include_directories(include)

# opencv
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
    message("OpenCV_LIBS: " ${OpenCV_LIBS})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${OpenCV_LIBS})

# onnxruntime include
include_directories(${ONNXRuntime_DIR}/include)
# onnxruntime lib
link_directories(${ONNXRuntime_DIR}/lib)
link_libraries(onnxruntime.lib onnxruntime_providers_cuda.lib onnxruntime_providers_shared.lib onnxruntime_providers_tensorrt.lib)

add_executable(main main.cpp utils.cpp opencv_utils.cpp inference.hpp)
add_executable(test test.cpp)
