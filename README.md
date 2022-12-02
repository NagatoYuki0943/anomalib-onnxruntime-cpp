# 说明

> 适用于anomalib导出的onnx格式的模型



# 下载onnxruntime和opencv

> [onnxruntime下载地址](https://github.com/microsoft/onnxruntime/releases)
>
> 暂时不清楚 gpu cuda tensorrt 版本的区别
>
> [onnxruntime文档](https://onnxruntime.ai/docs/)

> [opencv下载地址](https://opencv.org/releases/)

## 配置环境变量

```yaml
#opencv
D:\ai\opencv\build\x64\vc15\bin

#onnxruntime
D:\ai\onnxruntime\lib
```



# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json

# include

```cpp
//https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/include/providers.h
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "cpu_provider_factory.h"

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif
#ifdef USE_DNNL
#include "dnnl_provider_factory.h"
#endif
#ifdef USE_NUPHAR
#include "nuphar_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "tensorrt_provider_factory.h"
#endif
#ifdef USE_DML
#include "dml_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "migraphx_provider_factory.h"
#endif
```





# Cmake

> cmake和vs的代码一致，指示引入方式有差别
>
> cmake版本要设置 `CMakeLists.txt` 中 opencv，openvino的路径为自己的路径

```cmake
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
```

# VS

> 使用vs2019

属性

- C/C++

  - 附加包含目录 release debug 都包含

    ```python
    D:\ai\onnxruntime\include
    D:\ai\opencv\build\include
    ..\include	# rapidjson 为相对目录,可以更改为绝对目录
    ```
  
- 链接器

  - 附加包含目录 release debug 都包含

    ```python
    D:\ai\onnxruntime\lib
    D:\ai\opencv\build\x64\vc15\lib
    ```

  - 输入
  
    - 附加依赖项  release debug分开包含
  
      ```python
      # debug
      opencv_world460d.lib
      onnxruntime.lib
      onnxruntime_providers_cuda.lib
      onnxruntime_providers_shared.lib
      onnxruntime_providers_tensorrt.lib
      
      # release
      opencv_world460.lib
      onnxruntime.lib
      onnxruntime_providers_shared.lib
      onnxruntime_providers_cuda.lib
      onnxruntime_providers_tensorrt.lib
      ```

# 错误

### c2760 意外标记 “）”

属性

- C/C++
  - 语言 符合模式改为否

### 0xc000007b 0xC000007B

如果程序无法运行，将`onnxruntime\lib`下的`*.dll`文件复制到exe目录下可以解决

## 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
