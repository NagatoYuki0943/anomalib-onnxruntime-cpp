# 说明

> 适用于anomalib导出的onnx格式的模型
>
> 测试了patchcore,fastflow和efficient_ad模型

```yaml
# 模型配置文件中设置为onnx,导出openvino会导出onnx
optimization:
  export_mode: onnx # options: torch, onnx, openvino
```

# 其他推理方式

> [anomalib-onnxruntime-cpp](https://github.com/NagatoYuki0943/anomalib-onnxruntime-cpp)
>
> [anomalib-openvino-cpp](https://github.com/NagatoYuki0943/anomalib-openvino-cpp)
>
> [anomalib-tensorrt-cpp](https://github.com/NagatoYuki0943/anomalib-tensorrt-cpp)

# example

```C++
#include "inference.hpp"
#include <opencv2/opencv.hpp>


int main() {
    // patchcore模型训练配置文件删除了center_crop
    string model_path = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/model.onnx";
    string meta_path  = "D:/ml/code/anomalib/results/efficient_ad/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-onnxruntime-cpp/result"; // 注意目录不会自动创建,要手动创建才会保存
    string device     = "cuda";
    int threads       = 4;  // Ort::SessionOptions SetIntraOpNumThreads & SetInterOpNumThreads
    int gpu_mem_limit = 4;  // onnxruntime gpu memory limit
    bool efficient_ad = true; // 是否使用efficient_ad模型

    // 创建推理器
    auto inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit, efficient_ad);

    // 单张图片推理
    cv::Mat result = inference.single(image_path, save_dir);
    cv::resize(result, result, { 1500, 500 });
    cv::imshow("result", result);
    cv::waitKey(0);

    // 多张图片推理
    inference.multi(image_dir, save_dir);
    return 0;
}

```





# onnxruntime官方例子

> https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx
>
> 推荐 `squeezenet/main.cpp`

# 下载onnxruntime和opencv

> onnxruntime下载地址 https://github.com/microsoft/onnxruntime/releases
>
> onnxruntime文档 https://onnxruntime.ai/docs/
>
> onnxruntime使用gpu要安装cuda和cudnn
>
> https://developer.nvidia.com/cuda-toolkit
>
> https://developer.nvidia.cn/zh-cn/cudnn

> https://opencv.org

## 配置环境变量

```yaml
# opencv
$opencv_path\build\x64\vc16\bin

# onnxruntime
$onnxruntime_path\lib
```

# 关于include文件夹

> include文件夹是rapidjson的文件，用来解析json

# Cmake

> 设置 `CMakeLists.txt` 中 opencv，onnxruntime 路径为自己的路径

# 错误

### 0xc000007b 0xC000007B

如果程序无法运行，将`onnxruntime\lib`下的`*.dll`文件复制到exe目录下可以解决

# 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
