# 说明

> 适用于anomalib导出的onnx格式的模型

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

> cmake版本要设置 `CMakeLists.txt` 中 opencv，onnxruntime 路径为自己的路径



# 错误

### 0xc000007b 0xC000007B

如果程序无法运行，将`onnxruntime\lib`下的`*.dll`文件复制到exe目录下可以解决

## 查看是否缺失dll

> https://github.com/lucasg/Dependencies 这个工具可以查看exe工具是否缺失dll

# 第三方库

处理json使用了rapidjson https://rapidjson.org/zh-cn/

opencv方式参考了mmdeploy https://github.com/open-mmlab/mmdeploy/tree/master/csrc/mmdeploy/utils/opencv
