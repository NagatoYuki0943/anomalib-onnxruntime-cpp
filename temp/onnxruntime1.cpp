// https://zhuanlan.zhihu.com/p/513777076

#include "onnxruntime_cxx_api.h"

int main(){
  Ort::Env env;
  std::string weightFile = "./xxx.onnx";

  Ort::SessionOptions session_options;
  OrtCUDAProviderOptions options;
  options.device_id = 0;
  options.arena_extend_strategy = 0;
  //options.cuda_mem_limit = (size_t)1 * 1024 * 1024 * 1024;//onnxruntime1.7.0
  options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024; //onnxruntime1.8.1, onnxruntime1.9.0
  options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
  options.do_copy_in_default_stream = 1;
  session_options.AppendExecutionProvider_CUDA(options);

  Ort::Session session_{env, weightFile.c_str(), Ort::SessionOptions{nullptr}}; //CPU
  //Ort::Session session_{env, weightFile.c_str(), session_options}; //GPU

  static constexpr const int width_ = ; //模型input width
  static constexpr const int height_ = ; //模型input height
  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 3, height_, width_}; //NCHW, 1x3xHxW

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 3}; //模型output shape，此处假设是二维的(1,3)

  std::array<float, width_ * height_ * 3> input_image_{}; //输入图片，HWC
  std::array<float, 3> results_{}; //模型输出，注意和output_shape_对应

  std::string imgPath = "./xxx.jpg";
  cv::Mat img = cv::imread(imgPath);

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  const char* input_names[] = {"xxx"}; //输入节点名
  const char* output_names[] = {"xxx"}; //输出节点名

  //预处理
  cv::Mat img_f32;
  img.convertTo(img_f32, CV_32FC3);//转float

  //BGR2RGB,
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
        input_image_[i * img.cols + j + 0] = img_f32.at<cv::Vec3f>(i, j)[2];
        input_image_[i * img.cols + j + 1 * img.cols * img.rows] = img_f32.at<cv::Vec3f>(i, j)[1];
        input_image_[i * img.cols + j + 2 * img.cols * img.rows] = img_f32.at<cv::Vec3f>(i, j)[0];
    }
  }

  session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

  //获取output的shape
  Ort::TensorTypeAndShapeInfo shape_info = output_tensor_.GetTensorTypeAndShapeInfo();

  //获取output的dim
  size_t dim_count = shape_info.GetDimensionsCount();
  //std::cout<< dim_count << std::endl;

  //获取output的shape
  int64_t dims[2];
  shape_info.GetDimensions(dims, sizeof(dims) / sizeof(dims[0]));
  //std::cout<< dims[0] << "," << dims[1] << std::endl;

  //取output数据
  float* f = output_tensor_.GetTensorMutableData<float>();
  for(int i = 0; i < dims[1]; i++)
  {
    std::cout<< f[i]<< std::endl;
  }
}
