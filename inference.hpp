#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include "utils.h"

using namespace std;

class Inference {
private:
    bool efficient_ad;                                      // 是否使用efficient_ad模型
    MetaData meta{};                                        // 超参数
    Ort::Env env{};                                         // 三个ort参数
    Ort::AllocatorWithDefaultOptions allocator{};
    Ort::RunOptions runOptions{};
    Ort::Session session = Ort::Session(nullptr);           // onnxruntime session
    size_t input_nums{};                                    // 模型输入值数量
    size_t output_nums{};                                   // 模型输出值数量
    vector<const char*> input_node_names;                   // 输入节点名
    vector<Ort::AllocatedStringPtr> input_node_names_ptr;   // 输入节点名指针,保存它防止释放 https://github.com/microsoft/onnxruntime/issues/13651
    vector<vector<int64_t>> input_dims;                     // 输入形状
    vector<const char*> output_node_names;                  // 输出节点名
    vector<Ort::AllocatedStringPtr> output_node_names_ptr;  // 输入节点名指针
    vector<vector<int64_t>> output_dims;                    // 输出形状

public:
    /**
     * @param model_path    模型路径
     * @param meta_path     超参数路径
     * @param device        cpu or cuda or tensorrt 推理
     * @param threads       SetIntraOpNumThreads 线程数, defaults to 0
     * @param gpu_mem_limit 显存限制, only for cuda or tensorrt device, defaults to 2 GB
     * @param efficient_ad  是否使用efficient_ad模型
     */
    Inference(string& model_path, string& meta_path, string& device, int threads = 0, int gpu_mem_limit = 2, bool efficient_ad = false) {
        this->efficient_ad = efficient_ad;
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->get_model(model_path, device, threads, gpu_mem_limit);
        // 3.获取模型的输入输出
        this->get_onnx_info();
        // 4.模型预热
        this->warm_up();
    }

    /**
     * get onnx model
     * @param model_path    模型路径
     * @param device        使用的设备
     * @param threads       SetIntraOpNumThreads 线程数, defaults to 0
     * @param gpu_mem_limit 显存限制, only for cuda or tensorrt device, defaults to 2 GB
     */
    void get_model(string& model_path, string& device, int threads = 0, int gpu_mem_limit = 2) {
        // 获取可用的provider
        auto availableProviders = Ort::GetAvailableProviders();
        for (const auto& provider : availableProviders) {
            cout << provider << " ";
        }
        cout << endl;

        Ort::SessionOptions sessionOptions;
        // 使用0个线程执行op,若想提升速度，增加线程数
        sessionOptions.SetIntraOpNumThreads(threads);
        sessionOptions.SetInterOpNumThreads(threads);
        // ORT_ENABLE_ALL: 启用所有可能的优化
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (device == "cuda" || device == "tensorrt") {
            // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            // https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            if (device == "tensorrt") {
                // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
                // https://onnxruntime.ai/docs/api/c/struct_ort_tensor_r_t_provider_options.html
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = 0;
                trt_options.trt_max_workspace_size = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // gpu memory limit
                trt_options.trt_fp16_enable = 0;
                sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
            }
        }
        wchar_t* model_path1 = new wchar_t[model_path.size()];
        swprintf(model_path1, 4096, L"%S", model_path.c_str());
        // create session
        this->session = Ort::Session(this->env, model_path1, sessionOptions);
    }

    void get_onnx_info() {
        // 1. 获得模型有多少个输入和输出，一般是指对应网络层的数目, 如果是多输出网络，就会是对应输出的数目
        this->input_nums = this->session.GetInputCount();
        this->output_nums = this->session.GetOutputCount();
        printf("Number of inputs = %zu\n", this->input_nums); // Number of inputs = 1
        printf("Number of output = %zu\n", this->output_nums);// Number of output = 1

        // 2.获取输入输出name
        // 3.获取维度数量
        for (int i = 0; i < this->input_nums; i++) {
            // 输入变量名
            Ort::AllocatedStringPtr input_name = this->session.GetInputNameAllocated(i, this->allocator);
            this->input_node_names.push_back(input_name.get());
            this->input_node_names_ptr.push_back(move(input_name));

            // 输入形状
            auto input_shape_info = this->session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            this->input_dims.push_back(input_shape_info.GetShape());
        }

        for (int i = 0; i < this->output_nums; i++) {
            // 输出变量名
            Ort::AllocatedStringPtr output_name = this->session.GetOutputNameAllocated(i, this->allocator);
            this->output_node_names.push_back(output_name.get());
            this->output_node_names_ptr.push_back(move(output_name));

            // 输出形状
            auto output_shape_info = this->session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
            this->output_dims.push_back(output_shape_info.GetShape());
        }

        for (int i = 0; i < this->input_nums; ++i) {
            cout << "input_dims: ";
            for (const auto j : this->input_dims[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
        // test dynamic batch, only support 1 input and 1 output
        // this->input_dims[0] = { 1, 3, 256, 256 };

        for (int i = 0; i < this->output_nums; ++i) {
            cout << "output_dims: ";
            for (const auto j : this->output_dims[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
        // test dynamic batch, only support 1 input and 1 output
        // this->output_dims[0] = { 1, 1, 256, 256 };
    }

    /**
     * 模型预热
     */
    void warm_up() {
        // 输入数据
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }

    /**
     * 推理单张图片
     * @param image 原始图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result infer(cv::Mat& image) {
        // 1.保存图片原始高宽
        this->meta.image_size[0] = image.size().height;
        this->meta.image_size[1] = image.size().width;

        // 2.图片预处理
        cv::Mat resized_image = pre_process(image, this->meta, this->efficient_ad);
        cv::Mat blob = cv::dnn::blobFromImage(resized_image);

        // 3.从图像创建tensor
        // 3.1 申请内存空间
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 3.2 创建输入值
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_dims[0].data(), input_dims[0].size());
        // 3.3 推理 只传递输入
        vector<Ort::Value> output_tensors;
        try {
            output_tensors = session.Run(
                this->runOptions,
                this->input_node_names.data(),
                &input_tensor,
                this->input_nums,
                this->output_node_names.data(),
                this->output_nums
            );
        }
        catch (Ort::Exception& e) {
            cout << e.what() << endl;
        }

        // 4.将热力图转换为Mat
        auto* output0 = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]), CV_32FC1, output0);

        // 5.针对不同输出数量获取得分
        // efficient_ad模型有3个输出,不过只有第1个是anomaly_map,其余不用处理
        cv::Mat pred_score;
        if (this->output_nums == 2) {
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, output_tensors[1].GetTensorMutableData<float>());  // {1}
        }
        else {
            double _, maxValue;    // 最大值，最小值
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 6.后处理:标准化,缩放到原图
        vector<cv::Mat> post_mat = post_process(anomaly_map, pred_score, this->meta);
        anomaly_map = post_mat[0];
        float score = post_mat[1].at<float>(0, 0);

        // 7.返回结果
        return Result{ anomaly_map, score };
    }

    /**
     * 单张图片推理
     * @param image    RGB图片
     * @return      标准化的并所放到原图热力图和得分
     */
    Result single(cv::Mat& image) {
        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 1.推理单张图片
        Result result = this->infer(image);
        cout << "score: " << result.score << endl;

        // 2.生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;

        // 3.保存显示图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        // 拼接图片
        cv::Mat res;
        cv::hconcat(images, res);

        return Result{ res, result.score };
    }

    /**
     * 多张图片推理
     * @param image_dir 图片文件夹路径
     * @param save_dir  保存路径
     */
    void multi(string& image_dir, string& save_dir) {
        // 1.读取全部图片路径
        vector<cv::String> paths = getImagePaths(image_dir);

        vector<float> times;
        for (auto& image_path : paths) {
            // 2.读取单张图片
            cv::Mat image = readImage(image_path);

            // time
            auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            // 3.推理单张图片
            Result result = this->infer(image);
            cout << "score: " << result.score << endl;

            // 4.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
            vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
            // time
            auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            cout << "infer time: " << end - start << " ms" << endl;
            times.push_back(end - start);

            // 5.保存图片
            // 将mask转化为3通道,不然没法拼接图片
            cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
            // 拼接图片
            cv::Mat res;
            cv::hconcat(images, res);
            saveScoreAndImages(result.score, res, image_path, save_dir);
        }

        // 6.统计数据
        double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
        double avgValue = sumValue / times.size();                   // 求均值
        cout << "avg infer time: " << avgValue << " ms" << endl;
    }
};