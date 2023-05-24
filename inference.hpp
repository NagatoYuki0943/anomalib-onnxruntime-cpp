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
    MetaData meta{};                                        // ������
    Ort::Env env{};                                         // ����ort����
    Ort::AllocatorWithDefaultOptions allocator{};
    Ort::RunOptions runOptions{};
    Ort::Session session = Ort::Session(nullptr);           // onnxruntime session
    size_t input_nums{};                                    // ģ������ֵ����
    size_t output_nums{};                                   // ģ�����ֵ����
    vector<const char*> input_node_names;                   // ����ڵ���
    vector<Ort::AllocatedStringPtr> input_node_names_ptr;   // ����ڵ���ָ��,��������ֹ�ͷ� https://github.com/microsoft/onnxruntime/issues/13651
    vector<vector<int64_t>> input_dims;                     // ������״
    vector<const char*> output_node_names;                  // ����ڵ���
    vector<Ort::AllocatedStringPtr> output_node_names_ptr;  // ����ڵ���ָ��
    vector<vector<int64_t>> output_dims;                    // �����״

public:
    /**
     * @param model_path    ģ��·��
     * @param meta_path     ������·��
     * @param device        cpu or cuda or tensorrt ����
     * @param threads       SetIntraOpNumThreads �߳���, defaults to 0
     * @param gpu_mem_limit �Դ�����, only for cuda or tensorrt device, defaults to 2 GB
     */
    Inference(string& model_path, string& meta_path, string& device, int threads = 0, int gpu_mem_limit = 2) {
        // 1.��ȡmeta
        this->meta = getJson(meta_path);
        // 2.����ģ��
        this->session = this->get_model(model_path, device, threads, gpu_mem_limit);
        // 3.��ȡģ�͵��������
        this->get_onnx_info();
        // 4.ģ��Ԥ��
        this->warm_up();
    }

    /**
     * get onnx model
     * @param model_path    ģ��·��
     * @param device        ʹ�õ��豸
     * @param threads       SetIntraOpNumThreads �߳���, defaults to 0
     * @param gpu_mem_limit �Դ�����, only for cuda or tensorrt device, defaults to 2 GB
     */
    Ort::Session get_model(string& model_path, string& device, int threads = 0, int gpu_mem_limit = 2) {
        // ��ȡ���õ�provider
        auto availableProviders = Ort::GetAvailableProviders();
        for (const auto& provider : availableProviders) {
            cout << provider << " ";
        }
        cout << endl;
        // TensorrtExecutionProvider
        // CUDAExecutionProvider
        // CPUExecutionProvider

        Ort::SessionOptions sessionOptions;
        // ʹ��0���߳�ִ��op,���������ٶȣ������߳���
        sessionOptions.SetIntraOpNumThreads(threads);
        // ORT_ENABLE_ALL: �������п��ܵ��Ż�
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device == "cuda" || device == "tensorrt") {
            // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            // https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // 2GB
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            if (device == "tensorrt") {
                // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
                // https://onnxruntime.ai/docs/api/c/struct_ort_tensor_r_t_provider_options.html
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = 0;
                trt_options.trt_max_workspace_size = (size_t)gpu_mem_limit * 1024 * 1024 * 1024; // 2GB
                trt_options.trt_fp16_enable = 0;
                sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
            }
        }
        wchar_t* model_path1 = new wchar_t[model_path.size()];
        swprintf(model_path1, 2048, L"%S", model_path.c_str());
        // create session
        Ort::Session session = Ort::Session(this->env, model_path1, sessionOptions);
        return session;
    }

    void get_onnx_info() {
        // 1. ���ģ���ж��ٸ�����������һ����ָ��Ӧ��������Ŀ, ����Ƕ�������磬�ͻ��Ƕ�Ӧ�������Ŀ
        this->input_nums = session.GetInputCount();
        this->output_nums = session.GetOutputCount();
        printf("Number of inputs = %zu\n", this->input_nums); // Number of inputs = 1
        printf("Number of output = %zu\n", this->output_nums);// Number of output = 1

        // 2.��ȡ�������name
        // 3.��ȡά������
        for (int i = 0; i < this->input_nums; i++) {
            // ���������
            Ort::AllocatedStringPtr input_name = this->session.GetInputNameAllocated(i, this->allocator);
            this->input_node_names.push_back(input_name.get());
            this->input_node_names_ptr.push_back(move(input_name));

            // ������״
            auto input_shape_info = this->session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
            this->input_dims.push_back(input_shape_info.GetShape());
        }

        for (int i = 0; i < this->output_nums; i++) {
            // ���������
            Ort::AllocatedStringPtr output_name = this->session.GetOutputNameAllocated(i, this->allocator);
            this->output_node_names.push_back(output_name.get());
            this->output_node_names_ptr.push_back(move(output_name));

            // �����״
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

        for (int i = 0; i < this->output_nums; ++i) {
            cout << "output_dims: ";
            for (const auto j : this->output_dims[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
    }

    /**
     * ģ��Ԥ��
     */
    void warm_up() {
        // ��������
        cv::Size size = cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]);
        cv::Scalar color = cv::Scalar(0, 0, 0);
        cv::Mat input = cv::Mat(size, CV_8UC3, color);
        this->infer(input);
    }

    /**
     * ������ͼƬ
     * @param image ԭʼͼƬ
     * @return      ��׼���Ĳ����ŵ�ԭͼ����ͼ�͵÷�
     */
    Result infer(cv::Mat& image) {
        // 1.����ͼƬԭʼ�߿�
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.ͼƬԤ����
        cv::Mat resized_image;
        resized_image = pre_process(image, meta);
        // [H, W, C] -> [N, C, H, W]
        // ����ֻת��ά��,����Ԥ��������,python�汾�Ƿ�ʹ��openvinoͼƬԤ������Ҫ��һ��,C++ֻ���Լ���Ԥ������Ҫ��һ��
        // openvino���ʹ����һ���Ļ���Ҫ������������� u8 ת��Ϊ f32, Layout �� NHWC ��Ϊ NCHW  (38, 39��)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
                                               { this->meta.infer_size[1], this->meta.infer_size[0] },
                                               { 0, 0, 0 },
                                               false, false, CV_32F);

        // 3.��ͼ�񴴽�tensor
        // 3.1 �����ڴ�ռ�
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 3.2 ��������ֵ
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, resized_image.ptr<float>(), resized_image.total(), input_dims[0].data(), input_dims[0].size());
        // 3.3 ���� ֻ��������
        vector<Ort::Value> output_tensors;
        try {
            output_tensors = session.Run(this->runOptions, input_node_names.data(), &input_tensor, input_nums, output_node_names.data(), output_nums);
        }
        catch (Ort::Exception& e) {
            cout << e.what() << endl;
        }

        // 4.������ͼת��ΪMat
        auto* output0 = output_tensors[0].GetTensorMutableData<float>();
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
                                      CV_32FC1, output0);

        // 5.��Բ�ͬ���������ȡ�÷�
        cv::Mat pred_score;
        if (this->output_nums == 2) {
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, output_tensors[1].GetTensorMutableData<float>());  // {1}
        }
        else {
            double _, maxValue;    // ���ֵ����Сֵ
            cv::minMaxLoc(anomaly_map, &_, &maxValue);
            pred_score = cv::Mat(cv::Size(1, 1), CV_32FC1, maxValue);
        }
        cout << "pred_score: " << pred_score << endl;   // 4.0252275

        // 6.����:��׼��,���ŵ�ԭͼ
        vector<cv::Mat> result = post_process(anomaly_map, pred_score, meta);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 7.���ؽ��
        return Result{ anomaly_map, score };
    }
};