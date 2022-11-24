#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <numeric>
#include <vector>
#include <Windows.h>
#include"utils.h"


using namespace std;


class Inference {
private:
    MetaData meta{};                                        // 超参数
    Ort::Env env{};
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
     */
    Inference(const wchar_t* model_path, string& meta_path, string& device) {
        // 1.读取meta
        this->meta = getJson(meta_path);
        // 2.创建模型
        this->session = get_onnx_model(model_path, device);
        // 3.获取模型的输入输出
        this->get_onnx_info();
        // 5.模型预热
        this->warm_up();
    }

    /**
     * get openvino model
     * @param model_path 模型路径
     * @param device     使用的设备
     */
    Ort::Session get_onnx_model(const wchar_t* model_path, string& device) {
        // 获取可用的provider
        auto availableProviders = Ort::GetAvailableProviders();
        for (const auto& provider : availableProviders) {
            cout << provider << " ";
        }
        cout << endl;
        // TensorrtExecutionProvider
        // CUDAExecutionProvider
        // CPUExecutionProvider

        Ort::SessionOptions sessionOptions;
        // 使用1个线程执行op,若想提升速度，增加线程数
        sessionOptions.SetIntraOpNumThreads(1);
        // ORT_ENABLE_ALL: 启用所有可能的优化
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (device == "cuda" || device == "tensorrt") {
            // https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            // https://onnxruntime.ai/docs/api/c/struct_ort_c_u_d_a_provider_options.html
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.arena_extend_strategy = 0;
            cuda_options.gpu_mem_limit = (size_t)2 * 1024 * 1024 * 1024; // 2GB
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
            cuda_options.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            if (device == "tensorrt") {
                // https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
                // https://onnxruntime.ai/docs/api/c/struct_ort_tensor_r_t_provider_options.html
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = 0;
                trt_options.trt_max_workspace_size = (size_t)2 * 1024 * 1024 * 1024; // 2GB
                trt_options.trt_fp16_enable = 0;
                sessionOptions.AppendExecutionProvider_TensorRT(trt_options);
            }
        }
        // create session
        Ort::Session temp_session(nullptr);
        try {
            temp_session = Ort::Session{ this->env, model_path, sessionOptions };
        }
        catch (Ort::Exception& e) {
            cout << e.what() << endl;
        }
        return temp_session;
    }

    void get_onnx_info() {
        // 1. 获得模型有多少个输入和输出，一般是指对应网络层的数目, 如果是多输出网络，就会是对应输出的数目
        this->input_nums = session.GetInputCount();
        this->output_nums = session.GetOutputCount();
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
            Ort::AllocatedStringPtr output_name = this->session.GetOutputNameAllocated(i, allocator);
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

        for (int i = 0; i < this->output_nums; ++i) {
            cout << "output_dims: ";
            for (const auto j : this->output_dims[i]) {
                cout << j << " ";
            }
            cout << endl;
        }
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
        this->meta.image_size[0] = image.size[0];
        this->meta.image_size[1] = image.size[1];

        // 2.图片预处理
        cv::Mat resized_image;
        resized_image = pre_process(image, meta);
        // [H, W, C] -> [N, C, H, W]
        // 这里只转换维度,其他预处理都做了,python版本是否使用openvino图片预处理都需要这一步,C++只是自己的预处理需要这一步
        // openvino如果使用这一步的话需要将输入的类型由 u8 转换为 f32, Layout 由 NHWC 改为 NCHW  (38, 39行)
        resized_image = cv::dnn::blobFromImage(resized_image, 1.0,
            { this->meta.infer_size[1], this->meta.infer_size[0] },
            { 0, 0, 0 },
            false, false, CV_32F);

        // 3.从图像创建tensor
        // 3.1 申请内存空间
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 3.2 创建输入值
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, resized_image.ptr<float>(), resized_image.total(), input_dims[0].data(), input_dims[0].size());
        // 3.3 推理 只传递输入
        vector<Ort::Value> output_tensors;
        try {
            output_tensors = session.Run(this->runOptions, input_node_names.data(), &input_tensor, input_nums, output_node_names.data(), output_nums);
        }
        catch (Ort::Exception& e) {
            cout << e.what() << endl;
        }

        // 4.将热力图转换为Mat
        // result1.data<float>() 返回指针 放入Mat中不能解引用
        cv::Mat anomaly_map = cv::Mat(cv::Size(this->meta.infer_size[1], this->meta.infer_size[0]),
            CV_32FC1, output_tensors[0].GetTensorMutableData<float>());

        // 5.针对不同输出数量获取得分
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
        vector<cv::Mat> result = post_process(anomaly_map, pred_score, meta);
        anomaly_map = result[0];
        float score = result[1].at<float>(0, 0);

        // 7.返回结果
        return Result{ anomaly_map, score };
    }
};


/**
 * 单张图片推理
 * @param model_path    模型路径
 * @param meta_path     超参数路径
 * @param image_path    图片路径
 * @param save_dir      保存路径
 * @param device        cpu or cuda or tensorrt 推理
 */
void single(const wchar_t* model_path, string& meta_path, string& image_path, string& save_dir, string& device) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device);

    // 2.读取图片
    cv::Mat image = readImage(image_path);

    // time
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // 3.推理单张图片
    Result result = inference.infer(image);
    cout << "score: " << result.score << endl;

    // 4.生成其他图片(mask,mask边缘,热力图和原图的叠加)
    vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
    // time
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cout << "infer time:" << end - start << "ms" << endl;

    // 5.保存显示图片
    // 将mask转化为3通道,不然没法拼接图片
    cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
    saveScoreAndImage(result.score, images, image_path, save_dir);

    cv::imshow("result", images[2]);
    cv::waitKey(0);
}


/**
 * 多张图片推理
 * @param model_path    模型路径
 * @param meta_path     超参数路径
 * @param image_dir     图片文件夹路径
 * @param save_dir      保存路径
 * @param device        cpu or cuda or tensorrt 推理
 */
void multi(const wchar_t* model_path, string& meta_path, string& image_dir, string& save_dir, string& device) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device);

    // 2.读取全部图片路径
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 3.读取单张图片
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 4.推理单张图片
        Result result = inference.infer(image);
        cout << "score: " << result.score << endl;

        // 5.图片生成其他图片(mask,mask边缘,热力图和原图的叠加)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time:" << end - start << "ms" << endl;
        times.push_back(end - start);

        // 6.保存图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        saveScoreAndImage(result.score, images, image_path, save_dir);
    }

    // 6.统计数据
    double sumValue = accumulate(begin(times), end(times), 0.0);  // accumulate函数就是求vector和的函数；
    double meanValue = sumValue / times.size();                   // 求均值
    cout << "mean infer time: " << meanValue << endl;
}


int main() {
    const wchar_t* model_path = L"D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/run/optimization/model.onnx";
    string param_path = "D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/run/optimization/meta_data.json";
    string image_path = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir = "D:/ai/code/abnormal/anomalib-onnxruntime-cpp/vs/result"; // 注意目录不会自动创建,要手动创建才会保存
    string device = "cuda";
    // single(model_path, param_path, image_path, save_dir, device);
    multi(model_path, param_path, image_dir, save_dir, device);
    return 0;
}
