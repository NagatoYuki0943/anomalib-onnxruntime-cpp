#include <string>
#include <numeric>
#include <vector>
#include "inference.hpp"

using namespace std;

/**
 * 单张图片推理
 * @param model_path    模型路径
 * @param meta_path     超参数路径
 * @param image_path    图片路径
 * @param save_dir      保存路径
 * @param device        cpu or cuda or tensorrt 推理
 * @param threads       ort 线程数, defaults to 0
 * @param gpu_mem_limit 显存限制, only for cuda or tensorrt device, defaults to 2 GB
 */
void single(const wchar_t* model_path, string& meta_path, string& image_path, string& save_dir,
    string& device, int threads = 0, int gpu_mem_limit = 2) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit);

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
    cout << "infer time: " << end - start << " ms" << endl;

    // 5.保存显示图片
    // 将mask转化为3通道,不然没法拼接图片
    cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
    saveScoreAndImages(result.score, images, image_path, save_dir);

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
 * @param threads       ort 线程数, defaults to 0
 * @param gpu_mem_limit 显存限制, only for cuda or tensorrt device, defaults to 2 GB
 */
void multi(const wchar_t* model_path, string& meta_path, string& image_dir, string& save_dir,
    string& device, int threads = 0, int gpu_mem_limit = 2) {
    // 1.创建推理器
    Inference inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit);

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
        cout << "infer time: " << end - start << " ms" << endl;
        times.push_back(end - start);

        // 6.保存图片
        // 将mask转化为3通道,不然没法拼接图片
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        saveScoreAndImages(result.score, images, image_path, save_dir);
    }

    // 6.统计数据
    double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate函数就是求vector和的函数；
    double avgValue = sumValue / times.size();                   // 求均值
    cout << "avg infer time: " << avgValue << " ms" << endl;
}


int main() {
    const wchar_t* model_path = L"D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/256/optimization/model.onnx";
    string param_path = "D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/256/optimization/meta_data.json";
    string image_path = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir = "D:/ai/code/abnormal/anomalib-onnxruntime-cpp/vs/result"; // 注意目录不会自动创建,要手动创建才会保存
    string device = "cuda";
    single(model_path, param_path, image_path, save_dir, device);
    // multi(model_path, param_path, image_dir, save_dir, device);
    return 0;
}
