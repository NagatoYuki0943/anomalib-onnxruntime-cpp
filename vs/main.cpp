#include <string>
#include <numeric>
#include <vector>
#include "inference.hpp"

using namespace std;

/**
 * ����ͼƬ����
 * @param model_path    ģ��·��
 * @param meta_path     ������·��
 * @param image_path    ͼƬ·��
 * @param save_dir      ����·��
 * @param device        cpu or cuda or tensorrt ����
 * @param threads       ort �߳���, defaults to 0
 * @param gpu_mem_limit �Դ�����, only for cuda or tensorrt device, defaults to 2 GB
 */
void single(const wchar_t* model_path, string& meta_path, string& image_path, string& save_dir,
    string& device, int threads = 0, int gpu_mem_limit = 2) {
    // 1.����������
    Inference inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit);

    // 2.��ȡͼƬ
    cv::Mat image = readImage(image_path);

    // time
    auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // 3.������ͼƬ
    Result result = inference.infer(image);
    cout << "score: " << result.score << endl;

    // 4.��������ͼƬ(mask,mask��Ե,����ͼ��ԭͼ�ĵ���)
    vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
    // time
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    cout << "infer time: " << end - start << " ms" << endl;

    // 5.������ʾͼƬ
    // ��maskת��Ϊ3ͨ��,��Ȼû��ƴ��ͼƬ
    cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
    saveScoreAndImages(result.score, images, image_path, save_dir);

    cv::imshow("result", images[2]);
    cv::waitKey(0);
}


/**
 * ����ͼƬ����
 * @param model_path    ģ��·��
 * @param meta_path     ������·��
 * @param image_dir     ͼƬ�ļ���·��
 * @param save_dir      ����·��
 * @param device        cpu or cuda or tensorrt ����
 * @param threads       ort �߳���, defaults to 0
 * @param gpu_mem_limit �Դ�����, only for cuda or tensorrt device, defaults to 2 GB
 */
void multi(const wchar_t* model_path, string& meta_path, string& image_dir, string& save_dir,
    string& device, int threads = 0, int gpu_mem_limit = 2) {
    // 1.����������
    Inference inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit);

    // 2.��ȡȫ��ͼƬ·��
    vector<cv::String> paths = getImagePaths(image_dir);

    vector<float> times;
    for (auto& image_path : paths) {
        // 3.��ȡ����ͼƬ
        cv::Mat image = readImage(image_path);

        // time
        auto start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // 4.������ͼƬ
        Result result = inference.infer(image);
        cout << "score: " << result.score << endl;

        // 5.ͼƬ��������ͼƬ(mask,mask��Ե,����ͼ��ԭͼ�ĵ���)
        vector<cv::Mat> images = gen_images(image, result.anomaly_map, result.score);
        // time
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        cout << "infer time: " << end - start << " ms" << endl;
        times.push_back(end - start);

        // 6.����ͼƬ
        // ��maskת��Ϊ3ͨ��,��Ȼû��ƴ��ͼƬ
        cv::applyColorMap(images[0], images[0], cv::ColormapTypes::COLORMAP_JET);
        saveScoreAndImages(result.score, images, image_path, save_dir);
    }

    // 6.ͳ������
    double sumValue = accumulate(begin(times), end(times), 0.0); // accumulate����������vector�͵ĺ�����
    double avgValue = sumValue / times.size();                   // ���ֵ
    cout << "avg infer time: " << avgValue << " ms" << endl;
}


int main() {
    const wchar_t* model_path = L"D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/256/optimization/model.onnx";
    string param_path = "D:/ai/code/abnormal/anomalib/results/fastflow/mvtec/bottle/256/optimization/meta_data.json";
    string image_path = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir = "D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir = "D:/ai/code/abnormal/anomalib-onnxruntime-cpp/vs/result"; // ע��Ŀ¼�����Զ�����,Ҫ�ֶ������Żᱣ��
    string device = "cuda";
    single(model_path, param_path, image_path, save_dir, device);
    // multi(model_path, param_path, image_dir, save_dir, device);
    return 0;
}
