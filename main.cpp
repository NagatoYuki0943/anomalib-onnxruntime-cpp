#include <opencv2/opencv.hpp>
#include "inference.hpp"


int main() {
    // patchcoreģ��ѵ�������ļ�ɾ����center_crop
    string model_path = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/model.onnx";
    string meta_path  = "D:/ml/code/anomalib/results/patchcore/mvtec/bottle/run/weights/openvino/metadata.json";
    string image_path = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
    string image_dir  = "D:/ml/code/anomalib/datasets/MVTec/bottle/test/broken_large";
    string save_dir   = "D:/ml/code/anomalib-onnxruntime-cpp/result"; // ע��Ŀ¼�����Զ�����,Ҫ�ֶ������Żᱣ��
    string device     = "cuda";
    int threads       = 0;
    int gpu_mem_limit = 4;

    // ����������
    auto inference = Inference(model_path, meta_path, device, threads, gpu_mem_limit);

    // ����ͼƬ����
    cv::Mat result = inference.single(image_path, save_dir);
    cv::imshow("result", result);
    cv::waitKey(0);

    // ����ͼƬ����
    inference.multi(image_dir, save_dir);
    return 0;
}
