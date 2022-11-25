#include "utils.h"


MetaData getJson(const string& json_path) {
    FILE* fp;
    fopen_s(&fp, json_path.c_str(), "r");

    char readBuffer[1000];
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    rapidjson::Document doc;
    doc.ParseStream(is);
    fclose(fp);

    float image_threshold = doc["image_threshold"].GetFloat();
    float pixel_threshold = doc["pixel_threshold"].GetFloat();
    float min = doc["min"].GetFloat();
    float max = doc["max"].GetFloat();
    // �б�ֱ�ȡ��
    auto infer_size = doc["infer_size"].GetArray();
    int infer_height = infer_size[0].GetInt();
    int infer_width = infer_size[1].GetInt();

    // cout << image_threshold << endl;
    // cout << pixel_threshold << endl;
    // cout << min << endl;
    // cout << max << endl;
    // cout << infer_height << endl;
    // cout << infer_width << endl;

    return MetaData{ image_threshold, pixel_threshold, min, max, {infer_height, infer_width} };
}


vector<cv::String> getImagePaths(string& path) {
    vector<cv::String> paths;
    // for (auto& path : paths) {
    //     //cout << path << endl;
    //     // D:/ai/code/abnormal/anomalib/datasets/MVTec/bottle/test/broken_large\000.png
    // }
    cv::glob(path, paths, false);
    return paths;
}


cv::Mat readImage(string& path) {
    auto image = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_BGR2RGB);    // BGR2RGB
    return image;
}


void saveScoreAndImages(float score, vector<cv::Mat>& images, cv::String& image_path, string& save_dir) {
    // ��ȡͼƬ�ļ���
    // ��������ȷ������ʹ�� \ / ��Ϊ�ָ��������ҵ��ļ�����
    auto start = image_path.rfind('\\');
    if (start < 0 || start > image_path.length()) {
        start = image_path.rfind('/');
    }
    auto end = image_path.substr(start + 1).rfind('.');
    auto image_name = image_path.substr(start + 1).substr(0, end);  // 000

    // д��÷�
    ofstream ofs;
    ofs.open(save_dir + "/" + image_name + ".txt", ios::out);
    ofs << score;
    ofs.close();

    cv::Mat res;
    cv::hconcat(images, res);

    // д��ͼƬ
    cv::imwrite(save_dir + "/" + image_name + ".jpg", res);
}


cv::Mat pre_process(cv::Mat& image, MetaData& meta) {
    vector<float> mean = { 0.485, 0.456, 0.406 };
    vector<float> std = { 0.229, 0.224, 0.225 };

    // ���� w h
    cv::Mat resized_image = Resize(image, meta.infer_size[0], meta.infer_size[1], "bilinear");

    // ��һ��
    // convertToֱ�ӽ�����ֵ����255,normalize��NORM_MINMAX�ǽ�ԭʼ���ݷ�Χ�任��0~1֮��,convertTo���������ѧϰ������
    resized_image.convertTo(resized_image, CV_32FC3, 1.0 / 255, 0);
    //cv::normalize(resized_image, resized_image, 0, 1, cv::NormTypes::NORM_MINMAX, CV_32FC3);

    // ��׼��
    resized_image = Normalize(resized_image, mean, std);
    return resized_image;
}


cv::Mat cvNormalizeMinMax(cv::Mat& targets, float threshold, float min_val, float max_val) {
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    cv::Mat normalized1;
    // normalized = np.clip(normalized, 0, 1) ȥ��С��0�ʹ���1��
    // ����������: https://blog.csdn.net/simonyucsdy/article/details/106525717
    // ��������Ϊ1
    cv::threshold(normalized, normalized1, 1, 1, cv::ThresholdTypes::THRESH_TRUNC);
    // ��������Ϊ0
    cv::threshold(normalized1, normalized1, 0, 0, cv::ThresholdTypes::THRESH_TOZERO);
    return normalized1;
}


vector<cv::Mat> post_process(cv::Mat& anomaly_map, cv::Mat& pred_score, MetaData& meta) {
    // ��׼������ͼ�͵÷�
    anomaly_map = cvNormalizeMinMax(anomaly_map, meta.pixel_threshold, meta.min, meta.max);
    pred_score = cvNormalizeMinMax(pred_score, meta.image_threshold, meta.min, meta.max);

    // ��ԭ��ԭͼ�ߴ�
    anomaly_map = Resize(anomaly_map, meta.image_size[0], meta.image_size[1], "bilinear");

    // ��������ͼ�͵÷�
    return vector<cv::Mat>{anomaly_map, pred_score};
}


cv::Mat superimposeAnomalyMap(cv::Mat& anomaly_map, cv::Mat& origin_image) {
    auto anomaly = anomaly_map.clone();
    // ��һ����ͼƬЧ��������
    //python���룺 anomaly_map = (anomaly - anomaly.min()) / np.ptp(anomaly) np.ptp()����ʵ�ֵĹ��ܵ�ͬ��np.max(array) - np.min(array)
    double minValue, maxValue;    // ���ֵ����Сֵ
    cv::minMaxLoc(anomaly, &minValue, &maxValue);
    anomaly = (anomaly - minValue) / (maxValue - minValue);

    //ת��Ϊ����
    anomaly.convertTo(anomaly, CV_8UC1, 255, 0);
    //��ͨ��ת��Ϊ3ͨ��
    cv::applyColorMap(anomaly, anomaly, cv::ColormapTypes::COLORMAP_JET);
    //�ϲ�ԭͼ������ͼ
    cv::Mat combine;
    cv::addWeighted(anomaly, 0.4, origin_image, 0.6, 0, combine);

    return combine;
}


cv::Mat addLabel(cv::Mat& mixed_image, float score, int font) {
    string text = "Confidence Score " + to_string(score);
    int font_size = mixed_image.cols / 1024 + 1;
    int baseline = 0;
    int thickness = font_size / 2;
    cv::Size textsize = cv::getTextSize(text, font, font_size, thickness, &baseline);
    //cout << textsize << endl; //[1627 x 65]

    //����
    cv::rectangle(mixed_image, cv::Point(0, 0), cv::Point(textsize.width + 10, textsize.height + 10),
        cv::Scalar(225, 252, 134), cv::FILLED);

    //�������
    cv::putText(mixed_image, text, cv::Point(0, textsize.height + 10), font, font_size,
        cv::Scalar(0, 0, 0), thickness);

    return mixed_image;
}


cv::Mat compute_mask(cv::Mat& anomaly_map, float threshold, int kernel_size) {
    cv::Mat mask = anomaly_map.clone();
    // ��ֵ�� https://blog.csdn.net/weixin_42296411/article/details/80901080
    cv::threshold(mask, mask, threshold, 1, cv::ThresholdTypes::THRESH_BINARY);

    // ����������С��
    auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, { kernel_size, kernel_size }, { -1, -1 });
    cv::morphologyEx(mask, mask, cv::MorphTypes::MORPH_OPEN, kernel, { -1, -1 }, 1);

    // ���ŵ�255,ת��Ϊuint
    mask.convertTo(mask, CV_8UC1, 255, 0);

    return mask;
}


cv::Mat gen_mask_border(cv::Mat& mask, cv::Mat& image) {
    cv::Mat b = cv::Mat::zeros(mask.size[0], mask.size[1], CV_8UC1);
    cv::Mat g = b.clone();
    cv::Mat r = b.clone();

    // auto kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, {3, 3}, {-1, -1});
    // cv::morphologyEx(mask, mask_dilation, cv::MorphTypes::MORPH_CLOSE, kernel, {-1, -1}, 1);
    cv::Canny(mask, r, 128, 255, 3, false);

    // ����Ϊ3ͨ��ͼƬ
    vector<cv::Mat> rgb{ b, g, r };
    cv::Mat border;
    cv::merge(rgb, border);

    // ��Ե��ԭͼ���
    // border = image + border;
    cv::addWeighted(border, 0.4, image, 0.6, 0, border);
    return border;
}


vector<cv::Mat> gen_images(cv::Mat& image, cv::Mat& anomaly_map, float score, float threshold) {
    // 0.rgb2bgr
    cv::cvtColor(image, image, cv::ColorConversionCodes::COLOR_RGB2BGR);    // RGB2BGR

    // 1.����mask
    cv::Mat mask = compute_mask(anomaly_map, threshold);

    // 2.����mask��߽�
    cv::Mat border = gen_mask_border(mask, image);

    // 3.����ԭͼ������ͼ
    cv::Mat superimposed_map = superimposeAnomalyMap(anomaly_map, image);

    // 4.��ͼƬ��ӷ���
    superimposed_map = addLabel(superimposed_map, score);
    return vector<cv::Mat>{mask, border, superimposed_map};
}