#pragma once
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>
#include "rapidjson/document.h"         //https://github.com/Tencent/rapidjson
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filereadstream.h"
#include "opencv_utils.h"


using namespace std;

struct MetaData {
public:
    float image_threshold;
    float pixel_threshold;
    float min;
    float max;
    int infer_size[2];  // h w
    int image_size[2];  // h w
};


/**
 * ��ȡjson�����ļ�
 * @param json_path �����ļ�·��
 * @return
 */
MetaData getJson(const string& json_path);


/**
 * ��ȡ�ļ�����ȫ��ͼƬ�ľ���·��
 *
 * @param path          ͼƬ�ļ���·��
 * @return result       ȫ��ͼƬ����·���б�
 */
vector<cv::String> getImagePaths(string& path);



/**
 * ��ȡͼ�� BGR2RGB
 *
 * @param path  ͼƬ·��
 * @return      ͼƬ
 */
cv::Mat readImage(string& path);


/**
 * ����ͼƬ�ͷ���
 *
 * @param score      �÷�
 * @param mixed_image_with_label ��Ϻ��ͼƬ
 * @param image_path ����ͼƬ��·��
 * @param save_dir   �����·��
 */
void saveScoreAndImage(float score, vector<cv::Mat>& images, cv::String& image_path, string& save_dir);


/**
 * ͼƬԤ����
 * @param image Ԥ����ͼƬ
 * @return      ����Ԥ�����ͼƬ
 */
cv::Mat pre_process(cv::Mat& image, MetaData& meta);


/**
 * opencv��׼������ͼ
 *
 * @param targets       ����ͼ
 * @param threshold     ��ֵ,meta�еĲ���
 * @param min_val       ��Сֵ,meta�еĲ���
 * @param max_val       ���ֵ,meta�еĲ���
 * @return normalized   ������׼����Ľ��
 */
cv::Mat cvNormalizeMinMax(cv::Mat& targets, float threshold, float min_val, float max_val);


/**
 * ������,��׼������ͼ�͵÷�,��ԭ����ͼ��ԭͼ�ߴ�
 *
 * @param anomaly_map   δ������׼��������ͼ
 * @param pred_score    δ������׼���ĵ÷�
 * @param meta          meta������
 * @return result		����ͼ�͵÷�vector
 */
vector<cv::Mat> post_process(cv::Mat& anomaly_map, cv::Mat& pred_score, MetaData& meta);

/**
 * ����ͼƬ
 *
 * @param anomaly_map   ��Ϻ��ͼƬ
 * @param origin_image  ԭʼͼƬ
 * @return result       ���Ӻ��ͼ��
 */
cv::Mat superimposeAnomalyMap(cv::Mat& anomaly_map, cv::Mat& origin_image);


/**
 * ��ͼƬ��ӱ�ǩ
 *
 * @param mixed_image   ��Ϻ��ͼƬ
 * @param score         �÷�
 * @param font          ����
 * @return mixed_image  ��ӱ�ǩ��ͼ��
 */
cv::Mat addLabel(cv::Mat& mixed_image, float score, int font = cv::FONT_HERSHEY_PLAIN);


/**
 * ������������ͼ + �÷�
 */
struct Result{
public:
    cv::Mat anomaly_map;
    float score;
};


/**
 * ����mask
 *
 * @param anomaly_map ����ͼ
 * @param threshold   ��ֵ����ֵ
 * @param kernel_size ������kernel_size
 * @return mask
 */
cv::Mat compute_mask(cv::Mat& anomaly_map, float threshold=0.5, int kernel_size=1);


/**
 * ����mask�߽粢��ϵ�ԭͼ
 *
 * @param mask  mask
 * @param image ԭͼ
 * @return      ���mask�߽��ԭͼ
 */
cv::Mat gen_mask_border(cv::Mat& mask, cv::Mat& image);


/**
 * ����mask,mask��Ե,����ͼ��ԭͼ�ĵ���
 *
 * @param image        ԭͼ
 * @param anomaly_map  ����ͼ
 * @param score        �÷�
 * @param threshold    ����ͼ��ֵ����ֵ
 * @return
 */
vector<cv::Mat> gen_images(cv::Mat& image, cv::Mat& anomaly_map, float score, float threshold=0.5);