#include "cls_interface.h"
#include "tools.h"
#include <iostream>
#include <ctime>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

// ��batch ����
int main(int argc, char** argv)
{
	Facevisa_TensorRT_handle handle;
	Facevisa_Engine_Create(&handle, 0);
	// ͼƬ·��
	std::string root_dir = R"(D:\Workspace\task\001_tensorRT\classfication_interface\data\)";
	std::cout << root_dir << std::endl;

	std::vector<std::string> files_name;
	int files_number;
	read_files(root_dir + "*.bmp", files_name, &files_number);
	read_files(root_dir + "*.png", files_name, &files_number);
	read_files(root_dir + "*.jpg", files_name, &files_number);
	std::vector<int> compression_params;
	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
	compression_params.push_back(100);

	for (int index = 0; index < files_name.size(); index++)
	{
		std::cout << "[" << index << "/" << files_name.size() << "]" << " img name is: " << root_dir + files_name[index] << std::endl;
		//*it = "original_1106_1_D10502_55_72_20180621233457.jpg";
		std::string::size_type idx = files_name[index].rfind("/");
		std::string img_name = files_name[index].substr(idx + 1);
		cv::Mat img = cv::imread(root_dir + files_name[index]);

		if (0 == img.rows || 0 == img.cols || NULL == img.data)
			continue;

		// �������ۼ��
		Facevisa_TensorRT_result_s results;
		double start = clock();
		if (FACEVISA_OK != Facevisa_Engine_Inference(handle, img, &results)) {
			std::cout << "���ʧ�ܣ����� " << std::endl;
			continue;
		}
		double end = clock();

		std::cout << "��ߵȼ��� " << results.cls << std::endl;
		std::cout << "��ߵȼ��÷֣� " << results.score[results.cls] << std::endl;
		std::cout << "whloe time is: " << end - start << " ms!" << std::endl;
		std::cout << "---------------------" << std::endl;

		//cv::imshow("img_peel_dirty", img);
		//cv::waitKey(0);

	}
	Facevisa_Engine_Release(handle);
	return 0;
}


// ��batch ����  ע����Ҫ�� cls_interface.cpp �ļ��� kINPUT_BATCH_SIZE =2

//int main(int argc, char** argv)
//{
//	Facevisa_TensorRT_handle handle;
//	Facevisa_Engine_Create(&handle, 0);
//	// ͼƬ·��
//	std::string root_dir = R"(D:\Workspace\task\001_tensorRT\classfication_interface\data\)";
//	std::cout << root_dir << std::endl;
//
//	std::vector<std::string> files_name;
//	int files_number;
//	read_files(root_dir + "*.bmp", files_name, &files_number);
//	read_files(root_dir + "*.png", files_name, &files_number);
//	read_files(root_dir + "*.jpg", files_name, &files_number);
//	std::vector<int> compression_params;
//	compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
//	compression_params.push_back(100);
//
//	for (int index = 0; index < files_name.size(); index++)
//	{
//		std::cout << "[" << index << "/" << files_name.size() << "]" << " img name is: " << root_dir + files_name[index] << std::endl;
//		//*it = "original_1106_1_D10502_55_72_20180621233457.jpg";
//		std::string::size_type idx = files_name[index].rfind("/");
//		std::string img_name = files_name[index].substr(idx + 1);
//		cv::Mat img = cv::imread(root_dir + files_name[index]);
//
//		if (0 == img.rows || 0 == img.cols || NULL == img.data)
//			continue;
//
//		// �������ۼ��
//		std::vector<cv::Mat> imgs;
//		imgs.push_back(img);
//		imgs.push_back(img);
//		Facevisa_TensorRT_result_b results;
//		double start = clock();
//		if (FACEVISA_OK != Facevisa_Engine_Inference(handle, imgs, &results)) {
//			std::cout << "���ʧ�ܣ����� " << std::endl;
//			continue;
//		}
//		double end = clock();
//
//		for (int img_idx = 0; img_idx < 2; img_idx++) {
//			std::cout << "��ߵȼ��� " << results.cls[img_idx] << std::endl;
//			std::cout << "��ߵȼ��÷֣� " << results.score[img_idx][results.cls[img_idx]] << std::endl;
//			std::cout << "whloe time is: " << end - start << " ms!" << std::endl;
//			std::cout << "---------------------" << std::endl;
//		}
//		std::cout << "===========================" << std::endl;
//		//cv::imshow("img_peel_dirty", img);
//		//cv::waitKey(0);
//
//	}
//	Facevisa_Engine_Release(handle);
//	return 0;
//}



