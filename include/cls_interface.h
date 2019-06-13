#ifndef _FACEVISA_CLASSFICATION_INTERFACE_H_
#define _FACEVISA_CLASSFICATION_INTERFACE_H_

#include <stdio.h>
#include <opencv2/core/core.hpp>

#ifdef FACEVISA_CLASSFICATION_INTERFACE_H
#define FACEVISA_CLASSFICATION_API __declspec(dllexport)
#else
#define FACEVISA_CLASSFICATION_API __declspec(dllimport)
#endif

#define       FACEVISA_OK                     0x11120000
#define       FACEVISA_ALLOC_MEMORY_ERROR     0x11120001
#define       FACEVISA_PARAMETER_ERROR        0x11120002

typedef void * Facevisa_TensorRT_handle;

//��batch ���
typedef struct _Facevisa_TensorRT_result_batch_
{
	std::vector<int> cls;					// ȱ�������ĵȼ�, 0Ϊ������1Ϊdirty��
	std::vector<std::vector<float>> score;	// ȱ�ݵȼ�����Ӧ�ĵ÷�
}Facevisa_TensorRT_result_b;

//��batch ���
typedef struct _Facevisa_TensorRT_result_single_
{
	int cls;					// ȱ�������ĵȼ�, 0Ϊ������1Ϊdirty��
	std::vector<float> score;	// ȱ�ݵȼ�����Ӧ�ĵ÷�
}Facevisa_TensorRT_result_s;

int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id);

int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, const std::vector<cv::Mat> &images, Facevisa_TensorRT_result_b *results);
int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, const cv::Mat &image, Facevisa_TensorRT_result_s *results);

int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle);

#endif !_FACEVISA_CLASSFICATION_INTERFACE_H_