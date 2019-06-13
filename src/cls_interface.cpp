#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "common.h"
#include "cls_interface.h"
#include <iostream>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <windows.h>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
// Network details

static const int kINPUT_BATCH_SIZE = 1;			// Input image batch size
static const char* kINPUT_BLOB_NAME = "data";   // Input blob name
static const char* kOUTPUT_BLOB_NAME0 = "prob"; // Output blob name

static Logger gLogger;

// 工程相关变量容器
typedef struct TensorRTCaffeCantainer_ {
	DimsNCHW input_dims = DimsNCHW(1, 1, 1, 1);		// 输入的dim，初始化时根据prototxt自行读取，batch size大小为 kINPUT_BATCH_SIZE
	DimsNCHW output_dims = DimsNCHW(1, 1, 1, 1);	// 输出的dim，初始化时根据prototxt自行读取，batch size大小为 kINPUT_BATCH_SIZE
	int gpu_id;										// 配置 GPU 的 index	
	int buffers_size;								// 输入输出数据维度
	void** buffers;									// 输入输出数据，TensorRT 中前向函数需要
	nvinfer1::ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;
}TensorRTCaffeCantainer;

// 多batch 图像数据导入
// 注意：这里用的均值为（0，0，0），无scale
static int processImg(std::vector<cv::Mat> &imgs, int inputchannels, float *imgData) {
	int shift_data = 0;
	for (size_t index = 0; index < imgs.size(); index++) {
		cv::Mat float_img;
		cv::Mat img = imgs[index];
		std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
		if (3 == inputchannels) {
			if (1 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
			img.convertTo(float_img, CV_32F);
			//cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 117.0f, 123.0f);
			cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
			cv::Mat mean_(img.size(), CV_32FC3, meanValue);
			cv::subtract(float_img, mean_, float_img);
			cv::split(float_img, splitchannles);
		}
		else if (1 == inputchannels) {
			if (3 == img.channels())
				cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
			img.convertTo(float_img, CV_32F);
			//cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 0.0f, 0.0f);
			cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
			cv::Mat mean_(img.size(), CV_32FC1, meanValue);
			cv::subtract(float_img, mean_, float_img);
			splitchannles.emplace_back(float_img);
		}
		else {
			return FACEVISA_PARAMETER_ERROR;
		}
		shift_data = sizeof(float) * img.rows * img.cols;
		for (size_t i = 0; i < inputchannels; i++) {
			memcpy(imgData, splitchannles[i].data, shift_data);
			imgData += img.rows * img.cols;
		}
	}
	return FACEVISA_OK;
}

// 单batch 图像数据导入
// 注意：这里用的均值为（0，0，0），无scale
static int processImg(cv::Mat &img, int inputchannels, float *imgData) {
	int shift_data = 0;
	cv::Mat float_img;
	std::vector<cv::Mat> splitchannles = std::vector<cv::Mat>(inputchannels);
	if (3 == inputchannels) {
		if (1 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
		img.convertTo(float_img, CV_32F);
		//cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 117.0f, 123.0f);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
		cv::Mat mean_(img.size(), CV_32FC3, meanValue);
		cv::subtract(float_img, mean_, float_img);
		cv::split(float_img, splitchannles);
	}
	else if (1 == inputchannels) {
		if (3 == img.channels())
			cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		img.convertTo(float_img, CV_32F);
		//cv::Scalar_<float> meanValue = cv::Scalar_<float>(104.0f, 0.0f, 0.0f);
		cv::Scalar_<float> meanValue = cv::Scalar_<float>(0.0f, 0.0f, 0.0f);
		cv::Mat mean_(img.size(), CV_32FC1, meanValue);
		cv::subtract(float_img, mean_, float_img);
		splitchannles.emplace_back(float_img);
	}
	else {
		return FACEVISA_PARAMETER_ERROR;
	}
	shift_data = sizeof(float) * img.rows * img.cols;
	for (size_t i = 0; i < inputchannels; i++) {
		memcpy(imgData, splitchannles[i].data, shift_data);
		imgData += img.rows * img.cols;
	}
	
	return FACEVISA_OK;
}

// 模型初始化
int Facevisa_Engine_Create(Facevisa_TensorRT_handle *handle, int device_id) {
	if (NULL == *handle || NULL == handle)
	{
		return FACEVISA_PARAMETER_ERROR;
	}

	char moduleFileName[MAX_PATH];
	GetModuleFileNameA(0, moduleFileName, MAX_PATH);
	char * ptr = strrchr(moduleFileName, '\\');
	ptr++;
	strcpy(ptr, "templates\\");
	std::string root_dir = std::string(moduleFileName);
	std::string protostr = root_dir + "dirty.prototxt";
	std::string modelstr = root_dir + "dirty.caffemodel";

	// 初始化变量容器
	TensorRTCaffeCantainer *param = new (std::nothrow) TensorRTCaffeCantainer();
	if (NULL == param)
	{
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// 设置指定GPU, 备注未找到TensorRT的封装写法，先用cuda自己的调用方法
	if ((cudaSuccess == cudaSetDevice(device_id)) && (cudaSuccess == cudaFree(0)))
	{
		param->gpu_id = device_id;
	}
	else
	{
		param->gpu_id = -1;
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}

	// Create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	assert(builder != nullptr);

	// Parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	DataType dataType = DataType::kFLOAT;
	// 读取caffe model
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(protostr.c_str(), modelstr.c_str(), *network, dataType);

	// Specify which tensors are outputs
	network->markOutput(*blobNameToTensor->find(kOUTPUT_BLOB_NAME0));

	// Build the engine
	builder->setMaxBatchSize(kINPUT_BATCH_SIZE);
	builder->setMaxWorkspaceSize(36 << 20);
	builder->allowGPUFallback(true);

	ICudaEngine* engine;
	engine = builder->buildCudaEngine(*network);
	// 设置输入输出的buffers， 
	param->buffers_size = engine->getNbBindings();
	param->buffers = (void **)malloc(sizeof(float) * engine->getNbBindings());

	param->engine = engine;
	//assert(param->engine != nullptr);
	param->context = engine->createExecutionContext();
	// Create GPU buffers and a stream
	// input
	int index_in = engine->getBindingIndex(kINPUT_BLOB_NAME);
	Dims input_dim = engine->getBindingDimensions(index_in);
	param->input_dims.d[0] = engine->getMaxBatchSize();
	for (int dim_i = 0; dim_i < input_dim.nbDims; dim_i++) {
		param->input_dims.d[dim_i + 1] = input_dim.d[dim_i];
	}
	if (0 != cudaMalloc(&(param->buffers[index_in]), param->input_dims.n() * param->input_dims.c() * param->input_dims.h() * param->input_dims.w() * sizeof(float))) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	
	// output
	int index_out = engine->getBindingIndex(kOUTPUT_BLOB_NAME0);
	Dims output_dim = engine->getBindingDimensions(index_out);
	param->output_dims.d[0] = engine->getMaxBatchSize();
	for (int dim_i = 0; dim_i < input_dim.nbDims; dim_i++) {
		param->output_dims.d[dim_i + 1] = output_dim.d[dim_i];
	}
	if (0 != cudaMalloc(&(param->buffers[index_out]), param->output_dims.n() * param->output_dims.c() * param->output_dims.h() * param->output_dims.w() * sizeof(float))) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	if (0 != cudaStreamCreate(&param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	*handle = param;

	// release
	network->destroy();
	parser->destroy();
	builder->destroy();
	return FACEVISA_OK;
}

// 网络前向
static int Facevisa_Engine_Forward(Facevisa_TensorRT_handle handle, float* inputData, float* detectionOut) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	if (-1 == param->gpu_id) {
		return FACEVISA_PARAMETER_ERROR;
	}
	IExecutionContext& context = *(param->context);
	const ICudaEngine& engine = *(param->engine);
	if (engine.getNbBindings() != 2) {
		return FACEVISA_PARAMETER_ERROR;
	}

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	int index_in = engine.getBindingIndex(kINPUT_BLOB_NAME);
	if (0 != cudaMemcpyAsync(param->buffers[index_in], inputData, param->input_dims.n() * param->input_dims.c() * param->input_dims.h() * param->input_dims.w() * sizeof(float), cudaMemcpyHostToDevice, param->stream)) {
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	// 异步调用前向，同步使用execute
	context.enqueue(param->input_dims.n(), param->buffers, param->stream, nullptr);
	int index_out = engine.getBindingIndex(kOUTPUT_BLOB_NAME0);
	if (0 != cudaMemcpyAsync(detectionOut, param->buffers[index_out], param->output_dims.n() * param->output_dims.c() * param->output_dims.h() * param->output_dims.w() * sizeof(float), cudaMemcpyDeviceToHost, param->stream)){
		return FACEVISA_ALLOC_MEMORY_ERROR;
	}
	// 同步
	cudaStreamSynchronize(param->stream);
	return FACEVISA_OK;
}

// 多batch检测主接口
// 注意： 输入图像的images.size() 要与 batch size 一致
int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, const std::vector<cv::Mat> &images, Facevisa_TensorRT_result_b *results)
{
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	int width = param->input_dims.w();
	int height = param->input_dims.h();
	int channels = param->input_dims.c();
	int batch_size = param->input_dims.n();
	int output_size = param->output_dims.c() * param->output_dims.h() * param->output_dims.h();

	if (NULL == handle || images.size() != batch_size) {
		return FACEVISA_PARAMETER_ERROR;
	}

	std::vector<cv::Mat> img_resize(images.size());
	for (int img_idx = 0; img_idx < images.size(); img_idx++) {
		cv::resize(images[img_idx], img_resize[img_idx], cv::Size(width, height));
	}

	int shift_data = batch_size * channels * height * width * sizeof(float);
	float *input_data = (float *)malloc(shift_data);
	processImg(img_resize, channels, input_data);


	
	float* detectionOut = new float[batch_size * output_size];

	// forward
	double start = clock();
	int status = Facevisa_Engine_Forward(handle, input_data, detectionOut);
	if (FACEVISA_OK != status) {
		return status;
	}
	double end = clock();
	//std::cout << " Forward time is: " << end - start << " ms!" << std::endl;

	for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
		int max_ind = 0;
		float max_score = 0;
		std::vector<float> prob;
		for (int single_idx = 0; single_idx < output_size; single_idx++) {
			float single_score = detectionOut[single_idx*output_size + single_idx];
			prob.push_back(single_score);
			if (single_score > max_score) {
				max_ind = single_idx;
				max_score = single_score;
			}
		}
		results->cls.push_back(max_ind);
		results->score.push_back(prob);
	}

	return FACEVISA_OK;
}

// 单batch检测主接口
int Facevisa_Engine_Inference(Facevisa_TensorRT_handle handle, const cv::Mat &image, Facevisa_TensorRT_result_s *results)
{
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	int width = param->input_dims.w();
	int height = param->input_dims.h();
	int channels = param->input_dims.c();
	int batch_size = param->input_dims.n();
	int output_size = param->output_dims.c() * param->output_dims.h() * param->output_dims.h();

	if (NULL == handle) {
		return FACEVISA_PARAMETER_ERROR;
	}

	cv::Mat img_resize;
	cv::resize(image, img_resize, cv::Size(width, height));

	int shift_data = batch_size * channels * height * width * sizeof(float);
	float *input_data = (float *)malloc(shift_data);
	processImg(img_resize, channels, input_data);

	float* detectionOut = new float[batch_size * output_size];

	// forward
	double start = clock();
	int status = Facevisa_Engine_Forward(handle, input_data, detectionOut);
	if (FACEVISA_OK != status) {
		return status;
	}
	double end = clock();
	//std::cout << " Forward time is: " << end - start << " ms!" << std::endl;

	std::vector<float> prob;
	int max_ind = 0;
	float max_score = 0;
	for (int single_idx = 0; single_idx < output_size; single_idx++) {
		float single_score = detectionOut[single_idx];
		prob.push_back(single_score);
		if (single_score > max_score) {
			max_ind = single_idx;
			max_score = single_score;
		}
	}
	results->cls = max_ind;
	results->score = prob;

	return FACEVISA_OK;
}

// 内存释放
int Facevisa_Engine_Release(Facevisa_TensorRT_handle handle) {
	TensorRTCaffeCantainer *param = (TensorRTCaffeCantainer *)handle;
	nvinfer1::ICudaEngine *engine = param->engine;
	IExecutionContext *context = param->context;
	context->destroy();
	engine->destroy();
	cudaStreamDestroy(param->stream);
	for (size_t i = 0; i < param->buffers_size; i++) {
		cudaFree(param->buffers[i]);
	}
	free(param);
	handle = NULL;
	return FACEVISA_OK;
}

