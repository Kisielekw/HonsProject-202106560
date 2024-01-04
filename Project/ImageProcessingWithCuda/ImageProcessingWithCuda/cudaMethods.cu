
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageProcessing.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <iostream>

__device__ int Convolution(const unsigned char* imageIn,const int x, const int y, const float* kernal, const int width, const int height, unsigned int kernalSize)
{
	int sum = 0;

	int halfKernalSize = (kernalSize - 1) / 2;

	for (int ky = -halfKernalSize; ky <= halfKernalSize; ky++)
	{
		int curentY = ky + y;
		if (curentY >= height || curentY < 0)
			continue;
		for (int kx = -halfKernalSize; kx <= halfKernalSize; kx++)
		{
			int curentX = kx + x;
			if (curentX >= width || curentX < 0)
				continue;
			sum += imageIn[curentX + (curentY * width)] * kernal[(kx + halfKernalSize) + ((ky + halfKernalSize) * kernalSize)];
		}
	}

	return sum;
}

__device__ float GaussianFunction2D(const int x, const int y, const float sigma)
{
	double expPow = static_cast<double>((x * x) + (y * y)) / (2.0 * static_cast<double>(sigma * sigma));
	double exponentail = exp(-expPow);
	return exponentail / (2 * 3.1415 * (sigma * sigma));
}

__global__ void SobelBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	int sobelX, sobelY;

	float Gx[9] = {
		1, 0, -1,
		2, 0, -2,
		1, 0, -1
	};
	float Gy[9] = {
		1, 2, 1,
		0, 0, 0,
		-1, -2, -1
	};
	
	sobelX = Convolution(imageIn, x, y, Gx, width, height, 3);
	sobelY = Convolution(imageIn, x, y, Gy, width, height, 3);

	int magnitude = static_cast<int>(sqrt(static_cast<double>(sobelX * sobelX) + static_cast<double>(sobelY * sobelY)));
	if (magnitude > 255)
		magnitude = 255;
	imageOut[x + y * width] = static_cast<unsigned char>(magnitude);
}

__global__ void GaussBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma, const unsigned int kernalSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	int halfKernalSize = (kernalSize - 1) / 2;

	float* gaussKernel = (float*)malloc(sizeof(float) * kernalSize * kernalSize);

	for (int ky = -halfKernalSize; ky <= halfKernalSize; ky++)
	{
		for (int kx = -halfKernalSize; kx <= halfKernalSize; kx++)
		{
			gaussKernel[(kx + halfKernalSize) + ((ky + halfKernalSize) * kernalSize)] = GaussianFunction2D(kx, ky, sigma);
		}
	}

	float kernelSum = 0;
	for (int i = 0; i < 9; i++) {
		kernelSum += gaussKernel[i];
	}

	for (int i = 0; i < 9; i++) {
		gaussKernel[i] /= kernelSum;
	}

	imageOut[x + y * width] = Convolution(imageIn, x, y, gaussKernel, width, height, kernalSize);

	free(gaussKernel);
}

__global__ void DoGCuda(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd1, const float sd2, const unsigned int kernalSize)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	int halfKernalSize = (kernalSize - 1) / 2;

	float* gaussKernel1 = (float*)malloc(sizeof(float) * kernalSize * kernalSize);
	float* gaussKernel2 = (float*)malloc(sizeof(float) * kernalSize * kernalSize);

	for (int ky = -halfKernalSize; ky <= halfKernalSize; ky++)
	{
		for (int kx = -halfKernalSize; kx <= halfKernalSize; kx++)
		{
			gaussKernel1[(kx + halfKernalSize) + ((ky + halfKernalSize) * kernalSize)] = GaussianFunction2D(kx, ky, sd1);
			gaussKernel2[(kx + halfKernalSize) + ((ky + halfKernalSize) * kernalSize)] = GaussianFunction2D(kx, ky, sd2);
		}
	}

	float kernelSum1 = 0;
	float kernelSum2 = 0;
	for (int i = 0; i < 9; i++) {
		kernelSum1 += gaussKernel1[i];
		kernelSum2 += gaussKernel2[i];
	}

	for (int i = 0; i < 9; i++) {
		gaussKernel1[i] /= kernelSum1;
		gaussKernel2[i] /= kernelSum2;
	}

	imageOut[x + y * width] = Convolution(imageIn, x, y, gaussKernel1, width, height, kernalSize) - Convolution(imageIn, x, y, gaussKernel2, width, height, kernalSize);

	free(gaussKernel1);
	free(gaussKernel2);
}

__global__ void ThresholdCuda(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const unsigned char threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	if (imageIn[x + y * width] > threshold)
		imageOut[x + y * width] = 255;
	else
		imageOut[x + y * width] = 0;
}

std::chrono::microseconds cudaImageProcessing::Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(8, 8);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	SobelBlur <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height);

	cudaDeviceSynchronize();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration;
}

std::chrono::microseconds cudaImageProcessing::GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma, const unsigned int kernalSize)
{
	if (kernalSize % 2 != 1)
	{
		std::cout << "Kernal size must be odd" << std::endl;
		return std::chrono::microseconds(0);
	}

	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(8, 8);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	GaussBlur <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height, sigma, kernalSize);

	cudaDeviceSynchronize();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	return duration;
}

std::chrono::microseconds cudaImageProcessing::DoG(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sd1, const float sd2, const unsigned int kernalSize)
{
	if (kernalSize % 2 != 1)
	{
		std::cout << "Kernal size must be odd" << std::endl;
		return std::chrono::microseconds(0);
	}

	if (sd1 >= sd2)
	{
		std::cout << "sd1 must be less than sd2" << std::endl;
		return std::chrono::microseconds(0);
	}
		

	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(8, 8);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	DoGCuda <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height, sd1, sd2, kernalSize);

	cudaDeviceSynchronize();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

std::chrono::microseconds cudaImageProcessing::Threshold(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const unsigned char threshold)
{
	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(8, 8);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	ThresholdCuda <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height, threshold);

	cudaDeviceSynchronize();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}