
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageProcessing.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>

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

__global__ void PixelClusterSelectGrayscale(const unsigned char* imageIn, int width, int height, int k, unsigned int* pixelCluserIDs, unsigned char* clustriodVaules, unsigned int* clusterSums, unsigned int* clusterAmount)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	unsigned char min = 255;
	int clusterID = 0;

	for (int i = 0; i < k; i++)
	{
		unsigned char distance = abs(imageIn[x + y * width] - clustriodVaules[i]);
		if (distance < min)
		{
			min = distance;
			clusterID = i;
		}
	}

	atomicAdd(clusterSums + clusterID, static_cast<unsigned int>(imageIn[x + y * width]));
	atomicAdd(clusterAmount + clusterID, 1);
	pixelCluserIDs[x + y * width] = clusterID;
}

__global__ void ClustriodMovementGrayscale(unsigned char* clustriodValues, unsigned int* clusterSums, unsigned int* clusterAmount, int k)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	if (x > k)
		return;
	
	clustriodValues[x] = clusterSums[x] / clusterAmount[x];
	clusterSums[x] = 0;
	clusterAmount[x] = 0;
}

__global__ void SetPixelsGrayscale(unsigned char* imageOut, int width, int height, unsigned int* pixelClusterIDs, unsigned char* clustriodValues)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	imageOut[x + y * width] = clustriodValues[pixelClusterIDs[x + y * width]];
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

std::chrono::microseconds cudaImageProcessing::KMeansGrayscale(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const int k, const int iterationNum)
{
	unsigned char* clustriodValues = new unsigned char[k];
	unsigned int* clusterSums = new unsigned int[k] {0};
	unsigned int* clusterAmount = new unsigned int[k] {0};

	for (int i = 0; i < k; i++)
	{
		clustriodValues[i] = static_cast<unsigned char>(rand() % 256);
	}

	unsigned char* cudaImageIn = nullptr;
	unsigned char* cudaImageOut = nullptr;
	unsigned char* cudaClustriodValues = nullptr;
	unsigned int* cudaPixelClusterIDs = nullptr;
	unsigned int* cudaClusterSums = nullptr;
	unsigned int* cudaClusterAmount = nullptr;

	cudaMalloc((void**)&cudaImageIn, sizeof(unsigned char) * width * height);
	cudaMalloc((void**)&cudaImageOut, sizeof(unsigned char) * width * height);
	cudaMalloc((void**)&cudaClustriodValues, sizeof(unsigned char) * k);
	cudaMalloc((void**)&cudaPixelClusterIDs, sizeof(unsigned int) * width * height);
	cudaMalloc((void**)&cudaClusterSums, sizeof(unsigned int) * k);
	cudaMalloc((void**)&cudaClusterAmount, sizeof(unsigned int) * k);

	cudaMemcpy((void*)cudaImageIn, imageIn, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)cudaClustriodValues, imageIn, sizeof(unsigned char) * k, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)cudaClusterSums, clusterSums, sizeof(unsigned int) * k, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)cudaClusterAmount, clusterSums, sizeof(unsigned int) * k, cudaMemcpyHostToDevice);

	delete[] clustriodValues;
	delete[] clusterSums;

	dim3 pixelBlockDim(8, 8);
	dim3 clusterBlockDim(k);
	dim3 pixelGridDim(ceil(float(width) / float(pixelBlockDim.x)), ceil(float(height) / float(pixelBlockDim.y)));
	dim3 clusterGridDim(1);

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iterationNum; i++)
	{
		PixelClusterSelectGrayscale <<< pixelGridDim, pixelBlockDim >>> (cudaImageIn, width, height, k, cudaPixelClusterIDs, cudaClustriodValues, cudaClusterSums, cudaClusterAmount);
		cudaDeviceSynchronize();

		ClustriodMovementGrayscale <<< clusterGridDim, clusterBlockDim >>> (cudaClustriodValues, cudaClusterSums, cudaClusterAmount, k);
		cudaDeviceSynchronize();
	}

	SetPixelsGrayscale <<< pixelGridDim, pixelBlockDim >>> (cudaImageOut, width, height, cudaPixelClusterIDs, cudaClustriodValues);
	cudaDeviceSynchronize();

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	cudaMemcpy(imageOut, cudaImageOut, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);
	cudaFree(cudaClustriodValues);
	cudaFree(cudaPixelClusterIDs);
	cudaFree(cudaClusterSums);
	cudaFree(cudaClusterAmount);
	
	return duration;
}