
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageProcessing.h"

#include <stdio.h>
#include <cmath>
#include <algorithm>

__device__ int Convolution(const unsigned char* imageIn,const int x, const int y, const float* kernal, const int width, const int height)
{
	int sum = 0;

	for (int ky = -1; ky < 2; ky++)
	{
		int curentY = ky + y;
		if (curentY >= height || curentY < 0)
			continue;
		for (int kx = -1; kx < 2; kx++)
		{
			int curentX = kx + x;
			if (curentX >= width || curentX < 0)
				continue;
			sum += imageIn[curentX + (curentY * width)] * kernal[(kx + 1) + ((ky + 1) * 3)];
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
	
	sobelX = Convolution(imageIn, x, y, Gx, width, height);
	sobelY = Convolution(imageIn, x, y, Gy, width, height);

	int magnitude = static_cast<int>(sqrt(static_cast<double>(sobelX * sobelX) + static_cast<double>(sobelY * sobelY)));
	if (magnitude > 255)
		magnitude = 255;
	imageOut[x + y * width] = static_cast<unsigned char>(magnitude);
}

__global__ void GaussBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > width || y > height)
		return;

	float gaussKernel[9] = {
		GaussianFunction2D(-1, -1, sigma), GaussianFunction2D(0, -1, sigma), GaussianFunction2D(1, -1, sigma),
		GaussianFunction2D(-1, 0, sigma), GaussianFunction2D(0, 0, sigma), GaussianFunction2D(1, 0, sigma),
		GaussianFunction2D(-1, 1, sigma), GaussianFunction2D(0, 1, sigma), GaussianFunction2D(1, 1, sigma)
	};

	float kernelSum = 0;
	for (int i = 0; i < 9; i++) {
		kernelSum += gaussKernel[i];
	}

	for (int i = 0; i < 9; i++) {
		gaussKernel[i] /= kernelSum;
	}

	imageOut[x + y * width] = Convolution(imageIn, x, y, gaussKernel, width, height);
}

void cudaImageProcessing::Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(9, 9);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	SobelBlur <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);
}

void cudaImageProcessing::GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma)
{
	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(9, 9);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	GaussBlur <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height, sigma);

	cudaDeviceSynchronize();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);
}