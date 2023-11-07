
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

__global__ void Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
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

void cudaImageProcessing::SobelBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	unsigned char* cudaImageIn = NULL;
	unsigned char* cudaImageOut = NULL;

	cudaMalloc((void**)&cudaImageIn, width * height * sizeof(unsigned char));
	cudaMalloc((void**)&cudaImageOut, width * height * sizeof(unsigned char));

	cudaMemcpy(cudaImageIn, imageIn, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 blockDim(9, 9);
	dim3 gridDim(ceil(float(width) / float(blockDim.x)), ceil(float(height) / float(blockDim.y)));

	Sobel <<< gridDim, blockDim >>> (cudaImageIn, cudaImageOut, width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(imageOut, cudaImageOut, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(cudaImageIn);
	cudaFree(cudaImageOut);
}