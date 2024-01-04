#include "ImageProcessing.h"
#include <cmath>
#include <algorithm>

std::chrono::microseconds cppImageProcessing::Sobel(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height)
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

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
	
	for (int i = 0; i < 2; i++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int sobelX = Convolution(imageIn, Gx, x, y, width, height);
				int sobelY = Convolution(imageIn, Gy, x, y, width, height);

				int magnitude = static_cast<int>(sqrt(static_cast<double>((sobelX * sobelX) + (sobelY * sobelY))));

				imageOut[x + (y * width)] = static_cast<unsigned char>(std::min(255, std::max(0, magnitude)));
			}
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

std::chrono::microseconds cppImageProcessing::GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma)
{
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

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			imageOut[x + y * width] = Convolution(imageIn, gaussKernel, x, y, width, height);
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

float cppImageProcessing::GaussianFunction2D(const int x, const int y, const float sigma)
{
	double expPow = static_cast<double>((x * x) + (y * y)) / (2.0 * static_cast<double>(sigma * sigma));
	double exponentail = exp(-expPow);
	return exponentail / (2 * 3.1415 * (sigma * sigma));
}

int cppImageProcessing::Convolution(const unsigned char* imageIn, const float* kernal, const int x, const int y, const int width, const int height)
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