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
				int sobelX = Convolution(imageIn, Gx, x, y, width, height, 3);
				int sobelY = Convolution(imageIn, Gy, x, y, width, height, 3);

				int magnitude = static_cast<int>(sqrt(static_cast<double>((sobelX * sobelX) + (sobelY * sobelY))));

				imageOut[x + (y * width)] = static_cast<unsigned char>(std::min(255, std::max(0, magnitude)));
			}
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

std::chrono::microseconds cppImageProcessing::GaussianBlur(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const float sigma, const unsigned int kernalSize)
{
	if(kernalSize % 2 != 1)
		throw std::invalid_argument("Kernal size must be odd");

	float* gaussKernel = new float[kernalSize * kernalSize];

	int halfKernalSize = (kernalSize - 1) / 2;

	for (int y = -halfKernalSize; y <= halfKernalSize; y++)
	{
		for (int x = -halfKernalSize; x <= halfKernalSize; x++)
		{
			gaussKernel[(x + halfKernalSize) + ((y + halfKernalSize) * kernalSize)] = GaussianFunction2D(x, y, sigma);
		}
	}

	float kernelSum = 0;
	for (int i = 0; i < kernalSize * kernalSize; i++) {
		kernelSum += gaussKernel[i];
	}

	for (int i = 0; i < kernalSize * kernalSize; i++) {
		gaussKernel[i] /= kernelSum;
	}

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			imageOut[x + y * width] = Convolution(imageIn, gaussKernel, x, y, width, height, kernalSize);
		}
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	std::chrono::microseconds duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	return duration;
}

std::chrono::microseconds cppImageProcessing::KMeansGrayscale(const unsigned char* imageIn, unsigned char* imageOut, const int width, const int height, const int k, const int iterationNum)
{
	struct Clustriod
	{
		unsigned int clustriodID;
		unsigned char value;
	};

	struct Pixel
	{
		unsigned int x, y;
		unsigned int clustriodID;
		unsigned char value;
	};

	Clustriod* clustriods = new Clustriod[k];

	for (int i = 0; i < k; i++)
	{
		clustriods[i].clustriodID = i;
		clustriods[i].value = static_cast<unsigned char>(rand() % 256);
	}

	Pixel* pixels = new Pixel[width * height];
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			pixels[x + (y * width)].x = x;
			pixels[x + (y * width)].y = y;
			pixels[x + (y * width)].value = imageIn[x + (y * width)];
			pixels[x + (y * width)].clustriodID = 0;
		}
	}

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < iterationNum; i++)
	{
		for (int pixel = 0; pixel < width * height; pixel++)
		{
			unsigned char minDist = 255;

			for (int clustriod = 0; clustriod < k; clustriod++)
			{
				unsigned char distance = abs(clustriods[clustriod].value - pixels[pixel].value);
				if (minDist > distance)
				{
					minDist = distance;
					pixels[pixel].clustriodID = clustriods[clustriod].clustriodID;
				}
			}
		}

		bool change = false;

		for (int clustriod = 0; clustriod < k; clustriod++)
		{
			int sum = 0;
			int amount = 0;
			for (int pixel = 0; pixel < width * height; pixel++)
			{
				if (pixels[pixel].clustriodID == clustriods[clustriod].clustriodID)
				{
					sum += pixels[pixel].value;
					amount++;
				}
			}

			if (amount != 0)
			{
				clustriods[clustriod].value = sum / amount;
				change = true;
			}
		}
		
		if (!change)
			break;
	}

	for (int i = 0; i < width * height; i++)
	{
		imageOut[pixels[i].x + pixels[i].y * width] = clustriods[pixels[i].clustriodID].value;
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

int cppImageProcessing::Convolution(const unsigned char* imageIn, const float* kernal, const int x, const int y, const int width, const int height, const unsigned int kernalSize)
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