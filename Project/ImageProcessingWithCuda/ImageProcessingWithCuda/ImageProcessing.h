#pragma once
#include <chrono>

static class cppImageProcessing
{
public:
	static std::chrono::microseconds GaussianBlur(const unsigned char*, unsigned char*, const int, const int, const float, const unsigned int);
	static std::chrono::microseconds Sobel(const unsigned char*, unsigned char*, const int, const int);
private:
	static int Convolution(const unsigned char*, const float*, const int, const int, const int, const int, const unsigned int);
	static float GaussianFunction2D(const int, const int, const float);
};

static class cudaImageProcessing
{
public:
	static std::chrono::microseconds GaussianBlur(const unsigned char*, unsigned char*, const int, const int, const float, const unsigned int);
	static std::chrono::microseconds Sobel(const unsigned char*, unsigned char*, const int, const int);
};