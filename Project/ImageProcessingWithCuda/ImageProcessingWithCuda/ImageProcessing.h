#pragma once

static class cppImageProcessing
{
public:
	static void GaussianBlur(const unsigned char*, unsigned char*, const int, const int, const float);
	static void SobelBlur(const unsigned char*, unsigned char*, const int, const int);
private:
	static int Convolution(const unsigned char*, const float*, const int, const int, const int, const int);
	static float GaussianFunction2D(const int, const int, const float);
};

static class cudaImageProcessing
{
public:
	static void GaussianBlur(const unsigned char*, unsigned char*, const int, const int, const float);
	static void SobelBlur(const unsigned char*, unsigned char*, const int, const int);
};