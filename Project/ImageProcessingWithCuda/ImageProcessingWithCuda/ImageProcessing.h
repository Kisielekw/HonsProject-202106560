#pragma once

static class cppImageProcessing
{
public:
	static void SobelBlur(const unsigned char*, unsigned char*, const int, const int);
private:
	static int Convolution(const unsigned char*, const float*, const int, const int, const int, const int);
};

static class cudaImageProcessing
{
public:
	static void SobelBlur(const unsigned char*, unsigned char*, const int, const int);
};