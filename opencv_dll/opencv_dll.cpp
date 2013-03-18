// opencv_dll.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#include "Recog.h"

#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define HAAR_HEAD_SIZE CAM_HEIGHT/3
#define HAAR_EYE_SIZE CAM_WIDTH/22


using namespace cv;
using namespace std;

#ifdef DLL1_EXPORTS
#define DLLEXPORT __declspec(dllexport)
#else
//#define DLLEXPORT __declspec(dllimport)
#define DLLEXPORT 
#endif

extern "C++" DLLEXPORT void TestGetSize(Mat& im, int & val1, int & val2)
{
	val1 = im.cols;
	val2 = im.rows;
}

extern "C++" DLLEXPORT IRecog* InitRecog(void)
{
	return new Recog;
}