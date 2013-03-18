
#include "stdafx.h"

#include "Recog.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
/*
#define RESULT_NO		0
#define RESULT_LEFT		1
#define RESULT_RIGHT	2
#define RESULT_DOWN			3
#define RESULT_UP			4
#define RESULT_CLOCKWISE		5
#define RESULT_ANTICLOCKWISE	6
*/

void TestGetSize(Mat& im, int & val1, int & val2);

IRecog* InitRecog(void);