#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "../opencv_dll/opencv_dll.h"

using namespace cv;
using namespace std;

#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define HAAR_HEAD_SIZE CAM_HEIGHT/3
#define HAAR_EYE_SIZE CAM_WIDTH/22

#define WIN_NAME "Face Detection"

typedef void ( WINAPIV* LPFN_DISPLAYHELLO ) ( ); 

/*int TestDll()
{
	Mat im = imread("111.jpg");
	int v1, v2;
	TestGetSize(im, v1, v2);
	return v1;
}*/

int main()
{
	//-- 1. Подключение модуля распознавания
	IRecog* recog_core = InitRecog();

	//-- 2. Capture frame
	CvCapture* capture;
	capture = cvCaptureFromCAM( -1 );
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

	//-- 3. Video writer
	const char *filename = "capture.avi";
	double fps = 7;
	CvSize size = cvSize(320, 240);
	CvVideoWriter *writer = cvCreateVideoWriter(filename, CV_FOURCC('X','V','I','D'), fps, size, 1);

	//-- 4. Process every frame
	if( capture )
	{
		std::vector<Rect> faces;
		Mat frame;
		while( true )
		{
			//-- 4.1 Get frame
			frame = cvQueryFrame( capture );

			// result variables
			vector<pair<int, int> > face_part_coords;
			int result_code;
			string text;

			//-- 4.2 Process frame
			recog_core->ProcessImage(frame, face_part_coords, result_code);
			
			if(result_code == 1) text = "rotated left";
			if(result_code == 2) text = "rotated right";
			if(result_code == 3) text = "inclined right";
			if(result_code == 4) text = "inclined left";

			//-- 4.3 Video output
			cv::putText(frame, text, cv::Point( 15,35), 1, 2,cv::Scalar(0,255,10), 2, 7,false);
			imshow(WIN_NAME, frame);

			//-- 4.4 Write video
			cvWriteFrame(writer, &(IplImage)frame);

			//-- 4.5 Wait stop
			int c = waitKey(10);
			if((char)c == 'c'){break;}
		}
	}

	cvReleaseCapture( &capture );
	cvReleaseVideoWriter(&writer);
	cvDestroyWindow("capture");



	return 0;
}