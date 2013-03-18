#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>

#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define HAAR_HEAD_SIZE CAM_HEIGHT/3
#define HAAR_EYE_SIZE CAM_WIDTH/22

using namespace cv;
using namespace std;

class IRecog
{
public:
	virtual void ProcessImage(Mat& TheFrame, vector<pair<int, int> > & FacePartCoords, int & ResultCode)=0;
};

class Recog: public IRecog
{

public:
	Recog();

    ~Recog();
	void ProcessImage(Mat& TheFrame, vector<pair<int, int> > & FacePartCoords, int & ResultCode)
	{
		Mat frame = TheFrame;
		ResultCode = 0;

		//-- Output Data:
		vector<pair<int, int> > ResultVector;
		bool rotate_left = false, rotate_right = false, up = false, down = false, bent_left = false, bent_right = false;
		//-- Variables:
		double angle_eye_mouth=0, angle_eye_mouth_1=0, angle_eye_mouth_2=0, angle_eye_right_left=0;
		//bool haar_face_finded = haar_face_finded_old;
		Mat frame_gray;
		vector<Rect> faces;

		//-- Use Haar Detector
		flip(frame, frame, 1);
		cvtColor( frame, frame_gray, CV_BGR2GRAY );
		mFaceCascade.detectMultiScale( frame_gray, faces, 1.3, 3, 0, Size(HAAR_HEAD_SIZE, HAAR_HEAD_SIZE) );
		if(faces.size() == 0)
		{
			mFaceProfCascade.detectMultiScale(frame_gray, faces, 1.3, 3, 0, Size(HAAR_HEAD_SIZE, HAAR_HEAD_SIZE));
			if(faces.size() == 1)rotate_left = true;
		}
		if(faces.size() == 0)
		{
			flip(frame_gray, frame_gray, 1);
			mFaceProfCascade.detectMultiScale(frame_gray, faces, 1.3, 3, 0, Size(HAAR_HEAD_SIZE, HAAR_HEAD_SIZE));
			if(faces.size() == 1)rotate_right = true;
		}

		//-- Searching face pats
		if(faces.size() > 0)
		{
			Rect face_roi_rect = faces[0];
			Rect nose_roi_rect = faces[0];
			Rect mouth_roi_rect = faces[0];
			face_roi_rect.y += faces[0].height/8;
			face_roi_rect.height -= faces[0].height/2;
			nose_roi_rect.y += faces[0].height/4;
			nose_roi_rect.height -= faces[0].height/4*1.5;
			mouth_roi_rect.y += faces[0].height/2;
			mouth_roi_rect.height -= faces[0].height/2;
				
			Mat faceROI			= frame_gray(face_roi_rect);
			Mat faceROI_nose	= frame_gray(nose_roi_rect);
			Mat faceROI_mouth	= frame_gray(mouth_roi_rect);
			Mat faceROI_faceupdown	= frame_gray;
			vector<Rect> eyes, eyes2, noses, noses2, mouths, mouths2, faceup, faceup2, facedown, facedown2;
			mHogEye.detectMultiScale		(faceROI,		eyes,	0.2,	Size(8, 8),	Size(16, 16),	1.03,	2.0);
			mHogNose.detectMultiScale	(faceROI_nose,	noses,	0.5,	Size(8, 8),	Size(16, 16),	1.03,	2.0);
			mHogMouth.detectMultiScale	(faceROI_mouth,	mouths,	0.25,	Size(8, 8),	Size(16, 16),	1.03,	2.0);

			ProcessVector(eyes,		eyes2,		face_roi_rect);
			ProcessVector(noses,	noses2,		nose_roi_rect);
			ProcessVector(mouths,	mouths2,	mouth_roi_rect);

			//rectangle(frame, face_roi_rect,	Scalar(255, 255, 255),	1, 8, 0 );
			//rectangle(frame, nose_roi_rect,	Scalar(255, 0, 0),		1, 8, 0 );
			//rectangle(frame, mouth_roi_rect,	Scalar(0, 0, 255),		1, 8, 0 );

			DrawVector(frame, eyes2, face_roi_rect, Scalar(255,255,255), 2, rotate_right);
			DrawVector(frame, noses2, nose_roi_rect, Scalar(255,0,0), 1, rotate_right);
			DrawVector(frame, mouths2, mouth_roi_rect, Scalar(0,0,255), 1, rotate_right);

			//Подготовка результата
			for(unsigned int i = 0; i < eyes2.size(); i++)	eyes2[i].y += face_roi_rect.y;
			for(unsigned int i = 0; i < noses2.size(); i++)	noses2[i].y += nose_roi_rect.y;
			for(unsigned int i = 0; i < mouths2.size(); i++)	mouths2[i].y += mouth_roi_rect.y;
			PrepareResult(ResultVector, eyes2, noses2, mouths2);

			//Вычисление углов
			if(eyes2.size()==2 && mouths2.size() == 1)
			{
				//правый
				angle_eye_mouth_2 = CalculateAngle(ResultVector[1], ResultVector[3]);
				//левый
				angle_eye_mouth_1 = CalculateAngle(ResultVector[0], ResultVector[3]);
				angle_eye_mouth_1 = 180 - angle_eye_mouth_1;
			}
		
			if(eyes2.size()==1 && mouths2.size() == 1)
			{
				angle_eye_mouth = CalculateAngle(ResultVector[0], ResultVector[3]);
			}

			if(rotate_left==true)ResultCode = 1;
			if(rotate_right==true)ResultCode = 2;
			if((angle_eye_mouth_2-angle_eye_mouth_1) > 15)ResultCode = 3;
			if((angle_eye_mouth_1-angle_eye_mouth_2) > 15)ResultCode = 4;
			if(angle_eye_mouth != 0 && angle_eye_mouth < 50)ResultCode = 3;
			if(angle_eye_mouth != 0 && angle_eye_mouth > 125)ResultCode = 4;
		}
	}

private:
	HOGDescriptor mHogEye, mHogNose, mHogMouth;
	CascadeClassifier mFaceCascade, mFaceProfCascade;

private:

	void ProcessVector(vector<Rect>& noses, vector<Rect>& noses2, Rect Roi)
	{
		if(noses.size() > 0)
		{
			for(unsigned int i = 0; i < noses.size(); i++)
			{
				if(noses[i].width > Roi.width/2 || (noses[i].x<=0 || noses[i].y<=0) || ( (noses[i].x+noses[i].width) >= Roi.width || (noses[i].y+noses[i].height) >= Roi.height) )
				{
					continue;
				}
				noses2.push_back(noses[i]);
			}

			for(unsigned int i = 0; i < noses2.size(); i++)
			{ 
				Rect nose_ = noses2[i];
				nose_.x += Roi.x;
				nose_.y += (Roi.y);
			}
		}
	}

	void DrawVector(Mat& Frame, vector<Rect>& DrawRectVector, Rect Offset, Scalar Color,unsigned  int HowMatch = 0, bool RotateRight = false)
	{
		for(unsigned int i = 0; i < DrawRectVector.size(); i++)
		{
			//if(RotateRight == true)
				//Offset.x = 320 - Offset.x - DrawRectVector[i].width;
			Rect rect_ = DrawRectVector[i];
			rect_.x += Offset.x;
			rect_.y += Offset.y;

			if(RotateRight == true)
				rect_.x = 320 - rect_.x - rect_.width;
			if(HowMatch > 0)
				if(i >= HowMatch)
					break;
			rectangle(Frame, rect_, Color);
			circle(Frame, Point(rect_.x+rect_.width/2, rect_.y+rect_.height/2), 2, Color, CV_FILLED);
		}
	}

	pair<int, int> RectToPoint(Rect rect)
	{
		return pair<int, int>(rect.x + rect.width/2, rect.y + rect.height/2);
	}

	// Подготовка конечных данных
	void PrepareResult(vector<pair<int, int> >& ResultVector, const vector<Rect>& eyes2, const vector<Rect>& noses2, const vector<Rect>& mouths2)
	{
		//Глаза: левый, затем правый
		if(eyes2.size() == 2)
		{
			if(eyes2[1].x > eyes2[0].x)
			{
				ResultVector.push_back(RectToPoint(eyes2[0]));
				ResultVector.push_back(RectToPoint(eyes2[1]));
			}
			else
			{
				ResultVector.push_back(RectToPoint(eyes2[0]));
				ResultVector.push_back(RectToPoint(eyes2[1]));
			}
		}
		else
		{
			if(eyes2.size()==0)
			{
				ResultVector.push_back(pair<int, int> (0, 0));
				ResultVector.push_back(pair<int, int> (0, 0));
			}
			if(eyes2.size()==1)
			{
				ResultVector.push_back(RectToPoint(eyes2[0]));
				ResultVector.push_back(pair<int, int> (0, 0));
			}
		}
		//Нос
		if(noses2.size()==1)
		{
			ResultVector.push_back(RectToPoint(noses2[0]));
		}
		else
		{
			ResultVector.push_back(pair<int, int> (0, 0));
		}
		//Рот
		if(mouths2.size()==1)
		{
			ResultVector.push_back(RectToPoint(mouths2[0]));
		}
		else
		{
			ResultVector.push_back(pair<int, int> (0, 0));
		}
	}

	double CalculateAngle(pair<int, int> point1, pair<int, int> point2)
	{
		int dx = point1.first - point2.first;
		int dy = point2.second - point1.second;
	
		double giptnz =  sqrt( (double)dx*dx + (double)dy*dy);
		return acos(dx/giptnz)/3.14*180;
		return dx/giptnz/3.14*180;
	}

	void LoadHogDescriptors()
	{
#ifdef _DEBUG
		InitHogDescriptor(mHogEye,		"C:/_hog/my_photos/_eyes");
		InitHogDescriptor(mHogNose,		"C:/_hog/my_photos/_nose");
		InitHogDescriptor(mHogMouth,	"C:/_hog/my_photos/_mouth");
#else
		InitHogDescriptor(mHogEye,		"hog/_eyes");
		InitHogDescriptor(mHogNose,		"hog/_nose");
		InitHogDescriptor(mHogMouth,	"hog/_mouth");
#endif
	}

	void InitHogDescriptor(HOGDescriptor& hog, string VectorFile)
	{
		std::ifstream fin(VectorFile);
		std::string input;
		getline(fin, input);
	
		std::replace( input.begin(), input.end(), ',', '.');

		std::vector<float> output;
		std::stringstream ss(input);
		for(std::string::size_type p0=0,p1=input.find(' ');
			p1!=std::string::npos || p0!=std::string::npos;
			(p0=(p1==std::string::npos)?p1:++p1),p1=input.find(' ', p0) )
			output.push_back( strtod(input.c_str()+p0,NULL) );

		hog.derivAperture = 1;
		hog.blockSize = Size(16, 16);
		hog.cellSize = Size(16, 16);
		hog.blockStride = Size(8, 8);
		hog.winSize = Size(32, 32);
		hog.setSVMDetector(output);
	}

	void LoadHaarCascades()
	{
		String face_cascade_name = "cascades/haarcascade_frontalface_default.xml";
		String face_prof_cascade_name = "cascades/haarcascade_profileface.xml";

		if( !mFaceCascade.load		( face_cascade_name ) )			{ printf("--(!)Error loading\n"); return; };
		if( !mFaceProfCascade.load	( face_prof_cascade_name ) )	{ printf("--(!)Error loading\n"); return; };
	}
};