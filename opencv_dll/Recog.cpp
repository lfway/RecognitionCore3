#include "stdafx.h"

#include "Recog.h"

Recog::~Recog()
{

}

Recog::Recog()
{
	LoadHogDescriptors();
	LoadHaarCascades();
}