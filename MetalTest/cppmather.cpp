#include "cppmather.h"

CPPMather::CPPMather(int start)
{
	value = start;
}

void CPPMather::add(int val)
{
	value += val;
}

int CPPMather::val()
{
	return value;
}