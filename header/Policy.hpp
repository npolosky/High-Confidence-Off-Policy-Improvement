// Author: npolosky
#pragma once

#include <stdafx.h>

/*		Header for the Policy abstract base class

	:memberFn getParameters: parameter getter
	:memberFn setParameters: parameter setter
	:memberFn getProb: returns the probability of a particular action in a particular state
*/

class Policy
{
public:
	virtual void setParameters(std::vector<double> params) = 0;
	virtual std::vector<double> getParameters() = 0;
	virtual double getProb(std::vector<double> state, int action) = 0;
};