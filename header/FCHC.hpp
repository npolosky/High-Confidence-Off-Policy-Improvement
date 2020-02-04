// Author: npolosky
#pragma once

#include "stdafx.h"

/*		Header for the FCHC class which implements the First Choice Hill Climbing black box optimization algorithm

	:memberFn FCHC: constructor
	:memberFn train: runs the optimization algorithm
	:memberFn getParameters: parameter getter

	:hiddenVar numEpisodes: the number of episodes used to evaluate a given parameter setting
	:hiddenVar sigma: "temperature" parameter
	:hiddenVar bestParams: the parameter setting that has achieved the best expected discounted return 
	:hiddenVar bestJ: the expected discounted return achieved by the bestParams
	:hiddenVAr evalFunction: a function pointer to the function used to evaluated parameter settings
*/

class FCHC
{
public:
	FCHC(std::vector<double> initparams, double initsigma, double (*f)(std::vector<double> p, int n, mt19937_64 &generator, bool r), int nEpisodes);
	void train(mt19937_64 &generator);
	std::vector<double> getParameters();
private:
	int numEpisodes;
	double (*evalFunction)(std::vector<double> p, int n, mt19937_64 &generator, bool r);
	std::vector<double> bestParams;
	double bestJ = -DBL_MAX;
	double sigma;
};