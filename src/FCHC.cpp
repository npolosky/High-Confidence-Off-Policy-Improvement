// Author: npolosky
#include "stdafx.h"

using namespace std;

/*		Constructor for the First Choice Hill Climbing class.
	
	:param initparams: the initial parameter vector
	:param initsigma: the intial sigma value
	:param f: the parameter evaluation function we are optimizing
	:param nEpisodes: the number of episodes to evaluate each parameter setting for

*/ 
FCHC::FCHC(std::vector<double> initparams, double initsigma, double (*f)(std::vector<double> p, int n, mt19937_64 &generator, bool r), int nEpisodes)
{
	bestParams = initparams;
	sigma = initsigma;
	evalFunction = f;
	numEpisodes = nEpisodes;
}

/*		returns the best parameters found by the FCHC algorithm
*/
std::vector<double>
FCHC::getParameters()
{
	return bestParams;
}

/*		runs the FCHC optimization algorithm

	:param generator: a RNG used for generating new parameter samples
*/
void
FCHC::train(mt19937_64 &generator)
{
	std::vector<double> new_theta(bestParams.size());
	for(int i = 0; i < bestParams.size(); i++)
	{
		std::normal_distribution<double> distribution(bestParams[i], sigma);
		new_theta[i] = distribution(generator);
	}
	double newJ = evalFunction(new_theta, numEpisodes, generator, false);
	if(newJ > bestJ)
	{
		cout << "new best: " << newJ << endl;
		bestParams = new_theta;
		for(auto d: bestParams)
			cout << d << ", ";
		cout << endl;
		bestJ = newJ;
	}
}