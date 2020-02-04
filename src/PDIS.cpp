// Author: npolosky
#include "stdafx.h"


/*		Implementation of Per-Decision Importance Sampling (PDIS) algorithm

	:param D: the data. In this case a vector of histories generated by the behavior policy
	:param e_params: the evaluation policy parameters to evaluate
	:param E: the evaluation policy object

	Returns the estimated sample mean and standard deviation for the expected discounted return
	of the evaluation policy
*/
std::pair<double, double>
PDIS(const std::vector<std::vector<double>> &D, const std::vector<double> e_params, Policy &E)
{
	E.setParameters(e_params);
	std::vector<double> pdis_array(D.size(), 0.0);
	std::vector<double> importance_weight(D.size(), 1.0);
	#pragma omp parallel for
	for(int i = 0; i < D.size(); i++)
	{
		std::vector<double> history = D[i];
		for(int j = 0; j < history.size(); j+=4)
		{
			std::vector<double> state(1);
			state[0] = history[j];
			// the following line is commented out because the behavior policy action probabilities were
			// computed and stored in the histories data structure before runnig PDIS
			// importance_weight[i] *= (E.getProb(state, history[j+1]) / B.getProb(state, history[j+1]));
			importance_weight[i] *= (E.getProb(state, history[j+1]) / history[j+3]);
			pdis_array[i] += importance_weight[i] * history[j+2];
		}
	}

	double sample_mean = mean(pdis_array);
	double total = 0.0;
	for(auto d: pdis_array)
		total += (d-sample_mean) * (d-sample_mean);
	double sample_stddev = sqrt(total / ((double)pdis_array.size() - 1.0));

	return std::pair<double, double>(sample_mean, sample_stddev);
}

/*		Implements the High Confidence Off-Policy Evaluation (HCOPE) algorithm using 
		a Student's t distribution to compute confidence bounds

	:param theta: the parameter vector to evaluate
	:param params: pointer to the data and evaluation criteria parameters
	:param generator: a RNG

	Returns the lower bound on the expected discounted return of the policy parameterized
	by theta if it is above the constraint. Other wise returns the barrier function with 
	some shaping using the expected discounted return estimate.
*/
double
HCOPE(const VectorXd &theta, const void * params[], mt19937_64& generator)
{
	std::vector<double> epolicy_vec(theta.size());
	for(int i = 0; i < theta.size(); i++)
		epolicy_vec[i] = theta[i];

	const std::vector<std::vector<double>>* Dc = (const std::vector<std::vector<double>>*)params[0];
	const int* sSize = (const int*)params[1];
	const double* delta = (const double*)params[2];
	const double* c = (const double*)params[3];
	Policy* E = (Policy*)params[4];

	std::pair<double, double> mean_dev = PDIS(*Dc, epolicy_vec, *E);

	double result;
	double ttest_estimate = mean_dev.first - 2.0*(mean_dev.second / sqrt(*sSize))*tinv(1.0 - (*delta), (unsigned int)(*sSize) - 1u);
	if(ttest_estimate < (*c))
		result = -100000.0 + ttest_estimate;
	else
		result = mean_dev.first;
	return result;
}

/*		Tests whether a particular parameter vector passes the safety test

	:param theta: the parameters to test
	:param Ds: the safety data to test the parameters on
	:param delta: confidence interval used in the Student's t distribution
	:param c: the expected dsicounted return minimum constraint
	:param E: evluation policy object

	Returns true if the parameter vector passes the safety test and false otherwise.
*/
bool
safetyTest(VectorXd theta, std::vector<std::vector<double>> &Ds, double delta, double c, Policy &E)
{
	std::vector<double> epolicy_vec(theta.size());
	for(int i = 0; i < theta.size(); i++)
		epolicy_vec[i] = theta[i];

	std::pair<double, double> mean_dev = PDIS(Ds, epolicy_vec, E);

	double ttest_estimate = mean_dev.first - (mean_dev.second / sqrt(Ds.size()))*tinv(1.0 - delta, (unsigned int)(Ds.size()) - 1u);
	return (ttest_estimate >= c);
}

/*		Implements the High Confidence Off-Policy Improvement (HCOPI) algorithm

	:param Dc: data to evaluate candidate solutions on
	:param Ds: data used in the safety test
	:param delta: confidence interval used in the Student's t distribution
	:param c: the expected dsicounted return minimum constraint
	:param e_params: initial evaluation policy parameters
	:param E: evluation policy object
	:param generator: a RNG

	Returns the best parameters found by the algorithm and a boolean variable denoting
	whether or not these parameters passed the safety test.
*/
std::pair<VectorXd, bool>
HCOPI(std::vector<std::vector<double>> &Dc, std::vector<std::vector<double>> &Ds, double delta, double c, std::vector<double> e_params, Policy &E, mt19937_64 &generator)
{
	VectorXd initsol(e_params.size());
	for(int i = 0; i < e_params.size(); i++)
		initsol[i] = e_params[i];

	const VectorXd initialSolution = initsol;
	double initialSigma = 2.0*(initialSolution.dot(initialSolution) + 1.0);	// A heuristic to select the width of the search based on the weight magnitudes we expect to see.
	int numIterations = 100;
	bool minimize = false;

	int sSize = Ds.size();

	const void* params[6];
	params[0] = &Dc;
	params[1] = &sSize;
	params[2] = &delta;
	params[3] = &c;
	params[4] = &E;

	std::pair<VectorXd, bool> result;
	result.first = CMAES(initialSolution, initialSigma, numIterations, HCOPE, params, minimize, generator);
	result.second = safetyTest(result.first, Ds, delta, c, E);
	return result;
}