// Author: npolosky
#include "stdafx.h"

using namespace std;

/*		Constructor for the FnApproxSoftmax class

	:param sDim: dimensionality of states in the underlying MDP
	:param nActions: number of actions in the underlying MDP
	:param iOrder: independent order of the FourierBasis
	:param dOrder: dependent order of the FourierBasis
	:param params: initial parameter vector for the policy; params.size() == nactions*nstates
*/
FnApproxSoftmax::FnApproxSoftmax(int sDim, int nActions, int iOrder, int dOrder, std::vector<double> params)
{
	fb.init(sDim, iOrder, dOrder);
	numFeatures = fb.getNumOutputs();

	stateDim = sDim;
	numActions = nActions;
	setParameters(params);

	ud = uniform_real_distribution<double>(0, 1);
	sigma = 1.0;
}

/*		Parameter getter function. This also describes how the paramter vector maps to states and actions
*/
std::vector<double> FnApproxSoftmax::getParameters()
{
	std::vector<double> params(numActions*numFeatures);
	for(int i = 0; i < numActions; i++)
		for(int j = 0; j < numFeatures; j++)
			params.push_back(parameters[i][j]);
	return params;
}

/*		Parameter setter function. This also describes how the paramter vector maps to states and actions
*/
void FnApproxSoftmax::setParameters(std::vector<double> params)
{
	if(params.size() == 0)
	{
		params.resize(numActions * numFeatures);
		std::fill(params.begin(), params.end(), 0.1);
	}
	std::vector<std::vector<double>> new_params(numActions);
	for(int i = 0; i < numActions; i++)
	{
		std::vector<double> w_a(numFeatures);
		for(int j = 0; j < numFeatures; j++)
			w_a[j] = params[(i*numFeatures) + j];
		new_params[i] = w_a;
	}
	parameters = new_params;
}

/*		Returns an action given a state.

	:param state: vector representation of the current state
	:param generator: RNG used to sample action from softmax distribution
*/
int FnApproxSoftmax::getAction(std::vector<double> state, std::mt19937_64 & generator)
{
	std::vector<double> actionprob = getActionProb(state);
	double sample = ud(generator);
	double total = 0.0;

	for(int i = 0; i < actionprob.size(); i++)
	{
		total += actionprob[i];
		if(sample < total)
			return i;
	}
	return numActions - 1;
}

/*		Returns a vector of action probabilities given a state.

	:param state: vector representation of the current state
*/
std::vector<double> FnApproxSoftmax::getActionProb(std::vector<double> state)
{
	std::vector<double> phi = fb.basify(state);
	std::vector<double> actionprob(numActions, 0.0);
	for(int i = 0; i < numActions; i++)
		for(int j = 0; j < numFeatures; j++)
			actionprob[i] += parameters[i][j] * phi[j];

	double sum_of_elems = 0.0;
	for(auto &d : actionprob)
	{
		d = exp(sigma * d);
		sum_of_elems += d;
	}
	for(auto &d : actionprob)
		d = d / sum_of_elems;
	return actionprob;
}

/*		Returns the probability of an action in a given state.

	:param state: vector representation of the current state
	:param action: the action to evaluate the policy at
*/
double FnApproxSoftmax::getProb(std::vector<double> state, int action)
{
	return getActionProb(state)[action];
}