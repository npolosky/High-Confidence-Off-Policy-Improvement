// Author: npolosky
#include "stdafx.h"

using namespace std;

/*		Constructor for the TabularSoftmax class

	:param nstates: number of states in the underlying MDP
	:param nactions: number of actions in the underlying MDP
	:param params: initial parameter vector for the policy; params.size() == nactions*nstates
*/
TabularSoftmax::TabularSoftmax(int nstates, int nactions, std::vector<double> params)
{
	numStates = nstates;
	numActions = nactions;
	setParameters(params);
	ud = uniform_real_distribution<double>(0, 1);
	sigma = 1.0;
}

/*		Parameter getter function. This also describes how the paramter vector maps to states and actions
*/
std::vector<double> TabularSoftmax::getParameters()
{
	std::vector<double> params(numStates*numActions);
	for(int i = 0; i < numStates; i++)
		for(int j = 0; j < numActions; j++)
			params.push_back(parameters[i][j]);
	return params;
}

/*		Parameter setter function. This also describes how the paramter vector maps to states and actions
*/
void TabularSoftmax::setParameters(std::vector<double> params)
{
	std::vector<std::vector<double>> new_params(numStates);
	for(int i = 0; i < numStates; i++)
	{
		std::vector<double> actions(numActions);
		for(int j = 0; j < numActions; j++)
			actions[j] = params[(i*numActions) + j];
		new_params[i] = actions;
	}
	parameters = new_params;
}

/*		Returns an action given a state.

	:param state: vector representation of the current state
	:param generator: RNG used to sample action from softmax distribution
*/
int TabularSoftmax::getAction(std::vector<double> state, std::mt19937_64 & generator)
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
std::vector<double> TabularSoftmax::getActionProb(std::vector<double> state)
{
	std::vector<double> actionParams = parameters[state[0]];
	double sum_of_elems = 0.0;
	for(auto &d : actionParams)
	{
		d = exp(sigma * d);
		sum_of_elems += d;
	}
	for(auto &d : actionParams)
		d = d / sum_of_elems;
	return actionParams;
}

/*		Returns the probability of an action in a given state.

	:param state: vector representation of the current state
	:param action: the action to evaluate the policy at
*/
double TabularSoftmax::getProb(std::vector<double> state, int action)
{
	return getActionProb(state)[action];
}