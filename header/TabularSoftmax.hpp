// Author: npolosky
#pragma once

#include "stdafx.h"

/*		Header for the TabularSoftmax class which implements a Policy class as a tabular softmax policy

	:memberFn TabularSoftmax: constructor
	:memberFn getParameters: parameter getter
	:memberFn setParameters: parameter setter
	:memberFn getAction: returns an action given a state
	:memberFn getActionProb: returns a vector of action probabilities given a state
	:memberFn getProb: returns the probability of a particular action in a particular state

	:hiddenVar parameters: the parameters of the policy
	:hiddenVar sigma: "temperature" used in the softmax function
	:hiddenVar numStates: number of states in the underlying MDP
	:hiddenVar numActions: number of actions in the underlying MDP
	:hiddenVAr ud: a uniform distribution used for sampling actions
*/

class TabularSoftmax : public Policy
{
public:
	TabularSoftmax(int numStates, int numActions, std::vector<double> params);
	std::vector<double> getParameters();
	void setParameters(std::vector<double> params);
	int getAction(std::vector<double> state, std::mt19937_64 & generator);
	std::vector<double> getActionProb(std::vector<double> state);
	double getProb(std::vector<double> state, int action);
private:
	std::vector<std::vector<double>> parameters;
	double sigma;
	int numStates;
	int numActions;
	std::uniform_real_distribution<double> ud;
};
