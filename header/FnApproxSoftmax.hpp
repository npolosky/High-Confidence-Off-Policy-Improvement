// Author: npolosky
#pragma once

#include "stdafx.h"

using namespace std;

/*		Header for the FnApproxSoftmax class which implements a Policy class as a function approximation policy

	:memberFn FnApproxSoftmax: constructor
	:memberFn getParameters: parameter getter
	:memberFn setParameters: parameter setter
	:memberFn getAction: returns an action given a state
	:memberFn getActionProb: returns a vector of action probabilities given a state
	:memberFn getProb: returns the probability of a particular action in a particular state

	:hiddenVar fb: a FourierBasis object which is used to compute a feature vector representation of the current state
	:hiddenVar parameters: the parameters of the policy
	:hiddenVar sigma: "temperature" used in the softmax function
	:hiddenVar stateDim: the dimensionality of states in the underlying MDP
	:hiddenVar numActions: number of actions in the underlying MDP
	:hiddenVar numFeatures: number of features to compute using FourierBasis
	:hiddenVAr ud: a uniform distribution used for sampling actions
*/

class FnApproxSoftmax : public Policy
{
public:
	FnApproxSoftmax(int sDim, int nActions, int iOrder, int dOrder, std::vector<double> params);
	std::vector<double> getParameters();
	void setParameters(std::vector<double> params);
	int getAction(std::vector<double> state, std::mt19937_64 & generator);
	std::vector<double> getActionProb(std::vector<double> state);
	double getProb(std::vector<double> state, int action);
private:
	FourierBasis fb;
	std::vector<std::vector<double>> parameters;
	double sigma;
	int stateDim;
	int numActions;
	int numFeatures;
	std::uniform_real_distribution<double> ud;
};