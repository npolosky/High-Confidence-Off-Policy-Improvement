// Author: npolosky
#include <stdafx.h>

using namespace std;

/*		Function which runs an agent in the Gridworld environment

	:param params: parameters used in the policy
	:numEpisodes: number of episodes to run in the environment
	:param generator: a RNG
	:param record: true will record the histories of the agent in the environment

	Returns the expected discounted return of the agent in the Gridworld environment

*/
double runGridworld(std::vector<double> params, int numEpisodes, mt19937_64 &generator, bool record)
{
	ofstream out;
	if(record)
		out.open("data/gridworld.csv");
	std::vector<double> returns(numEpisodes, 0.0);
	int maxEpisodeLength = 10;

	Gridworld e;

	int m = 1;
	int a = e.getNumActions();
	int k = 1;

	if(record)
	{
		out << m << endl << a << endl << k << endl;
		for(int i = 0; i < params.size()-1; i++)
			out << params[i] << ',';
		out << params.back() << endl;
		out << numEpisodes << endl;
	}

	auto A = FnApproxSoftmax(m, a, 1, k, params);

	vector<double> state, nextState, policy_test;
	bool first = true;
	for(int i = 0; i < numEpisodes; i++)
	{
		bool inTerminalState = false;			
		// generator.seed(i);
		e.newEpisode(generator);	
		state = e.getState(generator);	
		for (int t = 0; (t < maxEpisodeLength) && (!inTerminalState); t++)
		{	
			int action = A.getAction(state, generator);		// Get the current action
			if(first)
				policy_test.push_back(A.getProb(state, action));
			double reward = e.update(action, generator);	
			returns[i] += reward;							
			nextState = e.getState(generator);			
			inTerminalState = e.inTerminalState();
			if(record)
			{
				for(auto s : state)
					out << s << ',';
				out << action << ',';
				out << reward;
				if(inTerminalState || (t+1) >= maxEpisodeLength)
					out << endl;
				else
					out << ',';				
			}
			state = nextState;														// Prepare for the next iteration of the loop with this line and the next.
		}
		first = false;
	}
	if(record)
	{
		for(int i = 0; i < policy_test.size()-1; i++)
			out << policy_test[i] << ',';
		out << policy_test.back() << endl;
	}
	out.close();
	return mean(returns);
}

/*		A function to test that the parameterization of the policy is corrrect

	:param history: data to evaluate policy probabilities on
	:param :policy_out: policy parameters to evaluate
	:param B: policy object used to evaluate policy_out
*/
void
policyTest(std::vector<double> history, std::vector<double> policy_out, Policy &B)
{
	for(int j = 0; j < history.size(); j+=3)
	{
		std::vector<double> state(1);
		state[0] = history[j];
		cout << "Diff: " << (B.getProb(state, history[j+1]) - policy_out[j/3]) << endl;
	}
}

/*		Fucntion used to parse a datafile

	:param datafile: name of the datafile to read from
	:param m: number of state features
	:param a: number of discrete actions
	:param k: order of the FourierBasis used by the behavior policy
	:param params: parameters of the behavior policy
	:param n: number of episodes of data
	:param p_test: a vector of aciton probabilities corresponding to the first history 
				   in the dataset used for testing policy parameterization

	Returns the data set.
*/
std::vector<std::vector<double>>
readDataFile(std::string dataFile, int &m, int &a, int &k, std::vector<double> &params, int &n, std::vector<double> &p_test)
{
	ifstream in(dataFile, std::ios::app);
	std::string line;
	vector<vector<double>> D;

	int step = 0;
	while(getline(in, line))
	{
		stringstream ss(line);
		string substr;
		if(step == 0)
			m = std::stoi(line);
		if(step == 1)
			a = std::stoi(line);
		if(step == 2)
			k = std::stoi(line);
		if(step == 3)
		{
			while(getline(ss, substr, ','))
				params.push_back(std::stod(substr));
		}
		if(step == 4)
			n = std::stoi(line);
		if(step > 4)
		{
			std::vector<double> history;
			while(getline(ss, substr, ','))
				history.push_back(stod(substr));
			D.push_back(history);
		}
		step++;
	}
	p_test = D.back();
	D.pop_back();
	in.close();
	return D;
}

std::vector<double> getPolicy(ifstream &in)
{
	std::string line;
	std::vector<double> v;
	while(getline(in, line, ','))
		v.push_back(std::stod(line));
	return v;
}

/* Function to compute behavior policy probabilities and store them in the dataset

	:param D: data set of histories
	:param params: behavior policy parameters
	:param B: behavior policy object

	Returns the expected discounted return of the behavior policy
*/
double augmentData(std::vector<std::vector<double>> &D, std::vector<double> params, Policy &B)
{
	B.setParameters(params);
	std::vector<double> returns(D.size(), 0.0);
	for(int i = 0; i < D.size(); i++)
	{
		std::vector<double> new_H;
		for(int j = 0; j < D[i].size(); j+=3)
		{
			std::vector<double> state(1);
			state[0] = D[i][j];
			returns[i] += D[i][j+2];
			new_H.push_back(D[i][j]);
			new_H.push_back(D[i][j+1]);
			new_H.push_back(D[i][j+2]);
			new_H.push_back(B.getProb(state, D[i][j+1]));
		}
		D[i] = new_H;
	}
	return mean(returns);
}

/*		This function drives the program and runs HCOPI on the data specified in the data/data.csv file
*/
int main(int argc, char * argv[])
{
	std::string dataFile = "data/data.csv";

	static mt19937_64 generator(time(NULL));

	int m;
	int a;
	int k;
	std::vector<double> behavior_parameters;
	int n;
	std::vector<double> policy_test;
	auto D = readDataFile(dataFile, m, a, k, behavior_parameters, n, policy_test);
	cout << "m: " << m << " a: " << a << " k: " << k << endl;

	auto agentB = FnApproxSoftmax(m, a, 1, k, behavior_parameters);
	double b_return = augmentData(D, behavior_parameters, agentB);
	cout << "b_return: " << b_return << endl;

	std::vector<std::vector<double>> Dc((int)(D.size()*0.7));
	std::vector<std::vector<double>> Ds;
	for(int i = 0; i < Dc.size(); i++)
		Dc[i] = D[i];
	for(int i = Dc.size(); i < D.size(); i++)
		Ds.push_back(D[i]);


	omp_set_nested(1);
	int numPolicies = 100;
	std::vector<std::pair<VectorXd, bool>> results(numPolicies);
	std::vector<double> returns(numPolicies);
	std::vector<double> deltas(numPolicies, 0.05);
	std::vector<double> c(numPolicies, 8.0);

	#pragma omp parallel for
	for(int trial = 0; trial < numPolicies; trial++)
	{
		auto agentE = FnApproxSoftmax(m, a, 1, k, behavior_parameters);
		results[trial] = HCOPI(Dc, Ds, deltas[trial], c[trial], behavior_parameters, agentE, generator);

	}
	cout << "Done optimizing" << endl;
	for(int i = 0; i < numPolicies; i++)
	{
		if(results[i].second)
		{
			std::string fileName = "output/" + to_string(i+1) + ".csv";
			ofstream out(fileName);
			for(int j = 0; j < results[i].first.size()-1; j++)
				out << results[i].first[j] << ',';
			out << results[i].first[results[i].first.size()-1] << endl;
			out.close();
		}
	}
	return 0;
}