#pragma once

#include <stdafx.h>

// Additional libraries that you will have to download.
// First is Eigen, which we use for linear algebra: http://eigen.tuxfamily.org/index.php?title=Main_Page
#include <Eigen/Dense>
#include <boost/math/special_functions/beta.hpp>
// Typically these shouldn't be in a .hpp file.
using namespace std;
using namespace Eigen;
using namespace boost::math;

// This function returns the inverse of Student's t CDF using the degrees of
// freedom in nu for the corresponding probabilities in p. That is, it is
// a C++ implementation of Matlab's tinv function: https://www.mathworks.com/help/stats/tinv.html
// To see how this was created, see the "quantile" block here: https://www.boost.org/doc/libs/1_58_0/libs/math/doc/html/math_toolkit/dist_ref/dists/students_t_dist.html
double tinv(double p, unsigned int nu);

// Get the sample standard deviation of the vector v (an Eigen::VectorXd)
double stddev(const VectorXd& v);

// Assuming v holds i.i.d. samples of a random variable, compute
// a (1-delta)-confidence upper bound on the expected value of the random
// variable using Student's t-test. That is:
// sampleMean + sampleStandardDeviation/sqrt(n) * tinv(1-delta, n-1),
// where n is the number of points in v.
//
// If numPoints is provided, then ttestUpperBound predicts what its output would be if it were to
// be run using a new vector, v, containing numPoints values sampled from the same distribution as
// the points in v.
double ttestUpperBound(const VectorXd& v, const double& delta, const int numPoints = -1);

/*
This function implements CMA-ES (http://en.wikipedia.org/wiki/CMA-ES). Return
value is the minimizer / maximizer. This code is written for brevity, not clarity.
See the link above for a description of what this code is doing.
*/
VectorXd CMAES(
	const VectorXd& initialMean,											// Starting point of the search
	const double& initialSigma,												// Initial standard deviation of the search around initialMean
	const unsigned int& numIterations,										// Number of iterations to run before stopping
	// f, below, is the function to be optimized. Its first argument is the solution, the middle arguments are variables required by f (listed below), and the last is a random number generator.
	double(*f)(const VectorXd& theta, const void* params[], mt19937_64& generator),
	const void* params[],													// Parrameters of f other than theta
	const bool& minimize,													// If true, we will try to minimize f. Otherwise we will try to maximize f
	mt19937_64& generator);