// Author: npolosky
#pragma once

#include <stdafx.h>

/*		Header declaring functions used in High Confidence Off-Policy Improvement (HCOPI)
*/

std::pair<double, double>
PDIS(const std::vector<std::vector<double>> &D, const std::vector<double> e_params, Policy &E);

double
HCOPE(const VectorXd &theta, const void * params[], mt19937_64& generator);

bool
safetyTest(VectorXd theta, std::vector<std::vector<double>> &Ds, double delta, double c, Policy &E);

std::pair<VectorXd, bool>
HCOPI(std::vector<std::vector<double>> &Dc, std::vector<std::vector<double>> &Ds, double delta, double c, std::vector<double> e_params, Policy &E, mt19937_64 &generator);