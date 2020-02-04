#pragma once

// This file has our include statements, so in other files we can just include this one.

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <algorithm>
#define _USE_MATH_DEFINES 
#include <math.h>
#include <time.h>
#include <climits>

// Tools
#include "MathUtils.hpp"
#include "FourierBasis.hpp"
#include "HelperFunctions.hpp"
#include "Policy.hpp"
#include "PDIS.hpp"
#include "TabularSoftmax.hpp"
#include "FnApproxSoftmax.hpp"

// Environments
#include "MountainCar.hpp"
#include "CartPole.hpp"
// #include "Acrobot.hpp"
#include "Gridworld.hpp"

// Agents
#include "QLearning.hpp"
// #include "Sarsa.hpp"
#include "FCHC.hpp"

#include <Eigen/Dense>
// Second is Boost, which we use for ibeta_inv: https://www.boost.org/
#include <boost/math/special_functions/beta.hpp>