// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>
#include <ctime>
#include <math.h>

#include "utils.h"
#include "randalgms.h"
#include "dane_omp.h"

using namespace std;
using namespace distropt;

//int main_test_dane(int argc, char* argv[])
int main(int argc, char* argv[])
{
	if (argc < 2) {
		std::cout << "Need to specify training data file." << std::endl;
		std::exit(0);
	}
	string data_file = argv[1];

	// read training and testing files
	vector<spmatT> labels;
	vector<spmatT> weights;
	vector<spmatT> values;
	vector<size_t> colidx;
	vector<size_t> rowptr;

	std::cout << "Loading training data ... " << std::endl;
	// news20 actually has column index starting from 0, rcv1 and most others start with 1. Need to revise code!
	//size_t n_examples = load_datafile(data_file, labels, weights, values, colidx, rowptr, false, true, true, 0);
	size_t n_examples = load_datafile(data_file, labels, weights, values, colidx, rowptr, false, true, true);
	SparseMatrixCSR X(values, colidx, rowptr, false);
	Vector y(labels);

	size_t dim = X.ncols();

	// training using SDCA and display error rates
	Vector w(dim);
	ISmoothLoss *f = new LogisticLoss();
	//ISmoothLoss *f = new SmoothedHinge();
	//ISmoothLoss *f = new SquareLoss();
	//IUnitRegularizer *g = new SquaredL2Norm();
	SquaredL2Norm g;

	X.normalize_rows();

	// -------------------------------------------------------------------------------------
	double lambda = 1.0e-5;
	size_t m = 16;

	dane_params params;
	params.avg_init = true;
	params.ini_itrs = 20;
	params.max_itrs = 100;
	params.eps_init = 1e-6;
	params.mu_dane = 10*lambda;
	params.local_solver = 's';
	params.sdca_epsilon = 1e-8;
	params.sdca_itrs = 20;
	params.sdca_update = 'b';
	params.sdca_display = false;

	dane_omp(X, y, *f, lambda, g, w, m, params);

	float train_err = binary_error_rate(X, y, w);
	std::cout << "Training error rate = " << train_err * 100 << " %" << std::endl;

	// do not forget to delete object pointers!
	delete f;

	return 0;
}