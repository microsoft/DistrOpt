// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mpi.h>
#include "functions.h"

using namespace functions;

namespace distropt {

	struct dane_params {
		bool   avg_init = false;
		int    ini_itrs = 10;
		double eps_init = 1e-6;
		double eps_obj = 1e-10;
		int    max_itrs = 100;
		double mu_dane = 0.0;
		char   local_solver = 's';	// 's' for sdca and 'b' for lbfgs
		double lbfgs_eps = 1e-6;
		double btls_max = 10;
		double btls_dec = 1e-4;
		double btls_rho = 0.8;
		double sdca_epsilon = 1e-6;
		int    sdca_itrs = 10;
		char   sdca_update = 'd';
		bool   sdca_display = false;
		bool   display = true;
		std::string filename = "temp.txt";
	};

	int dane_omp(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda, const SquaredL2Norm &g,
		Vector &wio, const int m, const dane_params &params);

	int dane_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, 
		const double lambda, const SquaredL2Norm &g, Vector &wio, const dane_params &params);
}