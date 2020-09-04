// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mpi.h>
#include "functions.h"

using namespace functions;

namespace distropt {

	struct disco_params {
		bool   avg_init = false;
		int    ini_itrs = 10;
		int    max_itrs = 100;
		bool   linesearch = true;
		double eps_obj = 1e-10;
		double eps_init = 1e-6;
		double lambda_0 = 1e-5;
		double beta = 0.1;
		char   spc_solver = 's';	// 's' for sdca and 'b' for lbfgs
		int    spc_pass = 3;
		double spc_eps = 1e-8;
		double btls_max = 10;
		double btls_dec = 1e-4;
		double btls_rho = 0.5;
		int    pcg_itrs = 100;
		double pcg_mu = 1e-6;
		bool   pcg_adpt = false;
		bool   pcg_polak = true;
		bool   pcg_display = false;
		bool   pcg_record = false;
		bool   display = true;
		std::string filename = "temp.txt";
	};

	int disco_omp(const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, const double lambda, const SquaredL2Norm &g,
		Vector &wio, const int m, const disco_params &params);

	int disco_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, 
		const double lambda, const SquaredL2Norm &g, Vector &wio, const disco_params &params);
}