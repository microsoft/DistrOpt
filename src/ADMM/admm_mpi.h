// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mpi.h>
#include "functions.h"

using namespace functions;

namespace distropt {

	struct admm_params {
		int    max_itrs = 100;
		double rho_Lagr = 0.01;
		double eps_primal = 1e-5;
		double eps_dual = 1e-5;
		char   local_solver = 's';		// 's' for sdca and 'b' for lbfgs
		int    local_itrs = 100;
		double local_epsl = 1e-6;
		bool   display = true;
		std::string filename = "temp.txt";
	};

	int admm_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const admm_params &params);
}