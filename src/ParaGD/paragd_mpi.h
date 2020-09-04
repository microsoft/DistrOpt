// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mpi.h>
#include "functions.h"

using namespace functions;

namespace distropt {

	struct paragd_params {
		int    max_itrs = 100;
		double eps_obj = 1e-10;
		double ls_incr = 2.0;
		double ls_decr = 2.0;
		bool   line_search = true;
		bool   display = true;
		std::string filename = "temp.txt";
	};

	int paragd_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const paragd_params &params);
}