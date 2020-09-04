// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mpi.h>
#include "functions.h"

using namespace functions;

namespace distropt {

	struct cocoa_params {
		int    max_itrs = 100;
		double eps_gap = 1e-8;
		int    local_itrs = 2;
		char   avg_or_sum = 's';
		bool   display = true;
		std::string filename = "temp.txt";
	};

	int cocoa_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const cocoa_params &params);
}