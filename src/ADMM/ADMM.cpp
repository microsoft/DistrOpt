// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "floattype.h"
#include "utils.h"
#include "randalgms.h"
#include "admm_mpi.h"

using namespace functions;
using namespace distropt;

int main(int argc, char* argv[])
{
	// In our implementation, only the main thread calls MPI, so we use MPI_THREAD_FUNNELED
	int mpi_thread_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_provided);

	MPI_Comm comm_world = MPI_COMM_WORLD;
	int mpi_size, mpi_rank;
	MPI_Comm_size(comm_world, &mpi_size);
	MPI_Comm_rank(comm_world, &mpi_rank);

	//---------------------------------------------------------------------------------------------
	// parse the input arguments
	std::string data_file;
	std::string init_mthd = "none";
	char loss_type, regu_type;
	double lambda, l1weight = 0;	// default l1 weight set to zero
	double rownorm = 1;				// default to normalize rows of X
	admm_params params;

	int must_count = 0;
	for (int i = 1; i < argc; i += 2) {
		std::string optstr(argv[i]);
		std::string valstr(argv[i + 1]);
		if (optstr == "-data") {
			data_file = valstr;
			must_count++;
		}
		else if (optstr == "-loss") {
			loss_type = valstr[0];
			must_count++;
		}
		else if (optstr == "-lambda") {
			lambda = std::stof(valstr);
			must_count++;
		}
		else if (optstr == "-regu") {
			regu_type = valstr[0];
			must_count++;
		}
		else if (optstr == "-l1") {
			l1weight = std::stof(valstr);
		}
		else if (optstr == "-init") {
			init_mthd = valstr;
		}
		else if (optstr == "-rho") {
			params.rho_Lagr = std::stof(valstr);
		}
		else if (optstr == "-maxitrs") {
			params.max_itrs = std::stoi(valstr);
		}
		else if (optstr == "-primaleps") {
			params.eps_primal = std::stof(valstr);
		}
		else if (optstr == "-dualeps") {
			params.eps_dual = std::stof(valstr);
		}
		else if (optstr == "-localsolver") {
			params.local_solver = valstr[0];
		}
		else if (optstr == "-localitrs") {
			params.local_itrs = std::stoi(valstr);
		}
		else if (optstr == "-localeps") {
			params.local_epsl = std::stof(valstr);
		}
		else if (optstr == "-rownorm") {
			rownorm = std::stof(valstr);
		}
		else if (optstr == "-display") {
			params.display = (std::stoi(valstr) > 0);
		}
		else {
			std::cout << "ADMM: Invalid arguments, please try again." << std::endl;
			std::exit(0);
		}
	}
	if (must_count < 4) {
		std::cout << "ADMM arguments: -data <string> -loss <char> -lambda <double> -regu <char>" << std::endl;
		std::cout << "                -l1 <double> -init <string> -rho<double> -maxitrs <int>" << std::endl;
		std::cout << "                -primaleps <double> -dualeps <double> -localitrs <int>" << std::endl;
		std::cout << "                -localeps <double> -rownorm <double> -display <int>" << std::endl;
		std::cout << "The first 4 options must be given, others can use default values." << std::endl;
		std::exit(0);
	}

	// generate the output file name reflecting the dataset, algorithm, loss+regu, and lambda value
	std::string data_name = data_file.substr(data_file.find_last_of("/\\") + 1);
	std::stringstream sslambda, ssrho;
	sslambda << std::scientific << std::setprecision(1) << lambda;
	std::string slambda = sslambda.str();
	slambda.erase(slambda.find_first_of("."), 2);
	ssrho << std::scientific << std::setprecision(1) << params.rho_Lagr;
	std::string srho = ssrho.str();
	srho.erase(srho.find_first_of("."), 2);
	params.filename = data_name + "_" + std::string(1, loss_type) + std::string(1, regu_type) + "_admm_" 
		+ params.local_solver + "_" + slambda + "_rho" + srho;

	//---------------------------------------------------------------------------------------------
	// read local data file
	string myfile = data_file + '_' + std::to_string(mpi_rank + 1);		// file labels start with 1

	std::vector<spmatT> labels;
	std::vector<spmatT> weights;
	std::vector<spmatT> values;
	std::vector<size_t> colidx;
	std::vector<size_t> rowptr;

	//std::cout << "Loading training data ... " << std::endl;
	double time_start = MPI_Wtime();
	size_t n_examples = load_datafile(myfile, labels, weights, values, colidx, rowptr, false, false, true);

	SparseMatrixCSR X(values, colidx, rowptr, false);
	Vector y(labels);
	if (rownorm > 0) { X.normalize_rows(rownorm); }

	MPI_Barrier(comm_world);

	if (mpi_rank == 0) {
		std::cout << "Loading files took " << MPI_Wtime() - time_start << " sec" << std::endl;
	}

	// use collective communications to get nTotalSamples and nAllFeatures
	size_t nSamples = X.nrows();
	size_t nFeatures = X.ncols();
	size_t N, D;

	time_start = MPI_Wtime();
	MPI_Allreduce(&nSamples,  &N, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_world);
	MPI_Allreduce(&nFeatures, &D, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm_world);

	// make sure everyone have the same feature dimension
	X.reset_ncols(D);

	if (mpi_rank == 0) {
		std::cout << "MPI_Allreduce took " << MPI_Wtime() - time_start << " sec" << std::endl;
		std::cout << "N = " << N << std::endl;
		std::cout << "D = " << D << std::endl;
	}

	//---------------------------------------------------------------------------------------------
	// construct loss function and regularization
	ISmoothLoss *f = nullptr;
	switch (loss_type)
	{
	case 'L':
	case 'l':
		f = new LogisticLoss();
		break;
	case 'S':
	case 's':
		f = new SmoothedHinge();
		break;
	case '2':
		f = new SquareLoss();
		break;
	default:
		throw std::runtime_error("Loss function type " + std::to_string(loss_type) + " not defined.");
	}

	IUnitRegularizer *g = nullptr;
	switch (regu_type)
	{
	case '2':
		g = new SquaredL2Norm();
		break;
	case 'E':
	case 'e':
		g = new ElasticNet(l1weight / lambda);
		break;
	default:
		throw std::runtime_error("Regularizer type " + std::to_string(regu_type) + " not defined.");
	}

	//---------------------------------------------------------------------------------------------
	// compute initial condition using different methods
	Vector w(D), a(X.nrows());
	if (init_mthd == "sdca") {
		bool display = false;
		if (mpi_rank == 0) display = true;
		randalgms::sdca(X, y, *f, lambda, *g, params.local_itrs, params.local_epsl, w, a, 'p', 'd', display);
	}
	else if (init_mthd == "cocoa") {
	}
	else{}

	// THIS STEP SHOULD BE LEFT INSIDE OF ADMM ALGORITHM? maybe not
	// Allreduce to compute the sum and then the average of local solutions
	int wlen = int(w.length());
	MPI_Allreduce(MPI_IN_PLACE, w.data(), wlen, MPI_VECTOR_TYPE, MPI_SUM, comm_world);
	// need to compute the average as starting points for all machines
	w.scale(1.0 / mpi_size);

	// ADMM algorithm ------------------------------------------------------------------------------
	MPI_Barrier(comm_world);
	admm_mpi(comm_world, N, X, y, *f, lambda, *g, w, params);

	// delete pointers
	delete f;
	delete g;

	// always call MPI_Finalize()
	MPI_Finalize();
	return 0;
}
