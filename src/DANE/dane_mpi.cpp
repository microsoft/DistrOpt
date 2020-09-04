// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <algorithm>

#include "timer.h"
#include "utils.h"
#include "randalgms.h"
#include "dane_omp.h"
#include "lbfgs_omp.h"

namespace distropt
{
	// formatted output: "iter  pcg_itrs   primal_obj    newton_dec    time" 
	void formatted_output_mpi(std::ostream &ofs, const int iters, const int epochs, const double stepsize, 
		const double primal_obj, const double t_dane)
	{
		ofs << std::setw(3) << iters
			<< std::setw(6) << epochs
			<< std::scientific << std::setprecision(1)
			<< std::setw(10) << stepsize
			<< std::fixed << std::setprecision(12)
			<< std::setw(17) << primal_obj
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_dane
			<< std::endl;
	}

	int dane_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y, const ISmoothLoss &f, 
		const double lambda, const SquaredL2Norm &g, Vector &wio, const dane_params &params)
	{
		int m, mpi_rank;	// number of machines and rank 0, ..., m-1
		MPI_Comm_size(comm_world, &m);
		MPI_Comm_rank(comm_world, &mpi_rank);

		// N is totoal number of examples on all machines, D is the feature dimension
		size_t Ni = X.nrows();
		size_t D = X.ncols();

		if (y.length() != Ni || wio.length() != D) {
			throw std::runtime_error("DANE: Input/output matrix and vector dimensions do not match.");
		}

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		HighResTimer timer;
		std::vector<int>    n_epochs;
		std::vector<double> primal_obj, t_dane, stepsizes;

		// create local regularized loss model
		RegularizedLoss localloss(X, y, f.symbol(), lambda, g.symbol());
		Vector w(D), grad(D), gradi(D), dw(D), Xiw(Ni), w_try(D);
		
		w.copy(wio);

		// first compute overall loss
		double sumloss = localloss.sum_loss(w);
		MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);

		double fw = sumloss / N + lambda*g(w);
		primal_obj.push_back(fw);
		t_dane.push_back(timer.seconds_from_start());
		n_epochs.push_back(0);
		stepsizes.push_back(1);

		bool display = params.display && mpi_rank == 0;
		if (display) {
			std::cout << std::endl << "iter passes stepsize      primal_obj     time" << std::endl;
			formatted_output_mpi(std::cout, 0, n_epochs[0], stepsizes[0], primal_obj[0], t_dane[0]);
		}

		// construct OffsetQuadratic as unit regularizer for PCG
		OffsetQuadratic q_regu(D);

		// DANE main loop
		double mu = params.mu_dane;
		int n_itrs = 0;
		for (n_itrs = 1; n_itrs <= params.max_itrs; n_itrs++)
		{
			// compute global gradient
			localloss.sum_grad(w, gradi);
			grad.copy(gradi);
			MPI_Allreduce(MPI_IN_PLACE, grad.data(), grad.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			grad.scale(1.0 / N);
			// need to include L2 regularization, but the same appears in both grad and gi[i]
			grad.axpy(lambda, w);

			double ci = double(m*Ni) / N;
			gradi.scale(double(m) / N);
			// need to include L2 regularization, but the same appears in both grad and gi[i] 
			gradi.axpy(ci*lambda, w);

			// solve the DANE local problem
			gradi.axpy(-1.0, grad);

			int local_passes;
			if (params.local_solver == 's') {
				gradi.axpy(mu, w);
				gradi.scale(1.0 / (ci*lambda + mu));

				q_regu.update_offset(gradi);
				double local_lambda = lambda + mu / ci;
				w_try.copy(w);
				local_passes = randalgms::sdca(X, y, f, local_lambda, q_regu, params.sdca_itrs, params.sdca_epsilon,
					w_try, Xiw, 'p', params.sdca_update, params.sdca_display);
			}
			else
			{
				// prepare for calling LBFGS
				auto localfval = [&localloss, &w, &gradi, &dw, ci, mu](const Vector &x) {
					dw.copy(x);
					dw.axpy(-1.0, w);
					return ci*localloss.regu_loss(x) - gradi.dot(x) + (mu / 2)*dw.dot(dw);
				};
				auto localgrad = [&localloss, &w, &gradi, &dw, ci, mu](const Vector &x, Vector &g) {
					dw.copy(x);
					dw.axpy(-1.0, w);
					localloss.regu_grad(x, g);
					g.scale(ci);
					g.axpy(-1.0, gradi);
					g.axpy(mu, dw);
					return ci*localloss.regu_loss(x) - gradi.dot(x) + (mu / 2)*dw.dot(dw);
				};

				lbfgs_params local_params;
				local_params.eps_grad = params.lbfgs_eps;
				local_params.display = false;
				local_passes = lbfgs_omp(localfval, localgrad, w, w_try, local_params);
			}

			// average w from all machines
			MPI_Allreduce(MPI_IN_PLACE, w_try.data(), w_try.length(), MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			w_try.scale(1.0 / m);
			Vector::elem_subtract(w_try, w, dw);


			// add backtracking line search to make sure!
			// backtracking line search (for nonconvex functions, need Wolfe conditions)
			double stepsize = 1.0;
			double ftry, gdw = grad.dot(dw);
			for (int lscnt = 0; lscnt < params.btls_max; lscnt++)
			{
				w_try.copy(w);
				w_try.axpy(stepsize, dw);
				// compute new primal and dual objective values
				sumloss = localloss.sum_loss(w_try);
				MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
				ftry = sumloss / N + lambda*g(w_try);
				if (ftry < fw + params.btls_dec*stepsize*gdw) {
					break;
				}
				stepsize *= params.btls_rho;
			}

			double df = abs(fw - ftry);

			w.copy(w_try);
			fw = ftry;


			// compute new primal and dual objective values
			//sumloss = localloss.sum_loss(w);
			//MPI_Allreduce(MPI_IN_PLACE, &sumloss, 1, MPI_DOUBLE, MPI_SUM, comm_world);
			//fw = sumloss / N + lambda*g(w);
			primal_obj.push_back(fw);
			n_epochs.push_back(local_passes);
			t_dane.push_back(timer.seconds_from_start());
			stepsizes.push_back(stepsize);

			if (display)
			{
				formatted_output_mpi(std::cout, n_itrs, n_epochs[n_itrs], stepsizes[n_itrs], primal_obj[n_itrs], t_dane[n_itrs]);
			}

			if (df < params.eps_obj) { break; }
		}

		//---------------------------------------------------------------------
		// write progress record in a file
		if (mpi_rank == 0) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "iter passes stepsize      primal_obj     time" << std::endl;
			for (int k = 0; k < primal_obj.size(); k++) {
				formatted_output_mpi(ofs, k, n_epochs[k], stepsizes[k], primal_obj[k], t_dane[k]);
			}
			ofs.close();
		}

		// --------------------------------------------------------------------------
		// return final iterate and number of iterations
		wio.copy(w);
		return n_itrs;
	}
}