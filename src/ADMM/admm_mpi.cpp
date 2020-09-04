// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <type_traits>

#include "admm_mpi.h"
#include "lbfgs_omp.h"
#include "regu_loss.h"
#include "randalgms.h"

namespace distropt
{
	// std::conditional does not work with MPI types, which are not exactly types
	//using MPI_FLOAT_TYPE = std::conditional<std::is_same<floatT, double>::value, MPI_DOUBLE, MPI_FLOAT>::type;

	// formatted output: "iter epochs   obj     err_conv    err_feas   sdca  t_sdca   t_comm     time" 
	void formatted_output(std::ostream &ofs, const int iters, const int epochs, const double primal_obj, const double err_primal,
		const double err_dual, const int sdca_epochs, const double t_sdca, const double t_comm, const double t_admm)
	{
		ofs << std::setw(3) << iters
			<< std::setw(6) << epochs
			<< std::fixed << std::setprecision(12) 
			<< std::setw(16) << primal_obj
			<< std::scientific << std::setprecision(3)
			<< std::setw(12) << err_primal
			<< std::setw(12) << err_dual
			<< std::setw(5) << sdca_epochs
			<< std::fixed << std::setprecision(3)
			<< std::setw(9) << t_sdca
			<< std::setw(9) << t_comm
			<< std::setw(9) << t_admm
			<< std::endl;
	}

	int admm_mpi(const MPI_Comm comm_world, const size_t N, const SparseMatrixCSR &X, const Vector &y,
		const ISmoothLoss &f, const double lambda, const IUnitRegularizer &g, Vector &wio, const admm_params &params)
	{
		int m, mpi_rank;	// number of machines and rank 0, ..., m-1
		MPI_Comm_size(comm_world, &m);
		MPI_Comm_rank(comm_world, &mpi_rank);

		// N is totoal number of examples on all machines, D is the feature dimension
		size_t Ni = X.nrows();
		size_t D = X.ncols();
		OffsetQuadratic q(D);						// quadratic function for updating v
		IUnitRegularizer &qq = q;					// parent reference for generic call of sdca
		double rho = params.rho_Lagr;				// penalty parameter in augmented Lagrangian
		double theta = ((double)Ni) / ((double)N);	// fraction of number of local examples

		// construct ADMM iterates and auxiliary variables
		Vector wi(D), ui(D), v(D), v_ui(D), v_pre(D), ri(D), uw(D), dwi(D);
		Vector Xiv(Ni);
		// initialize wi and v to be same as initial input wio
		wi.copy(wio);
		v.copy(wio);

		//---------------------------------------------------------------------
		// declare variables for tracking progress and time spent in different activities
		double t_comm_start, t_sdca_start;
		double t_admm_start = MPI_Wtime();
		int vlen = params.max_itrs + 1;
		std::vector<int>    n_epochs(vlen, 0), sdca_epochs(vlen, 0);
		std::vector<double> primal_obj(vlen, 0), err_primal(vlen, 0), err_dual(vlen, 0);
		std::vector<double> t_sdca(vlen, 0), t_comm(vlen, 0), t_admm(vlen, 0);

		// compute initial objective values and primal/dual residues
		X.aAxby(1.0, v, 0, Xiv);
		double sumloss = f.sum_loss(Xiv, y);
		Vector::elem_subtract(wi, v, ri);			// ri = wi - v
		double resi = pow(ri.norm2(), 2);			// t = ||ri||^2
		double loss_resi[2] = { sumloss, resi };
		// AllReduce to compute sum_loss and sum_i ||r_i||^2 
		t_comm_start = MPI_Wtime();
		MPI_Allreduce(MPI_IN_PLACE, loss_resi, 2, MPI_DOUBLE, MPI_SUM, comm_world);
		t_comm[0] = MPI_Wtime() - t_comm_start;

		primal_obj[0] = loss_resi[0] / N + lambda*g(v);
		err_primal[0] = std::sqrt(loss_resi[1]);	// e_primal = sqrt( sum_i || w_i - v ||^2 )
		Vector::elem_subtract(v, v_pre, ri);		// use ri as working memory for v - v_pre
		err_dual[0] = rho*std::sqrt(m)*ri.norm2();	// e_dual = rho * sqrt(m) * || v - v_pre ||

		n_epochs[0] = 1;
		t_admm[0] = MPI_Wtime() - t_admm_start;

		bool display = params.display && mpi_rank == 0;
		if (display) {
			std::cout << "iter epochs      obj        err_conv    err_feas   sdca  t_sdca   t_comm     time" << std::endl;
			formatted_output(std::cout, 0, n_epochs[0], primal_obj[0], err_primal[0], err_dual[0], sdca_epochs[0], t_sdca[0], t_comm[0], t_admm[0]);
		}

		RegularizedLoss localloss(X, y, f.symbol(), lambda, '2');

		// main loop of ADMM
		int n_iters = params.max_itrs;
		for (int k = 1; k <= params.max_itrs; k++)
		{
			// update scaled dual variable  ui := ui + wi - v
			Vector::elem_subtract(wi, v, ri);
			Vector::axpy(1.0, ri, ui);

			// compute new local solution: wi = argmin_w ( theta_i*Fi(X*w) + (rho/2)*||w-v+ui||^2)
			Vector::elem_subtract(v, ui, v_ui);
			q.update_offset(v_ui);

			// use SDCA to solve local problem, be careful about initial dual variable, output dual variable not used
			t_sdca_start = MPI_Wtime();
			if (params.local_solver == 'b') {
				// prepare for calling LBFGS
				auto localfval = [&localloss, &dwi, &v_ui, rho, theta](const Vector &x) {
					dwi.copy(x);
					dwi.axpy(-1.0, v_ui);
					return localloss.avg_loss(x) + (0.5*rho / theta)*dwi.dot(dwi);
				};
				auto localgrad = [&localloss, &dwi, &v_ui, rho, theta](const Vector &x, Vector &g) {
					dwi.copy(x);
					dwi.axpy(-1.0, v_ui);
					localloss.avg_grad(x, g);
					g.axpy(rho/theta, dwi);
					return localloss.avg_loss(x) + (0.5*rho / theta)*dwi.dot(dwi);
				};

				lbfgs_params local_params;
				local_params.eps_grad = params.local_epsl;
				local_params.display = false;
				sdca_epochs[k] = lbfgs_omp(localfval, localgrad, wi, wi, local_params);
			}
			else {
				Xiv.fill_zero();
				sdca_epochs[k] = randalgms::sdca(X, y, f, rho / theta, qq, params.local_itrs, params.local_epsl, wi, Xiv, 'p', 'd', false);
			}
			t_sdca[k] = MPI_Wtime() - t_sdca_start;
			n_epochs[k] = n_epochs[k - 1] + sdca_epochs[k];

			// AllReduce to compute uw = sum_i (ui + wi) and then scale to average
			Vector::elem_add(ui, wi, uw);		// uwi = ui + wi
			int uw_len = int(uw.length());
			t_comm_start = MPI_Wtime();
			MPI_Allreduce(MPI_IN_PLACE, uw.data(), uw_len, MPI_VECTOR_TYPE, MPI_SUM, comm_world);
			t_comm[k] += MPI_Wtime() - t_comm_start;
			uw.scale(1.0 / m);					// scale uw to be the average

			// compute new v using regularizer's prox operator
			v_pre.copy(v);						// copy previous v in order to compute residue
			g.prox(lambda / (m*rho), uw, v);

			// compute objective value via Allreduce, piggy-back feasibility error
			X.aAxby(1.0f, v, 0, Xiv);
			Vector::elem_subtract(wi, v, ri);			// ri = wi - v
			loss_resi[0] = f.sum_loss(Xiv, y);
			loss_resi[1] = pow(ri.norm2(), 2);			// resi = ||ri||^2
			// AllReduce to compute  sum_loss and sum_i ||r_i||^2
			t_comm_start = MPI_Wtime();
			MPI_Allreduce(MPI_IN_PLACE, loss_resi, 2, MPI_DOUBLE, MPI_SUM, comm_world);
			t_comm[k] += MPI_Wtime() - t_comm_start;

			primal_obj[k] = loss_resi[0] / N + lambda*g(v);
			err_primal[k] = std::sqrt(loss_resi[1]);	// e_primal = sqrt( sum_i || w_i - v ||^2 )
			Vector::elem_subtract(v, v_pre, ri);		// use ri as working memory for v - v_pre
			err_dual[k] = rho*std::sqrt(m)*ri.norm2();	// e_dual = rho * sqrt(m) * || v - v_pre ||

			t_admm[k] = MPI_Wtime() - t_admm_start;

			if (display) {
				formatted_output(std::cout, k, n_epochs[k], primal_obj[k], err_primal[k], err_dual[k], sdca_epochs[k], t_sdca[k], t_comm[k], t_admm[k]);
			}

			// check stopping criteria
			if (err_primal[k] <= params.eps_primal && err_dual[k] <= params.eps_dual)
			{
				n_iters = k;
				break;
			}
		}

		//---------------------------------------------------------------------
		// write progress record in a file
		if (mpi_rank == 0) {
			std::ofstream ofs(params.filename, std::ofstream::out);
			ofs << "iter epochs      obj        err_conv    err_feas   sdca  t_sdca   t_comm     time" << std::endl;
			for (int k = 0; k < primal_obj.size(); k++) {
				formatted_output(ofs, k, n_epochs[k], primal_obj[k], err_primal[k], err_dual[k], sdca_epochs[k], t_sdca[k], t_comm[k], t_admm[k]);
			}
			ofs.close();
		}

		// return final iterate and number of iterations
		wio.copy(v);
		return n_iters;
	}
}